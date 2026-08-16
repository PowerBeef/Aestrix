import Foundation
import Observation
import UIKit
import Photos
import ImarelloCore
#if !targetEnvironment(simulator)
import ImarelloRuntime
import ImarelloWeights
#endif

/// Main-actor UI state. Never holds `MLXArray`.
@MainActor
@Observable
final class GenerationModel {
    enum RunGate: Equatable {
        case simulator
        case missingWeights
        case ready
    }

    static let foxPlaceholder =
        "A red fox in a snowy forest at sunrise, photorealistic, golden rim light, shallow depth of field."

    var prompt = GenerationModel.foxPlaceholder
    var side = 512
    var seed: UInt64 = 42
    /// Editable digits for the seed field. `TextField(value:format:)` does not
    /// write back while a number pad is focused, so Generate can run the old seed.
    var seedText = "42"
    var lastImage: UIImage?
    var lastImageURL: URL?
    var lastSide = 512
    /// Seed that produced `lastImage`. Nil until a generate/edit finishes.
    var lastSeed: UInt64?
    var phase: PipelineProgress?
    var isRunning = false
    var errorMessage: String?
    var saveMessage: String?
    #if targetEnvironment(simulator)
    var gate: RunGate = .simulator
    #else
    var gate: RunGate = .missingWeights
    #endif

    var canGenerate: Bool {
        !isRunning
            && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && (gate == .ready || gate == .simulator)
    }

    var canEdit: Bool { canGenerate && lastImageURL != nil }

    /// Device-only. Simulator is a UI preview — no gate banner.
    var bannerText: String? {
        switch gate {
        case .missingWeights:
            return "Weights are not in the app container yet. See Docs/IOS.md."
        case .simulator, .ready:
            return nil
        }
    }

    var phaseLabel: String? {
        guard let phase else { return nil }
        switch phase.phase {
        case .preparing: return "Preparing…"
        case .encodingText: return "Encoding prompt…"
        case .encodingImage: return "Encoding image…"
        case .denoising:
            if phase.totalSteps > 0 {
                return "Denoising \(phase.step)/\(phase.totalSteps)"
            }
            return "Denoising…"
        case .decoding: return "Decoding…"
        case .finished: return "Done"
        }
    }

    var expectedModelsDirectory: URL {
        AppCache.resolvedDirectory("models")
    }

    private var runTask: Task<Void, Never>?

    #if !targetEnvironment(simulator)
    private var pipeline: ImarelloPipeline?
    #endif

    func refreshGate() {
        #if targetEnvironment(simulator)
        gate = .simulator
        #else
        let config = ImarelloConfig.autoDetectingTier()
        let klein = ModelPaths.resolveIfPresent(config: config)
        let small = ModelPaths.resolveSmallDecoderIfPresent(config: config)
        gate = (klein != nil && small != nil) ? .ready : .missingWeights
        #endif
        try? DeviceHarnessPaths.ensureDirectories()
    }

    /// Claim at most one inbox job. Safe to call from `.task` / `scenePhase`.
    func pollHarnessInbox() {
        guard !isRunning else { return }
        do {
            try DeviceHarnessPaths.ensureDirectories()
            let inbox = try FileManager.default.contentsOfDirectory(
                at: DeviceHarnessPaths.inbox(),
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            )
            .filter { $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            guard let url = inbox.first else { return }
            try claimAndRunHarnessJob(at: url)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Parse the seed field into `seed`. Call before generate/edit and when the
    /// number pad resigns — otherwise the last committed value is what runs.
    @discardableResult
    func commitSeedText() -> UInt64 {
        let digits = seedText.filter(\.isNumber)
        if let value = UInt64(digits), !digits.isEmpty {
            seed = value
        }
        seedText = String(seed)
        return seed
    }

    func randomizeSeed() {
        seed = UInt64.random(in: 0...9_999_999)
        seedText = String(seed)
    }

    func generate() {
        commitSeedText()
        guard canGenerate else { return }
        startRun { await self.performGenerate() }
    }

    func editLast() {
        commitSeedText()
        guard canEdit else { return }
        startRun { await self.performEdit() }
    }

    func cancel() {
        runTask?.cancel()
        runTask = nil
        isRunning = false
        phase = nil
    }

    func saveToPhotos() async {
        guard let url = lastImageURL else { return }
        saveMessage = nil
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else {
            errorMessage = "Photos access was denied. Enable it in Settings to save."
            return
        }
        do {
            try await PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAssetFromImage(atFileURL: url)
            }
            saveMessage = "Saved to Photos"
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func claimAndRunHarnessJob(at inboxURL: URL) throws {
        let data = try Data(contentsOf: inboxURL)
        let job = try DeviceHarnessPaths.jsonDecoder.decode(DeviceHarnessJob.self, from: data)
        let running = DeviceHarnessPaths.runningFile(id: job.id)
        try? FileManager.default.removeItem(at: running)
        try FileManager.default.moveItem(at: inboxURL, to: running)

        #if targetEnvironment(simulator)
        try writeHarnessResult(.skippedSimulator(job: job))
        try? FileManager.default.removeItem(at: running)
        #else
        prompt = job.prompt
        side = job.width
        seed = job.seed
        seedText = String(job.seed)
        startRun { await self.performHarnessJob(job, runningURL: running) }
        #endif
    }

    #if !targetEnvironment(simulator)
    private func performHarnessJob(_ job: DeviceHarnessJob, runningURL: URL) async {
        let started = Date()
        let iso = ISO8601DateFormatter().string(from: started)
        defer {
            finishRun()
            try? FileManager.default.removeItem(at: runningURL)
        }
        do {
            try job.validate(hasLastImage: lastImageURL != nil)
            try Task.checkCancellation()
            try await ensureReady()
            let metalNote = MetallibVerification.resolveFromBundles()
                .map { MetallibVerification.verify(url: $0).note }

            let url: URL
            switch job.mode {
            case .t2i:
                let out = try outputURL(prefix: "t2i")
                url = try await pipelineOrThrow().generate(
                    T2IRequest(
                        prompt: job.prompt,
                        width: job.width,
                        height: job.height,
                        steps: job.steps,
                        seed: job.seed,
                        outputURL: out,
                        textTokens: job.textTokens,
                        embedCache: true
                    ),
                    onProgress: { progress in
                        Task { @MainActor in
                            self.phase = progress
                        }
                    }
                )
            case .i2i:
                guard let source = lastImageURL else {
                    throw ImarelloError.imageLoadFailed(
                        path: "<last-in-app>",
                        reason: "i2i harness job needs a last generated PNG"
                    )
                }
                let out = try outputURL(prefix: "i2i")
                url = try await pipelineOrThrow().edit(
                    I2IRequest(
                        prompt: job.prompt,
                        imageURL: source,
                        strength: job.strength,
                        width: job.width,
                        height: job.height,
                        steps: job.steps,
                        seed: job.seed,
                        outputURL: out,
                        identity: .disabled,
                        textTokens: job.textTokens,
                        embedCache: true
                    ),
                    onProgress: { progress in
                        Task { @MainActor in
                            self.phase = progress
                        }
                    }
                )
            }
            applyResult(url: url, side: job.width)
            try writeHarnessResult(
                DeviceHarnessResult(
                    id: job.id,
                    status: .ok,
                    pngRelativePath: DeviceHarnessPaths.pngContainerPath(filename: url.lastPathComponent),
                    width: job.width,
                    height: job.height,
                    seed: job.seed,
                    elapsedSec: Date().timeIntervalSince(started),
                    startedAt: iso,
                    metallibNote: metalNote
                )
            )
        } catch is CancellationError {
            try? writeHarnessResult(
                DeviceHarnessResult(
                    id: job.id,
                    status: .failed,
                    error: "cancelled",
                    width: job.width,
                    height: job.height,
                    seed: job.seed,
                    elapsedSec: Date().timeIntervalSince(started),
                    startedAt: iso
                )
            )
        } catch {
            errorMessage = error.localizedDescription
            try? writeHarnessResult(
                DeviceHarnessResult(
                    id: job.id,
                    status: .failed,
                    error: error.localizedDescription,
                    width: job.width,
                    height: job.height,
                    seed: job.seed,
                    elapsedSec: Date().timeIntervalSince(started),
                    startedAt: iso
                )
            )
        }
    }
    #endif

    private func writeHarnessResult(_ result: DeviceHarnessResult) throws {
        try DeviceHarnessPaths.ensureDirectories()
        let data = try DeviceHarnessPaths.jsonEncoder.encode(result)
        try data.write(to: DeviceHarnessPaths.doneFile(id: result.id), options: .atomic)
    }

    private func startRun(_ work: @escaping @MainActor () async -> Void) {
        errorMessage = nil
        saveMessage = nil
        isRunning = true
        phase = PipelineProgress(phase: .preparing)
        runTask?.cancel()
        runTask = Task { @MainActor in
            await work()
        }
    }

    #if targetEnvironment(simulator)
    private func performGenerate() async {
        isRunning = false
        phase = nil
    }

    private func performEdit() async {
        isRunning = false
        phase = nil
    }
    #else
    private func performGenerate() async {
        defer { finishRun() }
        do {
            try Task.checkCancellation()
            try await ensureReady()
            let out = try outputURL(prefix: "t2i")
            let url = try await pipelineOrThrow().generate(
                T2IRequest(
                    prompt: prompt,
                    width: side,
                    height: side,
                    steps: 4,
                    seed: seed,
                    outputURL: out,
                    textTokens: .auto,
                    embedCache: true
                ),
                onProgress: { progress in
                    Task { @MainActor in
                        self.phase = progress
                    }
                }
            )
            applyResult(url: url, side: side)
        } catch is CancellationError {
            // cancelled
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func performEdit() async {
        defer { finishRun() }
        guard let source = lastImageURL else { return }
        do {
            try Task.checkCancellation()
            try await ensureReady()
            let canvas = lastSide
            let out = try outputURL(prefix: "i2i")
            let url = try await pipelineOrThrow().edit(
                I2IRequest(
                    prompt: prompt,
                    imageURL: source,
                    strength: 0.8,
                    width: canvas,
                    height: canvas,
                    steps: 4,
                    seed: seed,
                    outputURL: out,
                    identity: .disabled,
                    textTokens: .auto,
                    embedCache: true
                ),
                onProgress: { progress in
                    Task { @MainActor in
                        self.phase = progress
                    }
                }
            )
            applyResult(url: url, side: canvas)
        } catch is CancellationError {
            // cancelled
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func ensureReady() async throws {
        refreshGate()
        // Pipeline.snapshot is fixed at init. A harness/Generate before the
        // weight copy leaves a resident actor with snapshot == nil even after
        // files appear and the gate flips to .ready.
        if let existing = pipeline, await existing.hasLocalSnapshot == false {
            pipeline = nil
        }
        if pipeline == nil {
            pipeline = ImarelloPipeline(config: .autoDetectingTier())
        }
        guard let metal = MetallibVerification.resolveFromBundles() else {
            throw ImarelloError.notImplemented(
                "MLX metallib missing from the app bundle (no mlx-swift Cmlx library)"
            )
        }
        let check = MetallibVerification.verify(url: metal)
        if !check.productReady {
            throw ImarelloError.notImplemented("MLX metallib is not product-ready: \(check.note)")
        }
        guard gate == .ready else {
            throw ImarelloError.weightsNotFound(
                modelID: WeightPreset.bits4.defaultModelID,
                path: expectedModelsDirectory.path
            )
        }
        if await pipeline?.hasLocalSnapshot != true {
            throw ImarelloError.weightsNotFound(
                modelID: WeightPreset.bits4.defaultModelID,
                path: expectedModelsDirectory.path
            )
        }
    }

    private func pipelineOrThrow() throws -> ImarelloPipeline {
        guard let pipeline else {
            throw ImarelloError.notImplemented("Pipeline was not created")
        }
        return pipeline
    }

    private func applyResult(url: URL, side: Int) {
        lastImageURL = url
        lastImage = UIImage(contentsOfFile: url.path)
        lastSide = side
        lastSeed = seed
    }

    private func outputURL(prefix: String) throws -> URL {
        let dir = AppCache.directory("outputs")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("\(prefix)-\(Int(Date().timeIntervalSince1970)).png")
    }
    #endif

    private func finishRun() {
        isRunning = false
        if phase?.phase != .finished {
            phase = nil
        }
        runTask = nil
    }
}
