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
    var lastImage: UIImage?
    var lastImageURL: URL?
    var lastSide = 512
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
    }

    func generate() {
        guard canGenerate else { return }
        startRun { await self.performGenerate() }
    }

    func editLast() {
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
        if pipeline == nil {
            pipeline = ImarelloPipeline(config: .autoDetectingTier())
        }
        if let metal = MetallibVerification.resolveFromBundles() {
            let check = MetallibVerification.verify(url: metal)
            if !check.productReady {
                throw ImarelloError.notImplemented("MLX metallib is not product-ready: \(check.note)")
            }
        }
        refreshGate()
        guard gate == .ready else {
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
