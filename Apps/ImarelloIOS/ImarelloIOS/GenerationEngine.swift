import Foundation
import Observation
import ImarelloCore
#if !targetEnvironment(simulator)
import ImarelloRuntime
import ImarelloWeights
import ImarelloDirect
#endif

@MainActor
protocol GenerationServing: AnyObject {
    var gate: GenerationEngine.RunGate { get }
    var expectedModelsDirectory: URL { get }
    var allowsGenerationInCurrentProcess: Bool { get }
    func refreshGate()
    func generate(
        prompt: String, side: Int, seed: UInt64,
        steps: Int, textTokens: TextTokenMode,
        onProgress: @escaping @MainActor (PipelineProgress) -> Void
    ) async throws -> URL
    func edit(
        source: URL, prompt: String, side: Int, seed: UInt64,
        strength: Float, steps: Int, textTokens: TextTokenMode,
        onProgress: @escaping @MainActor (PipelineProgress) -> Void
    ) async throws -> URL
}

extension GenerationServing {
    func generate(
        prompt: String, side: Int, seed: UInt64,
        steps: Int = 4, textTokens: TextTokenMode = .full512,
        onProgress: @escaping @MainActor (PipelineProgress) -> Void
    ) async throws -> URL {
        try await generate(
            prompt: prompt, side: side, seed: seed, steps: steps,
            textTokens: textTokens, onProgress: onProgress
        )
    }

    func edit(
        source: URL, prompt: String, side: Int, seed: UInt64,
        strength: Float = 0.8, steps: Int = 4,
        textTokens: TextTokenMode = .full512,
        onProgress: @escaping @MainActor (PipelineProgress) -> Void
    ) async throws -> URL {
        try await edit(
            source: source, prompt: prompt, side: side, seed: seed,
            strength: strength, steps: steps, textTokens: textTokens,
            onProgress: onProgress
        )
    }
}

/// Pipeline lifecycle: the gate, readiness, and the two generation calls.
/// Behavior is a faithful extraction of the original GenerationModel —
/// including the pipeline-recreate when weights arrive after first launch.
/// Never holds `MLXArray`; the pipeline actor does.
@MainActor
@Observable
final class GenerationEngine: GenerationServing {
    enum RunGate: Equatable {
        case simulator
        case missingWeights
        case ready
    }

    #if targetEnvironment(simulator)
    private(set) var gate: RunGate = .simulator
    #else
    private(set) var gate: RunGate = .missingWeights
    #endif

    var expectedModelsDirectory: URL {
        AppCache.resolvedDirectory("models")
    }

    var allowsGenerationInCurrentProcess: Bool {
        #if targetEnvironment(simulator)
        false
        #else
        true
        #endif
    }

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

    // MARK: - Generation (device only; Simulator keeps these as chrome no-ops)

    #if targetEnvironment(simulator)
    func generate(
        prompt: String, side: Int, seed: UInt64,
        steps: Int = 4, textTokens: TextTokenMode = .full512,
        onProgress: @escaping @MainActor (PipelineProgress) -> Void
    ) async throws -> URL {
        throw CancellationError()
    }

    func edit(
        source: URL, prompt: String, side: Int, seed: UInt64,
        strength: Float = 0.8, steps: Int = 4, textTokens: TextTokenMode = .full512,
        onProgress: @escaping @MainActor (PipelineProgress) -> Void
    ) async throws -> URL {
        throw CancellationError()
    }
    #else
    /// Defaults are the product-locked in-app values; harness jobs pass the
    /// values from their job JSON so `--steps`/`--text-tokens` actually apply.
    func generate(
        prompt: String, side: Int, seed: UInt64,
        steps: Int = 4, textTokens: TextTokenMode = .full512,
        onProgress: @escaping @MainActor (PipelineProgress) -> Void
    ) async throws -> URL {
        try Task.checkCancellation()
        try await ensureReady()
        let out = try outputURL(prefix: "t2i")
        return try await withOrderedProgress(onProgress) { emit in
            try await pipelineOrThrow().generate(
                T2IRequest(
                    prompt: prompt,
                    width: side,
                    height: side,
                    steps: steps,
                    seed: seed,
                    outputURL: out,
                    textTokens: textTokens,
                    embedCache: true
                ),
                onProgress: emit
            )
        }
    }

    /// In-app I2I at the product-locked strength 0.8 (shown in the UI).
    func edit(
        source: URL, prompt: String, side: Int, seed: UInt64,
        strength: Float = 0.8, steps: Int = 4, textTokens: TextTokenMode = .full512,
        onProgress: @escaping @MainActor (PipelineProgress) -> Void
    ) async throws -> URL {
        try Task.checkCancellation()
        try await ensureReady()
        let out = try outputURL(prefix: "i2i")
        return try await withOrderedProgress(onProgress) { emit in
            try await pipelineOrThrow().edit(
                I2IRequest(
                    prompt: prompt,
                    imageURL: source,
                    strength: strength,
                    width: side,
                    height: side,
                    steps: steps,
                    seed: seed,
                    outputURL: out,
                    identity: .disabled,
                    textTokens: textTokens,
                    embedCache: true
                ),
                onProgress: emit
            )
        }
    }

    func ensureReady() async throws {
        refreshGate()
        guard let metal = MetallibVerification.resolveFromBundles() else {
            throw ImarelloError.metallibNotReady(
                "full mlx-swift 0.32.1 no-JIT iphoneos metallib is missing from the app bundle"
            )
        }
        let check = MetallibVerification.verify(url: metal)
        guard check.productReady else {
            throw ImarelloError.metallibNotReady(check.note)
        }
        // Pipeline.snapshot is fixed at init. A run before the weight copy
        // leaves a resident actor with snapshot == nil even after files appear.
        if let existing = pipeline, await existing.hasLocalSnapshot == false {
            pipeline = nil
        }
        if pipeline == nil {
            let config = ImarelloConfig.autoDetectingTier()
            guard let executable = Bundle.main.executableURL else {
                throw ImarelloError.metallibNotReady(
                    "the signed app executable URL is unavailable")
            }
            let artifacts = try DirectEngineArtifacts.resolve(relativeTo: executable)
            let snapshot = try ModelPaths.resolveOrThrow(config: config)
            let created = ImarelloPipeline(config: config)
            let backend = DirectT2IBackend(
                snapshot: snapshot,
                artifacts: artifacts,
                config: config
            )
            await created.setTextToImageBackend(backend)
            pipeline = created
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

    /// `AsyncStream.Continuation` is thread-safe and preserves yield order.
    /// A fresh stream identifies each run; finishing it rejects late callbacks
    /// from a cancelled or superseded pipeline invocation.
    private func withOrderedProgress<T>(
        _ onProgress: @escaping @MainActor (PipelineProgress) -> Void,
        operation: (@escaping @Sendable (PipelineProgress) -> Void) async throws -> T
    ) async throws -> T {
        let pair = AsyncStream.makeStream(
            of: PipelineProgress.self,
            bufferingPolicy: .unbounded
        )
        let delivery = Task { @MainActor in
            for await progress in pair.stream {
                guard !Task.isCancelled else { break }
                onProgress(progress)
            }
        }
        do {
            let result = try await operation { progress in
                pair.continuation.yield(progress)
            }
            pair.continuation.finish()
            await delivery.value
            return result
        } catch {
            pair.continuation.finish()
            delivery.cancel()
            await delivery.value
            throw error
        }
    }

    private func outputURL(prefix: String) throws -> URL {
        let dir = AppCache.directory("outputs")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // Millisecond suffix: an epoch-seconds-only stem collides when two runs
        // land in the same second, silently overwriting the earlier PNG.
        let now = Date().timeIntervalSince1970
        let millis = Int((now - now.rounded(.down)) * 1000)
        return dir.appendingPathComponent(
            "\(prefix)-\(Int(now))-\(String(format: "%03d", millis)).png")
    }
    #endif
}
