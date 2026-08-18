import Foundation
import Observation
import ImarelloCore
#if !targetEnvironment(simulator)
import ImarelloRuntime
import ImarelloWeights
#endif

/// Pipeline lifecycle: the gate, readiness, and the two generation calls.
/// Behavior is a faithful extraction of the original GenerationModel —
/// including the pipeline-recreate when weights arrive after first launch.
/// Never holds `MLXArray`; the pipeline actor does.
@MainActor
@Observable
final class GenerationEngine {
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
        return try await pipelineOrThrow().generate(
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
            onProgress: { progress in
                Task { @MainActor in onProgress(progress) }
            }
        )
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
        return try await pipelineOrThrow().edit(
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
            onProgress: { progress in
                Task { @MainActor in onProgress(progress) }
            }
        )
    }

    func ensureReady() async throws {
        refreshGate()
        // Pipeline.snapshot is fixed at init. A run before the weight copy
        // leaves a resident actor with snapshot == nil even after files appear.
        if let existing = pipeline, await existing.hasLocalSnapshot == false {
            pipeline = nil
        }
        if pipeline == nil {
            pipeline = ImarelloPipeline(config: .autoDetectingTier())
        }
        guard let metal = MetallibVerification.resolveFromBundles() else {
            throw ImarelloError.notImplemented(
                "MLX metallib missing from the app bundle (no mlx-swift Cmlx library)")
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
