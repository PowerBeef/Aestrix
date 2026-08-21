import Foundation
import ImarelloCore
import ImarelloWeights
import ImarelloRuntime

/// Whole-pipeline adapter for the current Direct engine.
///
/// The adapter itself is intentionally lightweight: construction retains no
/// model weights or Metal command resources. Until the V2 native cache and
/// token-mode paths land, capability selection keeps unsupported requests on
/// the built-in runtime path before either engine loads a stage.
public final class DirectT2IBackend: TextToImageGenerationBackend {
    public let identifier = "direct-v2-shell"

    private let snapshot: ModelSnapshot
    private let artifacts: DirectEngineArtifacts
    private let config: ImarelloConfig
    private let useNAXQmm: Bool

    public init(
        snapshot: ModelSnapshot,
        artifacts: DirectEngineArtifacts,
        config: ImarelloConfig,
        useNAXQmm: Bool = false
    ) {
        self.snapshot = snapshot
        self.artifacts = artifacts
        self.config = config
        self.useNAXQmm = useNAXQmm
    }

    public func supports(_ request: T2IRequest) -> Bool {
        Self.supportsRequest(request)
    }

    static func supportsRequest(_ request: T2IRequest) -> Bool {
        request.steps > 0
            && request.seed != nil
            && request.outputURL != nil
            && request.textTokens == .full512
            && !request.embedCache
            && request.padContent == .prompt
            && request.padKeep == nil
            && !request.padBias
    }

    public func generate(
        _ request: T2IRequest,
        onProgress: (@Sendable (PipelineProgress) -> Void)?,
        trace: PipelineTrace?
    ) throws -> URL {
        guard supports(request),
              let seed = request.seed,
              let outputURL = request.outputURL
        else {
            throw ImarelloError.invalidRequest(
                "Direct V2 backend was invoked for an unsupported request")
        }

        let pipeline = DirectPipeline(
            snapshot: snapshot, artifacts: artifacts, config: config)
        pipeline.useNAXQmm = useNAXQmm
        _ = try pipeline.generateSynchronously(
            prompt: request.prompt,
            width: request.width,
            height: request.height,
            steps: request.steps,
            seed: seed,
            outputURL: outputURL,
            onProgress: onProgress,
            trace: trace)
        return outputURL
    }
}
