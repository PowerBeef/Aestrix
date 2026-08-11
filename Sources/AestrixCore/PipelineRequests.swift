import Foundation

public struct T2IRequest: Sendable {
    public var prompt: String
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidance: Float
    public var seed: UInt64?
    /// Destination PNG path. If nil, pipeline chooses a timestamped file under Caches/Aestrix/outputs.
    public var outputURL: URL?

    public init(
        prompt: String,
        width: Int = 1024,
        height: Int = 1024,
        steps: Int = 4,
        guidance: Float = 1.0,
        seed: UInt64? = nil,
        outputURL: URL? = nil
    ) {
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.outputURL = outputURL
    }
}

public struct I2IRequest: Sendable {
    public var prompt: String
    /// Local file URL for the reference image (v1).
    public var imageURL: URL
    /// How hard to re-noise the source before denoising. **1.0** ≈ T2I from noise; **0.3–0.7** typical edits.
    public var strength: Float
    public var width: Int?
    public var height: Int?
    public var steps: Int
    public var guidance: Float
    public var seed: UInt64?
    public var outputURL: URL?

    public init(
        prompt: String,
        imageURL: URL,
        strength: Float = 0.8,
        width: Int? = nil,
        height: Int? = nil,
        steps: Int = 4,
        guidance: Float = 1.0,
        seed: UInt64? = nil,
        outputURL: URL? = nil
    ) {
        self.prompt = prompt
        self.imageURL = imageURL
        self.strength = strength
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.outputURL = outputURL
    }
}

public enum PipelinePhase: String, Sendable {
    case preparing
    case encodingText
    case encodingImage
    case denoising
    case decoding
    case finished
}

public struct PipelineProgress: Sendable {
    public var phase: PipelinePhase
    public var step: Int
    public var totalSteps: Int

    public init(phase: PipelinePhase, step: Int = 0, totalSteps: Int = 0) {
        self.phase = phase
        self.step = step
        self.totalSteps = totalSteps
    }
}
