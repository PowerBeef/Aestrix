import Foundation

/// How many text tokens are passed to the DiT.
///
/// FLUX.2 joint attention has no text mask, so padding tokens participate in attention.
/// `.auto` trims the padded 512-token sequence to the real prompt length (rounded up to
/// a multiple of 8) — a numerics-changing **experiment** that cuts joint sequence length.
public enum TextTokenMode: String, Sendable, Codable {
    /// Full 512 padded tokens (documented product default).
    case full512 = "512"
    /// Trim to real prompt length rounded up to a multiple of 8 (experimental).
    case auto
}

public struct T2IRequest: Sendable {
    public var prompt: String
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidance: Float
    public var seed: UInt64?
    /// Destination PNG path. If nil, pipeline chooses a timestamped file under Caches/Imarello/outputs.
    public var outputURL: URL?
    /// Text tokens passed to the DiT (default: full 512 padded).
    public var textTokens: TextTokenMode
    /// Cache prompt embeddings on disk and skip TE load+encode on repeat prompts.
    /// Default **false** so library/bench behavior is unchanged; the CLI enables it.
    public var embedCache: Bool

    public init(
        prompt: String,
        width: Int = 1024,
        height: Int = 1024,
        steps: Int = 4,
        guidance: Float = 1.0,
        seed: UInt64? = nil,
        outputURL: URL? = nil,
        textTokens: TextTokenMode = .full512,
        embedCache: Bool = false
    ) {
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.outputURL = outputURL
        self.textTokens = textTokens
        self.embedCache = embedCache
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
    /// Tier-B identity stack (reference latents, face mask, schedule curve). Default off.
    public var identity: IdentityPreserveConfig
    /// Text tokens passed to the DiT (default: full 512 padded).
    public var textTokens: TextTokenMode
    /// Cache prompt embeddings on disk and skip TE load+encode on repeat prompts.
    public var embedCache: Bool

    public init(
        prompt: String,
        imageURL: URL,
        strength: Float = 0.8,
        width: Int? = nil,
        height: Int? = nil,
        steps: Int = 4,
        guidance: Float = 1.0,
        seed: UInt64? = nil,
        outputURL: URL? = nil,
        identity: IdentityPreserveConfig = .disabled,
        textTokens: TextTokenMode = .full512,
        embedCache: Bool = false
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
        self.identity = identity
        self.textTokens = textTokens
        self.embedCache = embedCache
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
