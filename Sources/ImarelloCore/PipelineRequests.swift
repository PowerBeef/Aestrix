import Foundation

/// How many text tokens are passed to the DiT.
///
/// FLUX.2 joint attention has no text mask, so padding tokens participate in attention.
/// `.full512` is the **product default** (the distillation regime). `.auto` trims the
/// padded 512-token sequence to the real prompt length (rounded up to a multiple of 8) —
/// faster but weaker conditioning; it was default for one day and reverted 2026-08-16
/// after a vision regression. See `Docs/TEXT_TOKENS.md`.
public enum TextTokenMode: String, Sendable, Codable {
    /// Full 512 padded tokens (product default; `--text-tokens 512`).
    case full512 = "512"
    /// Trim to real prompt length rounded up to a multiple of 8 (opt-in speed path).
    case auto
}

public struct T2IRequest: Sendable {
    public var prompt: String
    public var width: Int
    public var height: Int
    public var steps: Int
    /// Currently ignored: distilled Klein runs guidance-free and the pipeline
    /// hardcodes `nil` (the DiT is built without guidance embeds).
    public var guidance: Float
    public var seed: UInt64?
    /// Destination PNG path. If nil, pipeline chooses a timestamped file under Caches/Imarello/outputs.
    public var outputURL: URL?
    /// Text tokens passed to the DiT (default: `.full512`, the product path).
    public var textTokens: TextTokenMode
    /// Cache prompt embeddings on disk and skip TE load+encode on repeat prompts.
    /// Default **false** so library/bench behavior is unchanged; the CLI enables it.
    public var embedCache: Bool

    /// EXPERIMENT (ENGINE_RESEARCH.md §5.1 R4): what the pad rows contain.
    public enum PadContentMode: String, Sendable, Codable {
        /// Prompt-conditioned pads (product behavior; how the model was distilled).
        case prompt
        /// Position-matched pads from an empty-prompt encode (content-vs-count test).
        case clean
    }

    public init(
        prompt: String,
        width: Int = 1024,
        height: Int = 1024,
        steps: Int = 4,
        guidance: Float = 1.0,
        seed: UInt64? = nil,
        outputURL: URL? = nil,
        textTokens: TextTokenMode = .full512,
        embedCache: Bool = false,
        padContent: PadContentMode = .prompt,
        padKeep: Int? = nil,
        padBias: Bool = false
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
        self.padContent = padContent
        self.padKeep = padKeep
        self.padBias = padBias
    }

    /// EXPERIMENT knobs (ENGINE_RESEARCH.md §5.1). Defaults are product behavior.
    public var padContent: PadContentMode
    /// R3: trim the text window to real tokens + this many pads (needs `.full512`).
    public var padKeep: Int?
    /// R3: add `ln(removed/kept)` to kept-pad attention logits (denominator compensation).
    public var padBias: Bool
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
    /// Currently ignored: distilled Klein runs guidance-free and the pipeline
    /// hardcodes `nil` (the DiT is built without guidance embeds).
    public var guidance: Float
    public var seed: UInt64?
    public var outputURL: URL?
    /// Tier-B identity stack (reference latents, face mask, schedule curve). Default off.
    public var identity: IdentityPreserveConfig
    /// Text tokens passed to the DiT (default: `.full512`, the product path).
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
