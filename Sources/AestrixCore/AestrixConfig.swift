import Foundation

public struct AestrixConfig: Sendable {
    /// Hugging Face repo id for a pre-quant package (never bf16).
    public var modelID: String
    public var weightPreset: WeightPreset
    public var memoryPolicy: MemoryPolicy
    public var tier: DeviceTier
    /// Hard clamp for max(width, height).
    public var maxSide: Int
    public var modelsDirectory: URL?

    public init(
        modelID: String? = nil,
        weightPreset: WeightPreset = .bits4,
        memoryPolicy: MemoryPolicy = .staged,
        tier: DeviceTier = .detect(),
        maxSide: Int? = nil,
        modelsDirectory: URL? = nil
    ) {
        self.weightPreset = weightPreset
        self.modelID = modelID ?? weightPreset.defaultModelID
        self.memoryPolicy = memoryPolicy
        self.tier = tier
        self.maxSide = maxSide ?? tier.defaultMaxSide
        self.modelsDirectory = modelsDirectory
    }

    public static func autoDetectingTier() -> AestrixConfig {
        AestrixConfig()
    }
}
