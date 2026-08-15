import Foundation

public struct ImarelloConfig: Sendable {
    /// Hugging Face repo id for a pre-quant package (never bf16).
    public var modelID: String
    /// Hugging Face git revision (commit SHA) this config expects.
    public var revision: String
    public var weightPreset: WeightPreset
    public var memoryPolicy: MemoryPolicy
    public var tier: DeviceTier
    /// Hard clamp for max(width, height).
    public var maxSide: Int
    public var modelsDirectory: URL?
    /// T2I / I2I decode graph. Encoder (I2I) is always the full klein AE.
    public var vaeDecoderVariant: VAEDecoderVariant

    public init(
        modelID: String? = nil,
        weightPreset: WeightPreset = .bits4,
        memoryPolicy: MemoryPolicy = .staged,
        tier: DeviceTier = .detect(),
        maxSide: Int? = nil,
        modelsDirectory: URL? = nil,
        revision: String? = nil,
        vaeDecoderVariant: VAEDecoderVariant = .smallDecoder
    ) {
        self.weightPreset = weightPreset
        self.modelID = modelID ?? weightPreset.defaultModelID
        self.revision = revision ?? weightPreset.pinnedRevision
        self.memoryPolicy = memoryPolicy
        self.tier = tier
        self.maxSide = maxSide ?? tier.defaultMaxSide
        self.modelsDirectory = modelsDirectory
        self.vaeDecoderVariant = vaeDecoderVariant
    }

    public static func autoDetectingTier() -> ImarelloConfig {
        ImarelloConfig()
    }

    /// Point this config at a product weight preset (repo + pinned revision).
    public mutating func apply(preset: WeightPreset) {
        weightPreset = preset
        modelID = preset.defaultModelID
        revision = preset.pinnedRevision
    }

    /// `hf download` that fetches the pinned revision into the default Imarello cache.
    public var downloadCommand: String {
        let dest = "~/Library/Caches/Imarello/models/\(modelID.replacingOccurrences(of: "/", with: "--"))"
        return "hf download \(modelID) --revision \(revision) --local-dir \(dest)"
    }
}
