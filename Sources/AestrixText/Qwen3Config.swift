import Foundation
import AestrixCore

/// Qwen3-4B configuration as used by FLUX.2-klein-4B text encoder packs.
public struct Qwen3Config: Sendable {
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var maxPositionEmbeddings: Int
    /// HF `hidden_states` indices to tap (embeddings = 0, after layer 0 = 1, …).
    public var tapHiddenStateIndices: [Int]

    public init(
        vocabSize: Int = 151_936,
        hiddenSize: Int = ModelConstants.qwenHiddenSize,
        intermediateSize: Int = 9_728,
        numHiddenLayers: Int = 36,
        numAttentionHeads: Int = 32,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 1_000_000,
        maxPositionEmbeddings: Int = 40_960,
        tapHiddenStateIndices: [Int] = ModelConstants.textEncoderLayers
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.tapHiddenStateIndices = tapHiddenStateIndices
    }

    public static let klein4B = Qwen3Config()

    /// Layers required for max tap index (HF index 27 ⇒ 0-based layers 0…26).
    public var layersNeededForTaps: Int {
        (tapHiddenStateIndices.max() ?? 0)
    }

    public var jointAttentionDim: Int {
        tapHiddenStateIndices.count * hiddenSize
    }
}
