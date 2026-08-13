/// FLUX.2-klein-4B distilled architecture constants.
public enum ModelConstants {
    public static let numDoubleBlocks = 5
    public static let numSingleBlocks = 20
    public static let numAttentionHeads = 24
    public static let attentionHeadDim = 128
    public static let innerDim = numAttentionHeads * attentionHeadDim // 3072
    public static let jointAttentionDim = 7680 // 3 × 2560 Qwen layers 9/18/27
    public static let inChannels = 128
    public static let maxSequenceLength = 512
    public static let ropeTheta: Float = 2000
    public static let ropeAxesDims: [Int] = [32, 32, 32, 32]
    public static let mlpRatio: Float = 3.0
    public static let timestepEmbedChannels = 256
    public static let vaeScaleFactor = 8
    public static let defaultSteps = 4
    public static let defaultGuidance: Float = 1.0
    public static let numTrainTimesteps = 1000
    public static let textEncoderLayers: [Int] = [9, 18, 27]
    public static let qwenHiddenSize = 2560
}
