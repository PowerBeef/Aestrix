import Foundation
import MLX
import MLXNN
import AestrixCore

/// FLUX.2 Klein MMDiT: 5 double + 20 single stream blocks.
public final class Flux2Transformer: Module {
    public let innerDim: Int
    public let outChannels: Int
    public let numLayers: Int
    public let numSingleLayers: Int
    public let jointAttentionDim: Int

    let posEmbed: Flux2PosEmbed
    @ModuleInfo(key: "time_guidance_embed") var timeGuidanceEmbed: Flux2TimestepGuidanceEmbeddings
    @ModuleInfo(key: "double_stream_modulation_img") var doubleStreamModulationImg: Flux2Modulation
    @ModuleInfo(key: "double_stream_modulation_txt") var doubleStreamModulationTxt: Flux2Modulation
    @ModuleInfo(key: "single_stream_modulation") var singleStreamModulation: Flux2Modulation
    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [Flux2TransformerBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks: [Flux2SingleTransformerBlock]
    @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public init(
        inChannels: Int = ModelConstants.inChannels,
        outChannels: Int? = nil,
        numLayers: Int = ModelConstants.numDoubleBlocks,
        numSingleLayers: Int = ModelConstants.numSingleBlocks,
        attentionHeadDim: Int = ModelConstants.attentionHeadDim,
        numAttentionHeads: Int = ModelConstants.numAttentionHeads,
        jointAttentionDim: Int = ModelConstants.jointAttentionDim,
        timestepGuidanceChannels: Int = ModelConstants.timestepEmbedChannels,
        mlpRatio: Float = ModelConstants.mlpRatio,
        axesDimsRope: [Int] = ModelConstants.ropeAxesDims,
        ropeTheta: Float = ModelConstants.ropeTheta,
        guidanceEmbeds: Bool = false
    ) {
        self.outChannels = outChannels ?? inChannels
        self.innerDim = numAttentionHeads * attentionHeadDim
        self.numLayers = numLayers
        self.numSingleLayers = numSingleLayers
        self.jointAttentionDim = jointAttentionDim

        self.posEmbed = Flux2PosEmbed(theta: ropeTheta, axesDim: axesDimsRope)
        self._timeGuidanceEmbed.wrappedValue = Flux2TimestepGuidanceEmbeddings(
            inChannels: timestepGuidanceChannels,
            embeddingDim: innerDim,
            guidanceEmbeds: guidanceEmbeds
        )
        self._doubleStreamModulationImg.wrappedValue = Flux2Modulation(dim: innerDim, modParamSets: 2)
        self._doubleStreamModulationTxt.wrappedValue = Flux2Modulation(dim: innerDim, modParamSets: 2)
        self._singleStreamModulation.wrappedValue = Flux2Modulation(dim: innerDim, modParamSets: 1)
        self._xEmbedder.wrappedValue = Linear(inChannels, innerDim, bias: false)
        self._contextEmbedder.wrappedValue = Linear(jointAttentionDim, innerDim, bias: false)
        let dim = innerDim
        let nHeads = numAttentionHeads
        let headDim = attentionHeadDim
        let ratio = mlpRatio
        self._transformerBlocks.wrappedValue = (0..<numLayers).map { _ in
            Flux2TransformerBlock(
                dim: dim,
                numAttentionHeads: nHeads,
                attentionHeadDim: headDim,
                mlpRatio: ratio
            )
        }
        self._singleTransformerBlocks.wrappedValue = (0..<numSingleLayers).map { _ in
            Flux2SingleTransformerBlock(
                dim: dim,
                numAttentionHeads: nHeads,
                attentionHeadDim: headDim,
                mlpRatio: ratio
            )
        }
        self._normOut.wrappedValue = AdaLayerNormContinuous(
            embeddingDim: innerDim, conditioningEmbeddingDim: innerDim)
        self._projOut.wrappedValue = Linear(innerDim, outChannels ?? inChannels, bias: false)
        super.init()
    }

    /// Forward pass (fp32 activations recommended for stability).
    public func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        timestep: MLXArray,
        imgIds: MLXArray,
        txtIds: MLXArray,
        guidance: MLXArray? = nil
    ) -> MLXArray {
        var ts = timestep
        if ts.ndim == 0 {
            ts = MLXArray.full([hiddenStates.dim(0)], values: ts)
        }
        ts = ts.asType(hiddenStates.dtype)
        // Scale to [0,1000] if needed
        let tMax = ts.max().item(Float.self)
        if tMax <= 1.0 {
            ts = ts * 1000.0
        }

        var g = guidance
        if var gv = g {
            if gv.ndim == 0 {
                gv = MLXArray.full([hiddenStates.dim(0)], values: gv)
            }
            gv = gv.asType(hiddenStates.dtype)
            let gMax = gv.max().item(Float.self)
            if gMax <= 1.0 {
                gv = gv * 1000.0
            }
            g = gv
        }

        var temb = timeGuidanceEmbed(ts, guidance: g).asType(.float32)

        var h = xEmbedder(hiddenStates)
        var e = contextEmbedder(encoderHiddenStates)

        var iIds = imgIds
        var tIds = txtIds
        if iIds.ndim == 3 { iIds = iIds[0] }
        if tIds.ndim == 3 { tIds = tIds[0] }

        let imgRope = posEmbed(iIds)
        let txtRope = posEmbed(tIds)
        let concatRope: (MLXArray, MLXArray) = (
            concatenated([txtRope.0, imgRope.0], axis: 0),
            concatenated([txtRope.1, imgRope.1], axis: 0)
        )

        let tembImg = doubleStreamModulationImg(temb)
        let tembTxt = doubleStreamModulationTxt(temb)

        for block in transformerBlocks {
            let out = block(
                hiddenStates: h,
                encoderHiddenStates: e,
                tembModParamsImg: tembImg,
                tembModParamsTxt: tembTxt,
                imageRotaryEmb: concatRope
            )
            e = out.encoder
            h = out.hidden
        }

        let txtLen = e.dim(1)
        h = concatenated([e, h], axis: 1)

        let tembSingle = singleStreamModulation(temb)[0]
        for block in singleTransformerBlocks {
            h = block(h, tembModParams: tembSingle, imageRotaryEmb: concatRope)
        }

        h = h[0..., txtLen..., 0...]
        h = normOut(h, textEmbeddings: temb)
        h = projOut(h)
        return h
    }
}
