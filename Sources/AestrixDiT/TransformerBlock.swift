import MLX
import MLXNN

/// Double-stream block: joint attention + dual FFNs with AdaLN modulation.
public final class Flux2TransformerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm1_context") var norm1Context: LayerNorm
    @ModuleInfo(key: "attn") var attn: Flux2Attention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "ff") var ff: Flux2FeedForward
    @ModuleInfo(key: "norm2_context") var norm2Context: LayerNorm
    @ModuleInfo(key: "ff_context") var ffContext: Flux2FeedForward

    public init(dim: Int, numAttentionHeads: Int, attentionHeadDim: Int, mlpRatio: Float = 3.0) {
        self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false, bias: false)
        self._norm1Context.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false, bias: false)
        self._attn.wrappedValue = Flux2Attention(dim: dim, heads: numAttentionHeads, dimHead: attentionHeadDim)
        self._norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false, bias: false)
        self._ff.wrappedValue = Flux2FeedForward(dim: dim, mult: mlpRatio)
        self._norm2Context.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false, bias: false)
        self._ffContext.wrappedValue = Flux2FeedForward(dim: dim, mult: mlpRatio)
        super.init()
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        tembModParamsImg: [(MLXArray, MLXArray, MLXArray)],
        tembModParamsTxt: [(MLXArray, MLXArray, MLXArray)],
        imageRotaryEmb: (MLXArray, MLXArray)?
    ) -> (encoder: MLXArray, hidden: MLXArray) {
        let (shiftMsa, scaleMsa, gateMsa) = tembModParamsImg[0]
        let (shiftMlp, scaleMlp, gateMlp) = tembModParamsImg[1]
        let (cShiftMsa, cScaleMsa, cGateMsa) = tembModParamsTxt[0]
        let (cShiftMlp, cScaleMlp, cGateMlp) = tembModParamsTxt[1]

        var h = hiddenStates
        var e = encoderHiddenStates

        var nh = norm1(h)
        nh = (1 + scaleMsa) * nh + shiftMsa
        var ne = norm1Context(e)
        ne = (1 + cScaleMsa) * ne + cShiftMsa

        let (attnH, attnE) = attn(
            hiddenStates: nh,
            encoderHiddenStates: ne,
            imageRotaryEmb: imageRotaryEmb
        )
        h = h + gateMsa * attnH
        e = e + cGateMsa * attnE

        nh = norm2(h)
        nh = (1 + scaleMlp) * nh + shiftMlp
        h = h + gateMlp * ff(nh)

        ne = norm2Context(e)
        ne = (1 + cScaleMlp) * ne + cShiftMlp
        e = e + cGateMlp * ffContext(ne)

        return (e, h)
    }
}

/// Single-stream block after concat.
public final class Flux2SingleTransformerBlock: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "attn") var attn: Flux2ParallelSelfAttention

    public init(dim: Int, numAttentionHeads: Int, attentionHeadDim: Int, mlpRatio: Float = 3.0) {
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false, bias: false)
        self._attn.wrappedValue = Flux2ParallelSelfAttention(
            dim: dim, heads: numAttentionHeads, dimHead: attentionHeadDim, mlpRatio: mlpRatio)
        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        tembModParams: (MLXArray, MLXArray, MLXArray),
        imageRotaryEmb: (MLXArray, MLXArray)?
    ) -> MLXArray {
        let (modShift, modScale, modGate) = tembModParams
        var nh = norm(hiddenStates)
        nh = (1 + modScale) * nh + modShift
        let attnOut = attn(nh, imageRotaryEmb: imageRotaryEmb)
        return hiddenStates + modGate * attnOut
    }
}
