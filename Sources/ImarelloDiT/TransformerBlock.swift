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

        var nh = ModulationOps.modApply(norm1(h), scaleMsa, shiftMsa)
        var ne = ModulationOps.modApply(norm1Context(e), cScaleMsa, cShiftMsa)
        let dump = ComputeDTypeProbe.enabled && !ComputeDTypeProbe.didDumpBlock

        let (attnH, attnE) = attn(
            hiddenStates: nh,
            encoderHiddenStates: ne,
            imageRotaryEmb: imageRotaryEmb
        )
        h = ModulationOps.gateAdd(h, gateMsa, attnH)
        e = ModulationOps.gateAdd(e, cGateMsa, attnE)

        if dump {
            ComputeDTypeProbe.record("block0.residual.h", hiddenStates)
            ComputeDTypeProbe.record("block0.after_adaln.nh", nh)
            ComputeDTypeProbe.record("block0.attn.out", attnH)
        }

        nh = ModulationOps.modApply(norm2(h), scaleMlp, shiftMlp)
        let ffH = ff(nh)
        h = ModulationOps.gateAdd(h, gateMlp, ffH)

        ne = ModulationOps.modApply(norm2Context(e), cScaleMlp, cShiftMlp)
        e = ModulationOps.gateAdd(e, cGateMlp, ffContext(ne))

        if dump {
            ComputeDTypeProbe.record("block0.ff.in", nh)
            ComputeDTypeProbe.record("block0.ff.out", ffH)
            ComputeDTypeProbe.record("block0.residual.out", h)
            ComputeDTypeProbe.didDumpBlock = true
        }

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
        let nh = ModulationOps.modApply(norm(hiddenStates), modScale, modShift)
        let attnOut = attn(nh, imageRotaryEmb: imageRotaryEmb)
        return ModulationOps.gateAdd(hiddenStates, modGate, attnOut)
    }
}
