import MLX
import MLXNN

/// Single-stream fused QKV + MLP projection attention.
public final class Flux2ParallelSelfAttention: Module {
    let heads: Int
    let dimHead: Int
    let innerDim: Int
    let mlpHiddenDim: Int

    @ModuleInfo(key: "to_qkv_mlp_proj") var toQkvMlpProj: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    let mlpAct = Flux2SwiGLU()
    @ModuleInfo(key: "to_out") var toOut: Linear

    public init(dim: Int, heads: Int, dimHead: Int, mlpRatio: Float = 3.0) {
        self.heads = heads
        self.dimHead = dimHead
        self.innerDim = heads * dimHead
        self.mlpHiddenDim = Int(Float(dim) * mlpRatio)
        self._toQkvMlpProj.wrappedValue = Linear(
            dim, innerDim * 3 + mlpHiddenDim * 2, bias: false)
        self._normQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self._normK.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self._toOut.wrappedValue = Linear(innerDim + mlpHiddenDim, dim, bias: false)
        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        imageRotaryEmb: (MLXArray, MLXArray)?
    ) -> MLXArray {
        let proj = toQkvMlpProj(hiddenStates)
        let splitQkv = split(proj, indices: [innerDim * 3], axis: -1)
        var qkv = splitQkv[0]
        let mlpHidden = splitQkv[1]
        let qkvParts = split(qkv, parts: 3, axis: -1)
        var query = qkvParts[0]
        var key = qkvParts[1]
        var value = qkvParts[2]

        let batch = query.dim(0)
        let seq = query.dim(1)
        query = query.reshaped([batch, seq, heads, dimHead]).transposed(0, 2, 1, 3)
        key = key.reshaped([batch, seq, heads, dimHead]).transposed(0, 2, 1, 3)
        value = value.reshaped([batch, seq, heads, dimHead]).transposed(0, 2, 1, 3)

        query = normQ(query.asType(.float32)).asType(.float32)
        key = normK(key.asType(.float32)).asType(.float32)

        if let (cos, sin) = imageRotaryEmb {
            (query, key) = AttentionUtils.applyRopeBSHD(query: query, key: key, cos: cos, sin: sin)
        }

        var attnOut = AttentionUtils.computeAttention(
            query: query, key: key, value: value,
            batchSize: batch, numHeads: heads, headDim: dimHead
        )
        let mlpOut = mlpAct(mlpHidden)
        attnOut = concatenated([attnOut, mlpOut], axis: -1)
        return toOut(attnOut)
    }
}
