import MLX
import MLXNN

/// Joint img+txt attention for double-stream blocks.
public final class Flux2Attention: Module {
    let heads: Int
    let dimHead: Int
    let innerDim: Int

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: Linear

    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm
    @ModuleInfo(key: "add_q_proj") var addQProj: Linear
    @ModuleInfo(key: "add_k_proj") var addKProj: Linear
    @ModuleInfo(key: "add_v_proj") var addVProj: Linear
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear

    public init(dim: Int, heads: Int, dimHead: Int) {
        self.heads = heads
        self.dimHead = dimHead
        self.innerDim = heads * dimHead
        self._toQ.wrappedValue = Linear(dim, innerDim, bias: false)
        self._toK.wrappedValue = Linear(dim, innerDim, bias: false)
        self._toV.wrappedValue = Linear(dim, innerDim, bias: false)
        self._normQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self._normK.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self._toOut.wrappedValue = Linear(innerDim, dim, bias: false)

        self._normAddedQ.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self._normAddedK.wrappedValue = RMSNorm(dimensions: dimHead, eps: 1e-5)
        self._addQProj.wrappedValue = Linear(dim, innerDim, bias: false)
        self._addKProj.wrappedValue = Linear(dim, innerDim, bias: false)
        self._addVProj.wrappedValue = Linear(dim, innerDim, bias: false)
        self._toAddOut.wrappedValue = Linear(innerDim, dim, bias: false)
        super.init()
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        imageRotaryEmb: (MLXArray, MLXArray)?
    ) -> (MLXArray, MLXArray) {
        let txtLen = encoderHiddenStates.dim(1)
        // Optionally decide the store dtype from the JOINT sequence: per-stream the
        // text side (512, not > threshold) lands f32, and concat + SDPA promotion
        // then drags the whole joint QKV back to f32 (six wasted casts per block,
        // f32 Steel for the 5 double blocks). Joint-based, both streams agree.
        let forcedDType: DType?
        if AttentionTuning.current.jointSeqF16 {
            let jointSeq = hiddenStates.dim(1) + txtLen
            forcedDType = jointSeq > AttentionTuning.current.f16SeqThreshold ? .float16 : .float32
        } else {
            forcedDType = nil
        }
        var (query, key, value) = AttentionUtils.processQKV(
            hiddenStates: hiddenStates,
            toQ: toQ, toK: toK, toV: toV,
            normQ: normQ, normK: normK,
            numHeads: heads, headDim: dimHead,
            forcedDType: forcedDType
        )
        let (encQ, encK, encV) = AttentionUtils.processQKV(
            hiddenStates: encoderHiddenStates,
            toQ: addQProj, toK: addKProj, toV: addVProj,
            normQ: normAddedQ, normK: normAddedK,
            numHeads: heads, headDim: dimHead,
            forcedDType: forcedDType
        )
        // Concat text then image on sequence axis (axis 2 in B,H,S,D), then RoPE.
        (query, key, value) = DiTOpProfile.time(
            .qkvRope,
            inputs: [query, key, value, encQ, encK, encV],
            sync: { [$0.0, $0.1, $0.2] }
        ) {
            var query = concatenated([encQ, query], axis: 2)
            var key = concatenated([encK, key], axis: 2)
            let value = concatenated([encV, value], axis: 2)
            if let (cos, sin) = imageRotaryEmb {
                (query, key) = AttentionUtils.applyRopeBSHD(query: query, key: key, cos: cos, sin: sin)
            }
            return (query, key, value)
        }

        if AttentionTuning.current.qkvCheckpoint, !DiTOpProfile.enabled {
            eval(query, key, value)
        }

        let joint = AttentionUtils.computeAttention(
            query: query, key: key, value: value,
            batchSize: hiddenStates.dim(0),
            numHeads: heads, headDim: dimHead
        )
        var encOut = joint[0..., ..<txtLen, 0...]
        var imgOut = joint[0..., txtLen..., 0...]
        // Deliberately below the global `linearChunkThreshold` (1536): the two
        // out-projections chunk from 1024 tokens. Bench sweeps of the global
        // tunable do NOT reach these calls — tune this constant explicitly.
        let outProjectionChunkThreshold = 1024
        encOut = AttentionUtils.linearChunkedSequence(
            toAddOut, encOut, threshold: outProjectionChunkThreshold)
        imgOut = AttentionUtils.linearChunkedSequence(
            toOut, imgOut, threshold: outProjectionChunkThreshold)
        return (imgOut, encOut)
    }
}
