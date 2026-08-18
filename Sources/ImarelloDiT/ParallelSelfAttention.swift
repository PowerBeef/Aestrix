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
        // Long sequences (1024² single-stream L≈4608): stream chunks end-to-end
        // through proj → split → norm/RoPE → SwiGLU so the [B, L, 27k] proj,
        // the [B, L, 18k] mlpHidden, and the [B, L, 12k] to_out input never
        // exist at full length. Same per-token ops — byte-identical; only the
        // small f16 q/k/v and per-chunk mlpOut survive the loop.
        let tuning = AttentionTuning.current
        if hiddenStates.dim(1) > tuning.linearChunkThreshold {
            return streamedForward(
                hiddenStates, imageRotaryEmb: imageRotaryEmb,
                chunkSize: max(1, tuning.linearChunkSize))
        }

        // S4 experiment: f16 proj output keeps the whole epilogue (SwiGLU →
        // concat → to_out input) in f16; QKV still norms in f32 below.
        let epilogueF16 = tuning.linearF16FullEpilogue
        let proj = DiTOpProfile.time(
            .qkvProj,
            inputs: [hiddenStates],
            sync: { [$0] }
        ) {
            AttentionUtils.linearChunkedSequence(
                toQkvMlpProj, hiddenStates, outputF16: epilogueF16)
        }
        let packed = DiTOpProfile.time(
            .qkvRope,
            inputs: [proj],
            sync: { [$0.0, $0.1, $0.2, $0.3] }
        ) {
            let splitQkv = split(proj, indices: [innerDim * 3], axis: -1)
            let qkv = splitQkv[0]
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

            // Norm in f32 for stability, then store Q/K/V in f16 for long sequences
            // (L≈4608 @ 1024²: f32 QKV alone ≈1.7 GB).
            let f16Thr = AttentionTuning.current.f16SeqThreshold
            let attnDType: DType = seq > f16Thr ? .float16 : .float32
            query = normQ(query.asType(.float32)).asType(attnDType)
            key = normK(key.asType(.float32)).asType(attnDType)
            value = value.asType(attnDType)

            if let (cos, sin) = imageRotaryEmb {
                (query, key) = AttentionUtils.applyRopeBSHD(query: query, key: key, cos: cos, sin: sin)
            }
            return (query, key, value, mlpHidden)
        }
        let query = packed.0
        let key = packed.1
        let value = packed.2
        let mlpHidden = packed.3
        let batch = query.dim(0)
        let seq = query.dim(2)

        // SwiGLU before attention: it halves 18432 → 9216, so the tensor held
        // live across Steel FA is mlpOut (162 MiB @1024²), not mlpHidden (324).
        let mlpOut = DiTOpProfile.time(
            .ffn,
            inputs: [mlpHidden],
            sync: { [$0] }
        ) {
            mlpAct(mlpHidden)
        }

        // Free proj graph before attention. Cache clear only on long sequences,
        // honoring the block clear interval (an interval of 2+ means "clear less",
        // which previously skipped only the transformer-loop clears, not these).
        if AttentionTuning.current.qkvCheckpoint {
            if !DiTOpProfile.enabled {
                eval(query, key, value, mlpOut)
            }
            if seq > AttentionTuning.current.blockCacheClearSeqThreshold,
                AttentionTuning.current.blockCacheClearInterval <= 1
            {
                Memory.clearCache()
            }
        }

        let attnOut = AttentionUtils.computeAttention(
            query: query, key: key, value: value,
            batchSize: batch, numHeads: heads, headDim: dimHead
        )
        return DiTOpProfile.time(
            .ffn,
            inputs: [attnOut, mlpOut],
            sync: { [$0] }
        ) {
            let fused = concatenated([attnOut.asType(mlpOut.dtype), mlpOut], axis: -1)
            return AttentionUtils.linearChunkedSequence(toOut, fused)
        }
    }

    /// Chunk-streamed forward for long sequences. Attention still sees full
    /// f16 Q/K/V (it needs every key); everything Linear/elementwise runs and
    /// materializes per 512-token chunk.
    private func streamedForward(
        _ hiddenStates: MLXArray,
        imageRotaryEmb: (MLXArray, MLXArray)?,
        chunkSize: Int
    ) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let seq = hiddenStates.dim(1)
        // Full-sequence decision, exactly as the unchunked path takes it.
        let attnDType: DType = seq > AttentionTuning.current.f16SeqThreshold ? .float16 : .float32
        let epilogueF16 = AttentionTuning.current.linearF16FullEpilogue

        var qChunks: [MLXArray] = []
        var kChunks: [MLXArray] = []
        var vChunks: [MLXArray] = []
        var mlpOutChunks: [MLXArray] = []
        var ranges: [(Int, Int)] = []
        var start = 0
        while start < seq {
            let end = min(start + chunkSize, seq)
            ranges.append((start, end))
            let piece = hiddenStates[0..., start ..< end, 0...]
            let proj = DiTOpProfile.time(.qkvProj, inputs: [piece], sync: { [$0] }) {
                AttentionUtils.applyLinear(toQkvMlpProj, piece, outputF16: epilogueF16)
            }
            let (q, k, v) = DiTOpProfile.time(
                .qkvRope, inputs: [proj], sync: { [$0.0, $0.1, $0.2] }
            ) {
                let splitQkv = split(proj, indices: [innerDim * 3], axis: -1)
                let qkvParts = split(splitQkv[0], parts: 3, axis: -1)
                let c = end - start
                var q = qkvParts[0].reshaped([batch, c, heads, dimHead]).transposed(0, 2, 1, 3)
                var k = qkvParts[1].reshaped([batch, c, heads, dimHead]).transposed(0, 2, 1, 3)
                var v = qkvParts[2].reshaped([batch, c, heads, dimHead]).transposed(0, 2, 1, 3)
                q = normQ(q.asType(.float32)).asType(attnDType)
                k = normK(k.asType(.float32)).asType(attnDType)
                v = v.asType(attnDType)
                if let (cos, sin) = imageRotaryEmb {
                    (q, k) = AttentionUtils.applyRopeBSHD(
                        query: q, key: k,
                        cos: cos[start ..< end, 0...], sin: sin[start ..< end, 0...])
                }
                return (q, k, v)
            }
            let mlpOut = DiTOpProfile.time(.ffn, inputs: [proj], sync: { [$0] }) {
                mlpAct(split(proj, indices: [innerDim * 3], axis: -1)[1])
            }
            // Materialize per chunk: this is the memory bound — proj/mlpHidden
            // temps free here instead of accumulating across the sequence.
            eval(q, k, v, mlpOut)
            qChunks.append(q)
            kChunks.append(k)
            vChunks.append(v)
            mlpOutChunks.append(mlpOut)
            start = end
        }

        let query = concatenated(qChunks, axis: 2)
        let key = concatenated(kChunks, axis: 2)
        let value = concatenated(vChunks, axis: 2)
        if AttentionTuning.current.qkvCheckpoint, !DiTOpProfile.enabled {
            eval(query, key, value)
        }
        if seq > AttentionTuning.current.blockCacheClearSeqThreshold,
            AttentionTuning.current.blockCacheClearInterval <= 1
        {
            Memory.clearCache()
        }

        let attnOut = AttentionUtils.computeAttention(
            query: query, key: key, value: value,
            batchSize: batch, numHeads: heads, headDim: dimHead
        )

        return DiTOpProfile.time(.ffn, inputs: [attnOut], sync: { [$0] }) {
            var outChunks: [MLXArray] = []
            outChunks.reserveCapacity(ranges.count)
            for (i, (s, e)) in ranges.enumerated() {
                let mlpOut = mlpOutChunks[i]
                let fused = concatenated(
                    [attnOut[0..., s ..< e, 0...].asType(mlpOut.dtype), mlpOut], axis: -1)
                let out = AttentionUtils.applyLinear(toOut, fused)
                eval(out)
                outChunks.append(out)
            }
            let result = concatenated(outChunks, axis: 1)
            eval(result)
            return result
        }
    }
}
