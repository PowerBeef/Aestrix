import Foundation
import MLX
import MLXNN
import MLXFast

enum AttentionUtils {
    static func processQKV(
        hiddenStates: MLXArray,
        toQ: Linear,
        toK: Linear,
        toV: Linear,
        normQ: RMSNorm,
        normK: RMSNorm,
        numHeads: Int,
        headDim: Int,
        forcedDType: DType? = nil
    ) -> (MLXArray, MLXArray, MLXArray) {
        let batch = hiddenStates.dim(0)
        let seq = hiddenStates.dim(1)

        // No eval here on the product path: JointAttention checkpoints after concat+RoPE.
        var (query, key, value) = DiTOpProfile.time(
            .qkvProj,
            inputs: [hiddenStates],
            sync: { [$0.0, $0.1, $0.2] }
        ) {
            (
                linearChunkedSequence(toQ, hiddenStates),
                linearChunkedSequence(toK, hiddenStates),
                linearChunkedSequence(toV, hiddenStates)
            )
        }

        (query, key, value) = DiTOpProfile.time(
            .qkvRope,
            inputs: [query, key, value],
            sync: { [$0.0, $0.1, $0.2] }
        ) {
            var query = query.reshaped([batch, seq, numHeads, headDim]).transposed(0, 2, 1, 3)
            var key = key.reshaped([batch, seq, numHeads, headDim]).transposed(0, 2, 1, 3)
            var value = value.reshaped([batch, seq, numHeads, headDim]).transposed(0, 2, 1, 3)

            // Prefer f16 for long sequences after f32 RMSNorm (memory).
            let f16Thr = AttentionTuning.current.f16SeqThreshold
            let attnDType: DType = forcedDType ?? (seq > f16Thr ? .float16 : query.dtype)
            query = normQ(query.asType(.float32)).asType(attnDType)
            key = normK(key.asType(.float32)).asType(attnDType)
            value = value.asType(attnDType)
            return (query, key, value)
        }
        let f16Thr = AttentionTuning.current.f16SeqThreshold
        let attnDType: DType = seq > f16Thr ? .float16 : query.dtype
        if ComputeDTypeProbe.enabled, !ComputeDTypeProbe.didDumpBlock, ComputeDTypeProbe.qkvRecords < 2 {
            let tag = ComputeDTypeProbe.qkvRecords == 0 ? "img" : "txt"
            ComputeDTypeProbe.record("qkv.\(tag).proj_q", query)
            ComputeDTypeProbe.record("qkv.\(tag).proj_k", key)
            ComputeDTypeProbe.record("qkv.\(tag).proj_v", value)
            ComputeDTypeProbe.recordNote(
                "qkv.\(tag)\tattnDType=\(ComputeDTypeProbe.dtypeName(attnDType)) seq=\(seq) f16Thr=\(f16Thr)")
            ComputeDTypeProbe.qkvRecords += 1
        }
        return (query, key, value)
    }

    static func computeAttention(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        batchSize: Int,
        numHeads: Int,
        headDim: Int
    ) -> MLXArray {
        let scale = 1.0 / Foundation.sqrt(Float(query.dim(-1)))
        // query/key/value: [B, H, S, D]
        let seq = query.dim(2)
        let tuning = AttentionTuning.current
        let useFusedFA: Bool = {
            switch tuning.backend {
            case .metalFA: return true
            case .mlx:
                // Product mlx: use Steel fused FA (full Q) when MLX supports it (D∈{64,80,128}).
                return MetalFlashAttention.steelHeadDims.contains(headDim) && seq > 8
            case .auto:
                return seq >= tuning.metalFAMinSeq && headDim <= 128
            }
        }()

        if useFusedFA {
            if ComputeDTypeProbe.enabled, !ComputeDTypeProbe.didDumpAttention {
                ComputeDTypeProbe.record("attn.steel.in.q", query)
                ComputeDTypeProbe.record("attn.steel.in.k", key)
                ComputeDTypeProbe.record("attn.steel.in.v", value)
                ComputeDTypeProbe.recordNote("attn.backend=steel seq=\(seq) headDim=\(headDim)")
                ComputeDTypeProbe.didDumpAttention = true
            }
            // Steel fused FA2 (simdgroup MMA) or hybrid FA2 — single logical forward.
            return DiTOpProfile.time(
                .steelFA,
                inputs: [query, key, value],
                sync: { [$0] }
            ) {
                // R3 EXPERIMENT: optional additive bias over joint keys; only
                // applied when its length matches this call's key sequence.
                var bias: MLXArray?
                if let b = AttentionTuning.experimentalAttnBias, b.dim(3) == key.dim(2) {
                    bias = b.asType(query.dtype)
                }
                var hidden = MetalFlashAttention.scaledDotProductAttention(
                    query: query, key: key, value: value, scale: scale, mask: bias
                )
                // [B, H, S, D] → [B, S, H*D]
                hidden = hidden.transposed(0, 2, 1, 3)
                return hidden.reshaped([batchSize, -1, numHeads * headDim])
            }
        }

        // Non-Steel head dims only (never reached on the product path: D=128 is
        // always Steel). The old query-chunked loop here was a pre-Steel
        // low-RAM hack — deleted 2026-08-18; plain SDPA covers the fallback.
        return DiTOpProfile.time(
            .steelFA,
            inputs: [query, key, value],
            sync: { [$0] }
        ) {
            var hidden = MLXFast.scaledDotProductAttention(
                queries: query,
                keys: key,
                values: value,
                scale: scale,
                mask: nil
            )
            hidden = hidden.transposed(0, 2, 1, 3)
            return hidden.reshaped([batchSize, -1, numHeads * headDim])
        }
    }

    /// Sequence Linear. Flattened last-dim GEMM; optional f16 `quantizedMM`.
    /// `outputF16` (S4): skip the f32 upcast in the qmm rescale — the caller
    /// keeps the epilogue chain in f16.
    static func applyLinear(
        _ linear: Linear, _ x: MLXArray, outputF16: Bool = false
    ) -> MLXArray {
        if x.ndim <= 2 {
            return applyLinearCore(linear, x, outputF16: outputF16)
        }
        let inDim = x.dim(-1)
        let x2 = x.reshaped([-1, inDim])
        let y2 = applyLinearCore(linear, x2, outputF16: outputF16)
        return y2.reshaped(Array(x.shape.dropLast()) + [y2.dim(-1)])
    }

    /// Compiled rescale halves of the f16 qmm sandwich. Same op order as the
    /// previous inline version — compile fuses div+cast and cast+mul into one
    /// kernel each, removing a full-size f32 temp per Linear call (that temp
    /// chain was the entire 512² watermark headroom; ENGINE_RESEARCH.md §2.3).
    private static let compiledPreScale: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        compile { (x: MLXArray, s: MLXArray) -> MLXArray in
            (x / s).asType(.float16)
        }
    }()
    private static let compiledPostScale: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        compile { (y: MLXArray, s: MLXArray) -> MLXArray in
            y.asType(.float32) * s
        }
    }()
    /// S4 full-f16 epilogue: rescale WITHOUT the f32 upcast — the result stays
    /// f16 for SwiGLU/concat/to_out-input. Overflow risk is the gate's job.
    private static let compiledPostScaleF16: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        compile { (y: MLXArray, s: MLXArray) -> MLXArray in
            y * s.asType(.float16)
        }
    }()
    /// S4 dynamic activation scale: `max(amax|x|/target, 1)` as a GPU scalar —
    /// no host sync; flows into the pre/post-scale graph like the constant did.
    private static let compiledDynamicScale: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        compile { (x: MLXArray, target: MLXArray) -> MLXArray in
            maximum(abs(x).max() / target, MLXArray(Float(1)))
        }
    }()
    nonisolated(unsafe) private static var scaleArrayCache: [Float: MLXArray] = [:]
    private static let scaleArrayLock = NSLock()
    private static func scaleArray(_ value: Float) -> MLXArray {
        scaleArrayLock.lock()
        defer { scaleArrayLock.unlock() }
        if let cached = scaleArrayCache[value] { return cached }
        let arr = MLXArray(value)
        eval(arr)
        scaleArrayCache[value] = arr
        return arr
    }

    /// 4-bit GEMM in f16 when `linearF16` is on. Affine scales are bf16 in the
    /// hub pack; `QuantizedLinear` would `promote(f16, bf16) → f32` unless we
    /// cast scales too (TransformerWeights pre-casts them at load — bf16→f16 is
    /// exact for this pack, min |scale| ≈ 1e-4 ≫ the f16 normal floor — so the
    /// asType below is a free no-op guard). Residual dtype is the caller's
    /// problem (AdaLN stays f32).
    private static func applyLinearCore(
        _ linear: Linear, _ x: MLXArray, outputF16: Bool = false
    ) -> MLXArray {
        let tuning = AttentionTuning.current
        guard tuning.linearF16, let q = linear as? QuantizedLinear else {
            let y = linear(x)
            return outputF16 ? y.asType(.float16) : y
        }
        // Scale down so f16 accumulate does not overflow (raw f16 qmm → TV static).
        // S4 dynamic mode normalizes per tensor (amax/target) instead of the flat 16.
        let scale = tuning.linearDynamicScale
            ? compiledDynamicScale(x, scaleArray(tuning.linearDynamicScaleTarget))
            : scaleArray(tuning.linearF16Scale)
        let x16 = compiledPreScale(x, scale)
        var y = quantizedMM(
            x16,
            q.weight,
            scales: q.scales.asType(.float16),
            biases: q.biases.map { $0.asType(.float16) },
            transpose: true,
            groupSize: q.groupSize,
            bits: q.bits,
            mode: q.mode
        )
        y = outputF16 ? compiledPostScaleF16(y, scale) : compiledPostScale(y, scale)
        // Bias must be added after the ×scale rescale: inside the sandwich it
        // would come out as scale·bias. (No DiT Linear routed here carries a
        // bias today; this keeps the helper correct if one ever does.)
        if let bias = q.bias {
            y = y + bias.asType(y.dtype)
        }
        return y
    }

    /// Apply a sequence Linear in chunks to avoid huge [B, L, out] intermediates (e.g. fused QKV+MLP).
    static func linearChunkedSequence(
        _ linear: Linear,
        _ x: MLXArray,
        chunkSize: Int? = nil,
        threshold: Int? = nil,
        outputF16: Bool = false
    ) -> MLXArray {
        // x: [B, S, C]
        let seq = x.dim(1)
        let thr = threshold ?? AttentionTuning.current.linearChunkThreshold
        let cs = max(1, chunkSize ?? AttentionTuning.current.linearChunkSize)
        if seq <= thr {
            return applyLinear(linear, x, outputF16: outputF16)
        }
        var parts: [MLXArray] = []
        parts.reserveCapacity((seq + cs - 1) / cs)
        var start = 0
        while start < seq {
            let end = min(start + cs, seq)
            let chunk = x[0..., start ..< end, 0...]
            // No per-chunk eval: lazy execution still runs chunk-by-chunk at the final
            // eval and frees each chunk's temps by refcount; per-chunk sync only stalled.
            parts.append(applyLinear(linear, chunk, outputF16: outputF16))
            start = end
        }
        let cat = concatenated(parts, axis: 1)
        eval(cat)
        return cat
    }

    /// Apply RoPE with cos/sin of shape [S, D/2] to Q/K of shape [B, H, S, D] (interleaved pairs).
    static func applyRopeBSHD(
        query: MLXArray,
        key: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        let outDtype = query.dtype
        let cosB = cos.reshaped([1, 1, cos.dim(0), cos.dim(1)])
        let sinB = sin.reshaped([1, 1, sin.dim(0), sin.dim(1)])

        // Keep rope output in the compact attention dtype when possible.
        let f16Thr = AttentionTuning.current.f16SeqThreshold
        let store = outDtype == .float32 && query.dim(2) > f16Thr ? DType.float16 : outDtype
        return (
            compiledRopeMix(query, cosB, sinB).asType(store),
            compiledRopeMix(key, cosB, sinB).asType(store)
        )
    }

    /// Compiled rotate-pairs mix (runs 50×/step). Same op sequence as the previous
    /// inline version — compile only fuses the elementwise chain per shape/dtype.
    private static let compiledRopeMix:
        @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray =
    {
        compile { (x: MLXArray, cosB: MLXArray, sinB: MLXArray) -> MLXArray in
            let xf = x.asType(.float32)
            let shape = xf.shape
            let x2 = xf.reshaped(Array(shape.dropLast()) + [-1, 2])
            let real = x2[.ellipsis, 0]
            let imag = x2[.ellipsis, 1]
            let out0 = real * cosB + (-imag) * sinB
            let out1 = imag * cosB + real * sinB
            return stacked([out0, out1], axis: -1).reshaped(shape)
        }
    }()
}
