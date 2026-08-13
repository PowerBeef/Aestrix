import Foundation
import MLX

/// Exact D=512 VAE mid-block attention: query-chunked f32 matmul + softmax.
///
/// Steel fused FA only covers D ∈ {64, 80, 128}. A full S×S score matrix at
/// 1024² encode (S=16384) is ~1.07 GiB. Chunks keep scores at `[Tq, S]`.
///
/// Default does **not** `eval` each chunk (that extra sync is unsafe on 8 GB).
public struct VAEAttentionConfig: Sendable, Equatable {
    public var queryChunkSize: Int
    /// When true, use `MLXFast.scaledDotProductAttention` (legacy fallback A/B).
    public var useMLXFast: Bool
    public var evalEachChunk: Bool
    public var clearCacheEachChunk: Bool

    public init(
        queryChunkSize: Int = 64,
        useMLXFast: Bool = false,
        evalEachChunk: Bool = false,
        clearCacheEachChunk: Bool = false
    ) {
        self.queryChunkSize = max(0, queryChunkSize)
        self.useMLXFast = useMLXFast
        self.evalEachChunk = evalEachChunk
        self.clearCacheEachChunk = clearCacheEachChunk
    }

    nonisolated(unsafe) public static var current = VAEAttentionConfig()

    public static func resetToDefault() {
        current = VAEAttentionConfig()
    }
}

public enum VAEAttention {
    /// q,k,v: `[B, H, S, D]`. Scores per chunk: `[B, H, Tq, S]` — never `[B, H, S, S]`.
    public static func scaledDotProductAttention(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        scale: Float
    ) -> MLXArray {
        let cfg = VAEAttentionConfig.current
        let outDtype = query.dtype
        let qf = query.asType(.float32)
        let kf = key.asType(.float32)
        let vf = value.asType(.float32)
        let S = qf.dim(2)
        let Tq = max(1, cfg.queryChunkSize == 0 ? S : cfg.queryChunkSize)
        let kT = kf.transposed(0, 1, 3, 2)

        var chunks: [MLXArray] = []
        var start = 0
        while start < S {
            let end = min(start + Tq, S)
            let qChunk = qf[0..., 0..., start ..< end, 0...]
            let scores = matmul(qChunk * scale, kT)
            let probs = softmax(scores, axis: -1)
            let out = matmul(probs, vf)
            if cfg.evalEachChunk {
                eval(out)
            }
            if cfg.clearCacheEachChunk {
                Memory.clearCache()
            }
            chunks.append(out)
            start = end
        }
        let cat = concatenated(chunks, axis: 2)
        eval(cat)
        return outDtype == .float32 ? cat : cat.asType(outDtype)
    }

    /// Full-row oracle (one chunk of size S).
    public static func referenceSDPA(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        scale: Float
    ) -> MLXArray {
        let saved = VAEAttentionConfig.current
        VAEAttentionConfig.current.queryChunkSize = max(1, query.dim(2))
        VAEAttentionConfig.current.evalEachChunk = false
        defer { VAEAttentionConfig.current = saved }
        return scaledDotProductAttention(query: query, key: key, value: value, scale: scale)
    }

    public static func scoreBytes(queryChunk: Int, keySeq: Int, dtypeBytes: Int = 4) -> Int {
        queryChunk * keySeq * dtypeBytes
    }
}
