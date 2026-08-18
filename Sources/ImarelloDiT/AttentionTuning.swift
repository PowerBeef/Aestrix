import Foundation
import MLX

/// Process-wide DiT attention / projection chunk knobs (bench sweeps + product defaults).
///
/// Not actor-isolated: set once before a generation/bench trial on the MLX thread.
public struct AttentionTuning: Sendable, Equatable, Codable {
    /// Above this sequence length, use query-chunked SDPA instead of one fused call.
    /// LEGACY (no effect since 2026-08-18): the query-chunked SDPA fallback was
    /// deleted — D=128 is always Steel. Kept for bench-report provenance compat.
    public var queryChunkThreshold: Int
    /// Query tokens per SDPA chunk when chunking is active.
    public var queryChunkSize: Int
    /// When seq **>** this, store Q/K/V (and RoPE out) in float16 after f32 RMSNorm.
    /// Product default **512** so 512² image tokens (seq=1024) use f16; text pad (512) stays f32.
    public var f16SeqThreshold: Int
    /// Run 4-bit `QuantizedLinear` (and SwiGLU) in float16. Residuals / AdaLN stay f32.
    /// Product default **true**: 4-bit qmm in f16 (scales cast f16; residual f32).
    /// `--attn-linear-compute f32` restores the old f32 GEMM path.
    public var linearF16: Bool
    /// Divide activations by this before f16 `quantizedMM`, multiply after (f32 out).
    /// Raw f16 GEMM overflowed to noise; **16** is the shipped rescue scale.
    public var linearF16Scale: Float
    /// Above this seq length, apply `Linear` in chunks along the sequence axis.
    public var linearChunkThreshold: Int
    /// Sequence tokens per Linear chunk.
    public var linearChunkSize: Int
    /// SDPA backend: `mlx` | `metal-fa` | `auto`.
    public var backend: AttentionBackend
    /// When `backend == .auto`, use Metal FA if joint seq ≥ this (default 1024).
    public var metalFAMinSeq: Int
    /// Per-block `Memory.clearCache()` in the DiT forward only when joint seq **>** this.
    /// Small canvases (512² joint=1536) keep the buffer pool warm; large canvases keep the
    /// aggressive clears that made 1024² fit on 8 GB.
    public var blockCacheClearSeqThreshold: Int
    /// When per-block cache clears are active, clear every N blocks (1 = every block).
    public var blockCacheClearInterval: Int
    /// Materialize Q/K/V (+ fused MLP hidden) inside attention forwards. Product default
    /// **true** (bounds lazy-graph peak). Must be **false** to trace a block under
    /// `MLX.compile` (eval inside a traced function breaks compilation) — spike use only.
    public var qkvCheckpoint: Bool
    /// When > 0, per-block checkpoints use `asyncEval` (schedule without a CPU stall)
    /// with a blocking `eval` every N blocks to cap graph depth. 0 = blocking every
    /// block (pre-Tier-2 behavior). Numerics identical; watermark must be re-gated.
    public var asyncEvalInterval: Int
    /// Decide the attention store dtype from the **joint** sequence (txt+img) instead of
    /// per stream. Double-stream text (seq 512) currently lands f32 and drags the whole
    /// joint QKV back to f32 through concat/SDPA promotion — 5 of 25 blocks run f32
    /// Steel plus six wasted casts per block. Pixel-changing: gate on eval + vision.
    public var jointSeqF16: Bool
    /// EXPERIMENT (S4, engine plan 2026-08-18): keep the single-stream fused-proj
    /// epilogue in f16 end-to-end (proj → SwiGLU → concat → to_out input; to_out
    /// output stays f32 for the residual). Pixel-changing: full gate + the
    /// schema-1.4 `unstructured_garbage` fail before any promotion.
    public var linearF16FullEpilogue: Bool
    /// EXPERIMENT (S4): per-tensor dynamic activation scale — `max(amax|x|/target, 1)`
    /// computed on-GPU per Linear call — instead of the flat `linearF16Scale`.
    public var linearDynamicScale: Bool
    /// Amax normalization target for the dynamic scale (activations land ≤ this
    /// magnitude before the f16 qmm; headroom for the f16 accumulator).
    public var linearDynamicScaleTarget: Float

    public init(
        queryChunkThreshold: Int = 1536,
        /// Phase C sweep (1024² warm): 512 beats 256 by ~1% denoise/step; 128 was neutral/slower.
        queryChunkSize: Int = 512,
        f16SeqThreshold: Int = 512,
        linearF16: Bool = true,
        linearF16Scale: Float = 16,
        linearChunkThreshold: Int = 1536,
        linearChunkSize: Int = 512,
        /// Default **mlx** (chunked SDPA). Use `metal-fa` / `auto` to exercise custom FA2;
        /// naive Metal FA is correct + low-temp memory but not yet MFA-speed at L≈4608.
        backend: AttentionBackend = .mlx,
        metalFAMinSeq: Int = 1024,
        blockCacheClearSeqThreshold: Int = 1536,
        /// 2 (2026-08-16 Tier-2, with EvalCachePolicy.product): every-block clears
        /// cost ~2% at 1024² post-streaming with no watermark benefit.
        blockCacheClearInterval: Int = 2,
        qkvCheckpoint: Bool = true,
        asyncEvalInterval: Int = 0,
        /// Product default **true** (2026-08-16 Tier-2): −1.3% e2e @1024², watermark
        /// −0.07 GiB, eval-regression 15/15 + vision pass. `false` restores per-stream
        /// dtype (f32 double-stream attention).
        jointSeqF16: Bool = true,
        linearF16FullEpilogue: Bool = false,
        linearDynamicScale: Bool = false,
        linearDynamicScaleTarget: Float = 64
    ) {
        self.queryChunkThreshold = queryChunkThreshold
        self.queryChunkSize = queryChunkSize
        self.f16SeqThreshold = f16SeqThreshold
        self.linearF16 = linearF16
        self.linearF16Scale = max(1, linearF16Scale)
        self.linearChunkThreshold = linearChunkThreshold
        self.linearChunkSize = linearChunkSize
        self.backend = backend
        self.metalFAMinSeq = metalFAMinSeq
        self.blockCacheClearSeqThreshold = blockCacheClearSeqThreshold
        self.blockCacheClearInterval = max(1, blockCacheClearInterval)
        self.qkvCheckpoint = qkvCheckpoint
        self.asyncEvalInterval = max(0, asyncEvalInterval)
        self.jointSeqF16 = jointSeqF16
        self.linearF16FullEpilogue = linearF16FullEpilogue
        self.linearDynamicScale = linearDynamicScale
        self.linearDynamicScaleTarget = max(1, linearDynamicScaleTarget)
    }

    /// Product defaults.
    public static let `default` = AttentionTuning()

    /// Active knobs for the process. Mutate only when no DiT forward is in flight.
    nonisolated(unsafe) public static var current: AttentionTuning = .default

    /// EXPERIMENT (ENGINE_RESEARCH.md §5.1 R3): additive bias over joint attention
    /// keys, `[1,1,1,jointSeq]`, set per generate by the pipeline. nil = product.
    nonisolated(unsafe) public static var experimentalAttnBias: MLXArray?

    public static func resetToDefault() {
        current = .default
    }

    /// Short label for bench reports.
    public var shortLabel: String {
        "\(backend.rawValue)/q\(queryChunkSize)/t\(queryChunkThreshold)/f16@\(f16SeqThreshold)/lin\(linearChunkSize)/qmm-\(linearF16 ? "f16" : "f32")"
            + (linearF16FullEpilogue ? "/epi-f16" : "")
            + (linearDynamicScale ? "/dynscale\(Int(linearDynamicScaleTarget))" : "")
    }
}
