import Foundation

/// Process-wide DiT attention / projection chunk knobs (bench sweeps + product defaults).
///
/// Not actor-isolated: set once before a generation/bench trial on the MLX thread.
public struct AttentionTuning: Sendable, Equatable, Codable {
    /// Above this sequence length, use query-chunked SDPA instead of one fused call.
    public var queryChunkThreshold: Int
    /// Query tokens per SDPA chunk when chunking is active.
    public var queryChunkSize: Int
    /// When seq **>** this, store Q/K/V (and RoPE out) in float16 after f32 RMSNorm.
    /// Product default **512** so 512² image tokens (seq=1024) use f16; text pad (512) stays f32.
    public var f16SeqThreshold: Int
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

    public init(
        queryChunkThreshold: Int = 1536,
        /// Phase C sweep (1024² warm): 512 beats 256 by ~1% denoise/step; 128 was neutral/slower.
        queryChunkSize: Int = 512,
        f16SeqThreshold: Int = 512,
        linearChunkThreshold: Int = 1536,
        linearChunkSize: Int = 512,
        /// Default **mlx** (chunked SDPA). Use `metal-fa` / `auto` to exercise custom FA2;
        /// naive Metal FA is correct + low-temp memory but not yet MFA-speed at L≈4608.
        backend: AttentionBackend = .mlx,
        metalFAMinSeq: Int = 1024,
        blockCacheClearSeqThreshold: Int = 1536,
        blockCacheClearInterval: Int = 1,
        qkvCheckpoint: Bool = true
    ) {
        self.queryChunkThreshold = queryChunkThreshold
        self.queryChunkSize = queryChunkSize
        self.f16SeqThreshold = f16SeqThreshold
        self.linearChunkThreshold = linearChunkThreshold
        self.linearChunkSize = linearChunkSize
        self.backend = backend
        self.metalFAMinSeq = metalFAMinSeq
        self.blockCacheClearSeqThreshold = blockCacheClearSeqThreshold
        self.blockCacheClearInterval = max(1, blockCacheClearInterval)
        self.qkvCheckpoint = qkvCheckpoint
    }

    /// Product defaults.
    public static let `default` = AttentionTuning()

    /// Active knobs for the process. Mutate only when no DiT forward is in flight.
    nonisolated(unsafe) public static var current: AttentionTuning = .default

    public static func resetToDefault() {
        current = .default
    }

    /// Short label for bench reports.
    public var shortLabel: String {
        "\(backend.rawValue)/q\(queryChunkSize)/t\(queryChunkThreshold)/f16@\(f16SeqThreshold)/lin\(linearChunkSize)"
    }
}
