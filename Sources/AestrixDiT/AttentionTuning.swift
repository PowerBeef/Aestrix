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
    public var f16SeqThreshold: Int
    /// Above this seq length, apply `Linear` in chunks along the sequence axis.
    public var linearChunkThreshold: Int
    /// Sequence tokens per Linear chunk.
    public var linearChunkSize: Int

    public init(
        queryChunkThreshold: Int = 1536,
        /// Phase C sweep (1024² warm): 512 beats 256 by ~1% denoise/step; 128 was neutral/slower.
        queryChunkSize: Int = 512,
        f16SeqThreshold: Int = 2048,
        linearChunkThreshold: Int = 1536,
        linearChunkSize: Int = 512
    ) {
        self.queryChunkThreshold = queryChunkThreshold
        self.queryChunkSize = queryChunkSize
        self.f16SeqThreshold = f16SeqThreshold
        self.linearChunkThreshold = linearChunkThreshold
        self.linearChunkSize = linearChunkSize
    }

    /// Product defaults (pre–Phase C baseline).
    public static let `default` = AttentionTuning()

    /// Active knobs for the process. Mutate only when no DiT forward is in flight.
    nonisolated(unsafe) public static var current: AttentionTuning = .default

    public static func resetToDefault() {
        current = .default
    }

    /// Short label for bench reports, e.g. `q256/t1536/f16@2048/lin512`.
    public var shortLabel: String {
        "q\(queryChunkSize)/t\(queryChunkThreshold)/f16@\(f16SeqThreshold)/lin\(linearChunkSize)"
    }
}
