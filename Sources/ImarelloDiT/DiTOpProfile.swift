import Foundation
import MLX

/// Off-by-default GPU-sync timer for ranking Steel FA vs FFN vs processQKV glue.
///
/// When `enabled` is false, `time` is a pure passthrough — no extra `eval`, no
/// numerics change. When enabled, each bucket flushes its inputs (untimed) then
/// times `body` + output `eval` so MLX lazy work is attributed to that bucket.
/// Extra syncs inflate wall time; use for ranking only, never as an e2e number.
public enum DiTOpProfile {
    public enum Bucket: String, Sendable, Codable, CaseIterable {
        /// Quant Linear projections: double-stream `to_q/k/v` (+ added) and
        /// single-stream fused `to_qkv_mlp_proj`.
        case qkvProj = "qkv_proj"
        /// True glue: reshape, RMSNorm, concat, RoPE, QKV checkpoint.
        case qkvRope = "qkv_rope"
        /// `computeAttention` (Steel fused FA on the product path).
        case steelFA = "steel_fa"
        /// Double-stream SwiGLU FFN + single-stream `mlpAct` + `to_out`.
        case ffn = "ffn"
    }

    private static let lock = NSLock()
    nonisolated(unsafe) public static var enabled = false
    nonisolated(unsafe) static var seconds: [Bucket: Double] = [:]
    nonisolated(unsafe) static var counts: [Bucket: Int] = [:]

    public static func reset() {
        lock.lock()
        enabled = false
        seconds = [:]
        counts = [:]
        lock.unlock()
    }

    public static func begin() {
        lock.lock()
        enabled = true
        seconds = [:]
        counts = [:]
        lock.unlock()
    }

    /// Always accumulates (for tests). `time` is the only GPU-sync entry point.
    public static func record(_ bucket: Bucket, seconds value: Double) {
        lock.lock()
        seconds[bucket, default: 0] += value
        counts[bucket, default: 0] += 1
        lock.unlock()
    }

    /// Time `body` and GPU-sync its outputs. No-op when disabled.
    public static func time<R>(
        _ bucket: Bucket,
        inputs: [MLXArray] = [],
        sync: (R) -> [MLXArray],
        _ body: () -> R
    ) -> R {
        lock.lock()
        let on = enabled
        lock.unlock()
        guard on else { return body() }
        if !inputs.isEmpty {
            eval(inputs)
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        let result = body()
        let outs = sync(result)
        if !outs.isEmpty {
            eval(outs)
        }
        record(bucket, seconds: CFAbsoluteTimeGetCurrent() - t0)
        return result
    }

    public static func snapshot(denoiseSeconds: Double? = nil) -> DiTOpProfileReport {
        lock.lock()
        let secondsCopy = seconds
        let countsCopy = counts
        lock.unlock()
        var bucketsMs: [String: Double] = [:]
        var countMap: [String: Int] = [:]
        var counted = 0.0
        for bucket in Bucket.allCases {
            let ms = (secondsCopy[bucket] ?? 0) * 1000
            bucketsMs[bucket.rawValue] = ms
            countMap[bucket.rawValue] = countsCopy[bucket] ?? 0
            counted += ms
        }
        let denoiseMs = denoiseSeconds.map { $0 * 1000 }
        let otherMs = denoiseMs.map { max(0, $0 - counted) }
        var shares: [String: Double] = [:]
        let denom = denoiseMs ?? counted
        if denom > 0 {
            for (key, ms) in bucketsMs {
                shares[key] = ms / denom
            }
            if let otherMs {
                shares["other"] = otherMs / denom
            }
        }
        var notes: [String] = [
            "ranking only: extra eval() at bucket boundaries inflates wall time",
            "qkv_proj includes single-stream fused QKV+MLP Linear",
            "qkv_rope is the fuse/compile-glue target (norm + RoPE + concat)",
        ]
        if let denoiseMs {
            notes.append(
                String(format: "denoise_ms=%.1f counted_ms=%.1f other_ms=%.1f", denoiseMs, counted, otherMs ?? 0)
            )
        }
        return DiTOpProfileReport(
            bucketsMs: bucketsMs,
            counts: countMap,
            countedMs: counted,
            denoiseMs: denoiseMs,
            otherMs: otherMs,
            shares: shares,
            notes: notes
        )
    }
}

public struct DiTOpProfileReport: Sendable, Codable, Equatable {
    public var bucketsMs: [String: Double]
    public var counts: [String: Int]
    public var countedMs: Double
    public var denoiseMs: Double?
    public var otherMs: Double?
    public var shares: [String: Double]
    public var notes: [String]

    public init(
        bucketsMs: [String: Double] = [:],
        counts: [String: Int] = [:],
        countedMs: Double = 0,
        denoiseMs: Double? = nil,
        otherMs: Double? = nil,
        shares: [String: Double] = [:],
        notes: [String] = []
    ) {
        self.bucketsMs = bucketsMs
        self.counts = counts
        self.countedMs = countedMs
        self.denoiseMs = denoiseMs
        self.otherMs = otherMs
        self.shares = shares
        self.notes = notes
    }

    /// Largest named bucket by milliseconds (not `other`).
    public var dominantBucket: String? {
        bucketsMs.max(by: { $0.value < $1.value })?.key
    }
}
