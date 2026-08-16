import Foundation

/// Process-wide eval / `Memory.clearCache` profile.
///
/// `product` matches today's 8 GB-safe constants on every machine.
/// `mid` is a bench-only relaxation for ≥16 GB hosts — never auto-selected.
public struct EvalCachePolicy: Sendable, Equatable, Codable {
    public var blockCacheClearInterval: Int
    public var pipelineCacheClampMinSide: Int
    public var denoiseCacheLimitBytes: UInt64?
    public var clearCacheAfterDenoiseStep: Bool

    public init(
        /// 2 (2026-08-16 Tier-2): with the chunk-streamed single blocks bounding
        /// transients, every-block clears + the 256 MiB clamp cost ~2% at 1024²
        /// for an identical watermark (measured 3.00 GiB either way).
        blockCacheClearInterval: Int = 2,
        pipelineCacheClampMinSide: Int = 768,
        denoiseCacheLimitBytes: UInt64? = 512 * 1_024 * 1_024,
        clearCacheAfterDenoiseStep: Bool = true
    ) {
        self.blockCacheClearInterval = max(1, blockCacheClearInterval)
        self.pipelineCacheClampMinSide = pipelineCacheClampMinSide
        self.denoiseCacheLimitBytes = denoiseCacheLimitBytes
        self.clearCacheAfterDenoiseStep = clearCacheAfterDenoiseStep
    }

    public static let product = EvalCachePolicy()
    /// Next relaxation step beyond product (≥16 GB bench hosts only).
    public static let mid = EvalCachePolicy(
        blockCacheClearInterval: 4,
        denoiseCacheLimitBytes: 1_024 * 1_024 * 1_024
    )

    public static func named(_ raw: String) -> EvalCachePolicy? {
        switch raw {
        case "product", "low": return .product
        case "mid": return .mid
        default: return nil
        }
    }

    public var profileName: String {
        if self == .product { return "product" }
        if self == .mid { return "mid" }
        return "custom"
    }

    /// Bench-only profiles that must not run on 8 GB without `--force`.
    public var requiresHighRAM: Bool { self != .product }

    /// Non-nil when this profile must not run on `tier` without `--force`.
    public func refusalReason(tier: DeviceTier, force: Bool) -> String? {
        guard requiresHighRAM, tier == .low, !force else { return nil }
        return "--eval-cache \(profileName) is ≥16 GB bench only (this host is tier \(tier.rawValue)). Pass --force if isolated."
    }

    nonisolated(unsafe) public static var current: EvalCachePolicy = .product

    public static func resetToDefault() {
        current = .product
    }
}
