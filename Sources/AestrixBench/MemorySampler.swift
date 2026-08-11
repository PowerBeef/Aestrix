import Foundation
import MLX
import AestrixCore
#if canImport(Metal)
import Metal
#endif

/// Process RSS + MLX memory snapshot for benchmarking.
public enum MemorySampler {
    public static var recommendedWorkingSetBytes: UInt64? {
        #if canImport(Metal)
        if let device = MTLCreateSystemDefaultDevice() {
            return UInt64(device.recommendedMaxWorkingSetSize)
        }
        #endif
        return nil
    }

    public static func sample(
        label: String,
        previousActive: UInt64? = nil,
        note: String? = nil
    ) -> MemoryPoint {
        let active = UInt64(Memory.activeMemory)
        let peak = UInt64(Memory.peakMemory)
        let cache = UInt64(Memory.cacheMemory)
        let rss = MemoryProbe.processResidentBytes()
        let rec = recommendedWorkingSetBytes
        let delta: Int64? = previousActive.map { Int64(active) - Int64($0) }
        let headroom: Int64? = rec.map { Int64($0) - Int64(active) }
        return MemoryPoint(
            label: label,
            rssBytes: rss,
            mlxActiveBytes: active,
            mlxCacheBytes: cache,
            mlxPeakBytes: peak,
            mlxActiveDeltaBytes: delta,
            recommendedWorkingSetBytes: rec,
            headroomBytes: headroom,
            note: note
        )
    }

    /// Reset MLX peak counter (assignment is the reset API).
    public static func resetPeak() {
        Memory.peakMemory = 0
    }

    public static func clearCache() {
        Memory.clearCache()
    }

    public static func applyCacheLimit(_ bytes: UInt64?) {
        guard let bytes else { return }
        Memory.cacheLimit = Int(bytes)
    }

    public static var cacheLimit: UInt64 {
        UInt64(Memory.cacheLimit)
    }

    public static var memoryLimit: UInt64 {
        UInt64(Memory.memoryLimit)
    }
}
