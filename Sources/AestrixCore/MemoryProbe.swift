import Foundation

/// Samples process memory for stage budgeting.
/// MLX `Memory.activeMemory` sampling is deferred to model-load phases (Phase 2+)
/// so the scaffold CLI does not require a prebuilt metallib at process start.
public struct MemorySample: Sendable, Equatable {
    public var label: String
    public var mlxActiveBytes: UInt64
    public var mlxCacheBytes: UInt64
    public var processResidentBytes: UInt64
    public var timestamp: Date

    public init(
        label: String,
        mlxActiveBytes: UInt64 = 0,
        mlxCacheBytes: UInt64 = 0,
        processResidentBytes: UInt64,
        timestamp: Date = Date()
    ) {
        self.label = label
        self.mlxActiveBytes = mlxActiveBytes
        self.mlxCacheBytes = mlxCacheBytes
        self.processResidentBytes = processResidentBytes
        self.timestamp = timestamp
    }
}

public final class MemoryProbe: @unchecked Sendable {
    private var samples: [MemorySample] = []
    private let lock = NSLock()

    public init() {}

    public func snapshot(label: String) -> MemorySample {
        let sample = MemorySample(
            label: label,
            processResidentBytes: Self.processResidentBytes()
        )
        lock.lock()
        samples.append(sample)
        lock.unlock()
        return sample
    }

    public func allSamples() -> [MemorySample] {
        lock.lock()
        defer { lock.unlock() }
        return samples
    }

    public func peakMLXActive() -> UInt64 {
        allSamples().map(\.mlxActiveBytes).max() ?? 0
    }

    public func peakProcessResident() -> UInt64 {
        allSamples().map(\.processResidentBytes).max() ?? 0
    }

    public func reset() {
        lock.lock()
        samples.removeAll()
        lock.unlock()
    }

    /// Best-effort cache clear; DiT module also calls MLX.Memory when linked.
    public static func clearMLXCache() {}

    public static func processResidentBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return UInt64(info.resident_size)
    }

    public static func formatBytes(_ bytes: UInt64) -> String {
        let gb = Double(bytes) / 1_073_741_824.0
        if gb >= 1 {
            return String(format: "%.2f GB", gb)
        }
        let mb = Double(bytes) / 1_048_576.0
        return String(format: "%.1f MB", mb)
    }
}
