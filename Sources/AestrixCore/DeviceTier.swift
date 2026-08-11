import Foundation

/// Hardware memory tier used to clamp resolution and memory policy.
public enum DeviceTier: String, Sendable, Codable, CaseIterable {
    case low
    case mid
    case high

    /// Peak process budget in bytes (planning gates; refine with real benches).
    public var peakBudgetBytes: UInt64 {
        switch self {
        case .low: return 6_500_000_000
        case .mid: return 12_000_000_000
        case .high: return 24_000_000_000
        }
    }

    public var defaultMaxSide: Int {
        switch self {
        case .low: return 512
        case .mid, .high: return 1024
        }
    }

    /// Classify from physical memory (bytes).
    public static func detect(physicalMemoryBytes: UInt64 = ProcessInfo.processInfo.physicalMemory) -> DeviceTier {
        let gb = Double(physicalMemoryBytes) / 1_073_741_824.0
        if gb < 12 { return .low }
        if gb < 20 { return .mid }
        return .high
    }
}
