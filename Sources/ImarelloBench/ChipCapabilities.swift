import Foundation
#if canImport(Metal)
import Metal
#endif

/// Runtime GPU capability probe backed by real Metal queries.
///
/// Replaces the device-name string heuristic as the primary signal (2026-08-18
/// engine plan, workstream N1). The name heuristic remains only as a fallback
/// for the Neural-Accelerator bit until the provisional family threshold is
/// validated on A19/M5 hardware — `appleGPUFamilyRaw` is exposed precisely so
/// device runs can tell us the true mapping.
public struct ChipCapabilities: Sendable, Codable, Equatable {
    public var gpuName: String?
    /// Highest Apple-family `MTLGPUFamily` raw value the device reports
    /// (1001 = apple1 … 1009 = apple9/M3-class; newer chips report higher).
    public var appleGPUFamilyRaw: Int?
    /// Metal 4 API generation is available (macOS 26 / iOS 26).
    public var metal4: Bool
    /// GPU Neural Accelerators (M5 / A19 class only; fp16+int8 tensor units).
    public var hasNeuralAccelerators: Bool
    public var physicalMemoryBytes: UInt64
    public var recommendedMaxWorkingSetBytes: UInt64?

    /// PROVISIONAL: first Apple GPU family raw value assumed to carry Neural
    /// Accelerators (apple11-class = A19/M5 per the public numbering). Verify
    /// against a real A19 device before gating any behavior on it.
    public static let provisionalNeuralAcceleratorFamilyRaw = 1011

    public static func detect() -> ChipCapabilities {
        let physical = ProcessInfo.processInfo.physicalMemory
        var caps = ChipCapabilities(
            gpuName: nil,
            appleGPUFamilyRaw: nil,
            metal4: false,
            hasNeuralAccelerators: false,
            physicalMemoryBytes: physical,
            recommendedMaxWorkingSetBytes: nil
        )
        if #available(macOS 26.0, iOS 26.0, *) {
            caps.metal4 = true
        }
        #if canImport(Metal)
        guard let device = MTLCreateSystemDefaultDevice() else { return caps }
        caps.gpuName = device.name
        caps.recommendedMaxWorkingSetBytes = UInt64(device.recommendedMaxWorkingSetSize)
        // Probe upward through the Apple family raw range; unknown values are
        // safely `false` from supportsFamily. Ceiling is generous on purpose.
        var highest: Int?
        for raw in 1001 ... 1020 {
            if let family = MTLGPUFamily(rawValue: raw), device.supportsFamily(family) {
                highest = raw
            }
        }
        caps.appleGPUFamilyRaw = highest
        let familySaysNA = (highest ?? 0) >= Self.provisionalNeuralAcceleratorFamilyRaw
        let name = device.name.uppercased()
        let nameSaysNA = name.contains("M5") || name.contains("M6") || name.contains("M7")
            || name.contains("A19")
        caps.hasNeuralAccelerators = familySaysNA || nameSaysNA
        #endif
        return caps
    }

    /// One-line summary for `imarello info` / logs.
    public var summary: String {
        let fam = appleGPUFamilyRaw.map(String.init) ?? "?"
        return "\(gpuName ?? "no-gpu") family_raw=\(fam) metal4=\(metal4) "
            + "neural_accelerators=\(hasNeuralAccelerators)"
    }
}
