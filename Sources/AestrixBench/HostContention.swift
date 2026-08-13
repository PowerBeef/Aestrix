import Foundation

#if os(macOS)
import Darwin
#endif

public struct ProcessCPUShare: Sendable, Codable, Equatable {
    public var pid: Int32
    public var name: String
    public var cpuPercent: Double
    public var rssBytes: UInt64?

    public init(pid: Int32, name: String, cpuPercent: Double, rssBytes: UInt64? = nil) {
        self.pid = pid
        self.name = name
        self.cpuPercent = cpuPercent
        self.rssBytes = rssBytes
    }
}

public struct HostContentionSnapshot: Sendable, Codable, Equatable {
    public var timestamp: TimeInterval
    public var thermalState: String
    public var lowPowerMode: Bool
    public var swapUsedBytes: UInt64?
    public var vmFreeBytes: UInt64?
    public var topProcesses: [ProcessCPUShare]
    public var contaminated: Bool
    public var notes: [String]

    public init(
        timestamp: TimeInterval = Date().timeIntervalSince1970,
        thermalState: String,
        lowPowerMode: Bool,
        swapUsedBytes: UInt64? = nil,
        vmFreeBytes: UInt64? = nil,
        topProcesses: [ProcessCPUShare] = [],
        contaminated: Bool,
        notes: [String] = []
    ) {
        self.timestamp = timestamp
        self.thermalState = thermalState
        self.lowPowerMode = lowPowerMode
        self.swapUsedBytes = swapUsedBytes
        self.vmFreeBytes = vmFreeBytes
        self.topProcesses = topProcesses
        self.contaminated = contaminated
        self.notes = notes
    }
}

/// Best-effort host snapshot recorded around each benchmark trial.
///
/// This records contention; it never terminates processes or silently discards trials.
public enum HostContention {
    public static func capture(topProcessCount: Int = 8) -> HostContentionSnapshot {
        let processInfo = ProcessInfo.processInfo
        let thermal = thermalState(processInfo.thermalState)
        let processes = topProcesses(limit: topProcessCount)
        let heavy = processes.contains {
            $0.pid != processInfo.processIdentifier && $0.cpuPercent >= 15
        }
        let thermalPressure = thermal != "nominal"
        let contaminated = thermalPressure || heavy || processInfo.isLowPowerModeEnabled

        var notes: [String] = []
        if thermalPressure {
            notes.append("thermal_state=\(thermal)")
        }
        if processInfo.isLowPowerModeEnabled {
            notes.append("low_power_mode=true")
        }
        if let process = processes.first(where: {
            $0.pid != processInfo.processIdentifier && $0.cpuPercent >= 15
        }) {
            notes.append(
                String(format: "competing_process=%@ pid=%d cpu=%.1f%%",
                    process.name, process.pid, process.cpuPercent))
        }
        let swap = swapUsedBytes()
        if let swap, swap > 0 {
            notes.append("swap_used_bytes=\(swap)")
        }

        return HostContentionSnapshot(
            thermalState: thermal,
            lowPowerMode: processInfo.isLowPowerModeEnabled,
            swapUsedBytes: swap,
            vmFreeBytes: vmFreeBytes(),
            topProcesses: processes,
            contaminated: contaminated,
            notes: notes
        )
    }

    static func parsePSOutput(_ output: String, limit: Int) -> [ProcessCPUShare] {
        output.split(whereSeparator: \.isNewline)
            .compactMap { line -> ProcessCPUShare? in
                let parts = line.split(
                    maxSplits: 3,
                    omittingEmptySubsequences: true,
                    whereSeparator: \.isWhitespace
                )
                guard parts.count == 4,
                      let pid = Int32(parts[0]),
                      let cpu = Double(parts[1]),
                      let rssKiB = UInt64(parts[2])
                else {
                    return nil
                }
                return ProcessCPUShare(
                    pid: pid,
                    name: String(parts[3]),
                    cpuPercent: cpu,
                    rssBytes: rssKiB * 1024
                )
            }
            .sorted { $0.cpuPercent > $1.cpuPercent }
            .prefix(max(0, limit))
            .map { $0 }
    }

    private static func thermalState(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    private static func topProcesses(limit: Int) -> [ProcessCPUShare] {
        #if os(macOS)
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/ps")
        task.arguments = ["-axo", "pid=,pcpu=,rss=,comm="]
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice
        do {
            try task.run()
            task.waitUntilExit()
            guard task.terminationStatus == 0 else { return [] }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            guard let output = String(data: data, encoding: .utf8) else { return [] }
            return parsePSOutput(output, limit: limit)
        } catch {
            return []
        }
        #else
        return []
        #endif
    }

    private static func swapUsedBytes() -> UInt64? {
        #if os(macOS)
        var usage = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        let result = sysctlbyname("vm.swapusage", &usage, &size, nil, 0)
        return result == 0 ? usage.xsu_used : nil
        #else
        return nil
        #endif
    }

    private static func vmFreeBytes() -> UInt64? {
        #if os(macOS)
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        var pageSize: vm_size_t = 0
        guard host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS else { return nil }
        return UInt64(stats.free_count) * UInt64(pageSize)
        #else
        return nil
        #endif
    }
}
