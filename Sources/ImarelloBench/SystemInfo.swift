import Foundation
#if canImport(Metal)
import Metal
#endif
import ImarelloCore

public enum SystemInfo {
    public static func snapshot(mlxCacheLimit: UInt64? = nil, mlxMemoryLimit: UInt64? = nil) -> SystemSnapshot {
        let processInfo = ProcessInfo.processInfo
        // Real Metal queries (family probe) with the name heuristic as fallback.
        let caps = ChipCapabilities.detect()
        let recommended = caps.recommendedMaxWorkingSetBytes
        let gpuName = caps.gpuName
        let metalSupport: String? = gpuName == nil ? nil : (caps.metal4 ? "Metal 4" : "Metal")
        let neural: Bool? = gpuName == nil ? nil : caps.hasNeuralAccelerators

        let thermal: String
        switch processInfo.thermalState {
        case .nominal: thermal = "nominal"
        case .fair: thermal = "fair"
        case .serious: thermal = "serious"
        case .critical: thermal = "critical"
        @unknown default: thermal = "unknown"
        }

        return SystemSnapshot(
            hostname: processInfo.hostName,
            osVersion: processInfo.operatingSystemVersionString,
            processId: processInfo.processIdentifier,
            physicalMemoryBytes: processInfo.physicalMemory,
            processorCount: processInfo.processorCount,
            recommendedMaxWorkingSetBytes: recommended,
            thermalState: thermal,
            mlxCacheLimitBytes: mlxCacheLimit,
            mlxMemoryLimitBytes: mlxMemoryLimit,
            imarelloGitSha: gitSha(),
            gpuName: gpuName,
            metalSupport: metalSupport,
            hasNeuralAccelerators: neural,
            appleGpuFamilyRaw: caps.appleGPUFamilyRaw
        )
    }

    private static func gitSha() -> String? {
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        task.arguments = ["rev-parse", "--short", "HEAD"]
        task.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice
        do {
            try task.run()
            task.waitUntilExit()
            guard task.terminationStatus == 0 else { return nil }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            return nil
        }
    }
}
