import Foundation
#if canImport(Metal)
import Metal
#endif
import ImarelloCore

public enum SystemInfo {
    public static func snapshot(mlxCacheLimit: UInt64? = nil, mlxMemoryLimit: UInt64? = nil) -> SystemSnapshot {
        let processInfo = ProcessInfo.processInfo
        var recommended: UInt64?
        var gpuName: String?
        var metalSupport: String?
        var neural: Bool?
        #if canImport(Metal)
        if let device = MTLCreateSystemDefaultDevice() {
            recommended = UInt64(device.recommendedMaxWorkingSetSize)
            gpuName = device.name
            // Metal 4 is the API generation on macOS 26; Neural Accelerators are M5+ only.
            if #available(macOS 26.0, iOS 26.0, *) {
                metalSupport = "Metal 4"
            } else {
                metalSupport = "Metal"
            }
            let name = device.name.uppercased()
            // Heuristic: Apple documents Neural Accelerators for M5 / A19-class, not M1–M4.
            neural = name.contains("M5") || name.contains("M6") || name.contains("M7")
                || name.contains("A19")
        }
        #endif

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
            hasNeuralAccelerators: neural
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
