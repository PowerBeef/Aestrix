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

        let git = gitState()
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
            imarelloGitSha: git.sha,
            imarelloGitDirty: git.dirty,
            gpuName: gpuName,
            metalSupport: metalSupport,
            hasNeuralAccelerators: neural,
            appleGpuFamilyRaw: caps.appleGPUFamilyRaw
        )
    }

    private static func gitState() -> (sha: String?, dirty: Bool?) {
        let sourceRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let executableRoot = URL(fileURLWithPath: CommandLine.arguments[0])
            .resolvingSymlinksInPath()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let candidates = [
            sourceRoot,
            executableRoot,
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
        ]
        guard let repository = candidates.first(where: {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent(".git").path)
        }) else { return (nil, nil) }

        func runGit(_ arguments: [String]) -> (status: Int32, output: String)? {
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/usr/bin/git")
            task.arguments = arguments
            task.currentDirectoryURL = repository
            let pipe = Pipe()
            task.standardOutput = pipe
            task.standardError = FileHandle.nullDevice
            do {
                try task.run()
                task.waitUntilExit()
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                return (
                    task.terminationStatus,
                    String(data: data, encoding: .utf8)?
                        .trimmingCharacters(in: .whitespacesAndNewlines) ?? "")
        } catch {
                return nil
            }
        }
        let shaRun = runGit(["rev-parse", "HEAD"])
        let dirtyRun = runGit(["status", "--porcelain", "--untracked-files=no"])
        return (
            shaRun?.status == 0 ? shaRun?.output : nil,
            dirtyRun?.status == 0 ? !(dirtyRun?.output.isEmpty ?? true) : nil)
    }
}
