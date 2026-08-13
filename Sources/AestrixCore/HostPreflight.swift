import Foundation

#if os(macOS)
import Darwin
#endif

/// Process-wide host checks so two Metal jobs cannot share an 8 GB Mac.
///
/// Acquires an exclusive lock under `~/Library/Caches/Aestrix/aestrix.lock`.
/// Warns on modest swap; fails hard when another live `aestrix` holds the lock
/// or swap is extreme (watchdog / WindowServer starvation class).
public enum HostPreflight {
    public static let swapWarnBytes: UInt64 = 256 * 1024 * 1024
    public static let swapFailBytes: UInt64 = 1536 * 1024 * 1024

    public struct Snapshot: Sendable, Equatable {
        public var physicalMemoryBytes: UInt64
        public var swapUsedBytes: UInt64
        public var otherAestrixPIDs: [Int32]
        public var notes: [String]

        public var isLowMemoryHost: Bool {
            physicalMemoryBytes < 12 * 1_073_741_824
        }
    }

    /// Exclusive lock held for the life of the process. Mutations only on `lockQueue`.
    nonisolated(unsafe) private static var lockHandle: FileHandle?
    private static let lockQueue = DispatchQueue(label: "aestrix.host-preflight")

    public static func snapshot() -> Snapshot {
        let swap = swapUsedBytes() ?? 0
        let others = siblingAestrixPIDs()
        var notes: [String] = []
        if swap >= swapWarnBytes {
            notes.append("swap_used=\(swap)")
        }
        if !others.isEmpty {
            notes.append("other_aestrix=\(others.map(String.init).joined(separator: ","))")
        }
        return Snapshot(
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            swapUsedBytes: swap,
            otherAestrixPIDs: others,
            notes: notes
        )
    }

    /// Call once from CLI commands that touch Metal / MLX.
    /// - Parameter force: skip swap-fail (still takes the lock).
    public static func acquireForCLI(force: Bool = false) throws {
        try lockQueue.sync {
            if lockHandle != nil { return }
            let snap = snapshot()
            if !snap.otherAestrixPIDs.isEmpty {
                throw AestrixError.anotherInstanceRunning(pids: snap.otherAestrixPIDs)
            }
            if snap.swapUsedBytes >= swapFailBytes, !force {
                throw AestrixError.hostMemoryPressure(
                    detail: "swap \(formatBytes(snap.swapUsedBytes)) exceeds "
                        + "\(formatBytes(swapFailBytes)); another GPU/editor job "
                        + "is likely active. Quit it or pass --force-headroom."
                )
            }
            try takeLockFile()
            if snap.swapUsedBytes >= swapWarnBytes {
                fputs(
                    "warning: host swap is \(formatBytes(snap.swapUsedBytes)) "
                        + "on a \(formatBytes(snap.physicalMemoryBytes)) machine. "
                        + "Avoid 1024² generate/bench until swap is 0.\n",
                    stderr
                )
            }
            if snap.isLowMemoryHost {
                fputs(
                    "note: 8 GB-class host — agents should prefer swift test / 512²; "
                        + "one Metal owner at a time (see AGENTS.md).\n",
                    stderr
                )
            }
        }
    }

    public static func formatBytes(_ bytes: UInt64) -> String {
        let mb = Double(bytes) / 1_048_576.0
        if mb >= 1024 {
            return String(format: "%.1f GiB", mb / 1024.0)
        }
        return String(format: "%.0f MiB", mb)
    }

    // MARK: - lock

    static func lockFileURL() -> URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let dir = caches.appendingPathComponent("Aestrix", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("aestrix.lock")
    }

    private static func takeLockFile() throws {
        let url = lockFileURL()
        let path = url.path
        if !FileManager.default.fileExists(atPath: path) {
            FileManager.default.createFile(atPath: path, contents: nil)
        }
        guard let handle = FileHandle(forUpdatingAtPath: path) else {
            throw AestrixError.hostMemoryPressure(detail: "could not open \(path)")
        }
        #if os(macOS)
        let fd = handle.fileDescriptor
        if flock(fd, LOCK_EX | LOCK_NB) != 0 {
            handle.closeFile()
            throw AestrixError.anotherInstanceRunning(pids: siblingAestrixPIDs())
        }
        let pid = String(ProcessInfo.processInfo.processIdentifier)
        try handle.truncate(atOffset: 0)
        handle.write(Data(pid.utf8))
        #endif
        lockHandle = handle
    }

    /// Other live processes whose comm contains `aestrix` (not this pid).
    public static func siblingAestrixPIDs() -> [Int32] {
        #if os(macOS)
        let selfPid = ProcessInfo.processInfo.processIdentifier
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/ps")
        task.arguments = ["-axo", "pid=,comm="]
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = FileHandle.nullDevice
        do {
            try task.run()
            task.waitUntilExit()
            guard task.terminationStatus == 0 else { return [] }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            guard let text = String(data: data, encoding: .utf8) else { return [] }
            return parsePSForAestrix(text, excluding: selfPid)
        } catch {
            return []
        }
        #else
        return []
        #endif
    }

    static func parsePSForAestrix(_ output: String, excluding selfPid: Int32) -> [Int32] {
        output.split(whereSeparator: \.isNewline).compactMap { line -> Int32? in
            let parts = line.split(maxSplits: 1, omittingEmptySubsequences: true, whereSeparator: \.isWhitespace)
            guard parts.count == 2, let pid = Int32(parts[0]), pid != selfPid else { return nil }
            let comm = String(parts[1])
            let base = (comm as NSString).lastPathComponent
            guard base == "aestrix" || comm.hasSuffix("/aestrix") else { return nil }
            return pid
        }
    }

    public static func swapUsedBytes() -> UInt64? {
        #if os(macOS)
        var usage = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        let result = sysctlbyname("vm.swapusage", &usage, &size, nil, 0)
        return result == 0 ? usage.xsu_used : nil
        #else
        return nil
        #endif
    }
}
