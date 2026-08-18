import Foundation

#if os(macOS)
import Darwin
#endif

/// Process-wide host checks so two Metal jobs cannot share an 8 GB Mac.
///
/// Acquires an exclusive lock under `~/Library/Caches/Imarello/imarello.lock`.
/// Fails only when another live `imarello` holds the lock. Swap is not a gate.
public enum HostPreflight {
    public struct Snapshot: Sendable, Equatable {
        public var physicalMemoryBytes: UInt64
        public var swapUsedBytes: UInt64
        public var otherImarelloPIDs: [Int32]
        public var notes: [String]

        public var isLowMemoryHost: Bool {
            physicalMemoryBytes < 12 * 1_073_741_824
        }
    }

    /// Exclusive lock held for the life of the process. Mutations only on `lockQueue`.
    nonisolated(unsafe) private static var lockHandle: FileHandle?
    private static let lockQueue = DispatchQueue(label: "imarello.host-preflight")

    public static func snapshot() -> Snapshot {
        let swap = swapUsedBytes() ?? 0
        let others = siblingImarelloPIDs()
        var notes: [String] = []
        if !others.isEmpty {
            notes.append("other_imarello=\(others.map(String.init).joined(separator: ","))")
        }
        return Snapshot(
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            swapUsedBytes: swap,
            otherImarelloPIDs: others,
            notes: notes
        )
    }

    /// Call once from CLI commands that touch Metal / MLX.
    public static func acquireForCLI() throws {
        try lockQueue.sync {
            if lockHandle != nil { return }
            let snap = snapshot()
            if !snap.otherImarelloPIDs.isEmpty {
                throw ImarelloError.anotherInstanceRunning(pids: snap.otherImarelloPIDs)
            }
            try takeLockFile()
        }
    }

    /// Hand the lock to serial child processes (res-ladder). The parent only
    /// spawns and waits after this — it must not touch Metal again.
    public static func releaseForSubprocesses() {
        lockQueue.sync {
            guard let handle = lockHandle else { return }
            #if os(macOS)
            flock(handle.fileDescriptor, LOCK_UN)
            #endif
            handle.closeFile()
            lockHandle = nil
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
        let dir = caches.appendingPathComponent("Imarello", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("imarello.lock")
    }

    private static func takeLockFile() throws {
        let url = lockFileURL()
        let path = url.path
        if !FileManager.default.fileExists(atPath: path) {
            FileManager.default.createFile(atPath: path, contents: nil)
        }
        guard let handle = FileHandle(forUpdatingAtPath: path) else {
            throw ImarelloError.hostMemoryPressure(detail: "could not open \(path)")
        }
        #if os(macOS)
        let fd = handle.fileDescriptor
        if flock(fd, LOCK_EX | LOCK_NB) != 0 {
            handle.closeFile()
            throw ImarelloError.anotherInstanceRunning(pids: siblingImarelloPIDs())
        }
        let pid = String(ProcessInfo.processInfo.processIdentifier)
        try handle.truncate(atOffset: 0)
        handle.write(Data(pid.utf8))
        #endif
        lockHandle = handle
    }

    /// Other live processes whose comm basename is exactly `imarello`/`aestrix`
    /// (not this pid).
    public static func siblingImarelloPIDs() -> [Int32] {
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
            return parsePSForProduct(text, excluding: selfPid)
        } catch {
            return []
        }
        #else
        return []
        #endif
    }

    static func parsePSForProduct(_ output: String, excluding selfPid: Int32) -> [Int32] {
        // A res-ladder parent hands the lock to its children and marks itself
        // ignorable — it only spawns and waits while a rung owns Metal.
        let ignoredPid = ProcessInfo.processInfo.environment["IMARELLO_IGNORE_PID"].flatMap(Int32.init)
        return output.split(whereSeparator: \.isNewline).compactMap { line -> Int32? in
            let parts = line.split(maxSplits: 1, omittingEmptySubsequences: true, whereSeparator: \.isWhitespace)
            guard parts.count == 2, let pid = Int32(parts[0]), pid != selfPid, pid != ignoredPid else { return nil }
            let comm = String(parts[1])
            let base = (comm as NSString).lastPathComponent
            // Exact comm only — not `imarello-helper`. Also match leftover `aestrix` binaries.
            guard base == "imarello" || base == "aestrix" else { return nil }
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
