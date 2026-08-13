import Testing
import Foundation
@testable import AestrixCore

@Suite("Host preflight")
struct HostPreflightTests {

    @Test("parsePSForAestrix ignores self and non-aestrix")
    func parsePS() {
        let selfPid: Int32 = 99
        let out = """
        99 /Users/me/.build/release/aestrix
        100 /usr/bin/swift
        101 aestrix
        102 /tmp/aestrix-helper
        103 Cursor Helper
        """
        let pids = HostPreflight.parsePSForAestrix(out, excluding: selfPid)
        #expect(pids == [101])
    }

    @Test("snapshot reports physical memory")
    func snapshotMemory() {
        let snap = HostPreflight.snapshot()
        #expect(snap.physicalMemoryBytes > 1_000_000_000)
        #expect(HostPreflight.formatBytes(256 * 1024 * 1024) == "256 MiB")
        #expect(HostPreflight.formatBytes(2 * 1024 * 1024 * 1024) == "2.0 GiB")
    }
}
