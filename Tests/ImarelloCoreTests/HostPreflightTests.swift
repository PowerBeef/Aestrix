import Testing
import Foundation
@testable import ImarelloCore

@Suite("Host preflight")
struct HostPreflightTests {

    @Test("parsePSForProduct ignores self and non-product")
    func parsePS() {
        let selfPid: Int32 = 99
        let out = """
        99 /Users/me/.build/release/imarello
        100 /usr/bin/swift
        101 imarello
        102 /tmp/imarello-helper
        103 Cursor Helper
        104 aestrix
        """
        let pids = HostPreflight.parsePSForProduct(out, excluding: selfPid)
        #expect(pids == [101, 104])
    }

    @Test("snapshot reports physical memory")
    func snapshotMemory() {
        let snap = HostPreflight.snapshot()
        #expect(snap.physicalMemoryBytes > 1_000_000_000)
        #expect(HostPreflight.formatBytes(256 * 1024 * 1024) == "256 MiB")
        #expect(HostPreflight.formatBytes(2 * 1024 * 1024 * 1024) == "2.0 GiB")
    }
}
