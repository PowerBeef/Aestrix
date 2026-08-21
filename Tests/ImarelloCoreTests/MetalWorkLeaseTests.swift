import Testing
@testable import ImarelloCore

@Suite("Process-wide Metal lease", .serialized)
struct MetalWorkLeaseTests {
    @Test("a detached owner cannot overlap an active lease")
    func contentionFailsClosed() async throws {
        let acquired = AsyncStream.makeStream(of: Void.self)
        let release = AsyncStream.makeStream(of: Void.self)
        let holder = Task.detached {
            try await MetalWorkLease.withLease {
                acquired.continuation.yield()
                for await _ in release.stream { break }
            }
        }
        var iterator = acquired.stream.makeAsyncIterator()
        _ = await iterator.next()

        await #expect(throws: ImarelloError.self) {
            try await Task.detached {
                try await MetalWorkLease.withLease { true }
            }.value
        }
        release.continuation.yield()
        release.continuation.finish()
        _ = try await holder.value
        acquired.continuation.finish()
    }
}
