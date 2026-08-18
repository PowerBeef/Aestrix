import Foundation

/// Monotonic stage timing helper.
public final class StageTimer: @unchecked Sendable {
    private var marks: [String: CFAbsoluteTime] = [:]
    private var completed: [String: Double] = [:]
    private let lock = NSLock()

    public init() {}

    public func begin(_ name: String) {
        lock.lock()
        marks[name] = CFAbsoluteTimeGetCurrent()
        lock.unlock()
    }

    @discardableResult
    public func end(_ name: String) -> Double {
        lock.lock()
        defer { lock.unlock() }
        let start = marks[name] ?? CFAbsoluteTimeGetCurrent()
        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
        completed[name] = ms
        marks[name] = nil
        return ms
    }

    /// Record an externally measured duration under `name`.
    public func record(_ name: String, milliseconds: Double) {
        lock.lock()
        completed[name] = milliseconds
        lock.unlock()
    }

    public func ms(_ name: String) -> Double? {
        lock.lock()
        defer { lock.unlock() }
        return completed[name]
    }

    public func allCompletedMs() -> [String: Double] {
        lock.lock()
        defer { lock.unlock() }
        return completed
    }

    public func measure<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
        begin(name)
        defer { _ = end(name) }
        return try body()
    }

    public func measureAsync<T: Sendable>(
        _ name: String,
        _ body: () async throws -> T
    ) async rethrows -> T {
        begin(name)
        defer { _ = end(name) }
        return try await body()
    }
}
