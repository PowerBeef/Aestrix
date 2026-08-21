import Foundation
import os

/// Process-wide ownership gate for work that may initialize MLX, submit Metal
/// commands, compile/load kernels, or mutate process-global MLX tuning state.
///
/// The lock only protects acquisition/release; it is never held across an
/// `await`. A task-local token makes intentional nesting (for example a
/// benchmark invoking a pipeline) reentrant without allowing another owner.
public enum MetalWorkLease {
    private final class Coordinator: Sendable {
        private let owner = OSAllocatedUnfairLock<UUID?>(initialState: nil)

        func acquire(_ token: UUID) throws {
            let acquired = owner.withLock { current in
                guard current == nil else { return false }
                current = token
                return true
            }
            guard acquired else { throw ImarelloError.concurrentMetalWorkNotAllowed }
        }

        func release(_ token: UUID) {
            owner.withLock { current in
                if current == token { current = nil }
            }
        }
    }

    private static let coordinator = Coordinator()
    @TaskLocal private static var currentToken: UUID?

    public static func withLease<T: Sendable>(
        _ operation: @Sendable () async throws -> T
    ) async throws -> T {
        if currentToken != nil {
            return try await operation()
        }
        let token = UUID()
        try coordinator.acquire(token)
        do {
            let result = try await $currentToken.withValue(token) {
                try await operation()
            }
            coordinator.release(token)
            return result
        } catch {
            coordinator.release(token)
            throw error
        }
    }

    public static func withLease<T>(
        _ operation: () throws -> T
    ) throws -> T {
        if currentToken != nil {
            return try operation()
        }
        let token = UUID()
        try coordinator.acquire(token)
        do {
            let result = try $currentToken.withValue(token, operation: operation)
            coordinator.release(token)
            return result
        } catch {
            coordinator.release(token)
            throw error
        }
    }
}
