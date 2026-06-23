import Foundation
import os

/// Lightweight structured logger. Uses `os.Logger` so logs land in Console.app and
/// Xcode without pulling in a third-party dependency.
public enum AestrixLog {
    private static let subsystem = "ai.aestrix.engine"
    private static let general = Logger(subsystem: subsystem, category: "engine")
    private static let memory = Logger(subsystem: subsystem, category: "memory")
    private static let perf = Logger(subsystem: subsystem, category: "perf")

    public static func info(_ message: String) { general.info("\(message, privacy: .public)") }
    public static func notice(_ message: String) { general.notice("\(message, privacy: .public)") }
    public static func error(_ message: String) { general.error("\(message, privacy: .public)") }

    public static func memory(_ message: String) { memory.debug("\(message, privacy: .public)") }
    public static func perf(_ message: String) { perf.debug("\(message, privacy: .public)") }
}
