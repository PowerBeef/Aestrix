import Foundation

/// Inbox job dropped into the iOS app container by `Scripts/ios-device-harness.sh`.
public struct DeviceHarnessJob: Sendable, Equatable, Codable {
    public enum Mode: String, Sendable, Equatable, Codable {
        case t2i
        case i2i
    }

    public var id: String
    public var mode: Mode
    public var prompt: String
    public var width: Int
    public var height: Int
    public var steps: Int
    public var seed: UInt64
    public var textTokens: TextTokenMode
    public var strength: Float

    public init(
        id: String,
        mode: Mode = .t2i,
        prompt: String,
        width: Int = 512,
        height: Int = 512,
        steps: Int = 4,
        seed: UInt64 = 42,
        textTokens: TextTokenMode = .full512,
        strength: Float = 0.8
    ) {
        self.id = id
        self.mode = mode
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.seed = seed
        self.textTokens = textTokens
        self.strength = strength
    }

    public static let allowedSides: Set<Int> = [512, 1024]
    public static let maximumJobIDLength = 128
    public static let maximumSeed: UInt64 = 9_999_999

    public static let defaultFoxPrompt =
        "A red fox in a snowy forest at sunrise, photorealistic, golden rim light, shallow depth of field."

    public static func sanitizedID(_ raw: String) -> String {
        let kept = raw.unicodeScalars.map { scalar -> Character in
            let ok = CharacterSet.alphanumerics.contains(scalar)
                || scalar == "." || scalar == "_" || scalar == "-"
            return ok ? Character(scalar) : "-"
        }
        let joined = String(kept)
        return joined.isEmpty ? "job" : joined
    }

    public func validate(hasLastImage: Bool) throws {
        let id = Self.sanitizedID(self.id)
        guard id == self.id,
              !self.id.hasPrefix("."),
              self.id != ".",
              self.id != "..",
              !self.id.isEmpty,
              self.id.count <= Self.maximumJobIDLength
        else {
            throw ImarelloError.notImplemented("harness job id must be [A-Za-z0-9._-]")
        }
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw ImarelloError.notImplemented("harness job prompt is empty")
        }
        guard width == height else {
            throw ImarelloError.invalidDimensions(width: width, height: height, reason: "must be square")
        }
        guard Self.allowedSides.contains(width) else {
            throw ImarelloError.invalidDimensions(
                width: width,
                height: height,
                reason: "device harness allows 512 or 1024 only"
            )
        }
        guard (1 ... 100).contains(steps) else {
            throw ImarelloError.invalidSteps(steps)
        }
        guard seed <= Self.maximumSeed else {
            throw ImarelloError.invalidRequest(
                "harness job seed must be between 0 and \(Self.maximumSeed)"
            )
        }
        guard strength.isFinite, strength > 0, strength <= 1 else {
            throw ImarelloError.invalidStrength(strength)
        }
        if mode == .i2i {
            guard hasLastImage else {
                throw ImarelloError.imageLoadFailed(
                    path: "<last-in-app>",
                    reason: "i2i harness job needs a last generated PNG (Edit last). No photo import."
                )
            }
        }
    }
}

/// Written to `jobs/done/{id}.json` when the app finishes (or skips) a job.
public struct DeviceHarnessResult: Sendable, Equatable, Codable {
    public enum Status: String, Sendable, Equatable, Codable {
        case ok
        case failed
        case skipped
    }

    public var id: String
    public var status: Status
    public var error: String?
    public var pngRelativePath: String?
    public var width: Int
    public var height: Int
    public var seed: UInt64
    public var elapsedSec: Double?
    public var startedAt: String?
    public var metallibNote: String?

    public init(
        id: String,
        status: Status,
        error: String? = nil,
        pngRelativePath: String? = nil,
        width: Int,
        height: Int,
        seed: UInt64,
        elapsedSec: Double? = nil,
        startedAt: String? = nil,
        metallibNote: String? = nil
    ) {
        self.id = id
        self.status = status
        self.error = error
        self.pngRelativePath = pngRelativePath
        self.width = width
        self.height = height
        self.seed = seed
        self.elapsedSec = elapsedSec
        self.startedAt = startedAt
        self.metallibNote = metallibNote
    }

    public static func skippedSimulator(job: DeviceHarnessJob) -> DeviceHarnessResult {
        DeviceHarnessResult(
            id: job.id,
            status: .skipped,
            error: "Simulator cannot run MLX. Use a physical iPhone.",
            width: job.width,
            height: job.height,
            seed: job.seed
        )
    }
}

/// On-disk layout under `Caches/Imarello/jobs/`.
public enum DeviceHarnessPaths {
    public static let containerJobsPrefix = "Library/Caches/Imarello/jobs"
    public static let containerOutputsPrefix = "Library/Caches/Imarello/outputs"

    public static func root() -> URL { AppCache.directory("jobs") }
    public static func inbox() -> URL { root().appendingPathComponent("inbox", isDirectory: true) }
    public static func running() -> URL { root().appendingPathComponent("running", isDirectory: true) }
    public static func done() -> URL { root().appendingPathComponent("done", isDirectory: true) }

    public static func ensureDirectories() throws {
        let fm = FileManager.default
        for dir in [inbox(), running(), done(), AppCache.directory("outputs")] {
            var isDir: ObjCBool = false
            if fm.fileExists(atPath: dir.path, isDirectory: &isDir), !isDir.boolValue {
                try fm.removeItem(at: dir)
            }
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
    }

    public static func inboxFile(id: String) -> URL {
        inbox().appendingPathComponent("\(DeviceHarnessJob.sanitizedID(id)).json")
    }

    public static func runningFile(id: String) -> URL {
        running().appendingPathComponent("\(DeviceHarnessJob.sanitizedID(id)).json")
    }

    public static func doneFile(id: String) -> URL {
        done().appendingPathComponent("\(DeviceHarnessJob.sanitizedID(id)).json")
    }

    public static func pngContainerPath(filename: String) -> String {
        "\(containerOutputsPrefix)/\(filename)"
    }

    public static let jsonEncoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }()

    public static let jsonDecoder = JSONDecoder()
}
