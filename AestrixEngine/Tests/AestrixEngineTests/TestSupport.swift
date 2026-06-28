import Foundation
import Testing
@testable import AestrixEngine

/// Shared test helpers. Fixtures live under `.build/fixtures/flux2-klein-4b-4bit`
/// (gitignored, cached across runs). `swift test` runs from the package dir.
enum TestFixtures {
    /// Override with `AESTRIX_FIXTURES=<abs path>` (e.g. under xcodebuild, whose cwd differs).
    static let root = URL(fileURLWithPath:
        ProcessInfo.processInfo.environment["AESTRIX_FIXTURES"]
        ?? ".build/fixtures/flux2-klein-4b-4bit")

    /// Generated outputs (e.g. test images) live under `outputs/` at the package root.
    /// Override with `AESTRIX_OUTPUTS=<abs path>`.
    static let outputsRoot: URL = {
        if let env = ProcessInfo.processInfo.environment["AESTRIX_OUTPUTS"], !env.isEmpty {
            return URL(fileURLWithPath: env)
        }
        // Anchor to the package root by walking up from this source file to Package.swift.
        // `#filePath` is stable across `swift test` and `xcodebuild` because it is the
        // compile-time source path, independent of the test runner's cwd.
        var dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        while dir.path != "/" {
            let packageManifest = dir.appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: packageManifest.path) {
                return dir.appendingPathComponent("outputs", isDirectory: true)
            }
            let parent = dir.deletingLastPathComponent()
            guard parent != dir else { break }
            dir = parent
        }
        fatalError("Could not locate Package.swift relative to the test source; set AESTRIX_OUTPUTS explicitly.")
    }()

    /// Ensure the given subset of model files is present (downloads + size-verifies).
    static func ensure(_ files: [ModelFile]) async throws {
        let downloader = ModelDownloader(
            files: files, baseURL: Flux2Klein4B4Bit.baseURL, destinationRoot: root)
        try await downloader.ensureDownloaded()
    }

    /// Pull a single file from the manifest by relative path.
    static func manifestFile(_ relativePath: String) -> ModelFile {
        guard let f = Flux2Klein4B4Bit.files.first(where: { $0.relativePath == relativePath }) else {
            fatalError("unknown manifest file: \(relativePath)")
        }
        return f
    }
}

/// Heavy tests download multi-MB/GB shards and/or run MLX — opt in with `AESTRIX_HEAVY_TESTS=1`.
let heavyEnabled = ProcessInfo.processInfo.environment["AESTRIX_HEAVY_TESTS"] != nil
