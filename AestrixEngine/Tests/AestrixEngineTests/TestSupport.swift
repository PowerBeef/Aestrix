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
