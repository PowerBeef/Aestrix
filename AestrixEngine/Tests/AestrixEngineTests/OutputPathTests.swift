import Foundation
import Testing
@testable import AestrixEngine

@Suite("Output paths")
struct OutputPathTests {

    @Test("outputsRoot is anchored to the package root")
    func outputsRootIsAnchoredToPackage() {
        let path = TestFixtures.outputsRoot.path
        #expect(path.hasSuffix("AestrixEngine/outputs"), "outputsRoot should resolve to AestrixEngine/outputs, got: \(path)")
    }

    #if canImport(AppKit)
    @Test("savePNG creates parent directories")
    func savePNGCreatesParentDirectory() throws {
        let image = AestrixImage(
            width: 4,
            height: 4,
            rgba: Array(repeating: 128, count: 4 * 4 * 4))
        let outURL = TestFixtures.outputsRoot
            .appendingPathComponent("test_save_png_creates_dir")
            .appendingPathComponent("first.png")
        _ = try image.savePNG(to: outURL)
        #expect(FileManager.default.fileExists(atPath: outURL.path))
        // Clean up
        try? FileManager.default.removeItem(at: outURL.deletingLastPathComponent())
    }
    #endif
}
