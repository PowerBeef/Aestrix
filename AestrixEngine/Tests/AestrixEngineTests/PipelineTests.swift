import Foundation
import Testing
import MLX
@testable import AestrixEngine

@Suite("End-to-end pipeline")
struct PipelineTests {

    @Test("generate a 256×256 image from a prompt", .enabled(if: heavyEnabled))
    func generateImage() async throws {
        // Ensure all model shards are present
        try await TestFixtures.ensure(Flux2Klein4B4Bit.files)

        let engine = AestrixEngine()
        let report = try await engine.load(modelDir: TestFixtures.root)
        #expect(report.isComplete, "missing files: \(report.missingFiles)")

        let config = GenConfig(width: 256, height: 256, steps: 4, seed: 42)
        let image = try await engine.generate(prompt: "a cat sitting on a windowsill", config: config)

        // Save to disk so we can see it (macOS only; savePNG is AppKit-gated)
        #if canImport(AppKit)
        let outURL = TestFixtures.outputsRoot.appendingPathComponent("first_image.png")
        try image.savePNG(to: outURL)
        print("🖼️ Image saved to \(outURL.path)")
        #endif

        #expect(image.width == 256)
        #expect(image.height == 256)
        #expect(image.rgba.count == 256 * 256 * 4)

        // Basic sanity: not all-black, not all-white, not all-same
        let firstByte = image.rgba[0]
        var allSame = true
        for byte in image.rgba.prefix(1000) {
            if byte != firstByte { allSame = false; break }
        }
        #expect(!allSame, "Image appears to be a solid color — pipeline may not be working")
    }
}
