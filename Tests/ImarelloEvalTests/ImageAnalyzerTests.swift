import Testing
import Foundation
import CoreGraphics
import ImageIO
@testable import ImarelloEval
import ImarelloCore

@Suite("Image analysis harness")
struct ImageAnalyzerTests {
    @Test func solidBlueIsBlueHueAndPromptMatches() throws {
        let url = try writeSolidPNG(r: 20, g: 40, b: 200, name: "solid_blue")
        defer { try? FileManager.default.removeItem(at: url) }

        let report = try ImageAnalyzer.analyze(
            imageURL: url,
            options: .init(prompt: "a cobalt blue ceramic mug on wood")
        )
        #expect(report.technical.dominantHue == "blue")
        #expect(report.promptAlignment.colorMatch == true)
        #expect(report.promptAlignment.requestedColors.contains("blue"))
        #expect(report.findings.contains { $0.code == "color_match" })
        #expect(report.overallScore > 50)
    }

    @Test func solidRedVsBluePromptFailsColor() throws {
        let url = try writeSolidPNG(r: 200, g: 30, b: 30, name: "solid_red")
        defer { try? FileManager.default.removeItem(at: url) }

        let report = try ImageAnalyzer.analyze(
            imageURL: url,
            options: .init(prompt: "a solid blue mug")
        )
        #expect(report.technical.dominantHue == "red")
        #expect(report.promptAlignment.colorMatch == false)
        #expect(report.findings.contains { $0.severity == .fail && $0.code == "color_mismatch" })
    }

    @Test func identicalImagesHaveHighSSIM() throws {
        let a = try writeSolidPNG(r: 100, g: 120, b: 140, name: "id_a")
        let b = try writeSolidPNG(r: 100, g: 120, b: 140, name: "id_b")
        defer {
            try? FileManager.default.removeItem(at: a)
            try? FileManager.default.removeItem(at: b)
        }
        let report = try ImageAnalyzer.analyze(
            imageURL: a,
            options: .init(referenceURL: b)
        )
        #expect(report.reference != nil)
        #expect(report.reference!.ssim > 0.99)
        #expect(report.reference!.meanAbsError < 1e-4)
    }

    @Test func differentColorsLowFidelity() throws {
        let a = try writeSolidPNG(r: 220, g: 20, b: 20, name: "diff_a")
        let b = try writeSolidPNG(r: 20, g: 20, b: 220, name: "diff_b")
        defer {
            try? FileManager.default.removeItem(at: a)
            try? FileManager.default.removeItem(at: b)
        }
        let report = try ImageAnalyzer.analyze(
            imageURL: a,
            options: .init(prompt: "red", referenceURL: b)
        )
        #expect(report.reference!.meanColorDistance > 0.5)
        #expect(report.reference!.fidelityScore < 70)
    }

    @Test func jsonRoundTrip() throws {
        let url = try writeSolidPNG(r: 10, g: 200, b: 30, name: "json")
        defer { try? FileManager.default.removeItem(at: url) }
        let report = try ImageAnalyzer.analyze(imageURL: url, options: .init(prompt: "green leaf"))
        let s = try ImageAnalysisReportBuilder.jsonString(report)
        #expect(s.contains("technical_score") || s.contains("technicalScore") || s.contains("overall"))
        #expect(s.contains("tile_seam") || s.contains("tileSeam") || s.contains("expects_vae"))
        #expect(!s.isEmpty)
    }

    @Test func largeCanvasExpectsVAETilingAndReportsSeamFields() throws {
        // 768+ triggers expectsVAETiling (Imarello auto-tiles decode).
        let url = try writeSolidPNG(r: 40, g: 80, b: 160, name: "large_blue", side: 800)
        defer { try? FileManager.default.removeItem(at: url) }
        let report = try ImageAnalyzer.analyze(imageURL: url, options: .init(prompt: "blue"))
        #expect(report.technical.expectsVAETiling == true)
        #expect(report.technical.tileSeamScore >= 0)
        #expect(report.findings.contains { $0.code == "vae_tile_expected" || $0.code == "possible_tile_seam" })
    }

    @Test func multiColorPromptMismatchIsWarnNotFail() throws {
        // Solid red image vs multi-color prompt that does not include red.
        let url = try writeSolidPNG(r: 200, g: 30, b: 30, name: "multi_red")
        defer { try? FileManager.default.removeItem(at: url) }
        let report = try ImageAnalyzer.analyze(
            imageURL: url,
            options: .init(prompt: "a blue bowl and a green table with purple flowers")
        )
        #expect(report.promptAlignment.colorMatch == false)
        #expect(report.promptAlignment.requestedColors.count > 1)
        let mm = report.findings.first { $0.code == "color_mismatch" }
        #expect(mm != nil)
        #expect(mm?.severity == .warn)
    }

    @Test func mergingVisionRaisesOverallWhenVisionStrong() throws {
        let url = try writeSolidPNG(r: 20, g: 40, b: 200, name: "vis_merge")
        defer { try? FileManager.default.removeItem(at: url) }
        let pixel = try ImageAnalyzer.analyze(
            imageURL: url,
            options: .init(prompt: "blue mug")
        )
        let assessment = VisionReview.Assessment(
            mode: .t2i,
            imagePath: url.path,
            prompt: "blue mug",
            caption: "A blue mug.",
            answers: ["subject": "mug", "color": "blue", "artifacts": "none"],
            findings: [
                .init(severity: .info, code: "subject_ok", message: "Clear mug subject.")
            ],
            visionScore: 95,
            verdict: "pass"
        )
        let merged = ImageAnalysisReportBuilder.mergingVision(pixel, assessment)
        #expect(merged.vision != nil)
        #expect(merged.findings.contains { $0.code == "vision_subject_ok" })
        #expect(merged.overallScore > pixel.overallScore * 0.4)
        #expect(merged.summary.contains("vision"))
    }

    // MARK: - Helpers

    private func writeSolidPNG(r: UInt8, g: UInt8, b: UInt8, name: String, side: Int = 64) throws -> URL {
        let w = side, h = side
        var rgba = [UInt8](repeating: 255, count: w * h * 4)
        for i in 0 ..< (w * h) {
            rgba[i * 4] = r
            rgba[i * 4 + 1] = g
            rgba[i * 4 + 2] = b
            rgba[i * 4 + 3] = 255
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGBitmapInfo.byteOrder32Big.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        )
        guard let ctx = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w * 4, space: cs, bitmapInfo: info.rawValue
        ), let cg = ctx.makeImage() else {
            throw ImarelloError.imageLoadFailed(path: name, reason: "test solid failed")
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("imarello_eval_\(name)_\(UUID().uuidString).png")
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, "public.png" as CFString, 1, nil
        ) else {
            throw ImarelloError.imageLoadFailed(path: url.path, reason: "dest")
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw ImarelloError.imageLoadFailed(path: url.path, reason: "finalize")
        }
        return url
    }
}
