import Testing
import Foundation
import CoreGraphics
import ImageIO
@testable import AestrixEval
import AestrixCore

/// P1: CI golden floors for synthetic fixtures (no model weights required).
///
/// Floors are intentionally conservative: they catch catastrophic regressions
/// (broken decode, all-black, color gate inversion), not aesthetic quality.
@Suite("Golden metric floors")
struct GoldenMetricFloorsTests {

    @Test("solid blue: technical floor + color match")
    func solidBlueFloors() throws {
        let url = try writeSolid(r: 20, g: 40, b: 200, name: "gold_blue", side: 128)
        defer { try? FileManager.default.removeItem(at: url) }
        let report = try ImageAnalyzer.analyze(
            imageURL: url,
            options: .init(prompt: "a cobalt blue ceramic mug", skipSemantic: true)
        )
        #expect(report.technical.technicalScore >= 35)
        #expect(report.technical.dominantHue == "blue")
        #expect(report.promptAlignment.colorMatch == true)
        #expect(report.overallScore >= 45)
        #expect(!report.findings.contains { $0.severity == .fail })
    }

    @Test("identical pair: SSIM and LPIPS-lite floors")
    func identicalPerceptualFloors() throws {
        let a = try writeSolid(r: 100, g: 120, b: 140, name: "gold_id_a")
        let b = try writeSolid(r: 100, g: 120, b: 140, name: "gold_id_b")
        defer {
            try? FileManager.default.removeItem(at: a)
            try? FileManager.default.removeItem(at: b)
        }
        let report = try ImageAnalyzer.analyze(
            imageURL: a,
            options: .init(referenceURL: b, skipSemantic: true)
        )
        let ref = try #require(report.reference)
        #expect(ref.ssim >= 0.99)
        #expect(ref.perceptualDistance <= 0.05)
        #expect(ref.perceptualScore >= 90)
        #expect(ref.fidelityScore >= 90)
    }

    @Test("opposite colors: low fidelity floor")
    func oppositeColorsLowFidelity() throws {
        let a = try writeSolid(r: 220, g: 20, b: 20, name: "gold_opp_a")
        let b = try writeSolid(r: 20, g: 20, b: 220, name: "gold_opp_b")
        defer {
            try? FileManager.default.removeItem(at: a)
            try? FileManager.default.removeItem(at: b)
        }
        let report = try ImageAnalyzer.analyze(
            imageURL: a,
            options: .init(prompt: "red", referenceURL: b, skipSemantic: true)
        )
        let ref = try #require(report.reference)
        #expect(ref.ssim < 0.5 || ref.meanColorDistance > 0.4)
        #expect(ref.perceptualDistance > 0.15)
        #expect(ref.fidelityScore < 75)
    }

    @Test("high strength + high SSIM + color edit → strength_too_low_for_edit")
    func strengthAwareGate() throws {
        let a = try writeSolid(r: 200, g: 30, b: 30, name: "str_a")
        let b = try writeSolid(r: 200, g: 30, b: 30, name: "str_b")
        defer {
            try? FileManager.default.removeItem(at: a)
            try? FileManager.default.removeItem(at: b)
        }
        // Identical images but strength 0.85 + blue prompt → should warn edit not applied
        let report = try ImageAnalyzer.analyze(
            imageURL: a,
            options: .init(
                prompt: "make the mug solid blue",
                referenceURL: b,
                i2iStrength: 0.85,
                skipSemantic: true
            )
        )
        #expect(report.findings.contains { $0.code == "strength_too_low_for_edit" })
    }

    @Test("schema version is 1.3+")
    func schemaVersion() throws {
        let url = try writeSolid(r: 10, g: 10, b: 10, name: "schema")
        defer { try? FileManager.default.removeItem(at: url) }
        let report = try ImageAnalyzer.analyze(imageURL: url, options: .init(skipSemantic: true))
        #expect(report.version == "1.3" || report.version.compare("1.3", options: .numeric) != .orderedAscending)
    }

    // MARK: - Helpers

    private func writeSolid(r: UInt8, g: UInt8, b: UInt8, name: String, side: Int = 64) throws -> URL {
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
            throw AestrixError.imageLoadFailed(path: name, reason: "solid write failed")
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("aestrix_gold_\(name)_\(UUID().uuidString).png")
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, "public.png" as CFString, 1, nil
        ) else {
            throw AestrixError.imageLoadFailed(path: url.path, reason: "dest")
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw AestrixError.imageLoadFailed(path: url.path, reason: "finalize")
        }
        return url
    }
}
