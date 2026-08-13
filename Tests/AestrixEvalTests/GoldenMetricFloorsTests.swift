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

    @Test("Docs/eval-floors.json is present and machine-independent")
    func floorsDocumentExists() throws {
        let doc = try loadFloorsDocument()
        #expect(doc.schemaVersion == "1.0")
        #expect(doc.machineIndependent)
        #expect(doc.floors.solidBlueColorMatch.requireColorMatch)
        // Flat fixture ceiling is Float(40).nextDown — a 40 floor can never pass.
        #expect(doc.floors.solidBlueColorMatch.minTechnicalScore < 40)
        #expect(doc.floors.identicalImages.minSsim >= 0.99)
        #expect(doc.floors.oppositeColors.maxFidelityScore <= 75)
        #expect(doc.floors.strengthAwareIdenticalColorEdit.expectFindingCode == "strength_too_low_for_edit")
    }

    @Test("solid blue: technical floor + color match")
    func solidBlueFloors() throws {
        let floor = try loadFloorsDocument().floors.solidBlueColorMatch
        let url = try writeSolid(r: 20, g: 40, b: 200, name: "gold_blue", side: 128)
        defer { try? FileManager.default.removeItem(at: url) }
        let report = try ImageAnalyzer.analyze(
            imageURL: url,
            options: .init(prompt: floor.prompt, skipSemantic: true)
        )
        #expect(report.technical.technicalScore >= Float(floor.minTechnicalScore))
        #expect(report.technical.dominantHue == "blue")
        #expect(report.promptAlignment.colorMatch == floor.requireColorMatch)
        #expect(report.overallScore >= Float(floor.minOverall))
        let failCount = report.findings.filter { $0.severity == .fail }.count
        #expect(failCount <= floor.maxFailFindings)
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
        let floor = try loadFloorsDocument().floors.identicalImages
        let ref = try #require(report.reference)
        #expect(ref.ssim >= Float(floor.minSsim))
        #expect(ref.perceptualDistance <= Float(floor.maxPerceptualDistance))
        #expect(ref.perceptualScore >= Float(floor.minPerceptualScore))
        #expect(ref.fidelityScore >= Float(floor.minFidelityScore))
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
        let floor = try loadFloorsDocument().floors.oppositeColors
        let ref = try #require(report.reference)
        #expect(ref.ssim < 0.5 || ref.meanColorDistance > 0.4)
        #expect(ref.perceptualDistance > Float(floor.minPerceptualDistance))
        #expect(ref.fidelityScore < Float(floor.maxFidelityScore))
    }

    @Test("high strength + high SSIM + color edit → strength_too_low_for_edit")
    func strengthAwareGate() throws {
        let a = try writeSolid(r: 200, g: 30, b: 30, name: "str_a")
        let b = try writeSolid(r: 200, g: 30, b: 30, name: "str_b")
        defer {
            try? FileManager.default.removeItem(at: a)
            try? FileManager.default.removeItem(at: b)
        }
        let floor = try loadFloorsDocument().floors.strengthAwareIdenticalColorEdit
        // Identical images but high strength + color-edit prompt → warn edit not applied
        let report = try ImageAnalyzer.analyze(
            imageURL: a,
            options: .init(
                prompt: "make the mug solid blue",
                referenceURL: b,
                i2iStrength: Float(floor.i2iStrength),
                skipSemantic: true
            )
        )
        #expect(report.findings.contains { $0.code == floor.expectFindingCode })
    }

    @Test("solid field occupies the vacuous ~40 technical bucket")
    func solidFieldVacuousTechnicalBucket() throws {
        let url = try writeSolid(r: 20, g: 40, b: 200, name: "vacuous_blue", side: 128)
        defer { try? FileManager.default.removeItem(at: url) }
        let pixels = try PixelBuffer.load(url: url, maxSide: 1024)
        var rMin = Float.greatestFiniteMagnitude, rMax: Float = -.greatestFiniteMagnitude
        var gMin = Float.greatestFiniteMagnitude, gMax: Float = -.greatestFiniteMagnitude
        var bMin = Float.greatestFiniteMagnitude, bMax: Float = -.greatestFiniteMagnitude
        for i in 0 ..< pixels.pixelCount {
            let r = pixels.rgb[i * 3], g = pixels.rgb[i * 3 + 1], b = pixels.rgb[i * 3 + 2]
            rMin = min(rMin, r); rMax = max(rMax, r)
            gMin = min(gMin, g); gMax = max(gMax, g)
            bMin = min(bMin, b); bMax = max(bMax, b)
        }
        let report = try ImageAnalyzer.analyze(
            imageURL: url,
            options: .init(prompt: "a cobalt blue ceramic mug", skipSemantic: true)
        )
        let t = report.technical
        // RGB is bit-identical after PNG; contrast/noise dust is Float32 reduction, not spatial variation.
        #expect(rMax - rMin < 1e-3)
        #expect(gMax - gMin < 1e-3)
        #expect(bMax - bMin < 1e-3)
        #expect(t.sharpnessLaplacianVar == 0)
        #expect(t.meanGradient == 0)
        #expect(t.stdLuminance < 1e-4)
        #expect(t.contrastRMS < 1e-4)
        #expect(t.clipBlackFraction == 0)
        #expect(t.clipWhiteFraction == 0)
        #expect(t.noiseProxy < 1e-6)
        #expect(t.luminanceEntropy == 0)
        #expect(t.expectsVAETiling == false)
        // compositeScore keeps only noise+clip+seam weights: 0.14+0.18+0.08.
        // That sum is 1 ULP below 0.4 in Float32, so 100*sum == Float(40).nextDown.
        #expect(t.technicalScore == Float(40).nextDown)
        #expect(t.technicalScore < 40)
        #expect(report.overallScore >= 45)
        #expect(report.promptAlignment.alignmentScore > 50)
    }

    @Test("schema version is 1.3+")
    func schemaVersion() throws {
        let url = try writeSolid(r: 10, g: 10, b: 10, name: "schema")
        defer { try? FileManager.default.removeItem(at: url) }
        let report = try ImageAnalyzer.analyze(imageURL: url, options: .init(skipSemantic: true))
        #expect(report.version == "1.3" || report.version.compare("1.3", options: .numeric) != .orderedAscending)
    }

    // MARK: - Helpers

    private struct EvalFloorsFile: Decodable {
        let schemaVersion: String
        let machineIndependent: Bool
        let floors: Floors
        struct Floors: Decodable {
            let solidBlueColorMatch: SolidBlue
            let identicalImages: Identical
            let oppositeColors: Opposite
            let strengthAwareIdenticalColorEdit: StrengthAware
        }
        struct SolidBlue: Decodable {
            let prompt: String
            let minTechnicalScore: Double
            let requireColorMatch: Bool
            let minOverall: Double
            let maxFailFindings: Int
        }
        struct Identical: Decodable {
            let minSsim: Double
            let maxPerceptualDistance: Double
            let minPerceptualScore: Double
            let minFidelityScore: Double
        }
        struct Opposite: Decodable {
            let maxFidelityScore: Double
            let minPerceptualDistance: Double
        }
        struct StrengthAware: Decodable {
            let i2iStrength: Double
            let expectFindingCode: String
        }
    }

    private func loadFloorsDocument() throws -> EvalFloorsFile {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Docs/eval-floors.json")
        let data = try Data(contentsOf: url)
        let dec = JSONDecoder()
        dec.keyDecodingStrategy = .convertFromSnakeCase
        return try dec.decode(EvalFloorsFile.self, from: data)
    }

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
