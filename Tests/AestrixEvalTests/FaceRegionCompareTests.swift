import Testing
import Foundation
import CoreGraphics
import ImageIO
@testable import AestrixCore
@testable import AestrixEval

@Suite("Face region compare")
struct FaceRegionCompareTests {

    @Test("solid images report zero faces and nil SSIM")
    func noFaceNilMetrics() throws {
        let a = try writeSolidPNG(r: 30, g: 80, b: 200)
        let b = try writeSolidPNG(r: 200, g: 40, b: 40)
        defer {
            try? FileManager.default.removeItem(at: a)
            try? FileManager.default.removeItem(at: b)
        }
        let m = try FaceRegionCompare.compare(generatedURL: a, referenceURL: b, maxSide: 128)
        #expect(m.generatedFaceCount == 0)
        #expect(m.referenceFaceCount == 0)
        #expect(m.ssim == nil)
        #expect(m.fidelityScore == nil)
    }

    private func writeSolidPNG(r: UInt8, g: UInt8, b: UInt8) throws -> URL {
        let width = 64, height = 64
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("aestrix-face-cmp-\(UUID().uuidString).png")
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0 ..< (width * height) {
            let o = i * 4
            rgba[o] = r; rgba[o + 1] = g; rgba[o + 2] = b
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let info = CGBitmapInfo.byteOrder32Big.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue))
        guard let ctx = CGContext(
            data: &rgba, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: cs, bitmapInfo: info.rawValue
        ), let cg = ctx.makeImage(),
           let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)
        else {
            throw AestrixError.imageLoadFailed(path: url.path, reason: "test PNG context failed")
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw AestrixError.imageLoadFailed(path: url.path, reason: "test PNG finalize failed")
        }
        return url
    }
}
