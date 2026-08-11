import Testing
import Foundation
import CoreGraphics
import ImageIO
@testable import AestrixCore
@testable import AestrixRuntime

@Suite("I2I identity preserve helpers")
struct IdentityPreserveTests {

    @Test("clean-pull alpha decays across steps")
    func cleanPullDecay() {
        let base: Float = 0.4
        let a0 = LatentOps.cleanPullAlpha(base: base, step: 0, totalSteps: 4, decay: true)
        let a3 = LatentOps.cleanPullAlpha(base: base, step: 3, totalSteps: 4, decay: true)
        #expect(abs(a0 - base) < 1e-5)
        #expect(a3 < a0)
        #expect(abs(a3 - base * 0.25) < 1e-4)

        let flat = LatentOps.cleanPullAlpha(base: base, step: 3, totalSteps: 4, decay: false)
        #expect(abs(flat - base) < 1e-5)
    }

    @Test("solid image yields zero face mask")
    func noFaceZeroMask() throws {
        let url = try writeSolidPNG(width: 128, height: 128, r: 40, g: 80, b: 200)
        defer { try? FileManager.default.removeItem(at: url) }
        let (mask, count) = try FaceIdentityMask.softPackedMask(
            imageURL: url, width: 128, height: 128)
        #expect(count == 0)
        #expect(mask.shape == [1, 64, 1])  // 128/16 = 8 → 8*8=64
        // All zeros
        let flat = mask.asArray(Float.self)
        #expect(flat.allSatisfy { abs($0) < 1e-6 })
    }

    @Test("portrait demo has a face mask when available")
    func demoPortraitFaceIfPresent() throws {
        let demo = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("outputs/demo/woman_t2i.png")
        guard FileManager.default.fileExists(atPath: demo.path) else {
            return  // skip if demo not in CWD
        }
        let (mask, count) = try FaceIdentityMask.softPackedMask(
            imageURL: demo, width: 512, height: 512)
        // Best-effort: if Vision finds a face, mask has mass.
        if count > 0 {
            let flat = mask.asArray(Float.self)
            let sum = flat.reduce(0, +)
            #expect(sum > 1)
            #expect(flat.max()! > 0.9)
        }
    }

    // MARK: - helpers

    private func writeSolidPNG(width: Int, height: Int, r: UInt8, g: UInt8, b: UInt8) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("aestrix-face-test-\(UUID().uuidString).png")
        let bytesPerRow = width * 4
        var rgba = [UInt8](repeating: 255, count: height * bytesPerRow)
        for i in 0 ..< (width * height) {
            let o = i * 4
            rgba[o] = r; rgba[o + 1] = g; rgba[o + 2] = b; rgba[o + 3] = 255
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        )
        guard let ctx = CGContext(
            data: &rgba, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: cs, bitmapInfo: bitmapInfo.rawValue
        ), let cg = ctx.makeImage() else {
            throw AestrixError.imageLoadFailed(path: url.path, reason: "test PNG context failed")
        }
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)
        else {
            throw AestrixError.imageLoadFailed(path: url.path, reason: "destination failed")
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw AestrixError.imageLoadFailed(path: url.path, reason: "finalize failed")
        }
        return url
    }
}
