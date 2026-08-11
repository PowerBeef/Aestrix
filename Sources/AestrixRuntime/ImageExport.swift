import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Accelerate
import MLX
import AestrixCore

/// Write MLX image tensors to PNG (macOS / iOS ImageIO).
enum ImageExport {
    /// - Parameter image: NCHW float RGB in roughly `[-1, 1]` (FLUX VAE convention) or `[0, 1]`.
    static func writePNG(_ image: MLXArray, to url: URL) throws {
        var x = image
        if x.ndim == 4 {
            x = x[0]  // CHW
        }
        precondition(x.ndim == 3, "expected CHW image, got shape \(x.shape)")

        // CHW → HWC
        x = x.transposed(1, 2, 0)
        eval(x)
        let h = x.dim(0)
        let w = x.dim(1)
        let c = x.dim(2)
        precondition(c == 3 || c == 1, "expected 1 or 3 channels, got \(c)")

        var floats = x.asArray(Float.self)
        let pixelCount = h * w

        // FLUX VAE decode is in [-1, 1]. Probe a few samples (avoid full min scan).
        let probe = min(256, floats.count)
        var minProbe = Float.infinity
        for i in 0 ..< probe {
            minProbe = min(minProbe, floats[i])
        }
        let isNegOneToOne = minProbe < -0.05

        if isNegOneToOne {
            var one: Float = 1
            var half: Float = 0.5
            vDSP_vsadd(floats, 1, &one, &floats, 1, vDSP_Length(floats.count))
            vDSP_vsmul(floats, 1, &half, &floats, 1, vDSP_Length(floats.count))
        }
        var zero: Float = 0
        var oneF: Float = 1
        vDSP_vclip(floats, 1, &zero, &oneF, &floats, 1, vDSP_Length(floats.count))
        var scale: Float = 255
        vDSP_vsmul(floats, 1, &scale, &floats, 1, vDSP_Length(floats.count))

        var bytes = [UInt8](repeating: 255, count: pixelCount * 4)
        if c == 3 {
            // Interleaved RGB → RGBA (opaque). Floats already scaled to [0, 255].
            for i in 0 ..< pixelCount {
                let o = i * 4
                let s = i * 3
                bytes[o] = UInt8(clamping: Int(floats[s].rounded()))
                bytes[o + 1] = UInt8(clamping: Int(floats[s + 1].rounded()))
                bytes[o + 2] = UInt8(clamping: Int(floats[s + 2].rounded()))
            }
        } else {
            for i in 0 ..< pixelCount {
                let u = UInt8(clamping: Int(floats[i].rounded()))
                let o = i * 4
                bytes[o] = u
                bytes[o + 1] = u
                bytes[o + 2] = u
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        )
        guard let ctx = CGContext(
            data: &bytes,
            width: w,
            height: h,
            bitsPerComponent: 8,
            bytesPerRow: w * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ), let cgImage = ctx.makeImage() else {
            throw AestrixError.unsupportedWeightFormat("failed to create CGImage for PNG export")
        }

        let dir = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL,
            UTType.png.identifier as CFString,
            1,
            nil
        ) else {
            throw AestrixError.unsupportedWeightFormat("CGImageDestination failed for \(url.path)")
        }
        CGImageDestinationAddImage(dest, cgImage, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw AestrixError.unsupportedWeightFormat("failed to write PNG at \(url.path)")
        }
    }
}
