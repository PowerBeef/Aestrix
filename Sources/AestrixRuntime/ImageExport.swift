import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import AestrixCore

/// Write MLX image tensors to PNG (macOS / iOS ImageIO).
enum ImageExport {
    /// - Parameter image: NCHW float RGB in roughly `[-1, 1]` or `[0, 1]`.
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
        // Detect range: if any < -0.05 treat as [-1,1]
        let minV = floats.min() ?? 0
        if minV < -0.05 {
            floats = floats.map { ($0 + 1) * 0.5 }
        }
        var bytes = [UInt8](repeating: 0, count: h * w * 4)
        for i in 0 ..< (h * w) {
            let r: Float
            let g: Float
            let b: Float
            if c == 3 {
                r = floats[i * 3]
                g = floats[i * 3 + 1]
                b = floats[i * 3 + 2]
            } else {
                r = floats[i]
                g = r
                b = r
            }
            bytes[i * 4] = clamp01(r)
            bytes[i * 4 + 1] = clamp01(g)
            bytes[i * 4 + 2] = clamp01(b)
            bytes[i * 4 + 3] = 255
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

    private static func clamp01(_ v: Float) -> UInt8 {
        let x = max(0, min(1, v))
        return UInt8(x * 255 + 0.5)
    }
}
