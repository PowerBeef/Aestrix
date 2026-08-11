import Foundation
import CoreGraphics
import ImageIO
import AestrixCore

/// Linear RGB float buffer in [0,1], row-major H×W×3 (no alpha).
public struct PixelBuffer: Sendable {
    public let width: Int
    public let height: Int
    /// Length = width * height * 3, planar interleaved RGB.
    public let rgb: [Float]

    public var pixelCount: Int { width * height }

    public init(width: Int, height: Int, rgb: [Float]) {
        precondition(rgb.count == width * height * 3)
        self.width = width
        self.height = height
        self.rgb = rgb
    }

    public static func load(url: URL, maxSide: Int? = 1024) throws -> PixelBuffer {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AestrixError.imageLoadFailed(path: url.path, reason: "file not found")
        }
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else {
            throw AestrixError.imageLoadFailed(path: url.path, reason: "not a decodable image")
        }
        return try from(cgImage: cg, maxSide: maxSide)
    }

    public static func from(cgImage: CGImage, maxSide: Int? = 1024) throws -> PixelBuffer {
        var w = cgImage.width
        var h = cgImage.height
        if let maxSide, max(w, h) > maxSide {
            let scale = Float(maxSide) / Float(max(w, h))
            w = max(1, Int(Float(w) * scale))
            h = max(1, Int(Float(h) * scale))
        }

        let bytesPerRow = w * 4
        var rgba = [UInt8](repeating: 0, count: h * bytesPerRow)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        )
        guard let ctx = CGContext(
            data: &rgba,
            width: w,
            height: h,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw AestrixError.imageLoadFailed(path: "", reason: "CGContext failed")
        }
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

        var rgb = [Float](repeating: 0, count: w * h * 3)
        for i in 0 ..< (w * h) {
            let o = i * 4
            rgb[i * 3] = Float(rgba[o]) / 255
            rgb[i * 3 + 1] = Float(rgba[o + 1]) / 255
            rgb[i * 3 + 2] = Float(rgba[o + 2]) / 255
        }
        return PixelBuffer(width: w, height: h, rgb: rgb)
    }

    public func luminance(at i: Int) -> Float {
        let r = rgb[i * 3]
        let g = rgb[i * 3 + 1]
        let b = rgb[i * 3 + 2]
        // Rec. 709
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    public func luminances() -> [Float] {
        (0 ..< pixelCount).map { luminance(at: $0) }
    }

    public func meanRGB() -> (r: Float, g: Float, b: Float) {
        var r: Float = 0, g: Float = 0, b: Float = 0
        let n = Float(pixelCount)
        for i in 0 ..< pixelCount {
            r += rgb[i * 3]
            g += rgb[i * 3 + 1]
            b += rgb[i * 3 + 2]
        }
        return (r / n, g / n, b / n)
    }
}
