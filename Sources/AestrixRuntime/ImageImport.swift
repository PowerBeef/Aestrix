import Foundation
import CoreGraphics
import ImageIO
import MLX
import AestrixCore

/// Load images from disk into MLX NCHW tensors for VAE encode.
enum ImageImport {
    /// Load RGB (or grayscale→RGB) image, center-crop/resize to `width`×`height` (multiples of 16),
    /// return NCHW float in **[-1, 1]**.
    static func loadNCHW(
        url: URL,
        width: Int,
        height: Int
    ) throws -> MLXArray {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AestrixError.imageLoadFailed(path: url.path, reason: "file not found")
        }
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else {
            throw AestrixError.imageLoadFailed(path: url.path, reason: "not a decodable image")
        }

        let target = try resizeAndCrop(cgImage, width: width, height: height)
        let w = target.width
        let h = target.height
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
            throw AestrixError.imageLoadFailed(path: url.path, reason: "CGContext failed")
        }
        ctx.draw(target, in: CGRect(x: 0, y: 0, width: w, height: h))

        // HWC RGB float [0,1] then to [-1,1]
        var floats = [Float](repeating: 0, count: 3 * h * w)
        for y in 0 ..< h {
            for x in 0 ..< w {
                let i = y * w + x
                let o = i * 4
                floats[0 * h * w + i] = Float(rgba[o]) / 255.0 * 2 - 1      // R plane
                floats[1 * h * w + i] = Float(rgba[o + 1]) / 255.0 * 2 - 1  // G
                floats[2 * h * w + i] = Float(rgba[o + 2]) / 255.0 * 2 - 1  // B
            }
        }
        return MLXArray(floats).reshaped([1, 3, h, w])
    }

    /// Snap dimensions down to multiple of 16 (VAE×patch).
    static func alignDimensions(width: Int, height: Int, maxSide: Int) -> (width: Int, height: Int) {
        let multiple = ModelConstants.vaeScaleFactor * 2  // 16
        var w = min(width, maxSide)
        var h = min(height, maxSide)
        w = max(multiple, (w / multiple) * multiple)
        h = max(multiple, (h / multiple) * multiple)
        return (w, h)
    }

    /// Prefer request size; else image native size (aligned).
    static func resolveCanvas(
        imageURL: URL,
        requestWidth: Int?,
        requestHeight: Int?,
        maxSide: Int
    ) throws -> (width: Int, height: Int) {
        if let rw = requestWidth, let rh = requestHeight {
            return alignDimensions(width: rw, height: rh, maxSide: maxSide)
        }
        guard let src = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let iw = props[kCGImagePropertyPixelWidth] as? Int,
              let ih = props[kCGImagePropertyPixelHeight] as? Int
        else {
            // Fallback default
            return alignDimensions(width: maxSide, height: maxSide, maxSide: maxSide)
        }
        let w = requestWidth ?? iw
        let h = requestHeight ?? ih
        return alignDimensions(width: w, height: h, maxSide: maxSide)
    }

    private static func resizeAndCrop(_ image: CGImage, width: Int, height: Int) throws -> CGImage {
        // Scale to cover target, then center crop.
        let sw = CGFloat(image.width)
        let sh = CGFloat(image.height)
        let tw = CGFloat(width)
        let th = CGFloat(height)
        let scale = max(tw / sw, th / sh)
        let rw = sw * scale
        let rh = sh * scale
        let ox = (tw - rw) / 2
        let oy = (th - rh) / 2

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        )
        guard let ctx = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else {
            throw AestrixError.imageLoadFailed(path: "", reason: "resize context failed")
        }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: ox, y: oy, width: rw, height: rh))
        guard let out = ctx.makeImage() else {
            throw AestrixError.imageLoadFailed(path: "", reason: "resize makeImage failed")
        }
        return out
    }
}
