import Foundation
import MLX

/// Converts MLX image tensors to platform-neutral `AestrixImage` (RGBA8 bytes).
enum ImageConversion {

    /// Convert an NHWC image tensor in approximately [-1, 1] to an `AestrixImage`.
    /// Clamps first, then renormalizes to [0, 255] (matches the mflux/diffusers decode order).
    static func toAestrixImage(_ pixels: MLXArray) -> AestrixImage {
        // pixels: [B, H, W, C] — take batch 0
        let img = pixels[0]
        let h = img.shape[0], w = img.shape[1], c = img.shape[2]

        // Clamp [-1, 1] → [0, 1] → [0, 255]
        let clamped = maximum(minimum(img, MLXArray(Float(1.0))), MLXArray(Float(-1.0)))
        let normalized = (clamped * MLXArray(Float(0.5)) + MLXArray(Float(0.5))) * MLXArray(Float(255.0))

        // Extract pixel values via item() — flatten and read one at a time.
        // For larger images this should be replaced with a batch CVPixelBuffer / CGImage path.
        let flat = normalized.reshaped([-1]).asType(.float32)
        eval(flat)
        let count = h * w * c
        var rgba = [UInt8](repeating: 0, count: h * w * 4)
        for i in 0..<(h * w) {
            for ch in 0..<min(c, 3) {
                let val = flat[i * c + ch].item(Float.self)
                rgba[i * 4 + ch] = UInt8(max(0, min(255, val.rounded())))
            }
            rgba[i * 4 + 3] = 255  // alpha
        }
        _ = count
        return AestrixImage(width: w, height: h, rgba: rgba)
    }
}
