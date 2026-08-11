import Foundation
import CoreGraphics
import Vision
import MLX
import AestrixCore

/// Soft face masks in **packed latent** space for regional I2I strength / clean-pull.
///
/// Uses Vision face detection on the same canvas the VAE sees. When no face is found,
/// returns an all-zero mask (no regional bias).
enum FaceIdentityMask {
    /// Build a soft face mask `[1, packedH·packedW, 1]` in float32, values in [0, 1].
    ///
    /// - Parameters:
    ///   - imageURL: Source image (resized/cropped like VAE encode).
    ///   - width / height: Pixel canvas (multiples of 16).
    ///   - expand: Extra margin around the face box (fraction of box size).
    ///   - softEdge: Softness of falloff outside the expanded box (fraction of packed min side).
    static func softPackedMask(
        imageURL: URL,
        width: Int,
        height: Int,
        expand: Float = 0.18,
        softEdge: Float = 0.12
    ) throws -> (mask: MLXArray, faceCount: Int) {
        let (packedH, packedW) = LatentOps.packedSpatial(width: width, height: height)
        let cg = try ImageImport.loadCGImage(url: imageURL, width: width, height: height)
        let boxes = detectFaces(in: cg)
        guard let primary = boxes.max(by: { $0.width * $0.height < $1.width * $1.height }) else {
            let zeros = [Float](repeating: 0, count: packedH * packedW)
            return (MLXArray(zeros).reshaped([1, packedH * packedW, 1]), 0)
        }

        // Vision: origin bottom-left, normalized. Convert to top-left pixel coords.
        let px = Float(primary.origin.x) * Float(width)
        let pyBottom = Float(primary.origin.y) * Float(height)
        let pw = Float(primary.width) * Float(width)
        let ph = Float(primary.height) * Float(height)
        let pyTop = Float(height) - pyBottom - ph

        // Expand box.
        let cx = px + pw * 0.5
        let cy = pyTop + ph * 0.5
        let ew = pw * (1 + expand)
        let eh = ph * (1 + expand)
        var x0 = cx - ew * 0.5
        var y0 = cy - eh * 0.5
        var x1 = cx + ew * 0.5
        var y1 = cy + eh * 0.5
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(Float(width), x1); y1 = min(Float(height), y1)

        // Map to packed coords (pixel / 16).
        let scale = Float(ModelConstants.vaeScaleFactor * 2)  // 16
        let lx0 = x0 / scale
        let ly0 = y0 / scale
        let lx1 = x1 / scale
        let ly1 = y1 / scale
        let soft = max(0.5, softEdge * Float(min(packedH, packedW)))

        var values = [Float](repeating: 0, count: packedH * packedW)
        for py in 0 ..< packedH {
            for px in 0 ..< packedW {
                // Cell center
                let cy = Float(py) + 0.5
                let cx = Float(px) + 0.5
                let dx: Float
                if cx < lx0 { dx = lx0 - cx }
                else if cx > lx1 { dx = cx - lx1 }
                else { dx = 0 }
                let dy: Float
                if cy < ly0 { dy = ly0 - cy }
                else if cy > ly1 { dy = cy - ly1 }
                else { dy = 0 }
                let dist = sqrt(dx * dx + dy * dy)
                let v: Float
                if dist <= 0 {
                    v = 1
                } else if dist >= soft {
                    v = 0
                } else {
                    // Cosine ease: 1 → 0 over soft edge
                    let t = dist / soft
                    v = 0.5 * (1 + cos(Float.pi * t))
                }
                values[py * packedW + px] = v
            }
        }
        let mask = MLXArray(values).reshaped([1, packedH * packedW, 1])
        return (mask, boxes.count)
    }

    /// Host-side face rectangles in Vision normalized coordinates (bottom-left origin).
    static func detectFaces(in image: CGImage) -> [CGRect] {
        let request = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return []
        }
        guard let results = request.results else { return [] }
        return results.map(\.boundingBox)
    }
}
