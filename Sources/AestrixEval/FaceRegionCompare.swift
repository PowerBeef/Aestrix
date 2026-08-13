import Foundation
import CoreGraphics
import Vision

/// Face-region full-reference metrics for identity-I2I regression gates.
///
/// The largest detected face is compared in each image after a modest crop expansion.
/// This is an image-quality signal, not a biometric identity classifier.
public enum FaceRegionCompare {
    public struct Metrics: Sendable, Codable, Equatable {
        public var generatedFaceCount: Int
        public var referenceFaceCount: Int
        public var ssim: Float?
        public var msSSIM: Float?
        public var perceptualScore: Float?
        public var fidelityScore: Float?
    }

    public static func compare(
        generatedURL: URL,
        referenceURL: URL,
        maxSide: Int = 1024,
        expansion: CGFloat = 0.18
    ) throws -> Metrics {
        let (_, generatedImage) = try PixelBuffer.loadWithCGImage(
            url: generatedURL, maxSide: maxSide)
        let (_, referenceImage) = try PixelBuffer.loadWithCGImage(
            url: referenceURL, maxSide: maxSide)
        let generatedFaces = detectFaces(in: generatedImage)
        let referenceFaces = detectFaces(in: referenceImage)

        guard let generatedBox = primaryFace(generatedFaces),
              let referenceBox = primaryFace(referenceFaces),
              let generatedCrop = crop(generatedImage, visionBox: generatedBox, expansion: expansion),
              let referenceCrop = crop(referenceImage, visionBox: referenceBox, expansion: expansion)
        else {
            return Metrics(
                generatedFaceCount: generatedFaces.count,
                referenceFaceCount: referenceFaces.count,
                ssim: nil,
                msSSIM: nil,
                perceptualScore: nil,
                fidelityScore: nil
            )
        }

        let generatedPixels = try PixelBuffer.from(cgImage: generatedCrop, maxSide: 384)
        let referencePixels = try PixelBuffer.from(cgImage: referenceCrop, maxSide: 384)
        let metrics = ReferenceCompare.compare(
            generated: generatedPixels, reference: referencePixels)
        return Metrics(
            generatedFaceCount: generatedFaces.count,
            referenceFaceCount: referenceFaces.count,
            ssim: metrics.ssim,
            msSSIM: metrics.msSSIM,
            perceptualScore: metrics.perceptualScore,
            fidelityScore: metrics.fidelityScore
        )
    }

    private static func detectFaces(in image: CGImage) -> [CGRect] {
        let request = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        do {
            try handler.perform([request])
            return request.results?.map(\.boundingBox) ?? []
        } catch {
            return []
        }
    }

    private static func primaryFace(_ boxes: [CGRect]) -> CGRect? {
        boxes.max { lhs, rhs in
            lhs.width * lhs.height < rhs.width * rhs.height
        }
    }

    private static func crop(
        _ image: CGImage,
        visionBox: CGRect,
        expansion: CGFloat
    ) -> CGImage? {
        let expanded = visionBox.insetBy(
            dx: -visionBox.width * expansion,
            dy: -visionBox.height * expansion
        ).intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        guard expanded.width > 0, expanded.height > 0 else { return nil }

        // Vision uses normalized bottom-left coordinates; CGImage cropping uses top-left.
        let pixelRect = CGRect(
            x: expanded.minX * CGFloat(image.width),
            y: (1 - expanded.maxY) * CGFloat(image.height),
            width: expanded.width * CGFloat(image.width),
            height: expanded.height * CGFloat(image.height)
        ).integral.intersection(
            CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard pixelRect.width >= 8, pixelRect.height >= 8 else { return nil }
        return image.cropping(to: pixelRect)
    }
}
