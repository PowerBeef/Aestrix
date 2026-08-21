import Foundation
import CoreGraphics
import ImageIO

/// Entry point for the image quality / accuracy harness.
public enum ImageAnalyzer {
    public struct Options: Sendable {
        public var prompt: String?
        public var referenceURL: URL?
        /// Downscale long side before metrics (speed). Default 1024.
        public var maxAnalysisSide: Int
        /// I2I denoise strength (0…1]. Enables strength-aware gates (P2).
        public var i2iStrength: Float?
        /// Prefer Core ML CLIP when models installed (P1). Default true.
        public var preferCoreMLCLIP: Bool
        /// Skip semantic/CLIP path (faster pure pixel).
        public var skipSemantic: Bool
        public var vaeTileConfiguration: VAETileEvaluationConfig

        public init(
            prompt: String? = nil,
            referenceURL: URL? = nil,
            maxAnalysisSide: Int = 1024,
            i2iStrength: Float? = nil,
            preferCoreMLCLIP: Bool = true,
            skipSemantic: Bool = false,
            vaeTileConfiguration: VAETileEvaluationConfig = .productDefault
        ) {
            self.prompt = prompt
            self.referenceURL = referenceURL
            self.maxAnalysisSide = maxAnalysisSide
            self.i2iStrength = i2iStrength
            self.preferCoreMLCLIP = preferCoreMLCLIP
            self.skipSemantic = skipSemantic
            self.vaeTileConfiguration = vaeTileConfiguration
        }
    }

    /// Analyze a generated image; optional prompt + reference for accuracy metrics.
    public static func analyze(imageURL: URL, options: Options = Options()) throws -> ImageAnalysisReport {
        let (pixels, cgImage) = try PixelBuffer.loadWithCGImage(
            url: imageURL, maxSide: options.maxAnalysisSide)
        let technical = TechnicalQuality.analyze(
            pixels, vaeTileConfiguration: options.vaeTileConfiguration)

        var refMetrics: ReferenceCompare.Metrics?
        var faceMetrics: FaceRegionCompare.Metrics?
        var refPath: String?
        if let refURL = options.referenceURL {
            let ref = try PixelBuffer.load(url: refURL, maxSide: options.maxAnalysisSide)
            refMetrics = ReferenceCompare.compare(generated: pixels, reference: ref)
            faceMetrics = try? FaceRegionCompare.compare(
                generatedURL: imageURL, referenceURL: refURL,
                maxSide: options.maxAnalysisSide)
            refPath = refURL.path
        }

        let promptMetrics = PromptAlignment.analyze(prompt: options.prompt, technical: technical)

        let semantic: SemanticAlignment.Metrics?
        if options.skipSemantic {
            semantic = nil
        } else {
            semantic = SemanticAlignment.analyze(
                pixels: pixels,
                prompt: options.prompt,
                cgImage: cgImage,
                preferCoreML: options.preferCoreMLCLIP
            )
        }

        return ImageAnalysisReportBuilder.build(
            imagePath: imageURL.path,
            referencePath: refPath,
            prompt: options.prompt,
            technical: technical,
            reference: refMetrics,
            faceRegion: faceMetrics,
            promptAlignment: promptMetrics,
            semantic: semantic,
            i2iStrength: options.i2iStrength,
            vaeTileConfiguration: options.vaeTileConfiguration
        )
    }
}
