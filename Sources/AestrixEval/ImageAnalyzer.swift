import Foundation

/// Entry point for the image quality / accuracy harness.
public enum ImageAnalyzer {
    public struct Options: Sendable {
        public var prompt: String?
        public var referenceURL: URL?
        /// Downscale long side before metrics (speed). Default 1024.
        public var maxAnalysisSide: Int

        public init(prompt: String? = nil, referenceURL: URL? = nil, maxAnalysisSide: Int = 1024) {
            self.prompt = prompt
            self.referenceURL = referenceURL
            self.maxAnalysisSide = maxAnalysisSide
        }
    }

    /// Analyze a generated image; optional prompt + reference for accuracy metrics.
    public static func analyze(imageURL: URL, options: Options = Options()) throws -> ImageAnalysisReport {
        let pixels = try PixelBuffer.load(url: imageURL, maxSide: options.maxAnalysisSide)
        let technical = TechnicalQuality.analyze(pixels)

        var refMetrics: ReferenceCompare.Metrics?
        var refPath: String?
        if let refURL = options.referenceURL {
            let ref = try PixelBuffer.load(url: refURL, maxSide: options.maxAnalysisSide)
            refMetrics = ReferenceCompare.compare(generated: pixels, reference: ref)
            refPath = refURL.path
        }

        let promptMetrics = PromptAlignment.analyze(prompt: options.prompt, technical: technical)

        return ImageAnalysisReportBuilder.build(
            imagePath: imageURL.path,
            referencePath: refPath,
            prompt: options.prompt,
            technical: technical,
            reference: refMetrics,
            promptAlignment: promptMetrics
        )
    }
}
