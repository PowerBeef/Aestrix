import Foundation

/// Runtime VAE tiling provenance recorded with image-analysis reports.
public struct VAETileEvaluationConfig: Sendable, Codable, Equatable {
    public var enabledThresholdLatentPixels: Int
    public var tileSize: Int
    public var overlap: Int
    public var blend: String
    public var outputScale: Int

    public init(
        enabledThresholdLatentPixels: Int = 128,
        tileSize: Int = 128,
        overlap: Int = 16,
        blend: String = "cosine",
        outputScale: Int = 8
    ) {
        self.enabledThresholdLatentPixels = enabledThresholdLatentPixels
        self.tileSize = tileSize
        self.overlap = overlap
        self.blend = blend
        self.outputScale = outputScale
    }

    public static let productDefault = VAETileEvaluationConfig()

    public func tiles(outputWidth: Int, outputHeight: Int) -> Bool {
        guard enabledThresholdLatentPixels > 0, outputScale > 0 else { return false }
        let latentWidth = (outputWidth + outputScale - 1) / outputScale
        let latentHeight = (outputHeight + outputScale - 1) / outputScale
        return max(latentWidth, latentHeight) >= enabledThresholdLatentPixels
    }
}
