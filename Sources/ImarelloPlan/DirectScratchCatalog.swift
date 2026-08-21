import Foundation
import ImarelloCore

public struct DirectScratchLayout: Codable, Sendable, Equatable {
    public var imageTokens: Int
    public var textTokens: Int
    public var persistentBytes: Int
    public var doubleStreamBytes: Int
    public var singleStreamBytes: Int
    public var rawPeakBytes: Int
}

public enum DirectScratchCatalog {
    /// Reproduces Direct V1's placement-heap formula independently of Metal.
    /// The default alignment matches the measured heap alignment on the M2
    /// reference host; runtime code must pass the device-reported alignment.
    public static func legacyLayout(
        width: Int,
        height: Int,
        textTokens: Int = ModelConstants.maxSequenceLength,
        alignment: Int = 4_096
    ) throws -> DirectScratchLayout {
        guard width > 0, height > 0,
              width.isMultiple(of: 16), height.isMultiple(of: 16),
              textTokens > 0, PlacementPlanner.isPowerOfTwo(alignment)
        else { throw PlanError.invalidRegion("canvas") }

        return try legacyLayout(
            imageTokens: (width / 16) * (height / 16),
            textTokens: textTokens,
            alignment: alignment)
    }

    public static func legacyLayout(
        imageTokens: Int,
        textTokens: Int = ModelConstants.maxSequenceLength,
        alignment: Int = 4_096
    ) throws -> DirectScratchLayout {
        guard imageTokens > 0, textTokens > 0,
              PlacementPlanner.isPowerOfTwo(alignment)
        else { throw PlanError.invalidRegion("token-count") }

        let jointTokens = imageTokens + textTokens
        let dimension = ModelConstants.innerDim
        let ffInner = dimension * 3
        let ffWide = ffInner * 2
        let projectionWidth = dimension * 9
        let concatenatedWidth = dimension * 4
        func aligned(_ bytes: Int) -> Int {
            PlacementPlanner.aligned(bytes, to: alignment)
        }
        func sum(_ sizes: [Int]) -> Int { sizes.reduce(0) { $0 + aligned($1) } }

        let persistent = aligned(imageTokens * dimension * 4) * 2
            + aligned(textTokens * dimension * 4) * 2
        let imageChunkRows = (imageTokens + 1) / 2
        let double = sum([
            imageTokens * dimension * 2,
            textTokens * dimension * 2,
            imageTokens * dimension * 2,
            textTokens * dimension * 2,
            jointTokens * dimension * 2,
            jointTokens * dimension * 2,
            jointTokens * dimension * 2,
            jointTokens * dimension * 2,
            imageTokens * dimension * 2,
            textTokens * dimension * 2,
            imageTokens * dimension * 4,
            textTokens * dimension * 4,
            imageChunkRows * ffWide * 2,
            textTokens * ffWide * 2,
            imageChunkRows * ffInner * 2,
            textTokens * ffInner * 2,
            imageTokens * dimension * 2,
            textTokens * dimension * 2,
        ])
        let jointChunkRows = (jointTokens + 1) / 2
        let single = sum([
            jointTokens * dimension * 2,
            jointChunkRows * projectionWidth * 2,
            jointTokens * dimension * 2,
            jointTokens * dimension * 2,
            jointTokens * dimension * 2,
            jointTokens * concatenatedWidth * 2,
            jointTokens * dimension * 2,
        ])
        return DirectScratchLayout(
            imageTokens: imageTokens,
            textTokens: textTokens,
            persistentBytes: persistent,
            doubleStreamBytes: double,
            singleStreamBytes: single,
            rawPeakBytes: persistent + max(double, single))
    }
}
