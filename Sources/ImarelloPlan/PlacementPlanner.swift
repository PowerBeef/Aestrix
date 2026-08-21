import Foundation

public struct PlacementOptions: Sendable, Equatable {
    public var aliasingEnabled: Bool
    public var minimumAlignment: Int
    public var canaryBytes: Int

    public init(
        aliasingEnabled: Bool = true,
        minimumAlignment: Int = 1,
        canaryBytes: Int = 0
    ) {
        self.aliasingEnabled = aliasingEnabled
        self.minimumAlignment = minimumAlignment
        self.canaryBytes = canaryBytes
    }
}

public enum PlacementPlanner {
    public static func place(
        _ regions: [TensorRegion], options: PlacementOptions = PlacementOptions()
    ) throws -> PlacementResult {
        guard isPowerOfTwo(options.minimumAlignment), options.canaryBytes >= 0 else {
            throw PlanError.invalidRegion("placement-options")
        }
        try validateRegions(regions)

        let ordered = regions.sorted {
            if $0.firstUse != $1.firstUse { return $0.firstUse < $1.firstUse }
            if $0.byteCount != $1.byteCount { return $0.byteCount > $1.byteCount }
            return $0.id < $1.id
        }
        var placed: [(region: TensorRegion, placement: RegionPlacement)] = []
        var peak = 0

        for region in ordered {
            let alignment = max(region.alignment, options.minimumAlignment)
            let prefix = aligned(options.canaryBytes, to: alignment)
            let reserved = aligned(prefix + region.byteCount + options.canaryBytes, to: alignment)
            let candidates = Set([0] + placed.map {
                aligned($0.placement.allocationOffset + $0.placement.reservedLength, to: alignment)
            })

            let legal = candidates.filter { candidate in
                placed.allSatisfy { existing in
                    let lifetimesConflict = !options.aliasingEnabled
                        || lifetimesOverlap(region, existing.region)
                    guard lifetimesConflict else { return true }
                    return !rangesOverlap(
                        candidate ..< candidate + reserved,
                        existing.placement.allocationOffset
                            ..< existing.placement.allocationOffset
                                + existing.placement.reservedLength)
                }
            }
            guard let allocationOffset = legal.min(by: { lhs, rhs in
                let lhsPeak = max(peak, lhs + reserved)
                let rhsPeak = max(peak, rhs + reserved)
                return lhsPeak == rhsPeak ? lhs < rhs : lhsPeak < rhsPeak
            }) else {
                throw PlanError.invalidPlacement(region.id)
            }

            let placement = RegionPlacement(
                regionID: region.id,
                offset: allocationOffset + prefix,
                allocationOffset: allocationOffset,
                reservedLength: reserved,
                canaryPrefixBytes: prefix,
                canarySuffixBytes: options.canaryBytes)
            placed.append((region, placement))
            peak = max(peak, allocationOffset + reserved)
        }

        let result = PlacementResult(
            peakBytes: peak,
            aliasingEnabled: options.aliasingEnabled,
            placements: placed.map(\.placement).sorted { $0.regionID < $1.regionID })
        try PlanVerifier.verify(regions: regions, placement: result)
        return result
    }

    static func lifetimesOverlap(_ lhs: TensorRegion, _ rhs: TensorRegion) -> Bool {
        lhs.firstUse <= rhs.lastUse && rhs.firstUse <= lhs.lastUse
    }

    static func rangesOverlap(_ lhs: Range<Int>, _ rhs: Range<Int>) -> Bool {
        lhs.lowerBound < rhs.upperBound && rhs.lowerBound < lhs.upperBound
    }

    static func aligned(_ value: Int, to alignment: Int) -> Int {
        (value + alignment - 1) / alignment * alignment
    }

    static func isPowerOfTwo(_ value: Int) -> Bool {
        value > 0 && value & (value - 1) == 0
    }

    private static func validateRegions(_ regions: [TensorRegion]) throws {
        var ids = Set<String>()
        for region in regions {
            guard ids.insert(region.id).inserted else {
                throw PlanError.duplicateRegion(region.id)
            }
            guard !region.id.isEmpty,
                  region.byteCount > 0,
                  isPowerOfTwo(region.alignment),
                  !region.logicalShape.isEmpty,
                  region.logicalShape.allSatisfy({ $0 > 0 }),
                  region.paddedShape.count == region.logicalShape.count,
                  zip(region.paddedShape, region.logicalShape).allSatisfy({ $0 >= $1 }),
                  region.firstUse >= 0,
                  region.firstUse <= region.lastUse
            else {
                throw PlanError.invalidRegion(region.id)
            }
        }
    }
}

public enum PlanVerifier {
    public static func verify(
        operations: [PlanOperation],
        regions: [TensorRegion],
        placement: PlacementResult
    ) throws {
        try verify(regions: regions, placement: placement)
        let regionIDs = Set(regions.map(\.id))
        var seenOperations = Set<String>()
        for operation in operations {
            guard seenOperations.insert(operation.id).inserted else {
                throw PlanError.duplicateOperation(operation.id)
            }
            for dependency in operation.dependencies where !seenOperations.contains(dependency) {
                throw PlanError.invalidDependency(
                    operation: operation.id, dependency: dependency)
            }
            for binding in operation.bindings where !regionIDs.contains(binding.regionID) {
                throw PlanError.missingRegion(
                    operation: operation.id, region: binding.regionID)
            }
        }
    }

    public static func verify(
        regions: [TensorRegion], placement: PlacementResult
    ) throws {
        var byID = [String: TensorRegion]()
        for region in regions {
            guard byID.updateValue(region, forKey: region.id) == nil else {
                throw PlanError.duplicateRegion(region.id)
            }
        }
        var placements = [String: RegionPlacement]()
        for item in placement.placements {
            guard placements.updateValue(item, forKey: item.regionID) == nil else {
                throw PlanError.invalidPlacement(item.regionID)
            }
        }
        for region in regions {
            guard let item = placements[region.id] else {
                throw PlanError.missingPlacement(region.id)
            }
            guard item.offset % region.alignment == 0,
                  item.allocationOffset >= 0,
                  item.offset >= item.allocationOffset,
                  item.reservedLength >= region.byteCount,
                  item.offset + region.byteCount
                    <= item.allocationOffset + item.reservedLength,
                  item.allocationOffset + item.reservedLength <= placement.peakBytes
            else {
                throw PlanError.invalidPlacement(region.id)
            }
        }
        guard placements.count == regions.count else {
            throw PlanError.invalidPlacement("unexpected-region")
        }

        for firstIndex in placement.placements.indices {
            for secondIndex in placement.placements.indices where secondIndex > firstIndex {
                let first = placement.placements[firstIndex]
                let second = placement.placements[secondIndex]
                guard let firstRegion = byID[first.regionID],
                      let secondRegion = byID[second.regionID]
                else {
                    throw PlanError.invalidPlacement("unknown-region")
                }
                let lifetimesConflict = !placement.aliasingEnabled
                    || PlacementPlanner.lifetimesOverlap(firstRegion, secondRegion)
                let bytesConflict = PlacementPlanner.rangesOverlap(
                    first.allocationOffset ..< first.allocationOffset + first.reservedLength,
                    second.allocationOffset ..< second.allocationOffset + second.reservedLength)
                if lifetimesConflict && bytesConflict {
                    throw PlanError.overlappingPlacements(first.regionID, second.regionID)
                }
            }
        }
    }
}
