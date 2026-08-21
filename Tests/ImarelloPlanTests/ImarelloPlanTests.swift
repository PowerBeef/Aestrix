import Foundation
import Testing
import ImarelloCore
@testable import ImarelloPlan

@Suite("Klein execution plan")
struct ImarelloPlanTests {
    @Test("model specification preserves product locks")
    func modelSpecification() {
        let spec = KleinModelSpecification.current
        #expect(spec.doubleBlocks == 5)
        #expect(spec.singleBlocks == 20)
        #expect(spec.attentionHeads == 24)
        #expect(spec.headDimension == 128)
        #expect(spec.jointAttentionDimension == 7_680)
        #expect(spec.maximumTextLength == 512)
        #expect(spec.textEncoderTaps == [9, 18, 27])
        #expect(spec.weightBits == 4)
    }

    @Test(
        "legacy Direct scratch catalog reproduces measured raw placements",
        arguments: [
            (side: 512, bytes: 188_743_680),
            (side: 768, bytes: 334_233_600),
            (side: 1_024, bytes: 537_919_488),
        ])
    func rawScratch(input: (side: Int, bytes: Int)) throws {
        let layout = try DirectScratchCatalog.legacyLayout(
            width: input.side, height: input.side)
        #expect(layout.rawPeakBytes == input.bytes)
        #expect(layout.imageTokens == (input.side / 16) * (input.side / 16))
    }

    @Test("legacy execution plan reproduces the placement peak and debug plan disables aliasing")
    func legacyPlanPlacement() throws {
        for (side, peak) in [(512, 188_743_680), (768, 334_233_600), (1_024, 537_919_488)] {
            let release = try DirectLegacyPlanFactory.make(width: side, height: side)
            #expect(release.placement.peakBytes == peak)
            #expect(release.placement.aliasingEnabled)
            #expect(try release.verifyDigest())
        }

        let debug = try DirectLegacyPlanFactory.make(
            width: 512, height: 512, debugNoAlias: true, canaryBytes: 64)
        #expect(!debug.placement.aliasingEnabled)
        #expect(debug.placement.peakBytes > 188_743_680)
        #expect(debug.placement.placements.allSatisfy { $0.canaryPrefixBytes == 4_096 })
    }

    @Test("odd rectangular packed grid uses ceiling chunk coverage")
    func oddRectangularCatalog() throws {
        let layout = try DirectScratchCatalog.legacyLayout(width: 528, height: 784)
        #expect(layout.imageTokens == 33 * 49)
        #expect(layout.rawPeakBytes > 0)
    }

    @Test("non-overlapping lifetimes alias while boundary overlaps do not")
    func deterministicAliasing() throws {
        let regions = [
            region("a", bytes: 4_096, first: 0, last: 1),
            region("b", bytes: 4_096, first: 2, last: 3),
            region("c", bytes: 4_096, first: 1, last: 2),
        ]
        let result = try PlacementPlanner.place(regions)
        let placements = Dictionary(
            uniqueKeysWithValues: result.placements.map { ($0.regionID, $0) })
        #expect(placements["a"]?.offset == 0)
        #expect(placements["b"]?.offset == 0)
        #expect(placements["c"]?.offset == 4_096)
        #expect(result.peakBytes == 8_192)

        let debug = try PlacementPlanner.place(
            regions,
            options: PlacementOptions(
                aliasingEnabled: false, minimumAlignment: 4_096, canaryBytes: 64))
        #expect(debug.peakBytes == 36_864)
        #expect(debug.placements.allSatisfy { $0.canaryPrefixBytes == 4_096 })
    }

    @Test("independent verifier rejects overlapping live allocations")
    func rejectsOverlap() {
        let regions = [
            region("a", bytes: 4_096, first: 0, last: 2),
            region("b", bytes: 4_096, first: 1, last: 3),
        ]
        let invalid = PlacementResult(
            peakBytes: 4_096,
            aliasingEnabled: true,
            placements: [
                RegionPlacement(
                    regionID: "a", offset: 0, allocationOffset: 0,
                    reservedLength: 4_096, canaryPrefixBytes: 0, canarySuffixBytes: 0),
                RegionPlacement(
                    regionID: "b", offset: 0, allocationOffset: 0,
                    reservedLength: 4_096, canaryPrefixBytes: 0, canarySuffixBytes: 0),
            ])
        #expect(throws: PlanError.overlappingPlacements("a", "b")) {
            try PlanVerifier.verify(regions: regions, placement: invalid)
        }
    }

    @Test("canonical plan digest is stable across region input ordering")
    func deterministicDigest() throws {
        let regions = [
            region("input", bytes: 4_096, first: 0, last: 0),
            region("output", bytes: 4_096, first: 0, last: 1),
        ]
        let placement = try PlacementPlanner.place(regions)
        let operation = PlanOperation(
            id: "op0",
            kernelIdentifier: "imarello_test_kernel",
            backend: .steel,
            stage: .dit,
            bindings: [
                PlanBinding(slot: 1, regionID: "output", access: .write),
                PlanBinding(slot: 0, regionID: "input", access: .read),
            ],
            constants: [
                PlanConstant(name: "rows", bitPattern: 16),
                PlanConstant(name: "cols", bitPattern: 32),
            ],
            dispatch: DispatchGeometry(grid: [16, 1, 1], threadgroup: [16, 1, 1]))
        let first = try ExecutionPlan.make(
            capabilities: PlanCapabilities(requiredFeatures: ["argument-tables"]),
            numericalProfile: .exactW4A16FP32Residual,
            operations: [operation], regions: regions, placement: placement)
        let second = try ExecutionPlan.make(
            capabilities: PlanCapabilities(requiredFeatures: ["argument-tables"]),
            numericalProfile: .exactW4A16FP32Residual,
            operations: [operation], regions: Array(regions.reversed()), placement: placement)
        #expect(first.digest == second.digest)
        #expect(try first.verifyDigest())
        #expect(try first.canonicalJSON() == second.canonicalJSON())
    }

    private func region(
        _ id: String, bytes: Int, first: Int, last: Int
    ) -> TensorRegion {
        TensorRegion(
            id: id,
            dataType: .float16,
            logicalShape: [bytes / 2],
            byteCount: bytes,
            alignment: 4_096,
            storage: .scratch,
            stage: .dit,
            firstUse: first,
            lastUse: last)
    }
}
