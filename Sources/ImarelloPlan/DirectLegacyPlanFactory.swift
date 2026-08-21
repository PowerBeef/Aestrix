import Foundation
import ImarelloCore

/// Deterministic placement proof for the current Direct V1 DiT scratch zones.
///
/// The three aggregate regions intentionally mirror the placement heap's
/// persistent, double-stream, and single-stream zones. The two compute phases
/// have disjoint lifetimes and therefore alias; debug plans disable that alias
/// and surround every region with canaries.
public enum DirectLegacyPlanFactory {
    public static func make(
        width: Int,
        height: Int,
        alignment: Int = 4_096,
        debugNoAlias: Bool = false,
        canaryBytes: Int = 0
    ) throws -> ExecutionPlan {
        let layout = try DirectScratchCatalog.legacyLayout(
            width: width, height: height, alignment: alignment)
        return try make(
            layout: layout, alignment: alignment,
            debugNoAlias: debugNoAlias, canaryBytes: canaryBytes)
    }

    public static func make(
        imageTokens: Int,
        alignment: Int = 4_096,
        debugNoAlias: Bool = false,
        canaryBytes: Int = 0
    ) throws -> ExecutionPlan {
        let layout = try DirectScratchCatalog.legacyLayout(
            imageTokens: imageTokens, alignment: alignment)
        return try make(
            layout: layout, alignment: alignment,
            debugNoAlias: debugNoAlias, canaryBytes: canaryBytes)
    }

    private static func make(
        layout: DirectScratchLayout,
        alignment: Int,
        debugNoAlias: Bool,
        canaryBytes: Int
    ) throws -> ExecutionPlan {
        let regions = [
            TensorRegion(
                id: "dit.residual-ping-pong",
                dataType: .uint8,
                logicalShape: [layout.persistentBytes],
                byteCount: layout.persistentBytes,
                alignment: alignment,
                storage: .scratch,
                stage: .dit,
                firstUse: 0,
                lastUse: 1),
            TensorRegion(
                id: "dit.double-stream-scratch",
                dataType: .uint8,
                logicalShape: [layout.doubleStreamBytes],
                byteCount: layout.doubleStreamBytes,
                alignment: alignment,
                storage: .scratch,
                stage: .dit,
                firstUse: 0,
                lastUse: 0),
            TensorRegion(
                id: "dit.single-stream-scratch",
                dataType: .uint8,
                logicalShape: [layout.singleStreamBytes],
                byteCount: layout.singleStreamBytes,
                alignment: alignment,
                storage: .scratch,
                stage: .dit,
                firstUse: 1,
                lastUse: 1),
        ]
        let placement: PlacementResult
        if !debugNoAlias, canaryBytes == 0 {
            // Preserve the legacy two-zone proof exactly: persistent first,
            // then the larger of two disjoint compute lifetimes.
            placement = PlacementResult(
                peakBytes: layout.rawPeakBytes,
                aliasingEnabled: true,
                placements: [
                    RegionPlacement(
                        regionID: "dit.residual-ping-pong",
                        offset: 0,
                        allocationOffset: 0,
                        reservedLength: layout.persistentBytes,
                        canaryPrefixBytes: 0,
                        canarySuffixBytes: 0),
                    RegionPlacement(
                        regionID: "dit.double-stream-scratch",
                        offset: layout.persistentBytes,
                        allocationOffset: layout.persistentBytes,
                        reservedLength: layout.doubleStreamBytes,
                        canaryPrefixBytes: 0,
                        canarySuffixBytes: 0),
                    RegionPlacement(
                        regionID: "dit.single-stream-scratch",
                        offset: layout.persistentBytes,
                        allocationOffset: layout.persistentBytes,
                        reservedLength: layout.singleStreamBytes,
                        canaryPrefixBytes: 0,
                        canarySuffixBytes: 0),
                ])
        } else {
            placement = try PlacementPlanner.place(
                regions,
                options: PlacementOptions(
                    aliasingEnabled: !debugNoAlias,
                    minimumAlignment: alignment,
                    canaryBytes: canaryBytes))
        }
        let operations = [
            PlanOperation(
                id: "dit.double-stream-phase",
                kernelIdentifier: "legacy-direct-double-stream",
                backend: .steel,
                stage: .dit,
                bindings: [],
                constants: [
                    PlanConstant(name: "imageTokens", bitPattern: UInt64(layout.imageTokens)),
                    PlanConstant(name: "textTokens", bitPattern: UInt64(layout.textTokens)),
                ],
                dispatch: DispatchGeometry(grid: [5, 1, 1], threadgroup: [1, 1, 1])),
            PlanOperation(
                id: "dit.single-stream-phase",
                kernelIdentifier: "legacy-direct-single-stream",
                backend: .steel,
                stage: .dit,
                bindings: [],
                constants: [
                    PlanConstant(name: "imageTokens", bitPattern: UInt64(layout.imageTokens)),
                    PlanConstant(name: "textTokens", bitPattern: UInt64(layout.textTokens)),
                ],
                dispatch: DispatchGeometry(grid: [20, 1, 1], threadgroup: [1, 1, 1]),
                dependencies: ["dit.double-stream-phase"]),
        ]
        return try ExecutionPlan.make(
            capabilities: PlanCapabilities(
                requiredFeatures: ["legacy-metal-compute"],
                optionalFeatures: ["metal4-argument-tables"]),
            numericalProfile: .exactW4A16FP32Residual,
            operations: operations,
            regions: regions,
            placement: placement)
    }
}
