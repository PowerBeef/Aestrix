import CryptoKit
import Foundation
import ImarelloCore

public enum PlanStage: String, Codable, Sendable, CaseIterable {
    case textEncoder
    case conditioner
    case dit
    case vae
    case export
}

public enum PlanDataType: String, Codable, Sendable, CaseIterable {
    case uint4
    case int4
    case uint8
    case int32
    case float16
    case bfloat16
    case float32

    public var bitsPerElement: Int {
        switch self {
        case .uint4, .int4: 4
        case .uint8: 8
        case .float16, .bfloat16: 16
        case .int32, .float32: 32
        }
    }
}

public enum PlanStorage: String, Codable, Sendable, CaseIterable {
    case scratch
    case stagePersistent
    case bridge
    case mappedWeight
    case output
}

public enum PlanBackend: String, Codable, Sendable, CaseIterable {
    case steel
    case nax
    case tensorOps
    case custom
}

public enum NumericalProfile: String, Codable, Sendable, CaseIterable {
    /// W4 weights, FP16/BF16 projections, FP32 residual accumulation.
    case exactW4A16FP32Residual
    case reference
    case researchApproximate
}

public enum PlanAccess: String, Codable, Sendable {
    case read
    case write
    case readWrite
}

public struct KleinModelSpecification: Codable, Sendable, Equatable {
    public var doubleBlocks: Int
    public var singleBlocks: Int
    public var attentionHeads: Int
    public var headDimension: Int
    public var jointAttentionDimension: Int
    public var maximumTextLength: Int
    public var textEncoderTaps: [Int]
    public var weightBits: Int

    public static let current = KleinModelSpecification(
        doubleBlocks: ModelConstants.numDoubleBlocks,
        singleBlocks: ModelConstants.numSingleBlocks,
        attentionHeads: ModelConstants.numAttentionHeads,
        headDimension: ModelConstants.attentionHeadDim,
        jointAttentionDimension: ModelConstants.jointAttentionDim,
        maximumTextLength: ModelConstants.maxSequenceLength,
        textEncoderTaps: ModelConstants.textEncoderLayers,
        weightBits: 4)
}

public struct PlanCapabilities: Codable, Sendable, Equatable {
    public var minimumOS: String
    public var requiredFeatures: [String]
    public var optionalFeatures: [String]

    public init(
        minimumOS: String = "26.2",
        requiredFeatures: [String] = [],
        optionalFeatures: [String] = []
    ) {
        self.minimumOS = minimumOS
        self.requiredFeatures = requiredFeatures.sorted()
        self.optionalFeatures = optionalFeatures.sorted()
    }
}

public struct TensorRegion: Codable, Sendable, Equatable {
    public var id: String
    public var dataType: PlanDataType
    public var logicalShape: [Int]
    public var paddedShape: [Int]
    public var byteCount: Int
    public var alignment: Int
    public var storage: PlanStorage
    public var stage: PlanStage
    public var firstUse: Int
    public var lastUse: Int

    public init(
        id: String,
        dataType: PlanDataType,
        logicalShape: [Int],
        paddedShape: [Int]? = nil,
        byteCount: Int,
        alignment: Int,
        storage: PlanStorage,
        stage: PlanStage,
        firstUse: Int,
        lastUse: Int
    ) {
        self.id = id
        self.dataType = dataType
        self.logicalShape = logicalShape
        self.paddedShape = paddedShape ?? logicalShape
        self.byteCount = byteCount
        self.alignment = alignment
        self.storage = storage
        self.stage = stage
        self.firstUse = firstUse
        self.lastUse = lastUse
    }
}

public struct PlanBinding: Codable, Sendable, Equatable {
    public var slot: Int
    public var regionID: String
    public var access: PlanAccess
    public var byteOffset: Int

    public init(slot: Int, regionID: String, access: PlanAccess, byteOffset: Int = 0) {
        self.slot = slot
        self.regionID = regionID
        self.access = access
        self.byteOffset = byteOffset
    }
}

public struct PlanConstant: Codable, Sendable, Equatable {
    public var name: String
    public var bitPattern: UInt64

    public init(name: String, bitPattern: UInt64) {
        self.name = name
        self.bitPattern = bitPattern
    }
}

public struct DispatchGeometry: Codable, Sendable, Equatable {
    public var grid: [Int]
    public var threadgroup: [Int]

    public init(grid: [Int], threadgroup: [Int]) {
        self.grid = grid
        self.threadgroup = threadgroup
    }
}

public struct PlanOperation: Codable, Sendable, Equatable {
    public var id: String
    public var kernelIdentifier: String
    public var backend: PlanBackend
    public var stage: PlanStage
    public var bindings: [PlanBinding]
    public var constants: [PlanConstant]
    public var dispatch: DispatchGeometry
    public var dependencies: [String]

    public init(
        id: String,
        kernelIdentifier: String,
        backend: PlanBackend,
        stage: PlanStage,
        bindings: [PlanBinding],
        constants: [PlanConstant] = [],
        dispatch: DispatchGeometry,
        dependencies: [String] = []
    ) {
        self.id = id
        self.kernelIdentifier = kernelIdentifier
        self.backend = backend
        self.stage = stage
        self.bindings = bindings.sorted {
            ($0.slot, $0.regionID, $0.byteOffset) < ($1.slot, $1.regionID, $1.byteOffset)
        }
        self.constants = constants.sorted { $0.name < $1.name }
        self.dispatch = dispatch
        self.dependencies = dependencies.sorted()
    }
}

public struct RegionPlacement: Codable, Sendable, Equatable {
    public var regionID: String
    public var offset: Int
    public var allocationOffset: Int
    public var reservedLength: Int
    public var canaryPrefixBytes: Int
    public var canarySuffixBytes: Int
}

public struct PlacementResult: Codable, Sendable, Equatable {
    public var peakBytes: Int
    public var aliasingEnabled: Bool
    public var placements: [RegionPlacement]
}

public struct ExecutionPlan: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var model: KleinModelSpecification
    public var capabilities: PlanCapabilities
    public var numericalProfile: NumericalProfile
    public var operations: [PlanOperation]
    public var regions: [TensorRegion]
    public var placement: PlacementResult
    public var digest: String

    public static func make(
        model: KleinModelSpecification = .current,
        capabilities: PlanCapabilities,
        numericalProfile: NumericalProfile,
        operations: [PlanOperation],
        regions: [TensorRegion],
        placement: PlacementResult
    ) throws -> ExecutionPlan {
        try PlanVerifier.verify(
            operations: operations, regions: regions, placement: placement)
        let payload = DigestPayload(
            schemaVersion: 1,
            model: model,
            capabilities: capabilities,
            numericalProfile: numericalProfile,
            operations: operations,
            regions: regions.sorted { $0.id < $1.id },
            placement: PlacementResult(
                peakBytes: placement.peakBytes,
                aliasingEnabled: placement.aliasingEnabled,
                placements: placement.placements.sorted { $0.regionID < $1.regionID }))
        let bytes = try Self.encoder.encode(payload)
        let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        return ExecutionPlan(
            schemaVersion: payload.schemaVersion,
            model: payload.model,
            capabilities: payload.capabilities,
            numericalProfile: payload.numericalProfile,
            operations: payload.operations,
            regions: payload.regions,
            placement: payload.placement,
            digest: digest)
    }

    public func canonicalJSON() throws -> Data {
        try Self.encoder.encode(self)
    }

    public func verifyDigest() throws -> Bool {
        let rebuilt = try Self.make(
            model: model,
            capabilities: capabilities,
            numericalProfile: numericalProfile,
            operations: operations,
            regions: regions,
            placement: placement)
        return rebuilt.digest == digest
    }

    private static var encoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }

    private struct DigestPayload: Codable {
        var schemaVersion: Int
        var model: KleinModelSpecification
        var capabilities: PlanCapabilities
        var numericalProfile: NumericalProfile
        var operations: [PlanOperation]
        var regions: [TensorRegion]
        var placement: PlacementResult
    }
}

public enum PlanError: Error, Sendable, Equatable, CustomStringConvertible {
    case duplicateRegion(String)
    case invalidRegion(String)
    case duplicateOperation(String)
    case missingRegion(operation: String, region: String)
    case invalidDependency(operation: String, dependency: String)
    case missingPlacement(String)
    case invalidPlacement(String)
    case overlappingPlacements(String, String)

    public var description: String {
        switch self {
        case .duplicateRegion(let id): "duplicate region: \(id)"
        case .invalidRegion(let id): "invalid region: \(id)"
        case .duplicateOperation(let id): "duplicate operation: \(id)"
        case .missingRegion(let operation, let region):
            "operation \(operation) references missing region \(region)"
        case .invalidDependency(let operation, let dependency):
            "operation \(operation) has invalid dependency \(dependency)"
        case .missingPlacement(let id): "missing placement: \(id)"
        case .invalidPlacement(let id): "invalid placement: \(id)"
        case .overlappingPlacements(let a, let b): "overlapping live placements: \(a), \(b)"
        }
    }
}
