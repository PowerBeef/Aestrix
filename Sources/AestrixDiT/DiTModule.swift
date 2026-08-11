import Foundation
import MLX
import MLXNN
import AestrixCore
import AestrixWeights

/// Loadable MMDiT wrapper for staged pipeline residency.
public final class DiTModule: LoadableModule, @unchecked Sendable {
    public let moduleName = "transformer"
    public private(set) var isLoaded = false

    public private(set) var model: Flux2Transformer?
    private let snapshot: ModelSnapshot?
    private let bits: Int
    private let groupSize: Int

    public init(
        snapshot: ModelSnapshot? = nil,
        bits: Int = TransformerWeights.defaultBits,
        groupSize: Int = TransformerWeights.defaultGroupSize
    ) {
        self.snapshot = snapshot
        self.bits = bits
        self.groupSize = groupSize
    }

    /// Staged load: with snapshot → quant weights; without → residency flag only (no Metal alloc).
    public func load() async throws {
        if isLoaded { throw AestrixError.moduleAlreadyLoaded(moduleName) }
        if let snapshot {
            try snapshot.validateLayout()
            model = try TransformerWeights.loadQuantized(
                from: snapshot.transformerDirectory,
                bits: bits,
                groupSize: groupSize
            )
        } else {
            // mem-selftest / dry stage: mark resident without constructing MLX modules.
            model = nil
        }
        isLoaded = true
    }

    /// Allocate random-init structure (requires working MLX Metal/CPU). Used by tests.
    public func loadStructureOnly() {
        model = Flux2Transformer()
        isLoaded = true
    }

    public func unload() async {
        let hadModel = model != nil
        model = nil
        isLoaded = false
        if hadModel {
            Memory.clearCache()
        }
    }

    /// Leaf parameter tensors currently resident (0 if dry-loaded).
    public var parameterLeafCount: Int {
        guard let model else { return 0 }
        return model.parameters().flattened().count
    }

    public func forward(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        timestep: MLXArray,
        imgIds: MLXArray,
        txtIds: MLXArray,
        guidance: MLXArray? = nil
    ) throws -> MLXArray {
        guard let model, isLoaded else {
            throw AestrixError.moduleNotLoaded(moduleName)
        }
        return model(
            hiddenStates: hiddenStates,
            encoderHiddenStates: encoderHiddenStates,
            timestep: timestep,
            imgIds: imgIds,
            txtIds: txtIds,
            guidance: guidance
        )
    }
}
