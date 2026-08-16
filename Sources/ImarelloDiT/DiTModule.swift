import Foundation
import MLX
import MLXNN
import ImarelloCore
import ImarelloWeights

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
        if isLoaded { throw ImarelloError.moduleAlreadyLoaded(moduleName) }
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

    /// Precompute RoPE for fixed img/txt ids (one denoise session).
    public func prepareRotaryEmbeddings(
        imgIds: MLXArray,
        txtIds: MLXArray
    ) throws -> (MLXArray, MLXArray) {
        guard let model, isLoaded else {
            throw ImarelloError.moduleNotLoaded(moduleName)
        }
        let rope = model.prepareRotaryEmbeddings(imgIds: imgIds, txtIds: txtIds)
        eval(rope.0, rope.1)
        return rope
    }

    /// Project Qwen 7680-d prompt embeds to DiT inner dim (once per generate).
    public func projectContext(_ encoderHiddenStates: MLXArray) throws -> MLXArray {
        guard let model, isLoaded else {
            throw ImarelloError.moduleNotLoaded(moduleName)
        }
        let projected = model.projectContext(encoderHiddenStates)
        eval(projected)
        return projected
    }

    /// Hoist timestep-only conditioning for a known step schedule (once per generate).
    public func precomputeStepConditioning(
        timesteps: [MLXArray],
        batch: Int,
        dtype: DType,
        guidance: MLXArray? = nil
    ) throws -> [Flux2StepConditioning] {
        guard let model, isLoaded else {
            throw ImarelloError.moduleNotLoaded(moduleName)
        }
        return model.precomputeStepConditioning(
            timesteps: timesteps, batch: batch, dtype: dtype, guidance: guidance)
    }

    public func forward(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        timestep: MLXArray,
        imgIds: MLXArray,
        txtIds: MLXArray,
        guidance: MLXArray? = nil,
        imageRotaryEmb: (MLXArray, MLXArray)? = nil,
        contextIsProjected: Bool = false,
        stepConditioning: Flux2StepConditioning? = nil,
        trace: PipelineTrace? = nil,
        stepIndex: Int? = nil
    ) throws -> MLXArray {
        guard let model, isLoaded else {
            throw ImarelloError.moduleNotLoaded(moduleName)
        }
        return model(
            hiddenStates: hiddenStates,
            encoderHiddenStates: encoderHiddenStates,
            timestep: timestep,
            imgIds: imgIds,
            txtIds: txtIds,
            guidance: guidance,
            imageRotaryEmb: imageRotaryEmb,
            contextIsProjected: contextIsProjected,
            stepConditioning: stepConditioning,
            trace: trace,
            stepIndex: stepIndex
        )
    }
}
