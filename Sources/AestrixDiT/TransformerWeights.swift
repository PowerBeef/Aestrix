import Foundation
import MLX
import MLXNN
import AestrixCore
import AestrixWeights

/// Load pre-quantized FLUX.2 transformer shards into `Flux2Transformer`.
public enum TransformerWeights {
    public static let defaultGroupSize = 64
    public static let defaultBits = 4

    /// Build a fresh Klein-4B transformer, convert Linears to QuantizedLinear, load weights.
    public static func loadQuantized(
        from directory: URL,
        bits: Int = defaultBits,
        groupSize: Int = defaultGroupSize
    ) throws -> Flux2Transformer {
        let model = Flux2Transformer()
        // Match hub 4-bit packs: all Linear layers quantized (including embedders).
        quantize(model: model, groupSize: groupSize, bits: bits) { _, _ in true }
        let arrays = try SafetensorsLoader.loadMergedArrays(in: directory)
        let nested = NestedDictionary<String, MLXArray>.unflattened(arrays)
        try model.update(parameters: nested, verify: [.all])
        eval(model)
        return model
    }

    public static func parameterKeyCount(_ model: Module) -> Int {
        model.parameters().flattened().count
    }
}
