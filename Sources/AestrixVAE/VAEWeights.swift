import Foundation
import MLX
import MLXNN
import AestrixCore
import AestrixWeights

public enum VAEWeights {
    public static let defaultGroupSize = 64
    public static let defaultBits = 4

    /// Load VAE: quantize mid-block attention Linears only (matches hub 4-bit packs), leave Conv2d full.
    public static func load(from directory: URL) throws -> Flux2VAE {
        let model = Flux2VAE()
        quantize(model: model, groupSize: defaultGroupSize, bits: defaultBits) { path, module in
            // Only attention projections are quantized in the community pack.
            guard module is Linear else { return false }
            return path.contains("attentions")
                && (path.contains("to_q") || path.contains("to_k") || path.contains("to_v")
                    || path.contains("to_out"))
        }
        let arrays = try SafetensorsLoader.loadMergedArrays(in: directory)
        // Drop unexpected dual keys if present (some packs include both .bias and quant .biases).
        let nested = NestedDictionary<String, MLXArray>.unflattened(arrays)
        try model.update(parameters: nested, verify: [.all])
        eval(model)
        return model
    }
}
