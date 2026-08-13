import Foundation
import MLX
import MLXNN
import ImarelloCore
import ImarelloWeights

/// Load pre-quantized Qwen3 TE shards into a pruned `Qwen3TextEncoder`.
public enum TextEncoderWeights {
    public static let defaultGroupSize = 64
    public static let defaultBits = 4

    /// Build Klein TE (layers through max tap), quantize Linear+Embedding, load hub weights.
    public static func loadQuantized(
        from directory: URL,
        config: Qwen3Config = .klein4B,
        bits: Int = defaultBits,
        groupSize: Int = defaultGroupSize
    ) throws -> Qwen3TextEncoder {
        let model = Qwen3TextEncoder(config: config, pruneToTaps: true)

        // Hub 4-bit packs quantize every Linear and the token embedding.
        quantize(model: model, groupSize: groupSize, bits: bits) { _, module in
            module is Linear || module is Embedding
        }

        var arrays = try SafetensorsLoader.loadMergedArrays(in: directory)
        arrays = sanitize(arrays, for: model)

        let nested = NestedDictionary<String, MLXArray>.unflattened(arrays)
        try model.update(parameters: nested, verify: [.all])
        eval(model)
        return model
    }

    /// Drop keys unused by the pruned encoder (deeper layers, final norm, rotary cache).
    public static func sanitize(
        _ weights: [String: MLXArray],
        for model: Qwen3TextEncoder
    ) -> [String: MLXArray] {
        let maxLayer = model.config.layersNeededForTaps // exclusive upper bound of layer index
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        for (key, value) in weights {
            if key.hasPrefix("rotary_emb") { continue }
            if key == "norm.weight" || key.hasPrefix("norm.") { continue }
            if key.hasPrefix("lm_head") { continue }

            if key.hasPrefix("layers.") {
                // layers.N....
                let rest = key.dropFirst("layers.".count)
                guard let dot = rest.firstIndex(of: ".") else { continue }
                let idxStr = rest[..<dot]
                guard let idx = Int(idxStr) else { continue }
                if idx >= maxLayer { continue }
            }
            out[key] = value
        }
        return out
    }

    public static func parameterKeyCount(_ model: Module) -> Int {
        model.parameters().flattened().count
    }
}
