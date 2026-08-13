import Foundation
import MLX
import MLXNN
import ImarelloCore
import ImarelloWeights

public enum VAEWeights {
    public static let defaultGroupSize = 64
    public static let defaultBits = 4

    /// Load full VAE (encode + decode). Used by I2I encode and generic load-vae.
    public static func load(from directory: URL) throws -> Flux2VAE {
        let model = Flux2VAE()
        quantizeAttentionLinears(model)
        let arrays = try SafetensorsLoader.loadMergedArrays(in: directory)
        let nested = NestedDictionary<String, MLXArray>.unflattened(arrays)
        try model.update(parameters: nested, verify: [.all])
        eval(model)
        return model
    }

    /// Load encoder + quant_conv + BN only (~67 MB vs ~165 MB full). I2I stage-0 encode.
    public static func loadEncodeOnly(from directory: URL) throws -> Flux2VAEEncoderOnly {
        let model = Flux2VAEEncoderOnly()
        quantizeAttentionLinears(model)
        let arrays = try SafetensorsLoader.loadMergedArrays(in: directory)
        var filtered: [String: MLXArray] = [:]
        filtered.reserveCapacity(arrays.count)
        for (key, value) in arrays {
            if key.hasPrefix("encoder.")
                || key.hasPrefix("quant_conv.")
                || key.hasPrefix("bn.")
            {
                filtered[key] = value
            }
        }
        precondition(
            !filtered.isEmpty,
            "no encode-side keys in VAE safetensors at " + directory.path)
        let nested = NestedDictionary<String, MLXArray>.unflattened(filtered)
        // Shape check only — decoder keys intentionally absent from this module.
        try model.update(parameters: nested, verify: [.shapeMismatch, .allModelKeysSet])
        eval(model)
        return model
    }

    /// Load decoder + BN + post_quant only (~97 MB vs ~165 MB full). T2I path.
    public static func loadDecodeOnly(from directory: URL) throws -> Flux2VAEDecoderOnly {
        let model = Flux2VAEDecoderOnly()
        quantizeAttentionLinears(model)
        let arrays = try SafetensorsLoader.loadMergedArrays(in: directory)
        var filtered: [String: MLXArray] = [:]
        filtered.reserveCapacity(arrays.count)
        for (key, value) in arrays {
            if key.hasPrefix("decoder.")
                || key.hasPrefix("post_quant_conv.")
                || key.hasPrefix("bn.")
            {
                filtered[key] = value
            }
        }
        precondition(!filtered.isEmpty, "no decode-side keys in VAE safetensors at \(directory.path)")
        let nested = NestedDictionary<String, MLXArray>.unflattened(filtered)
        // Shape check only — encoder keys intentionally absent from this module.
        try model.update(parameters: nested, verify: [.shapeMismatch, .allModelKeysSet])
        eval(model)
        return model
    }

    private static func quantizeAttentionLinears(_ model: Module) {
        quantize(model: model, groupSize: defaultGroupSize, bits: defaultBits) { path, module in
            guard module is Linear else { return false }
            return path.contains("attentions")
                && (path.contains("to_q") || path.contains("to_k") || path.contains("to_v")
                    || path.contains("to_out"))
        }
    }
}
