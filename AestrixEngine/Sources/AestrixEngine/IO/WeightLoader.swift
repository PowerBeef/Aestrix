import Foundation
import MLX

/// Quantization parameters read from a submodel's `index.json` metadata.
public struct QuantizationInfo: Sendable, Equatable {
    public let bits: Int
    public let groupSize: Int
    public init(bits: Int, groupSize: Int) {
        self.bits = bits
        self.groupSize = groupSize
    }
}

/// Codable view of a `<sub>/model.safetensors.index.json`:
/// `{ "metadata": { "quantization_level", "quantization_group_size", ... },
///    "weight_map": { "<tensor>": "<shard>" } }`.
struct SafetensorsIndex: Codable {
    struct Metadata: Codable {
        let quantization_level: String?
        let quantization_group_size: String?
        let mflux_version: String?
    }
    let metadata: Metadata?
    let weight_map: [String: String]
}

/// Loads the raw tensors of a submodel (`text_encoder` / `transformer` / `vae`) from its
/// sharded safetensors, using the `weight_map` in its `index.json`.
///
/// M1 returns raw `[String: MLXArray]` + quantization metadata only. Wiring these into
/// `QuantizedLinear` modules (weight/scales/biases → `MLXNN`) happens in M3/M4.
public enum WeightLoader {

    /// Loads all tensors for the submodel at `submodelDir` (the dir containing the shards +
    /// `model.safetensors.index.json`).
    public static func loadWeights(submodelDir: URL) throws -> (weights: [String: MLXArray], quantization: QuantizationInfo?) {
        let indexURL = submodelDir.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            return try loadSharded(submodelDir: submodelDir, indexURL: indexURL)
        }
        // Fallback: no index — load every *.safetensors in the directory.
        return (try loadAllSafetensors(in: submodelDir), nil)
    }

    /// Reads only a submodel's `index.json` — no weight loading. Useful for cheaply
    /// inspecting the weight map and quantization metadata before pulling multi-GB shards.
    public static func parseIndex(submodelDir: URL) throws -> (weightMap: [String: String], quantization: QuantizationInfo?) {
        let indexURL = submodelDir.appendingPathComponent("model.safetensors.index.json")
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            throw AestrixError.invalidInput("index.json not found at \(indexURL.path)")
        }
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: Data(contentsOf: indexURL))
        let quant = index.metadata.flatMap { m -> QuantizationInfo? in
            guard let bitsStr = m.quantization_level,
                  let gsStr = m.quantization_group_size,
                  let bits = Int(bitsStr), let groupSize = Int(gsStr) else { return nil }
            return QuantizationInfo(bits: bits, groupSize: groupSize)
        }
        return (index.weight_map, quant)
    }

    private static func loadSharded(submodelDir: URL, indexURL: URL) throws -> ([String: MLXArray], QuantizationInfo?) {
        let (weightMap, quant) = try parseIndex(submodelDir: submodelDir)

        // The set of shard files referenced by the weight_map.
        let shards = Set(weightMap.values).sorted()

        var weights: [String: MLXArray] = [:]
        weights.reserveCapacity(weightMap.count)
        for shard in shards {
            let shardURL = submodelDir.appendingPathComponent(shard)
            // loadArrays returns every tensor stored in the shard; we keep only those the
            // weight_map attributes to this shard.
            let shardTensors = try loadArrays(url: shardURL)
            for (tensor, shardFile) in weightMap where shardFile == shard {
                if let w = shardTensors[tensor] {
                    weights[tensor] = w
                }
            }
        }

        guard weights.count == weightMap.count else {
            throw AestrixError.weightLoadFailed(
                "expected \(weightMap.count) tensors, loaded \(weights.count)")
        }
        return (weights, quant)
    }

    private static func loadAllSafetensors(in dir: URL) throws -> [String: MLXArray] {
        let urls = (try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        var weights: [String: MLXArray] = [:]
        for url in urls { weights.merge(try loadArrays(url: url)) { _, new in new } }
        return weights
    }
}
