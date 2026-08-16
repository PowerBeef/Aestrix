import Foundation
import MLX
import MLXNN
import ImarelloCore
import ImarelloWeights

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
        precastQuantScalesToF16(model)
        eval(model)
        return model
    }

    /// Cast bf16 quant scales/biases to f16 once at load. The hot path re-cast
    /// them on every call (and every chunk at 1024²) — ~240 MB of tensors,
    /// hundreds of dispatches per step. Exact for this pack: min |scale| ≈ 1e-4,
    /// max < 0.11, well inside f16's normal range (probed 2026-08-16, all
    /// 60.5 M values; ENGINE_RESEARCH.md §4). Also makes MLX's internal
    /// `astype(scales)` inside every raw `linear()` call a no-op.
    private static func precastQuantScalesToF16(_ model: Module) {
        var replacements: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            if key.hasSuffix(".scales") || key.hasSuffix(".biases"), value.dtype == .bfloat16 {
                // context_embedder is the one quantized layer fed bf16 activations
                // (prompt embeds). bf16 scales keep its qmm promote at bf16;
                // f16 scales would flip it to f32 — different pixels. Skip it.
                if key.contains("context_embedder") { continue }
                replacements[key] = value.asType(.float16)
            }
        }
        guard !replacements.isEmpty else { return }
        let nested = NestedDictionary<String, MLXArray>.unflattened(replacements)
        model.update(parameters: nested)
    }

    public static func parameterKeyCount(_ model: Module) -> Int {
        model.parameters().flattened().count
    }
}
