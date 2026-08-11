import Foundation
import MLX
import MLXNN
import MLXFast
import AestrixCore

/// Qwen3 backbone used as a multi-layer feature extractor for FLUX.2-klein.
///
/// Diffusers taps HF `hidden_states[9/18/27]` (embeddings = index 0), concatenates on
/// the feature axis → `[B, L, 7680]`. Only layers `0 … maxTap-1` are required; deeper
/// layers and the final RMSNorm/lm_head are omitted for low-RAM staged residency.
public final class Qwen3TextEncoder: Module {
    public let config: Qwen3Config

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [Qwen3DecoderLayer]
    /// Present only when full 36-layer graph is built (not used for Klein taps).
    public let norm: RMSNorm?

    public init(config: Qwen3Config = .klein4B, pruneToTaps: Bool = true) {
        self.config = config
        let layerCount = pruneToTaps ? config.layersNeededForTaps : config.numHiddenLayers
        precondition(layerCount > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< layerCount).map { _ in Qwen3DecoderLayer(config) }
        // Final norm is only needed for LM head; Klein TE never uses it.
        self.norm = pruneToTaps
            ? nil
            : RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// Encode token ids → concatenated multi-layer embeddings `[B, L, jointDim]`.
    ///
    /// - Parameters:
    ///   - inputIds: `[B, L]` int32/int64 token ids (full 512 pad for Klein).
    ///   - attentionMask: optional `[B, L]` (1 = real, 0 = pad). When set, pad tokens
    ///     are blocked in attention while still producing a full-length sequence for DiT.
    public func encode(
        inputIds: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        var h = embedTokens(inputIds)
        let maskMode = Self.makeMaskMode(
            attentionMask: attentionMask,
            seqLen: h.dim(1),
            dtype: h.dtype
        )

        var taps: [Int: MLXArray] = [:]
        let needed = Set(config.tapHiddenStateIndices)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: maskMode)
            let hfIndex = i + 1 // HF: after layer i
            if needed.contains(hfIndex) {
                taps[hfIndex] = h
            }
        }

        var pieces: [MLXArray] = []
        pieces.reserveCapacity(config.tapHiddenStateIndices.count)
        for idx in config.tapHiddenStateIndices {
            guard let t = taps[idx] else {
                fatalError("Missing tap for HF hidden_states[\(idx)] — build more layers")
            }
            pieces.append(t)
        }
        // Concat along feature dim → [B, L, 3*hidden]
        return concatenated(pieces, axis: -1)
    }

    public func callAsFunction(
        inputIds: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        encode(inputIds: inputIds, attentionMask: attentionMask)
    }

    /// Build causal mask, optionally AND-ed with padding (additive `-inf` on blocked keys).
    /// Additive mask dtype must match SDPA activation dtype (usually bf16 for hub packs).
    static func makeMaskMode(
        attentionMask: MLXArray?,
        seqLen: Int,
        dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard let attentionMask else {
            return seqLen > 1 ? .causal : .none
        }
        // attentionMask: [B, L] → key validity [B, 1, 1, L]
        // Combine with causal: positions j > i blocked, and pad keys blocked.
        let b = attentionMask.dim(0)
        let l = attentionMask.dim(1)

        // Causal bool [L, L]
        let rows = MLXArray(0 ..< l).reshaped([l, 1])
        let cols = MLXArray(0 ..< l).reshaped([1, l])
        var causal = rows .>= cols // [L, L] bool

        // Pad: key allowed where mask == 1
        let keyOK = attentionMask.reshaped([b, 1, 1, l]) .> MLXArray(0) // [B,1,1,L]
        causal = causal.reshaped([1, 1, l, l])
        let allowed = causal .&& keyOK

        // Additive mask: 0 where allowed, large negative where blocked (cast to activation dtype).
        let neg = MLXArray(-1e4 as Float)
        let zero = MLXArray(0 as Float)
        let additive = MLX.where(allowed, zero, neg).asType(dtype)
        return .array(additive)
    }
}
