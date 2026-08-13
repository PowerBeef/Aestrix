import Foundation
import MLX
import MLXNN
import MLXFast
import ImarelloCore

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
        attentionMask: MLXArray? = nil,
        trace: PipelineTrace? = nil
    ) -> MLXArray {
        var h = embedTokens(inputIds)
        let maskMode = Self.makeMaskMode(
            attentionMask: attentionMask,
            seqLen: h.dim(1),
            dtype: h.dtype
        )
        let dens = trace?.density ?? .off
        // Checkpoint TE layers so 27-layer graph does not peak-accumulate.
        eval(h)
        trace?.probe("te.after_embed", phase: "te", minDensity: .blocks)

        var taps: [Int: MLXArray] = [:]
        let needed = Set(config.tapHiddenStateIndices)
        let sampleLayers: Set<Int> = dens == .max
            ? Set(0 ..< layers.count)
            : [0, 8, 17, layers.count - 1]
        // Eval every N layers to free intermediate graphs (N=1 is safest for RAM).
        let checkpointEvery = 1

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: maskMode)
            let hfIndex = i + 1 // HF: after layer i
            if needed.contains(hfIndex) {
                taps[hfIndex] = h
            }
            if (i + 1) % checkpointEvery == 0 {
                eval(h)
            }
            if dens.instrumentsDiTBlocks || dens == .max, sampleLayers.contains(i) {
                trace?.probe("te.after_layer_\(i)", phase: "te", block: i, minDensity: .blocks)
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
        let out = concatenated(pieces, axis: -1)
        eval(out)
        trace?.probe("te.after_concat_taps", phase: "te", minDensity: .blocks)
        return out
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

        // Causal [1,1,L,L] bool — cached for common Klein pad length 512.
        let causal = causalBoolMask(length: l)

        // Pad: key allowed where mask == 1
        let keyOK = attentionMask.reshaped([b, 1, 1, l]) .> MLXArray(0) // [B,1,1,L]
        let allowed = causal .&& keyOK

        // Additive mask: 0 where allowed, large negative where blocked (cast to activation dtype).
        let additive = MLX.where(allowed, Self.maskZero, Self.maskNeg).asType(dtype)
        return .array(additive)
    }

    // MLXArray is not Sendable; caches are read-only after init and only used on the TE encode path.
    nonisolated(unsafe) private static let maskZero = MLXArray(0 as Float)
    nonisolated(unsafe) private static let maskNeg = MLXArray(-1e4 as Float)

    /// Causal bool mask shaped `[1, 1, L, L]` (rows >= cols). Cached for L=512.
    nonisolated(unsafe) private static let causal512: MLXArray = {
        let l = ModelConstants.maxSequenceLength
        let rows = MLXArray(0 ..< l).reshaped([l, 1])
        let cols = MLXArray(0 ..< l).reshaped([1, l])
        let causal = (rows .>= cols).reshaped([1, 1, l, l])
        eval(causal)
        return causal
    }()

    private static func causalBoolMask(length: Int) -> MLXArray {
        if length == ModelConstants.maxSequenceLength {
            return causal512
        }
        let rows = MLXArray(0 ..< length).reshaped([length, 1])
        let cols = MLXArray(0 ..< length).reshaped([1, length])
        return (rows .>= cols).reshaped([1, 1, length, length])
    }
}
