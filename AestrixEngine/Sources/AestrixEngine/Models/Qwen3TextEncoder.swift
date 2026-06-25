import Foundation
import MLX
import MLXNN

// Qwen3-4B text encoder for FLUX.2 Klein, ported from mflux (`Qwen3TextEncoder` + the
// `qwen3_vl` components). Runs all 36 layers, extracts hidden states [9,18,27] (layers 8,17,26
// 0-indexed), and concatenates → ctx (B, 512, 7680 = 3 × 2560). Projection linears are 4-bit
// `QuantizedLinear`; norms/embed are bf16. RMSNorm + attention compute in float32 (mflux).
//
// Config (mflux): hidden 2560, 36 layers, 32 query heads, 8 KV heads (GQA), head_dim 128,
// intermediate 9728, rope_theta 1e6, rms_eps 1e-6, attention_bias false, vocab 151936.

private enum Qwen3Cfg {
    static let vocabSize = 151_936
    static let hidden = 2560
    static let layers = 36
    static let heads = 32
    static let kvHeads = 8
    static let headDim = 128
    static let intermediate = 9728
    static let ropeTheta: Float = 1_000_000.0
    static let rmsEps: Float = 1e-6
    static let promptLayers1 = [9, 18, 27]   // 1-indexed (HF) hidden-state indices to extract
}

// MARK: - RMSNorm (float32 compute, cast back to input dtype — matches mflux)

final class Qwen3RMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    private let eps: Float

    init(dimensions: Int, eps: Float = Qwen3Cfg.rmsEps) {
        self.eps = eps
        self._weight = ParameterInfo(wrappedValue: MLXArray.ones([dimensions]))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let xf = x.asType(.float32)
        let variance = mean(xf.square(), axis: -1, keepDims: true)
        let normed = xf * rsqrt(variance + eps)
        return (weight.asType(.float32) * normed).asType(dtype)
    }
}

// MARK: - RoPE (mflux Qwen3TextRotaryEmbedding + _apply_rotary_pos_emb/_rotate_half)

enum Qwen3RoPE {
    /// cos, sin for `positionIds` [B, seq]. Returns [B, seq, headDim].
    static func embeddings(positionIds: MLXArray, headDim: Int, base: Float, dtype: MLX.DType) -> (MLXArray, MLXArray) {
        let pos = positionIds.asType(.float32).expandedDimensions(axis: -1)     // [B, seq, 1]
        let exponents = MLX.arange(0, headDim, step: 2).asType(.float32) / Float(headDim)
        let invFreq = (MLXArray(base) ** exponents).reciprocal()                // 1 / base^(2i/d)
        let invFreqb = invFreq.expandedDimensions(axis: 0).expandedDimensions(axis: 0)  // [1,1,d/2]
        let freqs = pos * invFreqb                                              // [B, seq, d/2]
        let emb = concatenated([freqs, freqs], axis: -1)                        // [B, seq, d]
        return (cos(emb).asType(dtype), sin(emb).asType(dtype))
    }

    /// concat([-x2, x1]) over the last dim.
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.shape.last! / 2
        let x1 = x[.ellipsis, 0..<half]
        let x2 = x[.ellipsis, half...]
        return concatenated([-x2, x1], axis: -1)
    }

    /// Apply RoPE to q,k ([B, heads, seq, headDim]); cos/sin [B, seq, headDim].
    static func apply(q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
        let c = cos.expandedDimensions(axis: 1)   // [B, 1, seq, d]
        let s = sin.expandedDimensions(axis: 1)
        return (q * c + rotateHalf(q) * s, k * c + rotateHalf(k) * s)
    }
}

// MARK: - Attention (GQA + q/k_norm + RoPE + float32 SDPA)

final class Qwen3Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: QuantizedLinear
    @ModuleInfo(key: "k_proj") var kProj: QuantizedLinear
    @ModuleInfo(key: "v_proj") var vProj: QuantizedLinear
    @ModuleInfo(key: "o_proj") var oProj: QuantizedLinear
    @ModuleInfo(key: "q_norm") var qNorm: Qwen3RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Qwen3RMSNorm

    private let heads: Int, kvHeads: Int, kvGroups: Int, headDim: Int, scale: Float

    init(groupSize: Int = 64, bits: Int = 4) {
        self.heads = Qwen3Cfg.heads; self.kvHeads = Qwen3Cfg.kvHeads
        self.kvGroups = Qwen3Cfg.heads / Qwen3Cfg.kvHeads
        self.headDim = Qwen3Cfg.headDim; self.scale = 1.0 / sqrt(Float(Qwen3Cfg.headDim))
        let h = Qwen3Cfg.hidden, qo = Qwen3Cfg.heads * Qwen3Cfg.headDim, kv = Qwen3Cfg.kvHeads * Qwen3Cfg.headDim
        self._qProj.wrappedValue = QuantizedLinear(h, qo, bias: false, groupSize: groupSize, bits: bits)
        self._kProj.wrappedValue = QuantizedLinear(h, kv, bias: false, groupSize: groupSize, bits: bits)
        self._vProj.wrappedValue = QuantizedLinear(h, kv, bias: false, groupSize: groupSize, bits: bits)
        self._oProj.wrappedValue = QuantizedLinear(qo, h, bias: false, groupSize: groupSize, bits: bits)
        self._qNorm.wrappedValue = Qwen3RMSNorm(dimensions: Qwen3Cfg.headDim)
        self._kNorm.wrappedValue = Qwen3RMSNorm(dimensions: Qwen3Cfg.headDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?) -> MLXArray {
        let (b, seq) = (x.shape[0], x.shape[1])
        var q = qProj(x).reshaped([b, seq, heads, headDim])
        var k = kProj(x).reshaped([b, seq, kvHeads, headDim])
        let v = vProj(x).reshaped([b, seq, kvHeads, headDim])
        q = qNorm(q); k = kNorm(k)
        q = q.transposed(0, 2, 1, 3); k = k.transposed(0, 2, 1, 3)
        let vt = v.transposed(0, 2, 1, 3)
        (q, k) = Qwen3RoPE.apply(q: q, k: k, cos: cos, sin: sin)
        if kvGroups > 1 { k = Self.repeatKV(k, groups: kvGroups) }
        let vr = kvGroups > 1 ? Self.repeatKV(vt, groups: kvGroups) : vt
        let attn = scaledDotProductAttention(
            queries: q.asType(.float32), keys: k.asType(.float32), values: vr.asType(.float32),
            scale: scale, mask: mask).asType(q.dtype)
        let out = attn.transposed(0, 2, 1, 3).reshaped([b, seq, heads * headDim])
        return oProj(out)
    }

    private static func repeatKV(_ x: MLXArray, groups: Int) -> MLXArray {
        let s = x.shape; let (b, kv, seq, hd) = (s[0], s[1], s[2], s[3])
        return broadcast(x.expandedDimensions(axis: 2), to: [b, kv, groups, seq, hd]).reshaped([b, kv * groups, seq, hd])
    }
}

// MARK: - MLP (SwiGLU: silu(gate)*up → down)

final class Qwen3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: QuantizedLinear
    @ModuleInfo(key: "up_proj") var upProj: QuantizedLinear
    @ModuleInfo(key: "down_proj") var downProj: QuantizedLinear

    init(groupSize: Int = 64, bits: Int = 4) {
        let h = Qwen3Cfg.hidden, i = Qwen3Cfg.intermediate
        self._gateProj.wrappedValue = QuantizedLinear(h, i, bias: false, groupSize: groupSize, bits: bits)
        self._upProj.wrappedValue = QuantizedLinear(h, i, bias: false, groupSize: groupSize, bits: bits)
        self._downProj.wrappedValue = QuantizedLinear(i, h, bias: false, groupSize: groupSize, bits: bits)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

// MARK: - Decoder layer (input_norm → attn → +residual → post_norm → mlp → +residual)

final class Qwen3DecoderLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: Qwen3RMSNorm
    @ModuleInfo(key: "self_attn") var attn: Qwen3Attention
    @ModuleInfo(key: "post_attention_layernorm") var postAttnNorm: Qwen3RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Qwen3MLP

    override init() {
        self._inputNorm.wrappedValue = Qwen3RMSNorm(dimensions: Qwen3Cfg.hidden)
        self._attn.wrappedValue = Qwen3Attention()
        self._postAttnNorm.wrappedValue = Qwen3RMSNorm(dimensions: Qwen3Cfg.hidden)
        self._mlp.wrappedValue = Qwen3MLP()
        super.init()
    }
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?) -> MLXArray {
        var out = x + attn(inputNorm(x), cos: cos, sin: sin, mask: mask)
        out = out + mlp(postAttnNorm(out))
        return out
    }
}

// MARK: - Qwen3TextEncoder

/// Qwen3-4B text encoder → FLUX.2 context (B, 512, 7680) via hidden states [9,18,27].
public final class Qwen3TextEncoder: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: QuantizedEmbedding
    @ModuleInfo(key: "layers") var layers: [Qwen3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: Qwen3RMSNorm
    public let precision: MLX.DType

    /// - Parameter precision: running dtype for the data path (`.float32` for deterministic
    ///   parity, `.bfloat16` for production). RMSNorm/attention always compute in float32.
    public init(precision: MLX.DType = .bfloat16) {
        self.precision = precision
        self._embedTokens.wrappedValue = QuantizedEmbedding(
            embeddingCount: Qwen3Cfg.vocabSize, dimensions: Qwen3Cfg.hidden, groupSize: 64, bits: 4)
        self._layers.wrappedValue = (0..<Qwen3Cfg.layers).map { _ in Qwen3DecoderLayer() }
        self._norm.wrappedValue = Qwen3RMSNorm(dimensions: Qwen3Cfg.hidden)
        super.init()
    }

    public func loadWeights(_ flat: [String: MLXArray]) {
        update(parameters: NestedDictionary(flatWeights: flat))
    }

    /// The FLUX.2 context (B, seq, 7680): hidden states from layers [9,18,27], stacked & flattened.
    public func promptEmbeds(_ inputIds: MLXArray) -> MLXArray {
        let (b, seq) = (inputIds.shape[0], inputIds.shape[1])
        var h = embedTokens(inputIds).asType(precision)
        let pos = broadcast(arange(0, seq, dtype: .int32).expandedDimensions(axis: 0), to: [b, seq])
        let (cos, sin) = Qwen3RoPE.embeddings(positionIds: pos, headDim: Qwen3Cfg.headDim, base: Qwen3Cfg.ropeTheta, dtype: h.dtype)
        let mask = Self.causalMask(batch: b, seq: seq, dtype: h.dtype)
        var picked: [MLXArray] = []
        for (i, layer) in layers.enumerated() {
            h = layer(h, cos: cos, sin: sin, mask: mask)
            if Qwen3Cfg.promptLayers1.contains(i + 1) { picked.append(h) }   // layers 8,17,26 (0-idx)
        }
        // stack [B, 3, seq, hidden] → transpose [B, seq, 3, hidden] → reshape [B, seq, 3*hidden]
        let stacked = concatenated(picked.map { $0.expandedDimensions(axis: 1) }, axis: 1)  // [B,3,seq,hidden]
        let ctx = stacked.transposed(0, 2, 1, 3).reshaped([b, seq, picked.count * Qwen3Cfg.hidden])
        return ctx
    }

    /// Causal additive mask [B,1,seq,seq]: 0 on/below diagonal, large-negative above.
    private static func causalMask(batch: Int, seq: Int, dtype: MLX.DType) -> MLXArray {
        let upper = triu(MLXArray.ones([seq, seq]), k: 1)        // 1 strictly above diagonal
        let m = upper.asType(.float32) * MLXArray(-1e30)         // -1e30 above, 0 on/below
        return broadcast(m.expandedDimensions(axis: 0).expandedDimensions(axis: 0), to: [batch, 1, seq, seq]).asType(dtype)
    }
}
