import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Attention

/// Qwen3 GQA attention with per-head Q/K RMSNorm and non-traditional RoPE.
///
/// MLX Fast SDPA supports GQA without pre-tiling KV heads.
public final class Qwen3Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    public init(_ config: Qwen3Config) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / Foundation.sqrt(Float(config.headDim))

        let hidden = config.hiddenSize
        _qProj.wrappedValue = Linear(hidden, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(hidden, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(hidden, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, hidden, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    /// - Parameters:
    ///   - x: `[B, L, H]`
    ///   - mask: SDPA mask mode (`.causal` or additive `.array`)
    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal
    ) -> MLXArray {
        let b = x.dim(0)
        let l = x.dim(1)

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        // RMSNorm in float32 (matches DiT; avoids missing bf16 metallib kernels under swift build).
        let qDtype = queries.dtype
        let kDtype = keys.dtype
        queries = qNorm(queries.reshaped([b, l, numHeads, headDim]).asType(.float32))
            .asType(qDtype)
            .transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped([b, l, numKVHeads, headDim]).asType(.float32))
            .asType(kDtype)
            .transposed(0, 2, 1, 3)
        values = values.reshaped([b, l, numKVHeads, headDim]).transposed(0, 2, 1, 3)

        queries = rope(queries)
        keys = rope(keys)

        var out = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        out = out.transposed(0, 2, 1, 3).reshaped([b, l, numHeads * headDim])
        return oProj(out)
    }
}

// MARK: - MLP

public final class Qwen3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(_ config: Qwen3Config) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder layer

public final class Qwen3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3Attention
    let mlp: Qwen3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    public init(_ config: Qwen3Config) {
        _selfAttn.wrappedValue = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        _inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal
    ) -> MLXArray {
        let dtype = x.dtype
        let normedIn = inputLayernorm(x.asType(.float32)).asType(dtype)
        var h = x + selfAttn(normedIn, mask: mask)
        let normedPost = postAttentionLayernorm(h.asType(.float32)).asType(dtype)
        h = h + mlp(normedPost)
        return h
    }
}
