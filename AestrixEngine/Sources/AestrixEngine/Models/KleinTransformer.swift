import Foundation
import MLX
import MLXNN

// FLUX.2 Klein 4B diffusion transformer — the core denoising model.
// Ported faithfully from mflux (`Flux2Transformer`). 5 double-stream + 20 single-stream
// blocks, 4D RoPE (t,h,w,l), timestep/guidance embeddings, AdaLN modulation.
// All projections are 4-bit `QuantizedLinear`; norms are LayerNorm(affine=false).
//
// Config: inner_dim=3072 (24×128), joint_attention_dim=7680, in_channels=128,
// mlp_ratio=3.0, RoPE axes=(32,32,32,32), theta=2000.

private enum KleinCfg {
    static let innerDim = 3072
    static let heads = 24
    static let headDim = 128
    static let inChannels = 128
    static let jointAttnDim = 7680
    static let numDouble = 5
    static let numSingle = 20
    static let mlpRatio: Float = 3.0
    static let mlpHidden = 9216          // innerDim * 3
    static let ropeAxes = [32, 32, 32, 32]
    static let ropeTheta: Float = 2000
    static let timeChannels = 256
    static let lnEps: Float = 1e-6
    static let rmsEps: Float = 1e-5
}

// MARK: - RMSNorm (float32 compute, matches mflux nn.RMSNorm)

final class RMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    private let eps: Float
    init(dimensions: Int, eps: Float = KleinCfg.rmsEps) {
        self.eps = eps
        self._weight = ParameterInfo(wrappedValue: MLXArray.ones([dimensions]))
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let d = x.dtype
        let xf = x.asType(.float32)
        let v = mean(xf.square(), axis: -1, keepDims: true)
        return (weight.asType(.float32) * (xf * rsqrt(v + eps))).asType(d)
    }
}

// MARK: - 4D RoPE (PosEmbed4D + applyRopeBSHD)

/// Computes 4D RoPE cos/sin from position ids [N, 4]. Each axis (32 dims) produces
/// cos/sin [N, 16]; concatenated → [N, 64]. Applied to head_dim=128 via pairing.
enum RoPE4D {
    static func embed(_ ids: MLXArray) -> (MLXArray, MLXArray) {
        let pos = ids.asType(.float32)
        var cosParts: [MLXArray] = []
        var sinParts: [MLXArray] = []
        for (i, dim) in KleinCfg.ropeAxes.enumerated() {
            let scale = MLX.arange(0, dim, step: 2).asType(.float32) / Float(dim)
            let omega = MLXArray(KleinCfg.ropeTheta) ** (-scale)
            let p = pos[.ellipsis, i].expandedDimensions(axis: -1)  // [N, 1]
            let out = p * omega.expandedDimensions(axis: 0)          // [N, dim/2]
            cosParts.append(cos(out))
            sinParts.append(sin(out))
        }
        return (concatenated(cosParts, axis: -1), concatenated(sinParts, axis: -1))
    }

    /// Apply RoPE to q,k [B, S, H, D]. cos/sin: [S, D/2].
    /// Reshapes D → (D/2, 2) as (real, imag), rotates by (cos, sin).
    static func applyBSHD(_ q: MLXArray, _ k: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
        let cosB = cos.expandedDimensions(axis: 0).expandedDimensions(axis: 0)  // [1,1,S,D/2]
        let sinB = sin.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        func mix(_ x: MLXArray) -> MLXArray {
            let dtype = x.dtype
            let s = x.shape
            let d2 = s.last! / 2
            let xf = x.asType(.float32).reshaped(s.dropLast() + [d2, 2])
            let real = xf[.ellipsis, 0..<d2, 0]
            let imag = xf[.ellipsis, 0..<d2, 1]
            let r0 = real * cosB + (-imag) * sinB
            let r1 = imag * cosB + real * sinB
            return concatenated([r0.expandedDimensions(axis: -1), r1.expandedDimensions(axis: -1)], axis: -1)
                .reshaped(s).asType(dtype)
        }
        return (mix(q), mix(k))
    }
}

// MARK: - Timestep + Guidance Embedding

final class TimeGuidanceEmbed: Module {
    @ModuleInfo(key: "linear_1") var lin1: QuantizedLinear
    @ModuleInfo(key: "linear_2") var lin2: QuantizedLinear

    override init() {
        self._lin1.wrappedValue = QuantizedLinear(KleinCfg.timeChannels, KleinCfg.innerDim, bias: false, groupSize: 64, bits: 4)
        self._lin2.wrappedValue = QuantizedLinear(KleinCfg.innerDim, KleinCfg.innerDim, bias: false, groupSize: 64, bits: 4)
        super.init()
    }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        // Sinusoidal embedding (256-ch, flip_sin_to_cos).
        let half = KleinCfg.timeChannels / 2
        let freqs = exp(-log(Float(10000)) * (MLX.arange(0, half).asType(.float32) / Float(half)))
        let args = timestep.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        var emb = concatenated([sin(args), cos(args)], axis: -1)
        emb = concatenated([emb[.ellipsis, half...], emb[.ellipsis, 0..<half]], axis: -1)  // flip
        return lin2(silu(lin1(emb)))
    }
}

// MARK: - Modulation (AdaLN params: shift, scale, gate)

final class Modulation: Module {
    @ModuleInfo(key: "linear") var linear: QuantizedLinear
    let sets: Int

    init(sets: Int) {
        self.sets = sets
        self._linear.wrappedValue = QuantizedLinear(KleinCfg.innerDim, KleinCfg.innerDim * 3 * sets, bias: false, groupSize: 64, bits: 4)
        super.init()
    }

    /// Returns `sets` groups of (shift, scale, gate), each [B, 1, innerDim].
    func callAsFunction(_ temb: MLXArray) -> [(MLXArray, MLXArray, MLXArray)] {
        let mod = linear(silu(temb))
        let expanded = mod.ndim == 2 ? mod.expandedDimensions(axis: 1) : mod
        let dim = KleinCfg.innerDim
        var result: [(MLXArray, MLXArray, MLXArray)] = []
        for i in 0..<sets {
            let b = i * 3
            result.append((
                expanded[.ellipsis, (b * dim)..<((b + 1) * dim)],
                expanded[.ellipsis, ((b + 1) * dim)..<((b + 2) * dim)],
                expanded[.ellipsis, ((b + 2) * dim)..<((b + 3) * dim)],
            ))
        }
        return result
    }
}

// MARK: - FeedForward (SwiGLU)

final class FeedForward: Module {
    @ModuleInfo(key: "linear_in") var linIn: QuantizedLinear
    @ModuleInfo(key: "linear_out") var linOut: QuantizedLinear

    override init() {
        let inner = Int(Float(KleinCfg.innerDim) * KleinCfg.mlpRatio)
        self._linIn.wrappedValue = QuantizedLinear(KleinCfg.innerDim, inner * 2, bias: false, groupSize: 64, bits: 4)
        self._linOut.wrappedValue = QuantizedLinear(inner, KleinCfg.innerDim, bias: false, groupSize: 64, bits: 4)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = linIn(x)
        let half = h.shape.last! / 2
        let x1 = h[.ellipsis, 0..<half]
        let x2 = h[.ellipsis, half...]
        return linOut(silu(x1) * x2)
    }
}

// MARK: - Double-Stream Attention (img QKV + txt added_kv_proj)

final class DoubleStreamAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: QuantizedLinear
    @ModuleInfo(key: "to_k") var toK: QuantizedLinear
    @ModuleInfo(key: "to_v") var toV: QuantizedLinear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: QuantizedLinear
    @ModuleInfo(key: "add_q_proj") var addQ: QuantizedLinear
    @ModuleInfo(key: "add_k_proj") var addK: QuantizedLinear
    @ModuleInfo(key: "add_v_proj") var addV: QuantizedLinear
    @ModuleInfo(key: "norm_added_q") var normAddQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddK: RMSNorm
    @ModuleInfo(key: "to_add_out") var toAddOut: QuantizedLinear

    override init() {
        let d = KleinCfg.innerDim
        func ql(_ i: Int, _ o: Int) -> QuantizedLinear { QuantizedLinear(i, o, bias: false, groupSize: 64, bits: 4) }
        self._toQ.wrappedValue = ql(d, d); self._toK.wrappedValue = ql(d, d); self._toV.wrappedValue = ql(d, d)
        self._normQ.wrappedValue = RMSNorm(dimensions: KleinCfg.headDim)
        self._normK.wrappedValue = RMSNorm(dimensions: KleinCfg.headDim)
        self._toOut.wrappedValue = ql(d, d)
        self._addQ.wrappedValue = ql(d, d); self._addK.wrappedValue = ql(d, d); self._addV.wrappedValue = ql(d, d)
        self._normAddQ.wrappedValue = RMSNorm(dimensions: KleinCfg.headDim)
        self._normAddK.wrappedValue = RMSNorm(dimensions: KleinCfg.headDim)
        self._toAddOut.wrappedValue = ql(d, d)
        super.init()
    }

    func callAsFunction(_ img: MLXArray, _ txt: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
        let (b, h, hd) = (img.shape[0], KleinCfg.heads, KleinCfg.headDim)
        let txtLen = txt.shape[1]
        // Process img QKV → [B, H, S_img, D]
        var q = Self.proj(toQ, img, b, h, hd, normQ)
        var k = Self.proj(toK, img, b, h, hd, normK)
        let v = Self.proj(toV, img, b, h, hd, nil)
        // Process txt QKV → [B, H, S_txt, D]
        let eq = Self.proj(addQ, txt, b, h, hd, normAddQ)
        let ek = Self.proj(addK, txt, b, h, hd, normAddK)
        let ev = Self.proj(addV, txt, b, h, hd, nil)
        // Concat [txt, img] along sequence
        q = concatenated([eq, q], axis: 2)
        k = concatenated([ek, k], axis: 2)
        let vc = concatenated([ev, v], axis: 2)
        // RoPE
        (q, k) = RoPE4D.applyBSHD(q, k, cos: cos, sin: sin)
        // SDPA
        let scale: Float = 1.0 / sqrt(Float(hd))
        let attn = scaledDotProductAttention(queries: q, keys: k, values: vc, scale: scale, mask: nil)
            .transposed(0, 2, 1, 3).reshaped([b, -1, h * hd])
        // Split back [txt, img] along the seq axis (dim 1). Transpose to make seq last,
        // slice via .ellipsis, transpose back.
        let t = attn.transposed(0, 2, 1)  // [B, D, S]
        let txtOut = t[.ellipsis, 0..<txtLen].transposed(0, 2, 1)  // [B, txtLen, D]
        let imgOut = t[.ellipsis, txtLen..<(t.shape[2])].transposed(0, 2, 1)
        return (toOut(imgOut), toAddOut(txtOut))
    }

    private static func proj(_ linear: QuantizedLinear, _ x: MLXArray, _ b: Int, _ h: Int, _ hd: Int, _ norm: RMSNorm?) -> MLXArray {
        var p = linear(x).reshaped([b, x.shape[1], h, hd]).transposed(0, 2, 1, 3)  // [B, H, S, D]
        if let norm { p = norm(p.asType(.float32)).asType(p.dtype) }
        return p
    }
}

// MARK: - Single-Stream Attention (fused QKV + MLP)

final class SingleStreamAttention: Module {
    @ModuleInfo(key: "to_qkv_mlp_proj") var qkvMlpProj: QuantizedLinear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: QuantizedLinear

    override init() {
        let d = KleinCfg.innerDim
        let outDim = d * 3 + KleinCfg.mlpHidden * 2  // qkv(3×3072) + mlp_gate+up(2×9216)
        self._qkvMlpProj.wrappedValue = QuantizedLinear(d, outDim, bias: false, groupSize: 64, bits: 4)
        self._normQ.wrappedValue = RMSNorm(dimensions: KleinCfg.headDim)
        self._normK.wrappedValue = RMSNorm(dimensions: KleinCfg.headDim)
        self._toOut.wrappedValue = QuantizedLinear(d + KleinCfg.mlpHidden, d, bias: false, groupSize: 64, bits: 4)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (b, h, hd) = (x.shape[0], KleinCfg.heads, KleinCfg.headDim)
        let proj = qkvMlpProj(x)
        let qkvDim = KleinCfg.innerDim * 3
        let qkv = proj[.ellipsis, 0..<qkvDim]
        let mlpH = proj[.ellipsis, qkvDim...]
        var q = qkv[.ellipsis, 0..<KleinCfg.innerDim].reshaped([b, -1, h, hd]).transposed(0, 2, 1, 3)
        var k = qkv[.ellipsis, KleinCfg.innerDim..<(2 * KleinCfg.innerDim)].reshaped([b, -1, h, hd]).transposed(0, 2, 1, 3)
        let v = qkv[.ellipsis, (2 * KleinCfg.innerDim)..<(3 * KleinCfg.innerDim)].reshaped([b, -1, h, hd]).transposed(0, 2, 1, 3)
        q = normQ(q.asType(.float32)).asType(q.dtype)
        k = normK(k.asType(.float32)).asType(k.dtype)
        (q, k) = RoPE4D.applyBSHD(q, k, cos: cos, sin: sin)
        let scale: Float = 1.0 / sqrt(Float(hd))
        let attn = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
            .transposed(0, 2, 1, 3).reshaped([b, -1, h * hd])
        // SwiGLU MLP
        let mh = mlpH.shape.last!
        let half = mh / 2
        let mlpOut = silu(mlpH[.ellipsis, 0..<half]) * mlpH[.ellipsis, half...]
        return toOut(concatenated([attn, mlpOut], axis: -1))
    }
}

// MARK: - Double-Stream Block

final class DoubleStreamBlock: Module {
    @ModuleInfo(key: "attn") var attn: DoubleStreamAttention
    @ModuleInfo(key: "ff") var ff: FeedForward
    @ModuleInfo(key: "ff_context") var ffContext: FeedForward

    override init() {
        self._attn.wrappedValue = DoubleStreamAttention()
        self._ff.wrappedValue = FeedForward()
        self._ffContext.wrappedValue = FeedForward()
        super.init()
    }

    func callAsFunction(_ img: MLXArray, _ txt: MLXArray,
                        modImg: [(MLXArray, MLXArray, MLXArray)],
                        modTxt: [(MLXArray, MLXArray, MLXArray)],
                        cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
        let ln = { LayerNorm(dimensions: KleinCfg.innerDim, eps: KleinCfg.lnEps, affine: false) }
        _ = ln  // LayerNorm is stateless (affine=false); create inline below
        let (sMsa, scMsa, gMsa) = modImg[0]
        let (sMlp, scMlp, gMlp) = modImg[1]
        let (csMsa, cscMsa, cgMsa) = modTxt[0]
        let (csMlp, cscMlp, cgMlp) = modTxt[1]
        // AdaLN + attention
        var nh = LayerNorm(dimensions: KleinCfg.innerDim, eps: KleinCfg.lnEps, affine: false)(img)
        nh = (1 + scMsa) * nh + sMsa
        var nt = LayerNorm(dimensions: KleinCfg.innerDim, eps: KleinCfg.lnEps, affine: false)(txt)
        nt = (1 + cscMsa) * nt + csMsa
        let (attnImg, attnTxt) = attn(nh, nt, cos: cos, sin: sin)
        var hidden = img + gMsa * attnImg
        var ctx = txt + cgMsa * attnTxt
        // AdaLN + FF
        nh = LayerNorm(dimensions: KleinCfg.innerDim, eps: KleinCfg.lnEps, affine: false)(hidden)
        nh = (1 + scMlp) * nh + sMlp
        hidden = hidden + gMlp * ff(nh)
        nt = LayerNorm(dimensions: KleinCfg.innerDim, eps: KleinCfg.lnEps, affine: false)(ctx)
        nt = (1 + cscMlp) * nt + csMlp
        ctx = ctx + cgMlp * ffContext(nt)
        return (ctx, hidden)
    }
}

// MARK: - Single-Stream Block

final class SingleStreamBlock: Module {
    @ModuleInfo(key: "attn") var attn: SingleStreamAttention

    override init() {
        self._attn.wrappedValue = SingleStreamAttention()
        super.init()
    }

    func callAsFunction(_ x: MLXArray, modParams: (MLXArray, MLXArray, MLXArray),
                        cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (shift, scale, gate) = modParams
        var nh = LayerNorm(dimensions: KleinCfg.innerDim, eps: KleinCfg.lnEps, affine: false)(x)
        nh = (1 + scale) * nh + shift
        return x + gate * attn(nh, cos: cos, sin: sin)
    }
}

// MARK: - AdaLayerNormContinuous (final norm)

final class AdaLayerNormContinuous: Module {
    @ModuleInfo(key: "linear") var linear: QuantizedLinear

    override init() {
        self._linear.wrappedValue = QuantizedLinear(KleinCfg.innerDim, KleinCfg.innerDim * 2, bias: false, groupSize: 64, bits: 4)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ temb: MLXArray) -> MLXArray {
        let emb = linear(silu(temb).asType(temb.dtype))
        let dim = KleinCfg.innerDim
        let scale = emb[.ellipsis, 0..<dim]
        let shift = emb[.ellipsis, dim..<(2 * dim)]
        let normed = LayerNorm(dimensions: KleinCfg.innerDim, eps: KleinCfg.lnEps, affine: false)(x)
        return normed * (1 + scale.expandedDimensions(axis: 1)) + shift.expandedDimensions(axis: 1)
    }
}

// MARK: - KleinTransformer

public final class KleinTransformer: Module {
    @ModuleInfo(key: "time_guidance_embed") var timeEmbed: TimeGuidanceEmbed
    @ModuleInfo(key: "double_stream_modulation_img") var modImg: Modulation
    @ModuleInfo(key: "double_stream_modulation_txt") var modTxt: Modulation
    @ModuleInfo(key: "single_stream_modulation") var modSingle: Modulation
    @ModuleInfo(key: "x_embedder") var xEmbedder: QuantizedLinear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: QuantizedLinear
    @ModuleInfo(key: "transformer_blocks") var doubleBlocks: [DoubleStreamBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleBlocks: [SingleStreamBlock]
    @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: QuantizedLinear

    public override init() {
        self._timeEmbed.wrappedValue = TimeGuidanceEmbed()
        self._modImg.wrappedValue = Modulation(sets: 2)
        self._modTxt.wrappedValue = Modulation(sets: 2)
        self._modSingle.wrappedValue = Modulation(sets: 1)
        self._xEmbedder.wrappedValue = QuantizedLinear(KleinCfg.inChannels, KleinCfg.innerDim, bias: false, groupSize: 64, bits: 4)
        self._contextEmbedder.wrappedValue = QuantizedLinear(KleinCfg.jointAttnDim, KleinCfg.innerDim, bias: false, groupSize: 64, bits: 4)
        self._doubleBlocks.wrappedValue = (0..<KleinCfg.numDouble).map { _ in DoubleStreamBlock() }
        self._singleBlocks.wrappedValue = (0..<KleinCfg.numSingle).map { _ in SingleStreamBlock() }
        self._normOut.wrappedValue = AdaLayerNormContinuous()
        self._projOut.wrappedValue = QuantizedLinear(KleinCfg.innerDim, KleinCfg.inChannels, bias: false, groupSize: 64, bits: 4)
        super.init()
    }

    public func loadWeights(_ flat: [String: MLXArray]) {
        update(parameters: NestedDictionary(flatWeights: flat))
    }

    /// Forward: denoise `latent` guided by `ctx` at `timestep`.
    /// - Parameters:
    ///   - latent: [B, N_img, 128] (flattened noise or noisy latent).
    ///   - ctx: [B, 512, 7680] (text context from Qwen3).
    ///   - imgIds: [N_img, 4] (RoPE coords for image tokens: (t,h,w,l)).
    ///   - txtIds: [512, 4] (RoPE coords for text tokens).
    ///   - timestep: [B] scalar in [0, 1].
    /// - Returns: velocity prediction [B, N_img, 128].
    public func callAsFunction(_ latent: MLXArray, _ ctx: MLXArray, imgIds: MLXArray, txtIds: MLXArray, timestep: MLXArray) -> MLXArray {
        // Scale timestep (×1000 if ≤1, matching mflux).
        let ts = timestep * MLXArray(Float(1000.0))
        let temb = timeEmbed(ts)

        // Embed.
        var hidden = xEmbedder(latent)
        var encoder = contextEmbedder(ctx)

        // RoPE: compute for txt and img separately, then concat [txt, img].
        let (txtCos, txtSin) = RoPE4D.embed(txtIds)
        let (imgCos, imgSin) = RoPE4D.embed(imgIds)
        let cos = concatenated([txtCos, imgCos], axis: 0)
        let sin = concatenated([txtSin, imgSin], axis: 0)

        // Modulation.
        let modImgP = modImg(temb)
        let modTxtP = modTxt(temb)

        // Double-stream blocks.
        for block in doubleBlocks {
            (encoder, hidden) = block(hidden, encoder, modImg: modImgP, modTxt: modTxtP, cos: cos, sin: sin)
        }

        // Concat [txt, img] for single-stream.
        hidden = concatenated([encoder, hidden], axis: 1)
        let modSingleP = modSingle(temb)[0]

        // Single-stream blocks.
        for block in singleBlocks {
            hidden = block(hidden, modParams: modSingleP, cos: cos, sin: sin)
        }

        // Strip text tokens (first txtLen along seq axis), apply final norm + projection.
        let txtLen = encoder.shape[1]
        let ht = hidden.transposed(0, 2, 1)
        hidden = ht[.ellipsis, txtLen..<(ht.shape[2])].transposed(0, 2, 1)
        hidden = normOut(hidden, temb)
        return projOut(hidden)
    }
}
