import Foundation
import MLX
import MLXNN

/// GroupNorm matching mflux/diffusers: computed in float32, output cast to `precision`
/// (mflux casts to `ModelConfig.precision`). `precision` is threaded from `FluxVAE` so the
/// VAE can decode in float32 (diffusers `AutoencoderKLFlux2.force_upcast = true`) or bf16.
@inline(__always)
private func gnorm(_ gn: GroupNorm, _ x: MLXArray, _ precision: MLX.DType) -> MLXArray {
    gn(x.asType(.float32)).asType(precision)
}

// MARK: - ResnetBlock2D

/// diffusers `ResnetBlock2D` (no time embedding). NHWC. norm → silu → conv → norm → silu → conv + shortcut.
final class ResnetBlock2D: Module, UnaryLayer {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?
    private let precision: MLX.DType

    init(inChannels: Int, outChannels: Int, precision: MLX.DType = .bfloat16, groups: Int = 32) {
        self.precision = precision
        self._norm1.wrappedValue = GroupNorm(groupCount: groups, dimensions: inChannels, eps: 1e-6, pytorchCompatible: true)
        self._conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._norm2.wrappedValue = GroupNorm(groupCount: groups, dimensions: outChannels, eps: 1e-6, pytorchCompatible: true)
        self._conv2.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._convShortcut.wrappedValue = (inChannels != outChannels)
            ? Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
            : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = gnorm(norm1, x, precision)
        h = silu(h)
        h = conv1(h)
        h = gnorm(norm2, h, precision)
        h = silu(h)
        h = conv2(h)
        let shortcut = convShortcut?(x) ?? x
        return h + shortcut
    }
}

// MARK: - Down/Up samplers

/// diffusers/mflux `Downsample2D`: asymmetric pad (0,1) on H,W (bottom/right) then a
/// stride-2, padding=0 conv. (Symmetric padding=1 gives different border values.)
final class Downsample2D: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(channels: Int) {
        self._conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2, padding: 0)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        let (b, h, w, c) = (s[0], s[1], s[2], s[3])
        let row = MLXArray.zeros([b, 1, w, c], dtype: x.dtype)
        var p = concatenated([x, row], axis: 1)            // [b, h+1, w, c]
        let col = MLXArray.zeros([b, h + 1, 1, c], dtype: x.dtype)
        p = concatenated([p, col], axis: 2)                // [b, h+1, w+1, c]
        return conv(p)
    }
}

/// diffusers/mflux `Upsample2D`: nearest-neighbour ×2 via element repeat, then conv (pad 1).
final class Upsample2D: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(channels: Int) {
        self._conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)
        return conv(y)
    }
}

// MARK: - Encoder / Decoder blocks

/// Encoder down block: 2 resnets + optional downsampler.
final class DownEncoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "resnets") var resnets: [ResnetBlock2D]
    @ModuleInfo(key: "downsamplers") var downsamplers: [Downsample2D]?

    init(inChannels: Int, outChannels: Int, precision: MLX.DType, layersPerBlock: Int = 2, addDown: Bool) {
        var res: [ResnetBlock2D] = []
        for i in 0..<layersPerBlock {
            let inCh = (i == 0) ? inChannels : outChannels
            res.append(ResnetBlock2D(inChannels: inCh, outChannels: outChannels, precision: precision))
        }
        self._resnets.wrappedValue = res
        self._downsamplers.wrappedValue = addDown ? [Downsample2D(channels: outChannels)] : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        if let downs = downsamplers { for d in downs { h = d(h) } }
        return h
    }
}

/// Decoder up block: 3 resnets + optional upsampler. `inChannels`→`outChannels` on resnet 0.
final class UpDecoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "resnets") var resnets: [ResnetBlock2D]
    @ModuleInfo(key: "upsamplers") var upsamplers: [Upsample2D]?

    init(inChannels: Int, outChannels: Int, precision: MLX.DType, layersPerBlock: Int = 3, addUp: Bool) {
        var res: [ResnetBlock2D] = []
        for i in 0..<layersPerBlock {
            let inCh = (i == 0) ? inChannels : outChannels
            res.append(ResnetBlock2D(inChannels: inCh, outChannels: outChannels, precision: precision))
        }
        self._resnets.wrappedValue = res
        self._upsamplers.wrappedValue = addUp ? [Upsample2D(channels: outChannels)] : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets { h = r(h) }
        if let ups = upsamplers { for u in ups { h = u(h) } }
        return h
    }
}

// MARK: - Mid-block spatial self-attention

/// diffusers VAE mid-block `Attention`: group_norm → (B, H·W, C) tokens → Q/K/V →
/// single-head SDPA → to_out → reshape back → residual. mflux dequantizes the 4-bit
/// projections to bf16 `Linear` at load (tiny), so we use plain `Linear` here.
final class VAEMidAttention: Module, UnaryLayer {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    private let precision: MLX.DType
    private let scale: Float

    init(channels: Int, precision: MLX.DType) {
        self.precision = precision
        self.scale = 1.0 / sqrt(Float(channels))
        self._groupNorm.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels, eps: 1e-6, pytorchCompatible: true)
        func mk() -> Linear { Linear(channels, channels, bias: true) }
        self._toQ.wrappedValue = mk()
        self._toK.wrappedValue = mk()
        self._toV.wrappedValue = mk()
        self._toOut.wrappedValue = mk()
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        let (b, h, w, c) = (s[0], s[1], s[2], s[3])
        let residual = x
        let normed = gnorm(groupNorm, x, precision).reshaped([b, h * w, c])
        let q = toQ(normed).reshaped([b, 1, h * w, c])
        let k = toK(normed).reshaped([b, 1, h * w, c])
        let v = toV(normed).reshaped([b, 1, h * w, c])
        let attended = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let out = toOut(attended.reshaped([b, h, w, c]))
        return residual + out
    }
}

/// Mid block: resnet → attention → resnet (diffusers `UNetMidBlock2D`).
final class MidBlock: Module, UnaryLayer {
    @ModuleInfo(key: "resnets") var resnets: [ResnetBlock2D]
    @ModuleInfo(key: "attentions") var attentions: [VAEMidAttention]

    init(channels: Int, precision: MLX.DType) {
        self._resnets.wrappedValue = [
            ResnetBlock2D(inChannels: channels, outChannels: channels, precision: precision),
            ResnetBlock2D(inChannels: channels, outChannels: channels, precision: precision),
        ]
        self._attentions.wrappedValue = [VAEMidAttention(channels: channels, precision: precision)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = resnets[0](x)
        h = attentions[0](h)
        h = resnets[1](h)
        return h
    }
}

// MARK: - Encoder / Decoder

/// FLUX.2 VAE encoder. in 3-ch → 64 moments (2 × latent_channels=32).
final class FluxVAEEncoder: Module, UnaryLayer {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down_blocks") var downBlocks: [DownEncoderBlock]
    @ModuleInfo(key: "mid_block") var midBlock: MidBlock
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    private let precision: MLX.DType

    init(blockOutChannels: [Int] = [128, 256, 512, 512], precision: MLX.DType) {
        self.precision = precision
        let first = blockOutChannels[0]
        self._convIn.wrappedValue = Conv2d(inputChannels: 3, outputChannels: first, kernelSize: 3, padding: 1)
        var blocks: [DownEncoderBlock] = []
        for i in 0..<blockOutChannels.count {
            let inCh = blockOutChannels[max(i - 1, 0)]
            let outCh = blockOutChannels[i]
            blocks.append(DownEncoderBlock(inChannels: inCh, outChannels: outCh, precision: precision, addDown: i < blockOutChannels.count - 1))
        }
        self._downBlocks.wrappedValue = blocks
        self._midBlock.wrappedValue = MidBlock(channels: blockOutChannels.last!, precision: precision)
        self._convNormOut.wrappedValue = GroupNorm(groupCount: 32, dimensions: blockOutChannels.last!, eps: 1e-6, pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(inputChannels: blockOutChannels.last!, outputChannels: 64, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for b in downBlocks { h = b(h) }
        h = midBlock(h)
        h = gnorm(convNormOut, h, precision)
        h = silu(h).asType(precision)
        return convOut(h)
    }
}

/// FLUX.2 VAE decoder. in 32-ch latent → 3-ch image.
final class FluxVAEDecoder: Module, UnaryLayer {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_block") var midBlock: MidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [UpDecoderBlock]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    private let precision: MLX.DType

    init(blockOutChannels: [Int] = [128, 256, 512, 512], precision: MLX.DType) {
        self.precision = precision
        let rev = Array(blockOutChannels.reversed())  // [512, 512, 256, 128]
        let last = rev[0]                               // 512 (channels after conv_in)
        self._convIn.wrappedValue = Conv2d(inputChannels: 32, outputChannels: last, kernelSize: 3, padding: 1)
        self._midBlock.wrappedValue = MidBlock(channels: last, precision: precision)
        var blocks: [UpDecoderBlock] = []
        var prev = last
        for i in 0..<rev.count {
            let out = rev[i]
            blocks.append(UpDecoderBlock(inChannels: prev, outChannels: out, precision: precision, addUp: i < rev.count - 1))
            prev = out
        }
        self._upBlocks.wrappedValue = blocks
        self._convNormOut.wrappedValue = GroupNorm(groupCount: 32, dimensions: rev.last!, eps: 1e-6, pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(inputChannels: rev.last!, outputChannels: 3, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ z: MLXArray) -> MLXArray {
        var h = convIn(z)
        h = midBlock(h)
        for b in upBlocks { h = b(h) }
        h = gnorm(convNormOut, h, precision)
        h = silu(h).asType(precision)
        return convOut(h)
    }
}

// MARK: - FluxVAE

/// FLUX.2 Klein VAE (`AutoencoderKLFlux2`), 32-ch latent space.
///
/// `encode(_:)` returns the deterministic latent mode (32-ch, H/8, W/8) — the `bn` +
/// pixel-unshuffle to the transformer's 128-ch latent is applied at the pipeline level (M5).
/// The mid-block attention projections are 4-bit in the checkpoint but mflux dequantizes them
/// to bf16 `Linear` at load; convs/norms are bf16.
///
/// `precision` controls the running dtype (mflux `ModelConfig.precision`): `.bfloat16` for the
/// encode path, `.float32` for decode (diffusers `force_upcast = true`) to avoid bf16 drift.
public final class FluxVAE: Module {
    @ModuleInfo(key: "encoder") var encoder: FluxVAEEncoder
    @ModuleInfo(key: "decoder") var decoder: FluxVAEDecoder
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d

    public let precision: MLX.DType

    /// - Parameter precision: running dtype — `.float32` for decode (recommended), `.bfloat16` otherwise.
    public init(precision: MLX.DType = .float32) {
        self.precision = precision
        self._encoder.wrappedValue = FluxVAEEncoder(precision: precision)
        self._decoder.wrappedValue = FluxVAEDecoder(precision: precision)
        self._quantConv.wrappedValue = Conv2d(inputChannels: 64, outputChannels: 64, kernelSize: 1)
        self._postQuantConv.wrappedValue = Conv2d(inputChannels: 32, outputChannels: 32, kernelSize: 1)
        super.init()
    }

    /// Load weights from a `[String: MLXArray]` dict (as produced by `WeightLoader`).
    /// The 4-bit attention linears are dequantized to bf16 to match mflux; `bn.*` buffers are
    /// pipeline-level (no affine, applied in M5) and ignored here.
    public func loadWeights(_ flat: [String: MLXArray]) {
        let deq = WeightLoader.dequantized(flat, groupSize: 64, bits: 4)
        let filtered = deq.filter { !$0.key.hasPrefix("bn.") }
        update(parameters: NestedDictionary(flatWeights: filtered))
    }

    /// Encode an image (NHWC, `[-1, 1]`) to its latent mode `[B, H/8, W/8, 32]`.
    public func encode(_ x: MLXArray) -> MLXArray {
        let moments = quantConv(encoder(x))            // [B, H/8, W/8, 64]
        let mean = moments[.ellipsis, 0..<32]           // DiagonalGaussian.mode()
        return mean
    }

    /// Decode a latent `[B, H/8, W/8, 32]` to an image `[B, H, W, 3]` (≈ `[-1, 1]`).
    public func decode(_ z: MLXArray) -> MLXArray {
        decoder(postQuantConv(z))
    }
}
