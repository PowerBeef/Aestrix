import Foundation
import MLX
import MLXNN

// MARK: - ResnetBlock2D

/// diffusers `ResnetBlock2D` (no time embedding). NHWC. norm → silu → conv → norm → silu → conv + shortcut.
final class ResnetBlock2D: Module, UnaryLayer {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    init(inChannels: Int, outChannels: Int, groups: Int = 32) {
        self._norm1.wrappedValue = GroupNorm(groupCount: groups, dimensions: inChannels)
        self._conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._norm2.wrappedValue = GroupNorm(groupCount: groups, dimensions: outChannels)
        self._conv2.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._convShortcut.wrappedValue = (inChannels != outChannels)
            ? Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
            : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm1(x)
        h = silu(h)
        h = conv1(h)
        h = norm2(h)
        h = silu(h)
        h = conv2(h)
        let shortcut = convShortcut?(x) ?? x
        return h + shortcut
    }
}

// MARK: - Down/Up samplers

/// diffusers `Downsample2D`: stride-2 conv (3×3, pad 1).
final class Downsample2D: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(channels: Int) {
        self._conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2, padding: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

/// diffusers `Upsample2D`: nearest-neighbour ×2 then conv (3×3, pad 1).
final class Upsample2D: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(channels: Int) {
        self._conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        let (b, h, w, c) = (s[0], s[1], s[2], s[3])
        // Insert singleton axes on H and W, broadcast to 2×, then flatten back → nearest ×2.
        let y0 = x.reshaped([b, h, 1, w, 1, c])
        let y = broadcast(y0, to: [b, h, 2, w, 2, c]).reshaped([b, h * 2, w * 2, c])
        return conv(y)
    }
}

// MARK: - Encoder / Decoder blocks

/// Encoder down block: 2 resnets + optional downsampler.
final class DownEncoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "resnets") var resnets: [ResnetBlock2D]
    @ModuleInfo(key: "downsamplers") var downsamplers: [Downsample2D]?

    init(inChannels: Int, outChannels: Int, layersPerBlock: Int = 2, addDown: Bool) {
        var res: [ResnetBlock2D] = []
        for i in 0..<layersPerBlock {
            let inCh = (i == 0) ? inChannels : outChannels
            res.append(ResnetBlock2D(inChannels: inCh, outChannels: outChannels))
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

    init(inChannels: Int, outChannels: Int, layersPerBlock: Int = 3, addUp: Bool) {
        var res: [ResnetBlock2D] = []
        for i in 0..<layersPerBlock {
            let inCh = (i == 0) ? inChannels : outChannels
            res.append(ResnetBlock2D(inChannels: inCh, outChannels: outChannels))
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

// MARK: - Mid-block spatial self-attention (4-bit quantized Q/K/V/out)

/// diffusers VAE mid-block `Attention`: group_norm → (B, H·W, C) tokens → Q/K/V (quantized) →
/// single-head SDPA → to_out (quantized) → reshape back → residual.
final class VAEMidAttention: Module, UnaryLayer {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: QuantizedLinear
    @ModuleInfo(key: "to_k") var toK: QuantizedLinear
    @ModuleInfo(key: "to_v") var toV: QuantizedLinear
    @ModuleInfo(key: "to_out") var toOut: QuantizedLinear

    private let scale: Float

    init(channels: Int, groupSize: Int = 64, bits: Int = 4) {
        self.scale = 1.0 / sqrt(Float(channels))
        self._groupNorm.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels)
        func mk() -> QuantizedLinear { QuantizedLinear(channels, channels, bias: true, groupSize: groupSize, bits: bits) }
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
        var q = groupNorm(x).reshaped([b, h * w, c])
        // Q/K/V project to `channels` (single-head).
        let k = toK(q)
        let v = toV(q)
        q = toQ(q)
        // scores = softmax(Q·Kᵀ / √C) · V
        let scores = softmax(matmul(q, k.transposed(0, 2, 1)) * scale, axis: -1)
        var out = matmul(scores, v)
        out = toOut(out)
        return residual + out.reshaped([b, h, w, c])
    }
}

/// Mid block: resnet → attention → resnet (diffusers `UNetMidBlock2D`).
final class MidBlock: Module, UnaryLayer {
    @ModuleInfo(key: "resnets") var resnets: [ResnetBlock2D]
    @ModuleInfo(key: "attentions") var attentions: [VAEMidAttention]

    init(channels: Int) {
        self._resnets.wrappedValue = [
            ResnetBlock2D(inChannels: channels, outChannels: channels),
            ResnetBlock2D(inChannels: channels, outChannels: channels),
        ]
        self._attentions.wrappedValue = [VAEMidAttention(channels: channels)]
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

    /// `blockOutChannels` defaults to FLUX.2's `(128, 256, 512, 512)`.
    init(blockOutChannels: [Int] = [128, 256, 512, 512]) {
        let first = blockOutChannels[0]
        self._convIn.wrappedValue = Conv2d(inputChannels: 3, outputChannels: first, kernelSize: 3, padding: 1)
        var blocks: [DownEncoderBlock] = []
        for i in 0..<blockOutChannels.count {
            let inCh = blockOutChannels[max(i - 1, 0)]
            let outCh = blockOutChannels[i]
            blocks.append(DownEncoderBlock(inChannels: inCh, outChannels: outCh, addDown: i < blockOutChannels.count - 1))
        }
        self._downBlocks.wrappedValue = blocks
        self._midBlock.wrappedValue = MidBlock(channels: blockOutChannels.last!)
        self._convNormOut.wrappedValue = GroupNorm(groupCount: 32, dimensions: blockOutChannels.last!)
        self._convOut.wrappedValue = Conv2d(inputChannels: blockOutChannels.last!, outputChannels: 64, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for b in downBlocks { h = b(h) }
        h = midBlock(h)
        h = convNormOut(h)
        h = silu(h)
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

    init(blockOutChannels: [Int] = [128, 256, 512, 512]) {
        let rev = Array(blockOutChannels.reversed())  // [512, 512, 256, 128]
        let last = rev[0]                               // 512 (channels after conv_in)
        self._convIn.wrappedValue = Conv2d(inputChannels: 32, outputChannels: last, kernelSize: 3, padding: 1)
        self._midBlock.wrappedValue = MidBlock(channels: last)
        var blocks: [UpDecoderBlock] = []
        var prev = last
        for i in 0..<rev.count {
            let out = rev[i]
            blocks.append(UpDecoderBlock(inChannels: prev, outChannels: out, addUp: i < rev.count - 1))
            prev = out
        }
        self._upBlocks.wrappedValue = blocks
        self._convNormOut.wrappedValue = GroupNorm(groupCount: 32, dimensions: rev.last!)
        self._convOut.wrappedValue = Conv2d(inputChannels: rev.last!, outputChannels: 3, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ z: MLXArray) -> MLXArray {
        var h = convIn(z)
        h = midBlock(h)
        for b in upBlocks { h = b(h) }
        h = convNormOut(h)
        h = silu(h)
        return convOut(h)
    }
}

// MARK: - FluxVAE

/// FLUX.2 Klein VAE (`AutoencoderKLFlux2`), 32-ch latent space.
///
/// `encode(_:)` returns the deterministic latent mode (32-ch, H/8, W/8) — the `bn` +
/// pixel-unshuffle to the transformer's 128-ch latent is applied at the pipeline level (M5).
/// The mid-block attention projections (`to_q/k/v/out`) are 4-bit quantized; convs/norms are bf16.
public final class FluxVAE: Module {
    @ModuleInfo(key: "encoder") var encoder: FluxVAEEncoder
    @ModuleInfo(key: "decoder") var decoder: FluxVAEDecoder
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d

    public override init() {
        self._encoder.wrappedValue = FluxVAEEncoder()
        self._decoder.wrappedValue = FluxVAEDecoder()
        self._quantConv.wrappedValue = Conv2d(inputChannels: 64, outputChannels: 64, kernelSize: 1)
        self._postQuantConv.wrappedValue = Conv2d(inputChannels: 32, outputChannels: 32, kernelSize: 1)
        super.init()
    }

    /// Load weights from a `[String: MLXArray]` dict (as produced by `WeightLoader`).
    /// The `bn.*` buffers are pipeline-level and ignored here (no affine, applied in M5).
    public func loadWeights(_ flat: [String: MLXArray]) {
        let filtered = flat.filter { !$0.key.hasPrefix("bn.") }
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
