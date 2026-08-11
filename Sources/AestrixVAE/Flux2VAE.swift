import Foundation
import MLX
import MLXNN
import AestrixCore

/// FLUX.2 VAE: encoder / decoder + quant convs + BN stats for packed latents.
public final class Flux2VAE: Module {
    public static let latentChannels = 32
    public static let scalingFactor: Float = 1.0
    public static let shiftFactor: Float = 0.0

    @ModuleInfo(key: "encoder") var encoder: Flux2Encoder
    @ModuleInfo(key: "decoder") var decoder: Flux2Decoder
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
    @ModuleInfo(key: "bn") var bn: Flux2BatchNormStats

    public override init() {
        let lc = Self.latentChannels
        self._encoder.wrappedValue = Flux2Encoder(outChannels: lc)
        self._decoder.wrappedValue = Flux2Decoder(inChannels: lc)
        self._quantConv.wrappedValue = Conv2d(
            inputChannels: 2 * lc, outputChannels: 2 * lc, kernelSize: 1, stride: 1, padding: 0)
        self._postQuantConv.wrappedValue = Conv2d(
            inputChannels: lc, outputChannels: lc, kernelSize: 1, stride: 1, padding: 0)
        self._bn.wrappedValue = Flux2BatchNormStats(numFeatures: 4 * lc, eps: 1e-4)
        super.init()
    }

    /// Encode RGB image NCHW in roughly [-1,1] → latent NCHW (32 ch).
    public func encode(_ image: MLXArray) -> MLXArray {
        var img = image
        if img.ndim == 5 {
            img = img[0..., 0..., 0, 0..., 0...]
        }
        var enc = encoder(img)
        enc = enc.transposed(0, 2, 3, 1)
        enc = quantConv(enc)
        enc = enc.transposed(0, 3, 1, 2)
        let parts = split(enc, parts: 2, axis: 1)
        let mean = parts[0]
        return (mean - Self.shiftFactor) * Self.scalingFactor
    }

    /// Encode for DiT I2I: 32ch latent → 2×2 patchify → BN normalize → `[B, 128, H/16, W/16]`.
    /// Inverse of the first half of `decodePackedLatents`.
    public func encodePackedForDiT(_ image: MLXArray) -> MLXArray {
        let z32 = encode(image)
        var packed = Self.patchify(z32)
        let bnMean = bn.runningMean.reshaped([1, -1, 1, 1])
        let bnStd = sqrt(bn.runningVar.reshaped([1, -1, 1, 1]) + bn.eps)
        packed = (packed - bnMean) / bnStd
        return packed
    }

    /// Decode latent NCHW (32 ch) → RGB NCHW.
    public func decode(_ latents: MLXArray) -> MLXArray {
        var z = latents
        if z.ndim == 5 {
            z = z[0..., 0..., 0, 0..., 0...]
        }
        z = (z / Self.scalingFactor) + Self.shiftFactor
        z = z.transposed(0, 2, 3, 1)
        z = postQuantConv(z)
        z = z.transposed(0, 3, 1, 2)
        return decoder(z)
    }

    /// Decode packed DiT latents [B, C*4, H, W] using BN denorm + unpatchify.
    public func decodePackedLatents(_ packed: MLXArray) -> MLXArray {
        var p = packed
        if p.ndim == 5 {
            p = p[0..., 0..., 0, 0..., 0...]
        }
        let bnMean = bn.runningMean.reshaped([1, -1, 1, 1])
        let bnStd = sqrt(bn.runningVar.reshaped([1, -1, 1, 1]) + bn.eps)
        var latents = p * bnStd + bnMean
        latents = Self.unpatchify(latents)
        return decode(latents)
    }

    public static func unpatchify(_ latents: MLXArray) -> MLXArray {
        let batch = latents.dim(0)
        let numChannels = latents.dim(1)
        let height = latents.dim(2)
        let width = latents.dim(3)
        var x = latents.reshaped([batch, numChannels / 4, 2, 2, height, width])
        x = x.transposed(0, 1, 4, 2, 5, 3)
        return x.reshaped([batch, numChannels / 4, height * 2, width * 2])
    }

    public static func patchify(_ latents: MLXArray) -> MLXArray {
        // Inverse of unpatchify for encode→pack path (I2I).
        let batch = latents.dim(0)
        let c = latents.dim(1)
        let h = latents.dim(2)
        let w = latents.dim(3)
        var x = latents.reshaped([batch, c, h / 2, 2, w / 2, 2])
        x = x.transposed(0, 1, 3, 5, 2, 4)
        return x.reshaped([batch, c * 4, h / 2, w / 2])
    }
}
