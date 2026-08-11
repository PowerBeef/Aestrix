import Foundation
import MLX
import MLXNN
import AestrixCore

/// Decode-only VAE for T2I: omits encoder + quant_conv (~67 MB) to cut VAE-stage RAM on Tier L.
///
/// Weight keys match the hub pack prefixes: `decoder.*`, `post_quant_conv.*`, `bn.*`.
public final class Flux2VAEDecoderOnly: Module {
    public static let latentChannels = Flux2VAE.latentChannels

    @ModuleInfo(key: "decoder") var decoder: Flux2Decoder
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
    @ModuleInfo(key: "bn") var bn: Flux2BatchNormStats

    public override init() {
        let lc = Self.latentChannels
        self._decoder.wrappedValue = Flux2Decoder(inChannels: lc)
        self._postQuantConv.wrappedValue = Conv2d(
            inputChannels: lc, outputChannels: lc, kernelSize: 1, stride: 1, padding: 0)
        self._bn.wrappedValue = Flux2BatchNormStats(numFeatures: 4 * lc, eps: 1e-4)
        super.init()
    }

    /// Decode latent NCHW (32 ch) → RGB NCHW.
    public func decode(_ latents: MLXArray) -> MLXArray {
        var z = latents
        if z.ndim == 5 {
            z = z[0..., 0..., 0, 0..., 0...]
        }
        z = (z / Flux2VAE.scalingFactor) + Flux2VAE.shiftFactor
        z = z.transposed(0, 2, 3, 1)
        z = postQuantConv(z)
        z = z.transposed(0, 3, 1, 2)
        eval(z)
        Memory.clearCache()
        return decoder(z)
    }

    /// Decode packed DiT latents `[B, C*4, H, W]` using BN denorm + unpatchify.
    public func decodePackedLatents(
        _ packed: MLXArray,
        tileConfig: VAETileConfig = .default
    ) -> MLXArray {
        var p = packed
        if p.ndim == 5 {
            p = p[0..., 0..., 0, 0..., 0...]
        }
        let bnMean = bn.runningMean.reshaped([1, -1, 1, 1])
        let bnStd = sqrt(bn.runningVar.reshaped([1, -1, 1, 1]) + bn.eps)
        var latents = p * bnStd + bnMean
        latents = Flux2VAE.unpatchify(latents)
        eval(latents)
        Memory.clearCache()
        // 1024² → unpatchified 128×128×32; full UNet decode peaks hard on 8 GB.
        // Default: overlapping cosine-blended tiles (see `VAETileConfig`).
        if tileConfig.shouldTile(height: latents.dim(2), width: latents.dim(3)) {
            return Flux2VAE.decodeLatentsTiled(latents, decode: decode, config: tileConfig)
        }
        return decode(latents)
    }
}
