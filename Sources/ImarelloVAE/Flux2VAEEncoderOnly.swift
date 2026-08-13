import Foundation
import MLX
import MLXNN
import ImarelloCore

/// Encode-only VAE for the I2I stage-0 reference encode: omits the decoder (~97 MB)
/// so encode residency matches decode-only on the other end of the pipeline.
///
/// Weight keys match the hub pack prefixes: `encoder.*`, `quant_conv.*`, `bn.*`.
public final class Flux2VAEEncoderOnly: Module {
    public static let latentChannels = Flux2VAE.latentChannels

    @ModuleInfo(key: "encoder") var encoder: Flux2Encoder
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    @ModuleInfo(key: "bn") var bn: Flux2BatchNormStats

    public override init() {
        let lc = Self.latentChannels
        self._encoder.wrappedValue = Flux2Encoder(outChannels: lc)
        self._quantConv.wrappedValue = Conv2d(
            inputChannels: 2 * lc, outputChannels: 2 * lc, kernelSize: 1, stride: 1, padding: 0)
        self._bn.wrappedValue = Flux2BatchNormStats(numFeatures: 4 * lc, eps: 1e-4)
        super.init()
    }

    /// Encode RGB image NCHW in roughly [-1,1] → latent NCHW (32 ch). Mirrors `Flux2VAE.encode`.
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
        return (mean - Flux2VAE.shiftFactor) * Flux2VAE.scalingFactor
    }

    /// Encode for DiT I2I: 32ch latent → 2×2 patchify → BN normalize → `[B, 128, H/16, W/16]`.
    /// Mirrors `Flux2VAE.encodePackedForDiT`.
    public func encodePackedForDiT(_ image: MLXArray) -> MLXArray {
        let z32 = encode(image)
        var packed = Flux2VAE.patchify(z32)
        let bnMean = bn.runningMean.reshaped([1, -1, 1, 1])
        let bnStd = sqrt(bn.runningVar.reshaped([1, -1, 1, 1]) + bn.eps)
        packed = (packed - bnMean) / bnStd
        return packed
    }
}
