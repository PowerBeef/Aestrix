import Foundation
import MLX

/// Latent-space transforms that bridge the VAE (32-ch) and transformer (128-ch packed) spaces.
///
/// The VAE produces 32-channel latents at H/8 × W/8. Pixel-unshuffle (patchify) folds the
/// spatial dims ×2 into channels → 128-ch at H/16 × W/16. The transformer operates on these
/// 128-ch latents, flattened to a sequence. BatchNorm running stats (from the VAE checkpoint)
/// normalize the packed latents for the transformer and are reversed at decode time.
enum LatentTransforms {

    /// Flatten [B, C, H, W] → [B, H*W, C] (mflux `pack_latents`).
    static func pack(_ latents: MLXArray) -> MLXArray {
        let b = latents.shape[0], c = latents.shape[1], h = latents.shape[2], w = latents.shape[3]
        return latents.reshaped([b, c, h * w]).transposed(0, 2, 1)
    }

    /// Inverse of pack: [B, seq, C] → [B, C, H, W].
    static func unpack(_ packed: MLXArray, height: Int, width: Int) -> MLXArray {
        let b = packed.shape[0], c = packed.shape[2]
        return packed.transposed(0, 2, 1).reshaped([b, c, height, width])
    }

    /// Pixel-unshuffle: [B, 32, H, W] → [B, 128, H/2, W/2] (mflux `patchify_latents`).
    static func patchify(_ latents: MLXArray) -> MLXArray {
        let b = latents.shape[0], c = latents.shape[1], h = latents.shape[2], w = latents.shape[3]
        // reshape [B, C, H/2, 2, W/2, 2] → transpose [B, C, 2, 2, H/2, W/2] → reshape [B, C*4, H/2, W/2]
        return latents.reshaped([b, c, h / 2, 2, w / 2, 2])
            .transposed(0, 1, 3, 5, 2, 4)
            .reshaped([b, c * 4, h / 2, w / 2])
    }

    /// Pixel-shuffle (inverse of patchify): [B, 128, h, w] → [B, 32, h*2, w*2]
    /// (mflux `Flux2VAE._unpatchify_latents`).
    static func unpatchify(_ latents: MLXArray) -> MLXArray {
        let b = latents.shape[0], c = latents.shape[1], h = latents.shape[2], w = latents.shape[3]
        // reshape [B, C/4, 2, 2, h, w] → transpose [B, C/4, h, 2, w, 2] → reshape [B, C/4, h*2, w*2]
        return latents.reshaped([b, c / 4, 2, 2, h, w])
            .transposed(0, 1, 4, 2, 5, 3)
            .reshaped([b, c / 4, h * 2, w * 2])
    }

    /// Denormalize packed latents from transformer space to VAE space:
    /// `latents * std + mean` using the VAE BatchNorm running stats.
    static func bnDenormalize(_ packedLatents: MLXArray, runningMean: MLXArray, runningVar: MLXArray, eps: Float) -> MLXArray {
        // running_mean/var are [128] (128-ch packed space). Reshape to [1, 128, 1, 1] for NCHW.
        let mean = runningMean.reshaped([1, -1, 1, 1])
        let std = sqrt(runningVar.reshaped([1, -1, 1, 1]) + eps)
        return packedLatents * std + mean
    }
}
