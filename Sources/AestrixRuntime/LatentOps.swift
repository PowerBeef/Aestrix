import Foundation
import MLX
import AestrixCore
import AestrixVAE

/// Packed-latent helpers for FLUX.2 klein T2I (patchify / pack / ids / Euler).
enum LatentOps {
    /// Channels after 2×2 patchify: VAE latent 32 → 128.
    static let packedChannels = Flux2VAE.latentChannels * 4  // 128

    /// Spatial size of packed latents for a pixel canvas (H/16 × W/16).
    static func packedSpatial(width: Int, height: Int) -> (h: Int, w: Int) {
        let scale = ModelConstants.vaeScaleFactor * 2  // 16
        return (height / scale, width / scale)
    }

    /// Sample noise already in packed sequence form: `[B, H·W, 128]`.
    static func samplePackedNoise(
        batch: Int = 1,
        width: Int,
        height: Int,
        seed: UInt64?
    ) -> MLXArray {
        let (h, w) = packedSpatial(width: width, height: height)
        let shape = [batch, h * w, packedChannels]
        if let seed {
            MLXRandom.seed(seed)
        }
        return MLXRandom.normal(shape, dtype: .float32)
    }

    /// Image position ids `[1, H·W, 4]` with coords `(t=0, h, w, l=0)`.
    static func imageIds(width: Int, height: Int) -> MLXArray {
        let (h, w) = packedSpatial(width: width, height: height)
        let rows = Flux2RoPE.prepareGridIDs(height: h, width: w)
        let flat = rows.flatMap { $0 }
        return MLXArray(flat).reshaped([1, h * w, 4])
    }

    /// Text position ids `[1, L, 4]` with coords `(t=0, h=0, w=0, l=i)`.
    static func textIds(length: Int = ModelConstants.maxSequenceLength) -> MLXArray {
        let rows = Flux2RoPE.prepareTextIDs(length: length)
        let flat = rows.flatMap { $0 }
        return MLXArray(flat).reshaped([1, length, 4])
    }

    /// `[B, S, C]` → `[B, C, H, W]` (row-major H then W).
    static func unpackSequence(_ packed: MLXArray, height: Int, width: Int) -> MLXArray {
        let b = packed.dim(0)
        let c = packed.dim(2)
        // (B, S, C) → (B, C, S) → (B, C, H, W)
        return packed.transposed(0, 2, 1).reshaped([b, c, height, width])
    }

    /// `[B, C, H, W]` → `[B, H·W, C]`.
    static func packSpatial(_ spatial: MLXArray) -> MLXArray {
        let b = spatial.dim(0)
        let c = spatial.dim(1)
        let h = spatial.dim(2)
        let w = spatial.dim(3)
        return spatial.reshaped([b, c, h * w]).transposed(0, 2, 1)
    }

    /// Euler: `x + (σ_{t+1} − σ_t) · e`.
    ///
    /// Uses a compiled kernel so the cheap residual add is fused across denoise steps.
    static func eulerStep(
        sample: MLXArray,
        modelOutput: MLXArray,
        sigma: Float,
        sigmaNext: Float
    ) -> MLXArray {
        eulerStep(sample: sample, modelOutput: modelOutput, dt: MLXArray(sigmaNext - sigma))
    }

    /// Euler with a **precomputed** `dt` array (preferred in the denoise loop).
    static func eulerStep(
        sample: MLXArray,
        modelOutput: MLXArray,
        dt: MLXArray
    ) -> MLXArray {
        compiledEuler(sample, modelOutput, dt)
    }

    /// Precompute per-step `σ_{t+1} − σ_t` for a schedule (eval once before denoise).
    static func eulerDts(sigmas: [Float]) -> [MLXArray] {
        precondition(sigmas.count >= 2)
        var dts: [MLXArray] = []
        dts.reserveCapacity(sigmas.count - 1)
        for i in 0 ..< (sigmas.count - 1) {
            dts.append(MLXArray(sigmas[i + 1] - sigmas[i]))
        }
        eval(dts)
        return dts
    }

    /// Compiled once: `(sample, pred, dt) → sample + pred * dt`.
    private static let compiledEuler: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = {
        compile(shapeless: true) { (sample: MLXArray, pred: MLXArray, dt: MLXArray) -> MLXArray in
            sample + pred * dt
        }
    }()

    /// Flow-match scale noise (diffusers-style): `(1 − σ)·x₀ + σ·ε`.
    static func scaleNoise(clean: MLXArray, noise: MLXArray, sigma: Float) -> MLXArray {
        compiledScaleNoise(clean, noise, MLXArray(sigma))
    }

    private static let compiledScaleNoise: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = {
        compile(shapeless: true) { (clean: MLXArray, noise: MLXArray, sigma: MLXArray) -> MLXArray in
            clean * (1 - sigma) + noise * sigma
        }
    }()

    /// Sample noise matching an existing packed latent's shape.
    static func sampleNoiseLike(_ array: MLXArray, seed: UInt64?) -> MLXArray {
        if let seed {
            MLXRandom.seed(seed)
        }
        return MLXRandom.normal(array.shape, dtype: array.dtype)
    }
}
