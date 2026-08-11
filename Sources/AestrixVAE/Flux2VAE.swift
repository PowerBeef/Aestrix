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
        eval(z)
        Memory.clearCache()
        return decoder(z)
    }

    /// Decode packed DiT latents [B, C*4, H, W] using BN denorm + unpatchify.
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
        latents = Self.unpatchify(latents)
        eval(latents)
        Memory.clearCache()
        if tileConfig.shouldTile(height: latents.dim(2), width: latents.dim(3)) {
            return Self.decodeLatentsTiled(latents, decode: decode, config: tileConfig)
        }
        return decode(latents)
    }

    /// Low-RAM decode: tile NCHW latents, decode each, stitch RGB.
    /// - `.none`: non-overlapping N×N hard stitch (legacy 2×2).
    /// - `.cosine`: overlapping tiles + separable cosine blend (PDF / Draw Things style).
    public static func decodeLatentsTiled(
        _ latents: MLXArray,
        decode: (MLXArray) -> MLXArray,
        config: VAETileConfig = .default
    ) -> MLXArray {
        switch config.blend {
        case .none:
            return decodeLatentsTiledHard(latents, decode: decode, grid: config.noneGrid)
        case .cosine:
            return decodeLatentsTiledBlended(latents, decode: decode, config: config)
        }
    }

    /// Legacy API: non-overlapping grid stitch.
    public static func decodeLatentsTiled(
        _ latents: MLXArray,
        decode: (MLXArray) -> MLXArray,
        grid: Int
    ) -> MLXArray {
        decodeLatentsTiledHard(latents, decode: decode, grid: grid)
    }

    private static func decodeLatentsTiledHard(
        _ latents: MLXArray,
        decode: (MLXArray) -> MLXArray,
        grid: Int
    ) -> MLXArray {
        let h = latents.dim(2)
        let w = latents.dim(3)
        precondition(grid > 0 && h % grid == 0 && w % grid == 0, "latent H/W must divide tile grid")
        let th = h / grid
        let tw = w / grid
        var rows: [MLXArray] = []
        rows.reserveCapacity(grid)
        for yi in 0 ..< grid {
            var cols: [MLXArray] = []
            cols.reserveCapacity(grid)
            for xi in 0 ..< grid {
                let y0 = yi * th
                let x0 = xi * tw
                let tile = latents[0..., 0..., y0 ..< (y0 + th), x0 ..< (x0 + tw)]
                eval(tile)
                Memory.clearCache()
                let out = decode(tile)
                eval(out)
                Memory.clearCache()
                cols.append(out)
            }
            let row = concatenated(cols, axis: 3)
            eval(row)
            rows.append(row)
        }
        let rgb = concatenated(rows, axis: 2)
        eval(rgb)
        return rgb
    }

    private static func decodeLatentsTiledBlended(
        _ latents: MLXArray,
        decode: (MLXArray) -> MLXArray,
        config: VAETileConfig
    ) -> MLXArray {
        let h = latents.dim(2)
        let w = latents.dim(3)
        let batch = latents.dim(0)
        var tileSize = min(config.tileSize, h, w)
        if tileSize < 1 { tileSize = min(h, w) }
        var overlap = min(config.overlap, tileSize - 1)
        if overlap < 0 { overlap = 0 }

        let yStarts = VAETileMath.tileStarts(length: h, tileSize: tileSize, overlap: overlap)
        let xStarts = VAETileMath.tileStarts(length: w, tileSize: tileSize, overlap: overlap)

        // Infer RGB scale from first tile decode (FLUX.2 VAE is 8× after unpatchify).
        let y0First = yStarts[0]
        let x0First = xStarts[0]
        let thFirst = min(tileSize, h - y0First)
        let twFirst = min(tileSize, w - x0First)
        let firstTile = latents[0..., 0..., y0First ..< (y0First + thFirst), x0First ..< (x0First + twFirst)]
        eval(firstTile)
        Memory.clearCache()
        let firstOut = decode(firstTile)
        eval(firstOut)
        Memory.clearCache()
        let scaleY = firstOut.dim(2) / thFirst
        let scaleX = firstOut.dim(3) / twFirst
        precondition(scaleY > 0 && scaleX > 0, "VAE decode scale must be positive")
        let outH = h * scaleY
        let outW = w * scaleX
        let channels = firstOut.dim(1)

        var sumRGB = MLXArray.zeros([batch, channels, outH, outW], dtype: firstOut.dtype)
        var sumW = MLXArray.zeros([batch, 1, outH, outW], dtype: firstOut.dtype)

        func accumulate(tileY0: Int, tileX0: Int, out: MLXArray, th: Int, tw: Int) {
            let fadeTop = VAETileMath.leadingFade(origin: tileY0, overlap: overlap)
            let fadeBottom = VAETileMath.trailingFade(
                origin: tileY0, tileLen: th, fullLen: h, overlap: overlap)
            let fadeLeft = VAETileMath.leadingFade(origin: tileX0, overlap: overlap)
            let fadeRight = VAETileMath.trailingFade(
                origin: tileX0, tileLen: tw, fullLen: w, overlap: overlap)
            let maskLat = VAETileMath.weightMask(
                tileH: th, tileW: tw,
                fadeTop: fadeTop, fadeBottom: fadeBottom,
                fadeLeft: fadeLeft, fadeRight: fadeRight
            )
            let maskRGB = VAETileMath.upsampleNearest(
                mask: maskLat, height: th, width: tw, scaleY: scaleY, scaleX: scaleX
            )
            let rh = th * scaleY
            let rw = tw * scaleX
            let weight = MLXArray(maskRGB, [1, 1, rh, rw]).asType(out.dtype)
            let yRGB = tileY0 * scaleY
            let xRGB = tileX0 * scaleX
            let weighted = out * weight
            // Pad tile onto full canvas then add (no in-place slice write in MLX).
            let tileCanvas = putSlice(y0: yRGB, x0: xRGB, h: rh, w: rw, value: weighted, fullH: outH, fullW: outW)
            let wCanvas = putSlice(y0: yRGB, x0: xRGB, h: rh, w: rw, value: weight, fullH: outH, fullW: outW)
            sumRGB = sumRGB + tileCanvas
            sumW = sumW + wCanvas
            eval(sumRGB, sumW)
            Memory.clearCache()
        }

        // First tile already decoded.
        accumulate(tileY0: y0First, tileX0: x0First, out: firstOut, th: thFirst, tw: twFirst)
        Memory.clearCache()

        for (yi, y0) in yStarts.enumerated() {
            for (xi, x0) in xStarts.enumerated() {
                if yi == 0 && xi == 0 { continue }
                let th = min(tileSize, h - y0)
                let tw = min(tileSize, w - x0)
                let tile = latents[0..., 0..., y0 ..< (y0 + th), x0 ..< (x0 + tw)]
                eval(tile)
                Memory.clearCache()
                let out = decode(tile)
                eval(out)
                Memory.clearCache()
                accumulate(tileY0: y0, tileX0: x0, out: out, th: th, tw: tw)
            }
        }

        let eps = MLXArray(Float(1e-8)).asType(sumRGB.dtype)
        let rgb = sumRGB / maximum(sumW, eps)
        eval(rgb)
        Memory.clearCache()
        return rgb
    }

    /// Embed `value` […,h,w] into a full […,fullH,fullW] canvas at (y0,x0) via pad.
    private static func putSlice(
        y0: Int,
        x0: Int,
        h: Int,
        w: Int,
        value: MLXArray,
        fullH: Int,
        fullW: Int
    ) -> MLXArray {
        let padTop = y0
        let padBottom = fullH - (y0 + h)
        let padLeft = x0
        let padRight = fullW - (x0 + w)
        precondition(padTop >= 0 && padBottom >= 0 && padLeft >= 0 && padRight >= 0)
        if padTop == 0 && padBottom == 0 && padLeft == 0 && padRight == 0 {
            return value
        }
        return padded(
            value,
            widths: [
                IntOrPair((0, 0)),
                IntOrPair((0, 0)),
                IntOrPair((padTop, padBottom)),
                IntOrPair((padLeft, padRight)),
            ]
        )
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
