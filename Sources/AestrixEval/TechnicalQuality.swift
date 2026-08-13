import Foundation

/// No-reference technical quality metrics for a single image.
public enum TechnicalQuality {
    public struct Metrics: Sendable, Codable, Equatable {
        public var width: Int
        public var height: Int
        public var megapixels: Float

        /// Laplacian variance of luminance (higher = sharper). Typical: soft <50, ok 80–400, very sharp >600.
        public var sharpnessLaplacianVar: Float
        /// Mean absolute gradient magnitude (secondary sharpness).
        public var meanGradient: Float

        public var meanLuminance: Float
        public var stdLuminance: Float
        public var contrastRMS: Float

        /// Fraction of pixels near black / white (channel clip).
        public var clipBlackFraction: Float
        public var clipWhiteFraction: Float

        public var meanSaturation: Float
        public var meanRGB: [Float]
        /// Dominant coarse hue bucket: red/orange/yellow/green/cyan/blue/purple/neutral.
        public var dominantHue: String
        public var dominantHueFraction: Float
        /// Normalized weights for all hue buckets (incl. neutral); sums to ~1.
        public var hueWeights: [String: Float]
        /// Chromatic hues sorted by weight (excludes neutral) — better for subject color.
        public var topChromaticHues: [String]

        /// High-frequency residual after 3×3 box blur — proxy for grain/noise.
        public var noiseProxy: Float

        /// Shannon entropy of 64-bin luminance histogram (0–6-ish).
        public var luminanceEntropy: Float

        /// True when max(side) ≥ 768 — Aestrix VAE tiles unpatchified latents at spatial ≥ 96 (~768 px).
        public var expectsVAETiling: Bool
        /// Peak midline discontinuity / global gradient (higher ⇒ more tile-seam-like). Typical clean <1.5.
        public var tileSeamScore: Float
        /// Strongest vertical seam (column discontinuity).
        public var tileSeamVertical: Float
        /// Strongest horizontal seam (row discontinuity).
        public var tileSeamHorizontal: Float

        /// Composite 0…100 (heuristic).
        public var technicalScore: Float
    }

    public static func analyze(_ pixels: PixelBuffer) -> Metrics {
        let n = pixels.pixelCount
        let lum = pixels.luminances()
        let (meanL, stdL) = meanStd(lum)
        let sharp = laplacianVariance(lum, width: pixels.width, height: pixels.height)
        let grad = meanAbsGradient(lum, width: pixels.width, height: pixels.height)
        let (clipB, clipW) = clipFractions(pixels)
        let sat = meanSaturation(pixels)
        let mean = pixels.meanRGB()
        let hueStats = hueStatistics(pixels)
        let noise = noiseProxy(lum, width: pixels.width, height: pixels.height)
        let entropy = luminanceEntropy(lum)
        let seams = tileSeamDiscontinuities(lum, width: pixels.width, height: pixels.height, meanGradient: grad)
        let expectsTile = max(pixels.width, pixels.height) >= 768

        let score = compositeScore(
            sharp: sharp,
            contrast: stdL,
            clipB: clipB,
            clipW: clipW,
            noise: noise,
            entropy: entropy,
            tileSeam: expectsTile ? seams.score : 0
        )

        return Metrics(
            width: pixels.width,
            height: pixels.height,
            megapixels: Float(n) / 1_000_000,
            sharpnessLaplacianVar: sharp,
            meanGradient: grad,
            meanLuminance: meanL,
            stdLuminance: stdL,
            contrastRMS: stdL,
            clipBlackFraction: clipB,
            clipWhiteFraction: clipW,
            meanSaturation: sat,
            meanRGB: [mean.r, mean.g, mean.b],
            dominantHue: hueStats.dominant,
            dominantHueFraction: hueStats.dominantFraction,
            hueWeights: hueStats.weights,
            topChromaticHues: hueStats.topChromatic,
            noiseProxy: noise,
            luminanceEntropy: entropy,
            expectsVAETiling: expectsTile,
            tileSeamScore: seams.score,
            tileSeamVertical: seams.vertical,
            tileSeamHorizontal: seams.horizontal,
            technicalScore: score
        )
    }

    // MARK: - Internals

    private static func meanStd(_ x: [Float]) -> (Float, Float) {
        let n = Float(x.count)
        guard n > 0 else { return (0, 0) }
        let m = x.reduce(0, +) / n
        var v: Float = 0
        for e in x {
            let d = e - m
            v += d * d
        }
        return (m, sqrt(v / n))
    }

    private static func laplacianVariance(_ lum: [Float], width: Int, height: Int) -> Float {
        // Scale to ~0–255 so scores match OpenCV-style thresholds used in the report.
        // Discrete Laplacian kernel: [[0,1,0],[1,-4,1],[0,1,0]]
        var acc: Float = 0
        var count: Float = 0
        if height < 3 || width < 3 { return 0 }
        for y in 1 ..< (height - 1) {
            for x in 1 ..< (width - 1) {
                let i = y * width + x
                let c = lum[i] * 255
                let l = -4 * c
                    + lum[i - 1] * 255 + lum[i + 1] * 255
                    + lum[i - width] * 255 + lum[i + width] * 255
                acc += l * l
                count += 1
            }
        }
        return count > 0 ? acc / count : 0
    }

    private static func meanAbsGradient(_ lum: [Float], width: Int, height: Int) -> Float {
        var acc: Float = 0
        var count: Float = 0
        for y in 0 ..< (height - 1) {
            for x in 0 ..< (width - 1) {
                let i = y * width + x
                let dx = abs(lum[i + 1] - lum[i]) * 255
                let dy = abs(lum[i + width] - lum[i]) * 255
                acc += dx + dy
                count += 2
            }
        }
        return count > 0 ? acc / count : 0
    }

    private static func clipFractions(_ p: PixelBuffer) -> (Float, Float) {
        var black = 0, white = 0
        let n = p.pixelCount
        for i in 0 ..< n {
            let r = p.rgb[i * 3], g = p.rgb[i * 3 + 1], b = p.rgb[i * 3 + 2]
            if r < 0.02 && g < 0.02 && b < 0.02 { black += 1 }
            if r > 0.98 && g > 0.98 && b > 0.98 { white += 1 }
        }
        let nf = Float(n)
        return (Float(black) / nf, Float(white) / nf)
    }

    private static func meanSaturation(_ p: PixelBuffer) -> Float {
        var acc: Float = 0
        for i in 0 ..< p.pixelCount {
            let r = p.rgb[i * 3], g = p.rgb[i * 3 + 1], b = p.rgb[i * 3 + 2]
            let mx = max(r, max(g, b))
            let mn = min(r, min(g, b))
            acc += mx > 1e-6 ? (mx - mn) / mx : 0
        }
        return acc / Float(p.pixelCount)
    }

    private struct HueStats {
        var dominant: String
        var dominantFraction: Float
        var weights: [String: Float]
        var topChromatic: [String]
    }

    private static func hueStatistics(_ p: PixelBuffer) -> HueStats {
        // Prefer **chromatic** pixels (subject glazes) over brown wood / gray walls.
        // Center crop weighted higher for product-style subjects.
        var counts = [String: Float]()
        var totalWeight: Float = 0
        let cx = Float(p.width) / 2
        let cy = Float(p.height) / 2
        let centerR = 0.35 * Float(min(p.width, p.height))

        for i in 0 ..< p.pixelCount {
            let x = Float(i % p.width)
            let y = Float(i / p.width)
            let dist = sqrt(pow(x - cx, 2) + pow(y - cy, 2))
            let centerW: Float = dist < centerR ? 2.5 : 1.0

            let r = p.rgb[i * 3], g = p.rgb[i * 3 + 1], b = p.rgb[i * 3 + 2]
            let (h, s, v) = rgbToHSV(r, g, b)
            if s < 0.18 || v < 0.12 {
                counts["neutral", default: 0] += 0.2 * centerW
                totalWeight += 0.2 * centerW
                continue
            }
            let name: String
            if h < 15 || h >= 345 { name = "red" }
            else if h < 45 { name = "orange" }
            else if h < 70 { name = "yellow" }
            else if h < 160 { name = "green" }
            else if h < 200 { name = "cyan" }
            else if h < 260 { name = "blue" }
            else if h < 290 { name = "purple" }
            else { name = "magenta" }
            let chromaW = 1.0 + 3.0 * s
            counts[name, default: 0] += centerW * chromaW
            totalWeight += centerW * chromaW
        }

        var weights: [String: Float] = [:]
        if totalWeight > 0 {
            for (k, v) in counts { weights[k] = v / totalWeight }
        }
        let best = weights.max(by: { $0.value < $1.value })
        let chromatic = weights
            .filter { $0.key != "neutral" }
            .sorted { $0.value > $1.value }
            .map(\.key)

        return HueStats(
            dominant: best?.key ?? "neutral",
            dominantFraction: best?.value ?? 0,
            weights: weights,
            topChromatic: chromatic
        )
    }

    private static func rgbToHSV(_ r: Float, _ g: Float, _ b: Float) -> (Float, Float, Float) {
        let mx = max(r, max(g, b))
        let mn = min(r, min(g, b))
        let d = mx - mn
        let v = mx
        let s = mx > 1e-6 ? d / mx : 0
        var h: Float = 0
        if d > 1e-6 {
            if mx == r { h = 60 * (((g - b) / d).truncatingRemainder(dividingBy: 6)) }
            else if mx == g { h = 60 * (((b - r) / d) + 2) }
            else { h = 60 * (((r - g) / d) + 4) }
            if h < 0 { h += 360 }
        }
        return (h, s, v)
    }

    private static func noiseProxy(_ lum: [Float], width: Int, height: Int) -> Float {
        // Residual vs 3×3 box blur on a downsampled grid for speed.
        let step = max(1, min(width, height) / 128)
        var acc: Float = 0
        var count: Float = 0
        for y in stride(from: 1, to: height - 1, by: step) {
            for x in stride(from: 1, to: width - 1, by: step) {
                var s: Float = 0
                for dy in -1 ... 1 {
                    for dx in -1 ... 1 {
                        s += lum[(y + dy) * width + (x + dx)]
                    }
                }
                let blur = s / 9
                let res = lum[y * width + x] - blur
                acc += res * res
                count += 1
            }
        }
        return count > 0 ? sqrt(acc / count) : 0
    }

    private static func luminanceEntropy(_ lum: [Float]) -> Float {
        let bins = 64
        var hist = [Float](repeating: 0, count: bins)
        for v in lum {
            let b = min(bins - 1, max(0, Int(v * Float(bins))))
            hist[b] += 1
        }
        let n = Float(lum.count)
        var h: Float = 0
        for c in hist where c > 0 {
            let p = c / n
            h -= p * log2(p)
        }
        return h
    }

    /// Midline / third-line luminance jumps vs global gradient — catches hard VAE tile seams.
    /// Score ≈ max(line discontinuity) / meanGradient (unitless). Clean images usually < 1.5–2.
    private static func tileSeamDiscontinuities(
        _ lum: [Float], width: Int, height: Int, meanGradient: Float
    ) -> (vertical: Float, horizontal: Float, score: Float) {
        guard width >= 32, height >= 32 else {
            return (0, 0, 0)
        }
        let norm = max(meanGradient, 1e-4)
        // Candidate vertical seams (columns): halves and thirds — matches 2×2 / multi-tile layouts.
        let vCols = [width / 3, width / 2, (2 * width) / 3]
        var bestV: Float = 0
        for xc in vCols {
            guard xc > 0, xc < width - 1 else { continue }
            var acc: Float = 0
            var c: Float = 0
            for y in 0 ..< height {
                let i = y * width + xc
                acc += abs(lum[i] - lum[i - 1]) * 255
                c += 1
            }
            if c > 0 { bestV = max(bestV, (acc / c) / norm) }
        }
        let hRows = [height / 3, height / 2, (2 * height) / 3]
        var bestH: Float = 0
        for yr in hRows {
            guard yr > 0, yr < height - 1 else { continue }
            var acc: Float = 0
            var c: Float = 0
            for x in 0 ..< width {
                let i = yr * width + x
                acc += abs(lum[i] - lum[i - width]) * 255
                c += 1
            }
            if c > 0 { bestH = max(bestH, (acc / c) / norm) }
        }
        return (bestV, bestH, max(bestV, bestH))
    }

    private static func compositeScore(
        sharp: Float,
        contrast: Float,
        clipB: Float,
        clipW: Float,
        noise: Float,
        entropy: Float,
        tileSeam: Float
    ) -> Float {
        // Map features into 0…1 then weight.
        // Sharpness on 0–255 Laplacian variance scale (soft bokeh products often ~50–200).
        let sharpS = smoothstep(25, 200, sharp)
        let contrastS = smoothstep(0.05, 0.22, contrast)
        let clipPenalty = min(1, (clipB + clipW) * 8)
        let noiseS = 1 - smoothstep(0.01, 0.06, noise) // lower residual better for photoreal
        let entropyS = smoothstep(3.5, 5.5, entropy)
        // Mild penalty when midlines look like tile boundaries (cosine blend should keep this low).
        let seamPenalty = smoothstep(2.0, 4.5, tileSeam)

        // A perfectly flat, unclipped, non-tiled field zeros sharp/contrast/entropy.
        // Remaining mass is 0.14+0.18+0.08. In Float32 that left-fold is 1 ULP
        // below 0.4, so the score is Float(40).nextDown (39.999996) — never 40.0.
        // Do not set a solid-color fixture floor at 40.
        var s = 100 * (
            0.28 * sharpS
                + 0.18 * contrastS
                + 0.14 * noiseS
                + 0.14 * entropyS
                + 0.18 * (1 - clipPenalty)
                + 0.08 * (1 - seamPenalty)
        )
        s = max(0, min(100, s))
        return s
    }

    private static func smoothstep(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
        let t = max(0, min(1, (x - edge0) / (edge1 - edge0)))
        return t * t * (3 - 2 * t)
    }
}
