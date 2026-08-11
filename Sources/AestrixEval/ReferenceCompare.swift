import Foundation

/// Full-reference comparison between generated image and a reference (I2I source or gold).
public enum ReferenceCompare {
    public struct Metrics: Sendable, Codable, Equatable {
        public var ssim: Float
        public var mse: Float
        public var psnr: Float
        public var meanAbsError: Float
        /// L2 distance between mean RGB of gen vs ref (0…√3).
        public var meanColorDistance: Float
        /// Correlation of 24-bin RGB histograms (−1…1).
        public var histogramCorrelation: Float
        /// Approximate ΔE on mean colors (LAB-ish).
        public var meanDeltaE: Float
        /// 0…100 composite fidelity (higher = more like reference).
        public var fidelityScore: Float
    }

    public static func compare(generated: PixelBuffer, reference: PixelBuffer) -> Metrics {
        let g = resizeToMatch(generated, width: reference.width, height: reference.height)
        let r = reference
        let ssim = computeSSIM(g, r)
        let (mse, mae) = mseMae(g, r)
        let psnr: Float = mse > 1e-12 ? 10 * log10(1.0 / mse) : 99
        let mg = g.meanRGB()
        let mr = r.meanRGB()
        let mcd = sqrt(
            pow(mg.r - mr.r, 2) + pow(mg.g - mr.g, 2) + pow(mg.b - mr.b, 2)
        )
        let hist = histogramCorrelation(g, r)
        let de = deltaEApprox(mg, mr)

        // Fidelity: SSIM dominant; color distance penalizes global recolor (desired for identity, not for color edits).
        let ssimS = max(0, min(1, (ssim + 1) / 2)) // map theoretically [-1,1] → [0,1]; SSIM usually [0,1]
        let colorS = 1 - min(1, mcd / 0.6)
        let histS = max(0, min(1, (hist + 1) / 2))
        let score = 100 * (0.55 * ssimS + 0.25 * colorS + 0.20 * histS)

        return Metrics(
            ssim: ssim,
            mse: mse,
            psnr: psnr,
            meanAbsError: mae,
            meanColorDistance: mcd,
            histogramCorrelation: hist,
            meanDeltaE: de,
            fidelityScore: score
        )
    }

    // MARK: - Resize (nearest for speed / determinism)

    private static func resizeToMatch(_ src: PixelBuffer, width: Int, height: Int) -> PixelBuffer {
        if src.width == width && src.height == height { return src }
        var rgb = [Float](repeating: 0, count: width * height * 3)
        for y in 0 ..< height {
            let sy = min(src.height - 1, y * src.height / height)
            for x in 0 ..< width {
                let sx = min(src.width - 1, x * src.width / width)
                let si = (sy * src.width + sx) * 3
                let di = (y * width + x) * 3
                rgb[di] = src.rgb[si]
                rgb[di + 1] = src.rgb[si + 1]
                rgb[di + 2] = src.rgb[si + 2]
            }
        }
        return PixelBuffer(width: width, height: height, rgb: rgb)
    }

    private static func mseMae(_ a: PixelBuffer, _ b: PixelBuffer) -> (Float, Float) {
        var se: Float = 0
        var ae: Float = 0
        let n = Float(a.rgb.count)
        for i in 0 ..< a.rgb.count {
            let d = a.rgb[i] - b.rgb[i]
            se += d * d
            ae += abs(d)
        }
        return (se / n, ae / n)
    }

    /// Windowed SSIM on luminance (8×8 windows, uniform average).
    private static func computeSSIM(_ a: PixelBuffer, _ b: PixelBuffer) -> Float {
        let la = a.luminances()
        let lb = b.luminances()
        let w = a.width
        let h = a.height
        let win = 8
        let c1: Float = 0.01 * 0.01
        let c2: Float = 0.03 * 0.03
        var acc: Float = 0
        var count: Float = 0
        var y = 0
        while y + win <= h {
            var x = 0
            while x + win <= w {
                var meanA: Float = 0, meanB: Float = 0
                let n = Float(win * win)
                for dy in 0 ..< win {
                    for dx in 0 ..< win {
                        let i = (y + dy) * w + (x + dx)
                        meanA += la[i]
                        meanB += lb[i]
                    }
                }
                meanA /= n
                meanB /= n
                var varA: Float = 0, varB: Float = 0, cov: Float = 0
                for dy in 0 ..< win {
                    for dx in 0 ..< win {
                        let i = (y + dy) * w + (x + dx)
                        let da = la[i] - meanA
                        let db = lb[i] - meanB
                        varA += da * da
                        varB += db * db
                        cov += da * db
                    }
                }
                varA /= n
                varB /= n
                cov /= n
                let num = (2 * meanA * meanB + c1) * (2 * cov + c2)
                let den = (meanA * meanA + meanB * meanB + c1) * (varA + varB + c2)
                acc += num / max(den, 1e-12)
                count += 1
                x += win
            }
            y += win
        }
        return count > 0 ? acc / count : 0
    }

    private static func histogramCorrelation(_ a: PixelBuffer, _ b: PixelBuffer) -> Float {
        let bins = 8 // per channel → 8^3 = 512, use 8-bin per-channel 1D concat (24)
        func hist(_ p: PixelBuffer) -> [Float] {
            var h = [Float](repeating: 0, count: bins * 3)
            for i in 0 ..< p.pixelCount {
                for c in 0 ..< 3 {
                    let v = p.rgb[i * 3 + c]
                    let b = min(bins - 1, Int(v * Float(bins)))
                    h[c * bins + b] += 1
                }
            }
            let n = Float(p.pixelCount)
            return h.map { $0 / n }
        }
        let ha = hist(a)
        let hb = hist(b)
        let ma = ha.reduce(0, +) / Float(ha.count)
        let mb = hb.reduce(0, +) / Float(hb.count)
        var num: Float = 0, da: Float = 0, db: Float = 0
        for i in 0 ..< ha.count {
            let xa = ha[i] - ma
            let xb = hb[i] - mb
            num += xa * xb
            da += xa * xa
            db += xb * xb
        }
        let den = sqrt(da * db)
        return den > 1e-12 ? num / den : 0
    }

    private static func deltaEApprox(
        _ a: (r: Float, g: Float, b: Float),
        _ b: (r: Float, g: Float, b: Float)
    ) -> Float {
        // Cheap perceptual: weighted RGB Euclidean (not true CIEDE2000).
        let dr = a.r - b.r
        let dg = a.g - b.g
        let db = a.b - b.b
        return 100 * sqrt(0.3 * dr * dr + 0.59 * dg * dg + 0.11 * db * db)
    }
}
