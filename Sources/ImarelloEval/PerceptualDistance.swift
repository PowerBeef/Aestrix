import Foundation

/// LPIPS-style **always-on** perceptual distance (P2) without neural weights.
///
/// Combines multi-scale SSIM, multi-scale RGB L2, and gradient-map correlation.
/// Lower distance = more similar. Complements classic SSIM for I2I fidelity.
public enum PerceptualDistance {
    public struct Metrics: Sendable, Codable, Equatable {
        /// 0…~1+ perceptual distance (lower = more similar). ~0 identical, ~0.5 large change.
        public var distance: Float
        /// 0…100 score (higher = more similar): `100 * (1 - clamp(distance, 0, 1))`.
        public var score: Float
        public var msSSIM: Float
        public var multiScaleL2: Float
        public var gradientCorrelation: Float
        public var notes: [String]
    }

    public static func compare(generated: PixelBuffer, reference: PixelBuffer) -> Metrics {
        let g = resizeToMatch(generated, width: reference.width, height: reference.height)
        let r = reference

        let ms = multiScaleSSIM(g, r)
        let l2 = multiScaleL2(g, r)
        let gc = gradientCorrelation(g, r)

        // Distance blend (inspired by LPIPS: deep features → here multi-scale structure).
        let dist = max(0, 0.50 * (1 - ms) + 0.30 * min(1, l2 / 0.35) + 0.20 * (1 - max(0, gc)))
        let score = 100 * (1 - min(1, dist))
        return Metrics(
            distance: dist,
            score: score,
            msSSIM: ms,
            multiScaleL2: l2,
            gradientCorrelation: gc,
            notes: [
                "LPIPS-lite (MS-SSIM + multi-scale L2 + gradient corr); not AlexNet LPIPS.",
            ]
        )
    }

    // MARK: - Multi-scale SSIM

    private static func multiScaleSSIM(_ a: PixelBuffer, _ b: PixelBuffer) -> Float {
        let weights: [Float] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        var scaleA = a
        var scaleB = b
        var acc: Float = 0
        var wSum: Float = 0
        let levels = min(5, weights.count)
        for li in 0 ..< levels {
            let s = windowedSSIM(scaleA, scaleB)
            let w = weights[li]
            acc += w * s
            wSum += w
            if li + 1 < levels {
                scaleA = downsample2(scaleA)
                scaleB = downsample2(scaleB)
                if scaleA.width < 16 || scaleA.height < 16 { break }
            }
        }
        return wSum > 0 ? acc / wSum : 0
    }

    private static func windowedSSIM(_ a: PixelBuffer, _ b: PixelBuffer) -> Float {
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
                acc += den > 1e-12 ? num / den : 1
                count += 1
                x += win
            }
            y += win
        }
        return count > 0 ? acc / count : 0
    }

    // MARK: - Multi-scale L2

    private static func multiScaleL2(_ a: PixelBuffer, _ b: PixelBuffer) -> Float {
        var scaleA = a
        var scaleB = b
        var acc: Float = 0
        var levels = 0
        for _ in 0 ..< 4 {
            let n = Float(scaleA.rgb.count)
            var se: Float = 0
            for i in 0 ..< scaleA.rgb.count {
                let d = scaleA.rgb[i] - scaleB.rgb[i]
                se += d * d
            }
            acc += sqrt(se / n)
            levels += 1
            if scaleA.width < 16 { break }
            scaleA = downsample2(scaleA)
            scaleB = downsample2(scaleB)
        }
        return levels > 0 ? acc / Float(levels) : 0
    }

    // MARK: - Gradient correlation

    private static func gradientCorrelation(_ a: PixelBuffer, _ b: PixelBuffer) -> Float {
        let la = a.luminances()
        let lb = b.luminances()
        let w = a.width
        let h = a.height
        var magA: [Float] = []
        var magB: [Float] = []
        magA.reserveCapacity((w - 1) * (h - 1))
        magB.reserveCapacity((w - 1) * (h - 1))
        for y in 0 ..< (h - 1) {
            for x in 0 ..< (w - 1) {
                let i = y * w + x
                let ax = la[i + 1] - la[i]
                let ay = la[i + w] - la[i]
                let bx = lb[i + 1] - lb[i]
                let by = lb[i + w] - lb[i]
                magA.append(sqrt(ax * ax + ay * ay))
                magB.append(sqrt(bx * bx + by * by))
            }
        }
        // Flat/solid images have ~zero gradients: treat as perfect correlation if both flat.
        let meanA = magA.reduce(0, +) / Float(max(1, magA.count))
        let meanB = magB.reduce(0, +) / Float(max(1, magB.count))
        if meanA < 1e-6 && meanB < 1e-6 { return 1 }
        return pearson(magA, magB)
    }

    private static func pearson(_ a: [Float], _ b: [Float]) -> Float {
        let n = Float(a.count)
        guard n > 2 else { return 0 }
        let ma = a.reduce(0, +) / n
        let mb = b.reduce(0, +) / n
        var num: Float = 0, da: Float = 0, db: Float = 0
        for i in 0 ..< a.count {
            let x = a[i] - ma
            let y = b[i] - mb
            num += x * y
            da += x * x
            db += y * y
        }
        let d = sqrt(da * db)
        // Zero variance (constant gradients): identical constants → 1, else 0.
        if d <= 1e-12 { return abs(ma - mb) < 1e-6 ? 1 : 0 }
        return num / d
    }

    // MARK: - Resize helpers

    private static func downsample2(_ src: PixelBuffer) -> PixelBuffer {
        let w = max(1, src.width / 2)
        let h = max(1, src.height / 2)
        return resizeToMatch(src, width: w, height: h)
    }

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
}
