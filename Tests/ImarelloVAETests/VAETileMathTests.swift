import Testing
import Foundation
@testable import ImarelloVAE

@Suite("VAE tile math")
struct VAETileMathTests {

    @Test("tileStarts cover full length with end-aligned last tile")
    func tileStartsCover() {
        let starts = VAETileMath.tileStarts(length: 128, tileSize: 64, overlap: 8)
        #expect(starts.first == 0)
        #expect(starts.last == 128 - 64)
        // Every pixel covered by at least one tile
        var covered = [Bool](repeating: false, count: 128)
        for s in starts {
            for i in s ..< (s + 64) {
                covered[i] = true
            }
        }
        #expect(covered.allSatisfy { $0 })
    }

    @Test("tileStarts single tile when length <= tileSize")
    func tileStartsSingle() {
        #expect(VAETileMath.tileStarts(length: 48, tileSize: 64, overlap: 8) == [0])
    }

    @Test("cosine fade-in goes 0→1")
    func cosineFadeIn() {
        let f = VAETileMath.cosineFadeIn(overlap: 5)
        #expect(f.count == 5)
        #expect(abs(f[0] - 0) < 1e-5)
        #expect(abs(f[4] - 1) < 1e-5)
        #expect(f[2] > f[1] && f[3] > f[2])
    }

    @Test("neighboring 1D weights form partition of unity in overlap")
    func partitionOfUnity1D() {
        let length = 32
        let tile = 20
        let overlap = 8
        let starts = VAETileMath.tileStarts(length: length, tileSize: tile, overlap: overlap)
        #expect(starts.count >= 2)

        var accum = [Float](repeating: 0, count: length)
        for s in starts {
            let th = min(tile, length - s)
            let fadeStart = VAETileMath.leadingFade(origin: s, overlap: overlap)
            let fadeEnd = VAETileMath.trailingFade(
                origin: s, tileLen: th, fullLen: length, overlap: overlap)
            let w = VAETileMath.axisWeights(tileLen: th, fadeStart: fadeStart, fadeEnd: fadeEnd)
            for i in 0 ..< th {
                accum[s + i] += w[i]
            }
        }
        for (i, a) in accum.enumerated() {
            #expect(abs(a - 1) < 1e-4, "axis weight sum at \(i) = \(a)")
        }
    }

    @Test("2D separable weights sum ~1 on a small canvas")
    func partitionOfUnity2D() {
        let H = 32
        let W = 32
        let tile = 20
        let overlap = 8
        let yStarts = VAETileMath.tileStarts(length: H, tileSize: tile, overlap: overlap)
        let xStarts = VAETileMath.tileStarts(length: W, tileSize: tile, overlap: overlap)
        var accum = [Float](repeating: 0, count: H * W)
        for y0 in yStarts {
            for x0 in xStarts {
                let th = min(tile, H - y0)
                let tw = min(tile, W - x0)
                let mask = VAETileMath.weightMask(
                    tileH: th, tileW: tw,
                    fadeTop: VAETileMath.leadingFade(origin: y0, overlap: overlap),
                    fadeBottom: VAETileMath.trailingFade(
                        origin: y0, tileLen: th, fullLen: H, overlap: overlap),
                    fadeLeft: VAETileMath.leadingFade(origin: x0, overlap: overlap),
                    fadeRight: VAETileMath.trailingFade(
                        origin: x0, tileLen: tw, fullLen: W, overlap: overlap)
                )
                for dy in 0 ..< th {
                    for dx in 0 ..< tw {
                        accum[(y0 + dy) * W + (x0 + dx)] += mask[dy * tw + dx]
                    }
                }
            }
        }
        for (i, a) in accum.enumerated() {
            #expect(abs(a - 1) < 1e-3, "2D weight sum at \(i) = \(a)")
        }
    }

    @Test("upsampleNearest repeats each cell by scale")
    func upsample() {
        let mask: [Float] = [1, 2, 3, 4]
        let up = VAETileMath.upsampleNearest(mask: mask, height: 2, width: 2, scaleY: 2, scaleX: 2)
        #expect(up.count == 16)
        #expect(up[0] == 1 && up[1] == 1 && up[2] == 2 && up[3] == 2)
        #expect(up[4] == 1 && up[8] == 3 && up[15] == 4)
    }

    @Test("default config enables tiling at 1024-class latents")
    func defaultThreshold() {
        let c = VAETileConfig.default
        #expect(c.blend == .cosine)
        #expect(c.shouldTile(height: 128, width: 128))
        #expect(!c.shouldTile(height: 64, width: 64))
        // ~2 tiles per axis on 128² (not 3×3) so decode stays near hard-2×2 cost
        let starts = VAETileMath.tileStarts(
            length: 128, tileSize: c.tileSize, overlap: c.overlap)
        #expect(starts.count == 2)
    }
}
