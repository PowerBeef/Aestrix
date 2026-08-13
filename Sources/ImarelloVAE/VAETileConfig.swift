import Foundation

/// How overlapping VAE decode tiles are combined.
public enum VAETileBlend: String, Sendable, Codable, CaseIterable {
    /// Non-overlapping 2×2 (or N×N) hard stitch — lowest overhead, possible seams.
    case none
    /// Overlapping tiles with separable cosine ramps (partition of unity on 1D edges).
    case cosine
}

/// Low-RAM tiled VAE decode parameters (latent-space coordinates after unpatchify).
public struct VAETileConfig: Sendable, Equatable, Codable {
    /// When max(H, W) of unpatchified latents ≥ this, use tiling (default 96 → 768²+).
    public var enabledThreshold: Int
    /// Latent tile side length (default 64 → ~512 px RGB after 8× decode).
    public var tileSize: Int
    /// Latent overlap for `.cosine` blend (default 8 → ~64 px RGB). Ignored for `.none`.
    public var overlap: Int
    /// Blend mode.
    public var blend: VAETileBlend
    /// Grid for `.none` hard stitch (default 2 → 2×2).
    public var noneGrid: Int

    public init(
        enabledThreshold: Int = 96,
        /// Prefer ~2 tiles on 128² unpatchified latents (1024² image): 72+16 → starts [0,56].
        tileSize: Int = 72,
        overlap: Int = 16,
        blend: VAETileBlend = .cosine,
        noneGrid: Int = 2
    ) {
        self.enabledThreshold = enabledThreshold
        self.tileSize = tileSize
        self.overlap = overlap
        self.blend = blend
        self.noneGrid = noneGrid
    }

    public static let `default` = VAETileConfig()

    /// Legacy hard 2×2 without feather (pre–overlap path).
    public static let hard2x2 = VAETileConfig(
        enabledThreshold: 96,
        tileSize: 64,
        overlap: 0,
        blend: .none,
        noneGrid: 2
    )

    /// PDF-style 64 latent / 8 overlap (more tiles → slower decode; use for max seam quality).
    public static let fine64 = VAETileConfig(
        enabledThreshold: 96,
        tileSize: 64,
        overlap: 8,
        blend: .cosine,
        noneGrid: 2
    )

    public func shouldTile(height: Int, width: Int) -> Bool {
        height >= enabledThreshold || width >= enabledThreshold
    }
}

// MARK: - Pure math (unit-tested without MLX)

/// Tile layout and cosine blend windows in pure Swift for tests and MLX callers.
public enum VAETileMath {
    /// Origin indices along one spatial axis so tiles of `tileSize` cover `length` with `overlap`.
    public static func tileStarts(length: Int, tileSize: Int, overlap: Int) -> [Int] {
        precondition(length > 0 && tileSize > 0)
        precondition(overlap >= 0 && overlap < tileSize)
        if tileSize >= length { return [0] }
        let stride = max(1, tileSize - overlap)
        var starts: [Int] = []
        var s = 0
        while true {
            starts.append(s)
            if s + tileSize >= length { break }
            s += stride
            if s + tileSize > length {
                s = length - tileSize
            }
            if let last = starts.last, s <= last {
                // Degenerate: force end-aligned tile once.
                let end = length - tileSize
                if end != last { starts.append(end) }
                break
            }
        }
        // Dedupe consecutive duplicates.
        var unique: [Int] = []
        for x in starts {
            if unique.last != x { unique.append(x) }
        }
        return unique
    }

    /// Cosine fade-in: index 0 → ~0, index `overlap-1` → ~1. Empty if overlap ≤ 0.
    public static func cosineFadeIn(overlap: Int) -> [Float] {
        guard overlap > 0 else { return [] }
        if overlap == 1 { return [1] }
        return (0 ..< overlap).map { i in
            let t = Float(i) / Float(overlap - 1)
            return 0.5 * (1 - cos(Float.pi * t))
        }
    }

    /// 1D weights of length `tileLen` for a tile that may touch canvas edges.
    /// Outer edges that touch the canvas keep weight 1 (no fade-to-zero off-image).
    public static func axisWeights(
        tileLen: Int,
        fadeStart: Int,
        fadeEnd: Int
    ) -> [Float] {
        precondition(tileLen > 0)
        var w = [Float](repeating: 1, count: tileLen)
        let fadeIn = cosineFadeIn(overlap: fadeStart)
        for i in 0 ..< min(fadeIn.count, tileLen) {
            w[i] *= fadeIn[i]
        }
        let fadeOut = cosineFadeIn(overlap: fadeEnd)
        for i in 0 ..< min(fadeOut.count, tileLen) {
            w[tileLen - 1 - i] *= fadeOut[i]
        }
        return w
    }

    /// Separable 2D weight mask `H×W` (row-major) for one latent tile.
    public static func weightMask(
        tileH: Int,
        tileW: Int,
        fadeTop: Int,
        fadeBottom: Int,
        fadeLeft: Int,
        fadeRight: Int
    ) -> [Float] {
        let wy = axisWeights(tileLen: tileH, fadeStart: fadeTop, fadeEnd: fadeBottom)
        let wx = axisWeights(tileLen: tileW, fadeStart: fadeLeft, fadeEnd: fadeRight)
        var out = [Float](repeating: 0, count: tileH * tileW)
        var idx = 0
        for y in 0 ..< tileH {
            let yv = wy[y]
            for x in 0 ..< tileW {
                out[idx] = yv * wx[x]
                idx += 1
            }
        }
        return out
    }

    /// Upsample a latent-res mask to RGB res by integer nearest (repeat).
    public static func upsampleNearest(
        mask: [Float],
        height: Int,
        width: Int,
        scaleY: Int,
        scaleX: Int
    ) -> [Float] {
        precondition(mask.count == height * width)
        precondition(scaleY > 0 && scaleX > 0)
        let outH = height * scaleY
        let outW = width * scaleX
        var out = [Float](repeating: 0, count: outH * outW)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let v = mask[y * width + x]
                let y0 = y * scaleY
                let x0 = x * scaleX
                for dy in 0 ..< scaleY {
                    let row = (y0 + dy) * outW
                    for dx in 0 ..< scaleX {
                        out[row + x0 + dx] = v
                    }
                }
            }
        }
        return out
    }

    /// Overlap fade amount on the leading edge of a tile (not touching canvas start).
    public static func leadingFade(origin: Int, overlap: Int) -> Int {
        origin == 0 ? 0 : overlap
    }

    /// Overlap fade amount on the trailing edge (not covering canvas end).
    public static func trailingFade(origin: Int, tileLen: Int, fullLen: Int, overlap: Int) -> Int {
        origin + tileLen >= fullLen ? 0 : overlap
    }
}
