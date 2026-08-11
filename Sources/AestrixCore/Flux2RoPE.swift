import Foundation

/// 4-axis rotary position embeddings for FLUX.2 (T, H, W, L).
///
/// Matches mflux `Flux2PosEmbed`: for each axis dim, frequencies at even indices
/// `omega_i = theta^(-(2i/dim))`, then `cos(pos * omega)`, `sin(pos * omega)`.
/// Outputs are concatenated across axes.
public enum Flux2RoPE {
    public static let defaultTheta: Float = ModelConstants.ropeTheta
    public static let defaultAxesDims: [Int] = ModelConstants.ropeAxesDims

    /// Compute cos/sin for a batch of multi-axis position ids.
    /// - Parameter ids: shape conceptually `[..., numAxes]` as row-major floats
    /// - Returns: (cos, sin) each with last dim = sum(axesDims)/2 * 2 = sum(axesDims/2 * 1 for cos/sin pairs) = sum(axesDims)/2 for each of cos and sin? 
    ///   Actually mflux returns cos of shape `[..., sum(dim/2 for dim in axes)]` same for sin.
    public static func frequencies(
        ids: [[Float]],
        theta: Float = defaultTheta,
        axesDims: [Int] = defaultAxesDims
    ) -> (cos: [[Float]], sin: [[Float]]) {
        precondition(!ids.isEmpty)
        precondition(ids[0].count == axesDims.count, "ids last dim must match axes count")

        var allCos: [[Float]] = []
        var allSin: [[Float]] = []
        allCos.reserveCapacity(ids.count)
        allSin.reserveCapacity(ids.count)

        for pos in ids {
            var cosRow: [Float] = []
            var sinRow: [Float] = []
            for (axis, dim) in axesDims.enumerated() {
                let (c, s) = rope1D(dim: dim, pos: pos[axis], theta: theta)
                cosRow.append(contentsOf: c)
                sinRow.append(contentsOf: s)
            }
            allCos.append(cosRow)
            allSin.append(sinRow)
        }
        return (allCos, allSin)
    }

    /// Single-axis 1D RoPE frequencies for one position.
    public static func rope1D(dim: Int, pos: Float, theta: Float = defaultTheta) -> (cos: [Float], sin: [Float]) {
        precondition(dim % 2 == 0)
        let half = dim / 2
        var cosOut = [Float](repeating: 0, count: half)
        var sinOut = [Float](repeating: 0, count: half)
        for i in 0..<half {
            let scale = Float(2 * i) / Float(dim) // arange(0, dim, 2)/dim
            let omega = 1.0 / pow(theta, scale)
            let angle = pos * omega
            cosOut[i] = cos(angle)
            sinOut[i] = sin(angle)
        }
        return (cosOut, sinOut)
    }

    /// Apply complex rotation to interleaved pairs of a vector using cos/sin.
    /// `x` length must be 2 * cos.count (pairs).
    public static func applyRotary(x: [Float], cos: [Float], sin: [Float]) -> [Float] {
        precondition(x.count == cos.count * 2)
        precondition(cos.count == sin.count)
        var out = [Float](repeating: 0, count: x.count)
        for i in 0..<cos.count {
            let x0 = x[2 * i]
            let x1 = x[2 * i + 1]
            out[2 * i] = x0 * cos[i] - x1 * sin[i]
            out[2 * i + 1] = x0 * sin[i] + x1 * cos[i]
        }
        return out
    }

    /// Build flat img grid ids for packed latents: each row is `[t, h, w, layer]`.
    public static func prepareGridIDs(
        height: Int,
        width: Int,
        tCoord: Int = 0,
        layer: Int = 0
    ) -> [[Float]] {
        var ids: [[Float]] = []
        ids.reserveCapacity(height * width)
        for h in 0..<height {
            for w in 0..<width {
                ids.append([Float(tCoord), Float(h), Float(w), Float(layer)])
            }
        }
        return ids
    }

    /// Text token ids for FLUX.2: cartesian `(t=0, h=0, w=0, l=0..<length)`.
    /// Matches diffusers `_prepare_text_ids` — axis order is **T,H,W,L**.
    public static func prepareTextIDs(length: Int = ModelConstants.maxSequenceLength) -> [[Float]] {
        (0..<length).map { i in [0, 0, 0, Float(i)] }
    }
}
