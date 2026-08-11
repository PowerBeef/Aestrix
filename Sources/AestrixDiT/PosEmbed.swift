import Foundation
import MLX
import AestrixCore

/// 4-axis RoPE producing (cos, sin) for concatenated axes — mflux `Flux2PosEmbed`.
public final class Flux2PosEmbed {
    public let theta: Float
    public let axesDim: [Int]

    public init(theta: Float = ModelConstants.ropeTheta, axesDim: [Int] = ModelConstants.ropeAxesDims) {
        self.theta = theta
        self.axesDim = axesDim
    }

    /// - Parameter ids: [S, 4] or [B, S, 4] float/int positions
    /// - Returns: (cos, sin) each [S, sum(axisDim/2)]
    public func callAsFunction(_ ids: MLXArray) -> (MLXArray, MLXArray) {
        var pos = ids
        if pos.ndim == 3 {
            pos = pos[0]
        }
        pos = pos.asType(.float32)
        var cosOut: [MLXArray] = []
        var sinOut: [MLXArray] = []
        for (i, dim) in axesDim.enumerated() {
            let (c, s) = rope1D(dim: dim, pos: pos[.ellipsis, i])
            cosOut.append(c)
            sinOut.append(s)
        }
        return (concatenated(cosOut, axis: -1), concatenated(sinOut, axis: -1))
    }

    private func rope1D(dim: Int, pos: MLXArray) -> (MLXArray, MLXArray) {
        // scale = arange(0, dim, 2) / dim
        let half = dim / 2
        let indices = MLXArray(0..<half).asType(.float32) * 2.0 / Float(dim)
        let omega = 1.0 / pow(MLXArray(theta), indices)
        let out = pos.expandedDimensions(axis: -1) * omega.expandedDimensions(axis: 0)
        return (cos(out), sin(out))
    }
}
