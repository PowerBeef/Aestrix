import Foundation
import MLX
import ImarelloCore

/// 4-axis RoPE producing (cos, sin) for concatenated axes — mflux `Flux2PosEmbed`.
public final class Flux2PosEmbed {
    public let theta: Float
    public let axesDim: [Int]
    /// Cached `omega` per axis dim (invariant for a model instance).
    private let omegas: [MLXArray]

    public init(theta: Float = ModelConstants.ropeTheta, axesDim: [Int] = ModelConstants.ropeAxesDims) {
        self.theta = theta
        self.axesDim = axesDim
        self.omegas = axesDim.map { dim in
            let half = dim / 2
            let indices = MLXArray(0..<half).asType(.float32) * 2.0 / Float(dim)
            let omega = 1.0 / pow(MLXArray(theta), indices)
            eval(omega)
            return omega
        }
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
        cosOut.reserveCapacity(axesDim.count)
        sinOut.reserveCapacity(axesDim.count)
        for i in axesDim.indices {
            let (c, s) = rope1D(omega: omegas[i], pos: pos[.ellipsis, i])
            cosOut.append(c)
            sinOut.append(s)
        }
        return (concatenated(cosOut, axis: -1), concatenated(sinOut, axis: -1))
    }

    private func rope1D(omega: MLXArray, pos: MLXArray) -> (MLXArray, MLXArray) {
        let out = pos.expandedDimensions(axis: -1) * omega.expandedDimensions(axis: 0)
        return (cos(out), sin(out))
    }
}
