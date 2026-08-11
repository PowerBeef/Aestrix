import Foundation
import MLX
import MLXNN
import MLXFast

enum AttentionUtils {
    static func processQKV(
        hiddenStates: MLXArray,
        toQ: Linear,
        toK: Linear,
        toV: Linear,
        normQ: RMSNorm,
        normK: RMSNorm,
        numHeads: Int,
        headDim: Int
    ) -> (MLXArray, MLXArray, MLXArray) {
        let batch = hiddenStates.dim(0)
        let seq = hiddenStates.dim(1)

        var query = toQ(hiddenStates)
        var key = toK(hiddenStates)
        var value = toV(hiddenStates)

        query = query.reshaped([batch, seq, numHeads, headDim]).transposed(0, 2, 1, 3)
        key = key.reshaped([batch, seq, numHeads, headDim]).transposed(0, 2, 1, 3)
        value = value.reshaped([batch, seq, numHeads, headDim]).transposed(0, 2, 1, 3)

        let qDtype = query.dtype
        let kDtype = key.dtype
        query = normQ(query.asType(.float32)).asType(qDtype)
        key = normK(key.asType(.float32)).asType(kDtype)
        return (query, key, value)
    }

    static func computeAttention(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        batchSize: Int,
        numHeads: Int,
        headDim: Int
    ) -> MLXArray {
        let scale = 1.0 / Foundation.sqrt(Float(query.dim(-1)))
        var hidden = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: scale,
            mask: nil
        )
        hidden = hidden.transposed(0, 2, 1, 3)
        return hidden.reshaped([batchSize, -1, numHeads * headDim])
    }

    /// Apply RoPE with cos/sin of shape [S, D/2] to Q/K of shape [B, H, S, D] (interleaved pairs).
    static func applyRopeBSHD(
        query: MLXArray,
        key: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        let outDtype = query.dtype
        let cosB = cos.reshaped([1, 1, cos.dim(0), cos.dim(1)])
        let sinB = sin.reshaped([1, 1, sin.dim(0), sin.dim(1)])

        func mix(_ x: MLXArray) -> MLXArray {
            let xf = x.asType(.float32)
            let shape = xf.shape
            let x2 = xf.reshaped(Array(shape.dropLast()) + [-1, 2])
            let real = x2[.ellipsis, 0]
            let imag = x2[.ellipsis, 1]
            let out0 = real * cosB + (-imag) * sinB
            let out1 = imag * cosB + real * sinB
            let stacked = stacked([out0, out1], axis: -1)
            return stacked.reshaped(shape)
        }

        return (mix(query).asType(outDtype), mix(key).asType(outDtype))
    }
}
