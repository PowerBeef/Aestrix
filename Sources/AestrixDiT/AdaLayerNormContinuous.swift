import MLX
import MLXNN

/// Continuous AdaLN: SiLU(cond) → Linear → (scale, shift); x = LN(x)*(1+scale)+shift.
public final class AdaLayerNormContinuous: Module {
    let embeddingDim: Int
    @ModuleInfo(key: "linear") var linear: Linear
    let norm: LayerNorm

    public init(embeddingDim: Int, conditioningEmbeddingDim: Int) {
        self.embeddingDim = embeddingDim
        self._linear.wrappedValue = Linear(conditioningEmbeddingDim, embeddingDim * 2, bias: false)
        self.norm = LayerNorm(dimensions: embeddingDim, eps: 1e-6, affine: false, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, textEmbeddings: MLXArray) -> MLXArray {
        var te = linear(silu(textEmbeddings).asType(.float32))
        // mflux: scale = te[:, 0:dim], shift = te[:, dim:2dim]
        let scale = te[0..., 0..<embeddingDim]
        let shift = te[0..., embeddingDim..<(2 * embeddingDim)]
        return norm(x) * (1 + scale)[0..., .newAxis, 0...] + shift[0..., .newAxis, 0...]
    }
}
