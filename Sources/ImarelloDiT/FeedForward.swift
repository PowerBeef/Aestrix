import MLX
import MLXNN

/// SwiGLU: split last dim, silu(x1) * x2.
public final class Flux2SwiGLU: Module, UnaryLayer {
    public override init() {
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(x, parts: 2, axis: -1)
        return silu(parts[0]) * parts[1]
    }
}

/// Double-stream FFN: Linear(dim → 2*inner) → SwiGLU → Linear(inner → dim).
public final class Flux2FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    let act = Flux2SwiGLU()
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    public init(dim: Int, mult: Float = 3.0) {
        let inner = Int(Float(dim) * mult)
        self._linearIn.wrappedValue = Linear(dim, inner * 2, bias: false)
        self._linearOut.wrappedValue = Linear(inner, dim, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        linearOut(act(linearIn(x)))
    }
}
