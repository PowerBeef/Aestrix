import MLX
import MLXNN

/// SwiGLU: split last dim, silu(x1) * x2.
public final class Flux2SwiGLU: Module, UnaryLayer {
    public override init() {
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        Self.compiledSwiGLU(x)
    }

    /// Same ops as `silu(g) * up` (`g * sigmoid(g) * up`).
    /// Not shapeless: `split` cannot infer shapes under shapeless compile.
    /// Inlined sigmoid so we do not nest `compiledSilu` inside another compile.
    private static let compiledSwiGLU: @Sendable (MLXArray) -> MLXArray = {
        compile { (x: MLXArray) -> MLXArray in
            let parts = split(x, parts: 2, axis: -1)
            let g = parts[0]
            return (g * sigmoid(g)) * parts[1]
        }
    }()
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
        let hidden = AttentionUtils.linearChunkedSequence(linearIn, x)
        return AttentionUtils.linearChunkedSequence(linearOut, act(hidden))
    }
}
