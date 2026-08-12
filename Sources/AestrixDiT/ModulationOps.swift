import MLX

/// Compiled AdaLN elementwise helpers, shared across all 25 blocks (× steps).
///
/// Shapeless-safe: pure broadcasting elementwise chains with fixed ranks
/// (`scale`/`shift`/`gate` are `[1, 1, dim]`, activations `[B, S, dim]`).
/// Same op order as the previous inline expressions — parity preserved.
enum ModulationOps {
    /// `(1 + scale) * x + shift`
    static let modApply: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = {
        compile(shapeless: true) { (x: MLXArray, scale: MLXArray, shift: MLXArray) -> MLXArray in
            (1 + scale) * x + shift
        }
    }()

    /// `residual + gate * value`
    static let gateAdd: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = {
        compile(shapeless: true) { (residual: MLXArray, gate: MLXArray, value: MLXArray) -> MLXArray in
            residual + gate * value
        }
    }()
}
