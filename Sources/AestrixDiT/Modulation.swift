import MLX
import MLXNN

/// SiLU → Linear(dim → dim*3*sets) → split into (shift, scale, gate) per set.
public final class Flux2Modulation: Module {
    let modParamSets: Int
    @ModuleInfo(key: "linear") var linear: Linear

    public init(dim: Int, modParamSets: Int = 2) {
        self.modParamSets = modParamSets
        self._linear.wrappedValue = Linear(dim, dim * 3 * modParamSets, bias: false)
        super.init()
    }

    /// Returns `modParamSets` triples of (shift, scale, gate).
    public func callAsFunction(_ temb: MLXArray) -> [(MLXArray, MLXArray, MLXArray)] {
        var mod = silu(temb)
        mod = linear(mod)
        if mod.ndim == 2 {
            mod = mod.expandedDimensions(axis: 1)
        }
        let parts = split(mod, parts: 3 * modParamSets, axis: -1)
        var result: [(MLXArray, MLXArray, MLXArray)] = []
        result.reserveCapacity(modParamSets)
        for i in 0..<modParamSets {
            let base = 3 * i
            result.append((parts[base], parts[base + 1], parts[base + 2]))
        }
        return result
    }
}
