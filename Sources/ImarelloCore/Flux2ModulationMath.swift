import Foundation

/// Pure-math helpers for AdaLN-style modulation parameter layout.
///
/// Weight path (Phase 2) uses `SiLU(temb) → Linear(dim → dim*3*sets)` then split into
/// (shift, scale, gate) per set. Here we only validate split geometry without weights.
public enum Flux2ModulationMath {
    /// Number of scalars produced by the modulation linear for given dim and set count.
    public static func outputFeatures(dim: Int, modParamSets: Int) -> Int {
        dim * 3 * modParamSets
    }

    /// Split a flat modulation vector into `modParamSets` triples of (shift, scale, gate),
    /// each of length `dim`, matching mflux `mx.split(..., 3 * sets)`.
    public static func split(
        mod: [Float],
        dim: Int,
        modParamSets: Int
    ) -> [(shift: [Float], scale: [Float], gate: [Float])] {
        let expected = outputFeatures(dim: dim, modParamSets: modParamSets)
        precondition(mod.count == expected, "mod count \(mod.count) != \(expected)")
        var out: [(shift: [Float], scale: [Float], gate: [Float])] = []
        out.reserveCapacity(modParamSets)
        for set in 0..<modParamSets {
            let base = set * 3 * dim
            let shift = Array(mod[base..<(base + dim)])
            let scale = Array(mod[(base + dim)..<(base + 2 * dim)])
            let gate = Array(mod[(base + 2 * dim)..<(base + 3 * dim)])
            out.append((shift, scale, gate))
        }
        return out
    }

    /// Apply modulation: `(1 + scale) * norm + shift`.
    public static func apply(norm: [Float], shift: [Float], scale: [Float]) -> [Float] {
        precondition(norm.count == shift.count && shift.count == scale.count)
        return zip(zip(norm, scale), shift).map { ns, sh in
            let (n, sc) = ns
            return (1 + sc) * n + sh
        }
    }

    /// Gated residual: `x + gate * y`.
    public static func gatedResidual(x: [Float], y: [Float], gate: [Float]) -> [Float] {
        precondition(x.count == y.count && y.count == gate.count)
        return zip(zip(x, y), gate).map { xy, g in
            let (xi, yi) = xy
            return xi + g * yi
        }
    }

    public static func silu(_ x: Float) -> Float {
        x / (1 + exp(-x))
    }

    public static func silu(_ x: [Float]) -> [Float] {
        x.map(silu)
    }
}
