import Foundation

/// How I2I maps user `strength` ∈ (0,1] → starting noise fraction before μ-shift.
///
/// Higher start noise → more prompt freedom / color change; lower → more structure / identity.
public enum StrengthScheduleCurve: String, Sendable, CaseIterable {
    /// Default color-friendly curve: mid strengths still re-noise hard (`1 − (1−s)^1.5`).
    case colorEdit = "color"
    /// Milder start noise for identity-sensitive portrait edits (`1 − (1−s)^0.85`).
    case identityPreserve = "identity"
    /// Direct mapping: startT ≈ strength.
    case linear = "linear"

    /// Map strength → unshifted start time in (0, 1].
    public func startT(strength: Float) -> Float {
        let s = max(0.05, min(1.0, strength))
        switch self {
        case .colorEdit:
            // Concave-down: 0.65 → ~0.88, 0.8 → ~0.91
            return 1 - pow(1 - s, 1.5)
        case .identityPreserve:
            // Concave-up: less noise at mid/high strength (0.8 → ~0.74)
            return 1 - pow(1 - s, 0.85)
        case .linear:
            return s
        }
    }
}
