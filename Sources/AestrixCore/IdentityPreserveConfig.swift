import Foundation

/// Tier-B I2I identity controls: reference latents, face-regional strength, clean-latent pull.
///
/// Defaults are **off** so classic strength I2I is unchanged. Use ``identityPreset`` for
/// the full Tier-B stack recommended for people/character consistency.
public struct IdentityPreserveConfig: Sendable, Equatable {
    /// Concatenate clean reference tokens into DiT (t-axis RoPE 10, 20, …).
    public var useReferenceLatents: Bool
    /// Detect faces (Vision) and apply regional noise + optional clean pull.
    public var facePreserve: Bool
    /// Face region start-σ scale relative to global start σ (0…1]. Lower → more face fidelity.
    public var faceStrengthScale: Float
    /// Max blend of clean latent into face region after each Euler step (0 = off).
    public var cleanPullAlpha: Float
    /// Decay clean-pull across steps (stronger early for structure lock).
    public var cleanPullDecay: Bool
    /// Strength → start-noise curve.
    public var scheduleCurve: StrengthScheduleCurve
    /// Spatially downsample reference tokens by this factor (1 = full ref, 2 = quarter
    /// tokens). Cuts identity-I2I joint sequence cost; identity fidelity trade-off —
    /// experimental, default off.
    public var refDownsample: Int

    public init(
        useReferenceLatents: Bool = false,
        facePreserve: Bool = false,
        faceStrengthScale: Float = 0.45,
        cleanPullAlpha: Float = 0,
        cleanPullDecay: Bool = true,
        scheduleCurve: StrengthScheduleCurve = .colorEdit,
        refDownsample: Int = 1
    ) {
        self.useReferenceLatents = useReferenceLatents
        self.facePreserve = facePreserve
        self.faceStrengthScale = min(1, max(0, faceStrengthScale))
        self.cleanPullAlpha = min(1, max(0, cleanPullAlpha))
        self.cleanPullDecay = cleanPullDecay
        self.scheduleCurve = scheduleCurve
        self.refDownsample = max(1, refDownsample)
    }

    /// Classic v1 path: strength-only, color curve, no face/ref.
    public static let disabled = IdentityPreserveConfig()

    /// Full Tier B preset for people / character consistency.
    ///
    /// Clean-pull is intentionally modest so wardrobe/scene edits can still land;
    /// reference latents + face regional σ do the heavy identity work. Prefer
    /// strength **≥ 0.85** with this preset for large scene changes.
    public static let identityPreset = IdentityPreserveConfig(
        useReferenceLatents: true,
        facePreserve: true,
        faceStrengthScale: 0.5,
        cleanPullAlpha: 0.2,
        cleanPullDecay: true,
        scheduleCurve: .identityPreserve
    )

    /// Any Tier-B feature active (beyond classic color strength schedule).
    public var isActive: Bool {
        useReferenceLatents
            || facePreserve
            || cleanPullAlpha > 0
            || scheduleCurve != .colorEdit
    }
}
