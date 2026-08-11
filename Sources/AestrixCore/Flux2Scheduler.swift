import Foundation

/// Flow-matching Euler discrete scheduler for FLUX.2 klein (requires_sigma_shift).
///
/// Primary path matches the **linear mu / exponential time-shift** used by diffusers-style
/// FLUX.2 (xocialize port goldens):
/// - base sigmas: `linspace(1.0, 1/numSteps, numSteps)`
/// - `mu = linear_shift(image_seq_len)` with base 0.5 @ 256 → max 1.15 @ 4096
/// - `sigma' = exp(mu) / (exp(mu) + (1/t - 1)^1)`
/// - append terminal 0; timesteps = pre-terminal sigmas * 1000
///
/// Euler step: `latents + (sigma_{t+1} - sigma_t) * model_output`
public struct Flux2Scheduler: Sendable {
    public let numInferenceSteps: Int
    public let imageSeqLen: Int
    public let numTrainTimesteps: Int
    public let sigmas: [Float]
    public let timesteps: [Float]

    public init(
        numInferenceSteps: Int = ModelConstants.defaultSteps,
        imageSeqLen: Int,
        numTrainTimesteps: Int = ModelConstants.numTrainTimesteps,
        baseSeqLen: Int = 256,
        maxSeqLen: Int = 4096,
        baseShift: Float = 0.5,
        maxShift: Float = 1.15
    ) {
        precondition(numInferenceSteps >= 1)
        precondition(imageSeqLen > 0)
        self.numInferenceSteps = numInferenceSteps
        self.imageSeqLen = imageSeqLen
        self.numTrainTimesteps = numTrainTimesteps

        let mu = Self.linearMu(
            imageSeqLen: imageSeqLen,
            baseSeqLen: baseSeqLen,
            maxSeqLen: maxSeqLen,
            baseShift: baseShift,
            maxShift: maxShift
        )
        let (ts, sigs) = Self.sigmasAndTimesteps(
            numInferenceSteps: numInferenceSteps,
            mu: mu,
            numTrainTimesteps: numTrainTimesteps
        )
        self.timesteps = ts
        self.sigmas = sigs
    }

    /// Packed latent sequence length for a pixel canvas (VAE scale 8 + 2×2 pack).
    public static func imageSeqLen(width: Int, height: Int, vaeScale: Int = ModelConstants.vaeScaleFactor) -> Int {
        let latentH = height / (vaeScale * 2) // == height/16
        let latentW = width / (vaeScale * 2)
        return max(1, latentH * latentW)
    }

    public static func linearMu(
        imageSeqLen: Int,
        baseSeqLen: Int = 256,
        maxSeqLen: Int = 4096,
        baseShift: Float = 0.5,
        maxShift: Float = 1.15
    ) -> Float {
        let m = (maxShift - baseShift) / Float(maxSeqLen - baseSeqLen)
        let b = baseShift - m * Float(baseSeqLen)
        return m * Float(imageSeqLen) + b
    }

    public static func timeShiftExponential(mu: Float, t: Float, sigmaPower: Float = 1.0) -> Float {
        // exp(mu) / (exp(mu) + (1/t - 1)^power)
        let expMu = exp(mu)
        let inv = pow(1.0 / t - 1.0, sigmaPower)
        return expMu / (expMu + inv)
    }

    public static func sigmasAndTimesteps(
        numInferenceSteps: Int,
        mu: Float,
        numTrainTimesteps: Int = ModelConstants.numTrainTimesteps
    ) -> (timesteps: [Float], sigmas: [Float]) {
        // linspace(1.0, 1/N, N)
        var base: [Float] = []
        base.reserveCapacity(numInferenceSteps)
        if numInferenceSteps == 1 {
            base = [1.0]
        } else {
            let start: Float = 1.0
            let end = 1.0 / Float(numInferenceSteps)
            for i in 0..<numInferenceSteps {
                let t = Float(i) / Float(numInferenceSteps - 1)
                base.append(start + (end - start) * t)
            }
        }
        let shifted = base.map { timeShiftExponential(mu: mu, t: $0) }
        let timesteps = shifted.map { $0 * Float(numTrainTimesteps) }
        let sigmas = shifted + [0.0]
        return (timesteps, sigmas)
    }

    /// Euler update: `x + (sigma_next - sigma) * noise`.
    public func step(modelOutput: [Float], stepIndex: Int, sample: [Float]) -> [Float] {
        precondition(stepIndex >= 0 && stepIndex < numInferenceSteps)
        precondition(modelOutput.count == sample.count)
        let s = sigmas[stepIndex]
        let sNext = sigmas[stepIndex + 1]
        let dt = sNext - s
        return zip(sample, modelOutput).map { x, e in x + dt * e }
    }

    /// First schedule index for strength-based I2I when **slicing** the full T2I schedule.
    ///
    /// Prefer ``strengthSchedule`` for I2I — it always runs `numInferenceSteps` steps
    /// from a noise level derived from strength (better color/structure edits on few-step
    /// distilled models).
    public static func startStep(numInferenceSteps: Int, strength: Float) -> Int {
        let s = max(0, min(1, strength))
        let initTimestep = min(Int((Float(numInferenceSteps) * s).rounded(.towardZero)), numInferenceSteps)
        let effective = s > 0 && initTimestep == 0 ? 1 : initTimestep
        return max(numInferenceSteps - effective, 0)
    }

    /// Build an I2I Euler schedule that always uses **all** `numInferenceSteps` steps.
    ///
    /// - Maps strength → starting noise with a mild curve (mid strengths denoise harder;
    ///   needed for color changes on 4-step distilled Klein).
    /// - Base times: `linspace(startT, 1/N, N)` then the same exponential μ shift as T2I.
    /// - Returns sigmas with terminal 0, and timesteps = pre-terminal × 1000.
    public static func strengthSchedule(
        numInferenceSteps: Int,
        strength: Float,
        imageSeqLen: Int,
        numTrainTimesteps: Int = ModelConstants.numTrainTimesteps
    ) -> (sigmas: [Float], timesteps: [Float], startSigma: Float) {
        precondition(numInferenceSteps >= 1)
        let s = max(0.05, min(1.0, strength))
        // Curve: push mid-range upward so 0.65 ≈ 0.88 noise, 0.7 ≈ 0.91 (color edits).
        // f(s) = 1 - (1-s)^1.5  keeps f(1)=1, f(0)=0, concave-down.
        let startT = 1 - pow(1 - s, 1.5)
        let endT = 1.0 / Float(numInferenceSteps)

        var base: [Float] = []
        base.reserveCapacity(numInferenceSteps)
        if numInferenceSteps == 1 {
            base = [startT]
        } else {
            for i in 0 ..< numInferenceSteps {
                let u = Float(i) / Float(numInferenceSteps - 1)
                base.append(startT + (endT - startT) * u)
            }
        }

        let mu = linearMu(imageSeqLen: imageSeqLen)
        let shifted = base.map { timeShiftExponential(mu: mu, t: max($0, 1e-4)) }
        let timesteps = shifted.map { $0 * Float(numTrainTimesteps) }
        let sigmas = shifted + [0.0]
        return (sigmas, timesteps, shifted[0])
    }

    /// Scalar convenience for schedule tables in tests.
    public var sigmaTable: [Float] { sigmas }
    public var timestepTable: [Float] { timesteps }
}
