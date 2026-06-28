import Foundation
import MLX

/// Flow-match Euler discrete scheduler for FLUX.2 Klein.
///
/// Computes sigmas and timesteps using an empirical mu (based on image sequence length)
/// and exponential time-shift, matching mflux `FlowMatchEulerDiscreteScheduler`.
/// The denoise step is a simple Euler velocity update:
/// `latents += (sigmas[t+1] - sigmas[t]) * pred`.
public struct RectifiedFlowScheduler {
    public let sigmas: MLXArray     // [num_steps + 1], trailing 0
    public let timesteps: MLXArray  // [num_steps]

    public init(numSteps: Int = 4, imageSeqLen: Int) {
        let (s, t) = Self.compute(numSteps: numSteps, imageSeqLen: imageSeqLen)
        self.sigmas = s
        self.timesteps = t
    }

    /// One Euler step: `latents + (sigmas[t+1] - sigmas[t]) * noise`.
    public func step(noise: MLXArray, timestepIndex t: Int, latents: MLXArray) -> MLXArray {
        let dt = sigmas[t + 1] - sigmas[t]
        return latents + dt * noise
    }

    // MARK: - Schedule computation (ports mflux FlowMatchEulerDiscreteScheduler)
    //
    // FLUX.2-klein's model config has `requires_sigma_shift=True`, so mflux calls
    // `set_image_seq_len(image_seq_len)` → `get_timesteps_and_sigmas(...)`. That path is what
    // actually runs at generation time (the constructor path with `sigma_min` spacing + a
    // `stretch_to_terminal` is overridden). We port that exact path here:
    //   sigmas   = linspace(1.0, 1.0/num_steps, num_steps)
    //   mu       = empirical_mu(image_seq_len, num_steps)
    //   sigmas   = time_shift_exponential(mu, 1.0, sigmas)   # elementwise
    //   timesteps = sigmas * num_train_timesteps
    //   sigmas   = sigmas ++ [0.0]
    // (no terminal stretch.)

    private static func compute(numSteps: Int, imageSeqLen: Int) -> (MLXArray, MLXArray) {
        let (sigmas, timesteps) = computeFloats(numSteps: numSteps, imageSeqLen: imageSeqLen)
        return (MLXArray(sigmas), MLXArray(timesteps))
    }

    /// Pure-Swift schedule computation (no Metal) — exposed for the mflux-parity unit test.
    /// Returns `(sigmas, timesteps)` where `sigmas` has the trailing 0 appended.
    static func computeFloats(numSteps: Int, imageSeqLen: Int) -> (sigmas: [Float], timesteps: [Float]) {
        let numTrainTimesteps: Float = 1000

        // linspace(1.0, 1.0/num_steps, num_steps)
        let stop = 1.0 / Float(numSteps)
        var sigmasLinear: [Float] = []
        for i in 0..<numSteps {
            sigmasLinear.append(1.0 + Float(i) * (stop - 1.0) / Float(numSteps - 1))
        }

        let mu = empiricalMu(imageSeqLen: imageSeqLen, numSteps: numSteps)
        let sigmasShifted = sigmasLinear.map { timeShiftExp(mu: mu, sigmaPower: 1.0, t: $0) }

        let timesteps = sigmasShifted.map { $0 * numTrainTimesteps }
        let sigmasWithZero = sigmasShifted + [Float(0)]
        return (sigmasWithZero, timesteps)
    }

    private static func empiricalMu(imageSeqLen: Int, numSteps: Int) -> Float {
        let a1: Float = 8.73809524e-05, b1: Float = 1.89833333
        let a2: Float = 0.00016927, b2: Float = 0.45666666
        if imageSeqLen > 4300 {
            return a2 * Float(imageSeqLen) + b2
        }
        let m200 = a2 * Float(imageSeqLen) + b2
        let m10 = a1 * Float(imageSeqLen) + b1
        let a = (m200 - m10) / 190.0
        let b = m200 - 200.0 * a
        return a * Float(numSteps) + b
    }

    private static func timeShiftExp(mu: Float, sigmaPower: Float, t: Float) -> Float {
        return exp(mu) / (exp(mu) + pow(1.0 / t - 1.0, sigmaPower))
    }
}
