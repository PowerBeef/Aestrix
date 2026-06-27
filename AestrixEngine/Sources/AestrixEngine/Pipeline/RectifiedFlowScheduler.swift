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

    private static func compute(numSteps: Int, imageSeqLen: Int) -> (MLXArray, MLXArray) {
        let numTrainTimesteps = 1000
        let sigmaMin = 1.0 / Float(numTrainTimesteps)
        let sigmaMax: Float = 1.0

        // Linear sigmas
        var sigmasLinear: [Float] = []
        for i in 0..<numSteps {
            let t = sigmaMax - Float(i) * (sigmaMax - sigmaMin) / Float(numSteps - 1)
            sigmasLinear.append(t)
        }

        // Empirical mu from image_seq_len + num_steps
        let mu = empiricalMu(imageSeqLen: imageSeqLen, numSteps: numSteps)

        // Time-shift exponential
        let sigmasShifted = sigmasLinear.map { timeShiftExp(mu: mu, sigmaPower: 1.0, t: $0) }

        // Stretch to terminal (0.02)
        let terminal: Float = 0.02
        let oneMinusLast = 1.0 - sigmasShifted.last!
        let scaleFactor = oneMinusLast / (1.0 - terminal)
        let sigmasFinal = sigmasShifted.map { 1.0 - ((1.0 - $0) / scaleFactor) }

        let timesteps = sigmasFinal.map { $0 * Float(numTrainTimesteps) }
        let sigmasWithZero = sigmasFinal + [Float(0)]

        return (MLXArray(sigmasWithZero), MLXArray(timesteps))
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
