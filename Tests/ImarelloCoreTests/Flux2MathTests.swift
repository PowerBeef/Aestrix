import Testing
import Foundation
@testable import ImarelloCore

@Suite("FLUX.2 pure math")
struct Flux2MathTests {

    // MARK: - RoPE

    @Test("1D RoPE omega frequencies match theta scaling")
    func rope1DOmega() {
        let dim = 32
        let theta: Float = 2000
        let (cos0, sin0) = Flux2RoPE.rope1D(dim: dim, pos: 0, theta: theta)
        #expect(cos0.count == dim / 2)
        #expect(sin0.count == dim / 2)
        // pos=0 → all cos=1, sin=0
        for c in cos0 { #expect(abs(c - 1) < 1e-5) }
        for s in sin0 { #expect(abs(s) < 1e-5) }

        let (cos1, sin1) = Flux2RoPE.rope1D(dim: dim, pos: 1, theta: theta)
        // first frequency: omega = 1 / theta^0 = 1
        #expect(abs(cos1[0] - cos(1.0 as Float)) < 1e-5)
        #expect(abs(sin1[0] - sin(1.0 as Float)) < 1e-5)
        // second: omega = 1 / theta^(2/32) = 1 / theta^(1/16)
        let omega1 = 1.0 / pow(theta, Float(2) / Float(dim))
        #expect(abs(cos1[1] - cos(omega1)) < 1e-5)
        #expect(abs(sin1[1] - sin(omega1)) < 1e-5)
    }

    @Test("4-axis RoPE concatenates axis halves")
    func rope4AxisConcat() {
        let ids: [[Float]] = [[0, 1, 2, 3]]
        let (cos, sin) = Flux2RoPE.frequencies(ids: ids)
        let expectedHalf = ModelConstants.ropeAxesDims.reduce(0) { $0 + $1 / 2 }
        #expect(cos[0].count == expectedHalf)
        #expect(sin[0].count == expectedHalf)
        #expect(expectedHalf == 64) // 4 * 16
    }

    @Test("grid ids layout t,h,w,layer")
    func gridIDs() {
        let ids = Flux2RoPE.prepareGridIDs(height: 2, width: 3, tCoord: 0)
        #expect(ids.count == 6)
        #expect(ids[0] == [0, 0, 0, 0])
        #expect(ids[1] == [0, 0, 1, 0])
        #expect(ids[3] == [0, 1, 0, 0])
        #expect(ids[5] == [0, 1, 2, 0])
    }

    @Test("applyRotary rotates (1,0) by 90° to (0,1)")
    func applyRotaryUnit() {
        let out = Flux2RoPE.applyRotary(x: [1, 0], cos: [0], sin: [1])
        #expect(abs(out[0] - 0) < 1e-5)
        #expect(abs(out[1] - 1) < 1e-5)
    }

    // MARK: - Timestep embedding

    @Test("timestep embed shape and flip_sin_to_cos")
    func timestepEmbedShape() {
        let emb = Flux2TimestepEmbedding.embed(timesteps: [0, 500, 1000], dim: 256)
        #expect(emb.count == 3)
        #expect(emb[0].count == 256)
        // t=0 → all zeros in sin/cos args → cos=1 sin=0; with flip: first half cos=1, second sin=0
        for i in 0..<128 {
            #expect(abs(emb[0][i] - 1) < 1e-5)
        }
        for i in 128..<256 {
            #expect(abs(emb[0][i]) < 1e-5)
        }
    }

    @Test("timestep scale ×1000 when max≤1")
    func timestepScale() {
        let scaled = Flux2TimestepEmbedding.scaleTimestepsIfNeeded([0.5, 1.0])
        #expect(scaled == [500, 1000])
        let unscaled = Flux2TimestepEmbedding.scaleTimestepsIfNeeded([0.5, 1000])
        #expect(unscaled == [0.5, 1000])
    }

    // MARK: - Modulation geometry

    @Test("modulation split geometry double and single stream")
    func modulationSplit() {
        let dim = 4
        // double stream: 2 sets → 4*3*2 = 24
        let flat = (0..<24).map { Float($0) }
        let sets = Flux2ModulationMath.split(mod: flat, dim: dim, modParamSets: 2)
        #expect(sets.count == 2)
        #expect(sets[0].shift == [0, 1, 2, 3])
        #expect(sets[0].scale == [4, 5, 6, 7])
        #expect(sets[0].gate == [8, 9, 10, 11])
        #expect(sets[1].shift == [12, 13, 14, 15])

        let single = Flux2ModulationMath.split(mod: Array(flat.prefix(12)), dim: dim, modParamSets: 1)
        #expect(single.count == 1)
        #expect(Flux2ModulationMath.outputFeatures(dim: 3072, modParamSets: 2) == 3072 * 6)
    }

    @Test("modulation apply and gated residual")
    func modulationApply() {
        let y = Flux2ModulationMath.apply(norm: [1, 2], shift: [0.5, -0.5], scale: [1, 0])
        // (1+1)*1 + 0.5 = 2.5; (1+0)*2 - 0.5 = 1.5
        #expect(abs(y[0] - 2.5) < 1e-5)
        #expect(abs(y[1] - 1.5) < 1e-5)
        let r = Flux2ModulationMath.gatedResidual(x: [1, 1], y: [2, 3], gate: [0.5, 0])
        #expect(abs(r[0] - 2.0) < 1e-5)
        #expect(abs(r[1] - 1.0) < 1e-5)
    }

    // MARK: - Scheduler

    @Test("imageSeqLen is H/16 * W/16")
    func imageSeqLen() {
        #expect(Flux2Scheduler.imageSeqLen(width: 1024, height: 1024) == 4096)
        #expect(Flux2Scheduler.imageSeqLen(width: 512, height: 512) == 1024)
        #expect(Flux2Scheduler.imageSeqLen(width: 768, height: 512) == 48 * 32)
    }

    @Test("linear mu is 1.15 at 4096 tokens")
    func linearMuAt4096() {
        let mu = Flux2Scheduler.linearMu(imageSeqLen: 4096)
        #expect(abs(mu - 1.15) < 1e-5)
        let mu256 = Flux2Scheduler.linearMu(imageSeqLen: 256)
        #expect(abs(mu256 - 0.5) < 1e-5)
    }

    @Test("4-step 1024² schedule is monotone decreasing to 0")
    func schedule4Step1024() {
        let sched = Flux2Scheduler(numInferenceSteps: 4, imageSeqLen: 4096)
        #expect(sched.sigmas.count == 5) // N + terminal 0
        #expect(sched.timesteps.count == 4)
        #expect(abs(sched.sigmas.last! ) < 1e-7)
        for i in 0..<(sched.sigmas.count - 1) {
            #expect(sched.sigmas[i] >= sched.sigmas[i + 1] - 1e-6)
        }
        // first sigma after shift should be near 1
        #expect(sched.sigmas[0] > 0.9)
        // timesteps = sigma * 1000 before terminal
        #expect(abs(sched.timesteps[0] - sched.sigmas[0] * 1000) < 1e-3)
    }

    @Test("euler step uses sigma_next - sigma")
    func eulerStep() {
        let sched = Flux2Scheduler(numInferenceSteps: 4, imageSeqLen: 1024)
        let s0 = sched.sigmas[0]
        let s1 = sched.sigmas[1]
        let sample: [Float] = [1, 2, 3]
        let noise: [Float] = [0.1, 0.1, 0.1]
        let next = sched.step(modelOutput: noise, stepIndex: 0, sample: sample)
        let dt = s1 - s0
        for i in 0..<3 {
            #expect(abs(next[i] - (sample[i] + dt * noise[i])) < 1e-5)
        }
    }

    @Test("time shift formula at t=1 is 1")
    func timeShiftAtOne() {
        let v = Flux2Scheduler.timeShiftExponential(mu: 1.15, t: 1.0)
        #expect(abs(v - 1.0) < 1e-5)
    }

    @Test("I2I strength maps to schedule start step (legacy slice)")
    func strengthStartStep() {
        #expect(Flux2Scheduler.startStep(numInferenceSteps: 4, strength: 1.0) == 0)
        #expect(Flux2Scheduler.startStep(numInferenceSteps: 4, strength: 0.5) == 2)
        #expect(Flux2Scheduler.startStep(numInferenceSteps: 4, strength: 0.25) == 3)
        #expect(Flux2Scheduler.startStep(numInferenceSteps: 4, strength: 0.01) == 3)
        #expect(Flux2Scheduler.startStep(numInferenceSteps: 4, strength: 0.0) == 4)
    }

    @Test("I2I strength schedule always has N steps and ends at 0")
    func strengthScheduleFullSteps() {
        let (sigmas, timesteps, startSigma) = Flux2Scheduler.strengthSchedule(
            numInferenceSteps: 4, strength: 0.65, imageSeqLen: 1024)
        #expect(timesteps.count == 4)
        #expect(sigmas.count == 5)
        #expect(abs(sigmas.last!) < 1e-7)
        #expect(startSigma == sigmas[0])
        // Mid strength should still start fairly high (curve + shift).
        #expect(startSigma > 0.7)
        for i in 0 ..< (sigmas.count - 1) {
            #expect(sigmas[i] >= sigmas[i + 1] - 1e-6)
        }
        // Higher strength → higher start sigma
        let high = Flux2Scheduler.strengthSchedule(
            numInferenceSteps: 4, strength: 0.95, imageSeqLen: 1024)
        #expect(high.startSigma >= startSigma - 1e-5)
    }

    @Test("documented startT table matches StrengthScheduleCurve")
    func strengthCurveStartTMatchesDocs() {
        // Docs/I2I_STRENGTH.md — keep the table and this test in lockstep.
        let rows: [(Float, Float, Float)] = [
            (0.35, 0.476, 0.307),
            (0.50, 0.646, 0.445),
            (0.65, 0.793, 0.590),
            (0.75, 0.875, 0.692),
            (0.80, 0.911, 0.745),
            (0.85, 0.942, 0.801),
            (0.90, 0.968, 0.859),
            (1.00, 1.000, 1.000),
        ]
        for (s, color, identity) in rows {
            #expect(abs(StrengthScheduleCurve.colorEdit.startT(strength: s) - color) < 0.002)
            #expect(abs(StrengthScheduleCurve.identityPreserve.startT(strength: s) - identity) < 0.002)
            #expect(abs(StrengthScheduleCurve.linear.startT(strength: s) - s) < 1e-5)
        }
    }

    @Test("Identity schedule starts milder than color curve at same strength")
    func identityVsColorSchedule() {
        let s: Float = 0.8
        #expect(StrengthScheduleCurve.colorEdit.startT(strength: s)
            > StrengthScheduleCurve.identityPreserve.startT(strength: s) + 0.05)
        #expect(abs(StrengthScheduleCurve.linear.startT(strength: s) - s) < 1e-5)

        let color = Flux2Scheduler.strengthSchedule(
            numInferenceSteps: 4, strength: s, imageSeqLen: 1024, curve: .colorEdit)
        let id = Flux2Scheduler.strengthSchedule(
            numInferenceSteps: 4, strength: s, imageSeqLen: 1024, curve: .identityPreserve)
        #expect(color.startSigma > id.startSigma + 0.02)
    }

    @Test("grid ids honor tCoord for reference frames")
    func gridIDsReferenceT() {
        let ids = Flux2RoPE.prepareGridIDs(height: 1, width: 2, tCoord: 10)
        #expect(ids.count == 2)
        #expect(ids[0] == [10, 0, 0, 0])
        #expect(ids[1] == [10, 0, 1, 0])
    }

    @Test("IdentityPreserveConfig preset enables Tier-B stack")
    func identityPreset() {
        #expect(!IdentityPreserveConfig.disabled.isActive)
        let p = IdentityPreserveConfig.identityPreset
        #expect(p.useReferenceLatents)
        #expect(p.facePreserve)
        #expect(p.cleanPullAlpha > 0)
        #expect(p.scheduleCurve == .identityPreserve)
        #expect(p.isActive)
    }

    @Test("text ids use T,H,W,L with token on L axis")
    func textIDsLayout() {
        let ids = Flux2RoPE.prepareTextIDs(length: 3)
        #expect(ids[0] == [0, 0, 0, 0])
        #expect(ids[1] == [0, 0, 0, 1])
        #expect(ids[2] == [0, 0, 0, 2])
    }
}
