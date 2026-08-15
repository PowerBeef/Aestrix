import Testing
@testable import ImarelloDiT

@Suite("DiT op profile accumulator", .serialized)
struct DiTOpProfileTests {
    @Test("reset disables and zeros")
    func resetClears() {
        DiTOpProfile.begin()
        DiTOpProfile.record(.steelFA, seconds: 1)
        DiTOpProfile.reset()
        #expect(!DiTOpProfile.enabled)
        let snap = DiTOpProfile.snapshot()
        #expect(snap.countedMs == 0)
        #expect(snap.bucketsMs["steel_fa"] == 0)
    }

    @Test("shares and remainder against denoise wall")
    func sharesAndOther() {
        DiTOpProfile.reset()
        DiTOpProfile.record(.qkvProj, seconds: 0.40)
        DiTOpProfile.record(.qkvRope, seconds: 0.10)
        DiTOpProfile.record(.steelFA, seconds: 0.30)
        DiTOpProfile.record(.ffn, seconds: 0.20)
        let snap = DiTOpProfile.snapshot(denoiseSeconds: 1.20)
        #expect(abs(snap.countedMs - 1000) < 0.01)
        #expect(abs((snap.otherMs ?? -1) - 200) < 0.01)
        #expect(abs((snap.shares["qkv_proj"] ?? 0) - (400 / 1200)) < 1e-6)
        #expect(abs((snap.shares["other"] ?? 0) - (200 / 1200)) < 1e-6)
        #expect(snap.dominantBucket == "qkv_proj")
        #expect(snap.counts["ffn"] == 1)
    }

    @Test("time is a passthrough when disabled")
    func timeDisabledPassthrough() {
        DiTOpProfile.reset()
        let value = DiTOpProfile.time(.ffn, inputs: [], sync: { _ in [] }) { 7 }
        #expect(value == 7)
        #expect(DiTOpProfile.snapshot().countedMs == 0)
    }
}
