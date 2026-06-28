import Foundation
import Testing
@testable import AestrixEngine

/// Parity test for the rectified-flow scheduler against mflux
/// `FlowMatchEulerDiscreteScheduler.get_timesteps_and_sigmas` — the path FLUX.2-klein actually
/// uses (model config `requires_sigma_shift=True` → `set_image_seq_len`).
///
/// This is pure-Swift arithmetic (no Metal), so it runs under plain `swift test`.
/// Reference values were dumped from mflux 0.18.0:
///   `FlowMatchEulerDiscreteScheduler.get_timesteps_and_sigmas(image_seq_len, num_inference_steps)`.
@Suite("RectifiedFlowScheduler — mflux parity")
struct SchedulerParityTests {

    /// Tolerant comparison helper: relative/absolute for bf16→float rounded references.
    private func assertClose(_ got: [Float], _ want: [Float], tol: Float = 1e-3) {
        #expect(got.count == want.count, "len \(got.count) != \(want.count)")
        for i in 0..<min(got.count, want.count) {
            let g = got[i], w = want[i]
            let absErr = abs(g - w)
            let relErr = absErr / max(abs(w), 1e-6)
            #expect(absErr < tol || relErr < tol, "idx \(i): got \(g) want \(w)")
        }
    }

    @Test("sigmas/timesteps match mflux for 256×256 / 4 steps (the failing-image config)")
    func seq256Steps4() {
        let (sigmas, timesteps) = RectifiedFlowScheduler.computeFloats(numSteps: 4, imageSeqLen: 256)
        assertClose(timesteps, [1000.0, 955.3907, 877.1342, 704.1116])
        assertClose(sigmas,    [1.0, 0.9553908, 0.8771342, 0.7041116, 0.0])
    }

    @Test("sigmas/timesteps match mflux for 1024×1024 / 4 steps (default gen size)")
    func seq1024Steps4() {
        let (sigmas, timesteps) = RectifiedFlowScheduler.computeFloats(numSteps: 4, imageSeqLen: 1024)
        assertClose(timesteps, [1000.0, 958.0854, 883.9818, 717.4966])
        assertClose(sigmas,    [1.0, 0.9580854, 0.8839818, 0.7174966, 0.0])
    }

    @Test("sigmas/timesteps match mflux for 4096 seq (large image)")
    func seq4096Steps4() {
        let (sigmas, timesteps) = RectifiedFlowScheduler.computeFloats(numSteps: 4, imageSeqLen: 4096)
        assertClose(timesteps, [1000.0, 967.3839, 908.1439, 767.2000])
        assertClose(sigmas,    [1.0, 0.9673839, 0.9081439, 0.7672000, 0.0])
    }

    @Test("last sigma is NOT the terminal 0.02 (regression guard for the stretch bug)")
    func noTerminalCollapse() {
        let (sigmas, _) = RectifiedFlowScheduler.computeFloats(numSteps: 4, imageSeqLen: 256)
        // The buggy constructor-path collapsed the last non-zero sigma to ~0.02. The correct path
        // leaves it ~0.70 — the denoise loop's final step must move the latent substantially.
        #expect(sigmas[sigmas.count - 2] > 0.5, "final sigma \(sigmas[sigmas.count-2]) collapsed — stretch bug?")
    }
}
