import Foundation
import Testing
@testable import AestrixEngine

/// Package-level tests that do NOT execute MLX GPU ops.
///
/// Note: MLX execution requires an Xcode/xcodebuild build (SwiftPM CLI cannot build the
/// Metal shaders / `default.metallib`). The on-device MLX sanity check lives in the demo
/// app (`AestrixEngine.verifyIntegration()`), run via xcodebuild. Keep this suite MLX-free
/// so `swift test` stays green and fast.
@Suite("AestrixEngine config invariants")
struct IntegrationTests {

    @Test("GenConfig defaults match the FLUX.2-klein distilled recipe")
    func configDefaults() {
        let config = GenConfig()
        #expect(config.steps == 4)        // distilled rectified flow
        #expect(config.guidance == 1.0)   // no classifier-free guidance
        #expect(config.width == 1024)
        #expect(config.height == 1024)
        #expect(config.width % 16 == 0 && config.height % 16 == 0) // VAE 16× compression
    }

    // MARK: - MLX execution (requires xcodebuild, NOT `swift test`)

    @Test(
        "MLX executes a matmul on the GPU",
        .enabled(if: ProcessInfo.processInfo.environment["AESTRIX_RUN_MLX_TESTS"] != nil)
    )
    func mlxExecutes() async {
        // MLX GPU ops need a Metal library built by Xcode/xcodebuild; `swift test`
        // (SwiftPM CLI) cannot build it, so this test is skipped under `swift test`.
        // Run via: AESTRIX_RUN_MLX_TESTS=1 xcodebuild test -scheme AestrixEngine …
        let engine = AestrixEngine()
        let r = await engine.verifyIntegration()
        #expect(r.ok, "MLX matmul did not execute on-device")
        #expect(r.matmulRows == 512)
        #expect(r.resultSum.isFinite)
    }
}
