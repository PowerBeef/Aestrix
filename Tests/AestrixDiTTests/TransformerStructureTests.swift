import Testing
import Foundation
import MLX
import MLXNN
@testable import AestrixDiT
import AestrixCore

/// DiT tests allocate MLX arrays and require a working MLX backend (Metal metallib).
/// Enable with: `AESTRIX_MLX_TESTS=1 swift test --filter AestrixDiTTests`
private let mlxTestsEnabled = ProcessInfo.processInfo.environment["AESTRIX_MLX_TESTS"] == "1"

@Suite("DiT structure", .enabled(if: mlxTestsEnabled))
struct TransformerStructureTests {
    @Test("Klein-4B transformer builds with expected block counts")
    func buildsKlein4B() {
        let t = Flux2Transformer()
        #expect(t.numLayers == 5)
        #expect(t.numSingleLayers == 20)
        #expect(t.innerDim == 3072)
        #expect(t.jointAttentionDim == 7680)
    }

    @Test("parameter tree contains transformer_blocks and single_transformer_blocks")
    func parameterKeysPresent() {
        let t = Flux2Transformer()
        let flat = t.parameters().flattened().map(\.0)
        #expect(flat.contains { $0.hasPrefix("transformer_blocks.0") })
        #expect(flat.contains { $0.hasPrefix("single_transformer_blocks.0") })
        #expect(flat.contains { $0.hasPrefix("x_embedder") })
        #expect(flat.contains { $0.hasPrefix("context_embedder") })
        #expect(flat.contains { $0.hasPrefix("double_stream_modulation_img") })
        #expect(flat.contains { $0.hasPrefix("time_guidance_embed") })
        #expect(flat.contains { $0.hasPrefix("norm_out") })
        #expect(flat.contains { $0.hasPrefix("proj_out") })
        #expect(flat.contains { $0.hasPrefix("transformer_blocks.4") })
        #expect(flat.contains { $0.hasPrefix("single_transformer_blocks.19") })
        #expect(!flat.contains { $0.hasPrefix("transformer_blocks.5") })
    }

    @Test("quantize produces weight/scales keys for Linear layers")
    func quantizeAddsScales() {
        let t = Flux2Transformer()
        quantize(model: t, groupSize: 64, bits: 4) { _, _ in true }
        let keys = t.parameters().flattened().map(\.0)
        #expect(keys.contains { $0.contains("x_embedder.weight") })
        #expect(keys.contains { $0.contains("x_embedder.scales") })
        #expect(keys.contains { $0.contains("to_q.weight") } || keys.contains { $0.contains("to_qkv_mlp_proj.weight") })
    }

    @Test("DiTModule structure-only load/unload")
    func moduleLoadUnload() async throws {
        let m = DiTModule()
        m.loadStructureOnly()
        #expect(m.isLoaded)
        #expect(m.model != nil)
        await m.unload()
        #expect(!m.isLoaded)
        #expect(m.model == nil)
    }

    @Test("tiny random-init forward produces [B, img, 128]")
    func tinyForward() {
        let t = Flux2Transformer()
        let b = 1
        let img = 4
        let txt = 8
        let h = MLXArray.zeros([b, img, 128], dtype: .float32)
        let e = MLXArray.zeros([b, txt, 7680], dtype: .float32)
        let ts = MLXArray([Float(1000.0)])
        let imgIds = MLXArray.zeros([img, 4], dtype: .float32)
        let txtIds = MLXArray.zeros([txt, 4], dtype: .float32)
        let out = t(
            hiddenStates: h,
            encoderHiddenStates: e,
            timestep: ts,
            imgIds: imgIds,
            txtIds: txtIds,
            guidance: nil
        )
        eval(out)
        #expect(out.shape == [b, img, 128])
    }
}

@Suite("DiT dry residency")
struct DiTDryResidencyTests {
    @Test("load without snapshot marks loaded without allocating a model")
    func dryLoad() async throws {
        let m = DiTModule()
        try await m.load()
        #expect(m.isLoaded)
        #expect(m.model == nil)
        await m.unload()
        #expect(!m.isLoaded)
    }
}
