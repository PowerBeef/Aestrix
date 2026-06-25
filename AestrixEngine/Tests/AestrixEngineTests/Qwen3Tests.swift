import Foundation
import Testing
import MLX
@testable import AestrixEngine

@Suite("Qwen3TextEncoder")
struct Qwen3Tests {

    @Test("loads Qwen3-4B and encodes a prompt to ctx (1,512,7680)", .enabled(if: heavyEnabled))
    func encodePrompt() async throws {
        try await TestFixtures.ensure(Flux2Klein4B4Bit.textEncoderFiles + Flux2Klein4B4Bit.tokenizerFiles)

        // Tokenize the prompt → 512 ids (end-to-end: tokenizer → text encoder).
        let tokenizer = try await AestrixTokenizer(modelDir: TestFixtures.root)
        let ids = tokenizer.encode(prompt: "a beautiful mountain landscape")
        #expect(ids.count == 512)

        // Load the 4-bit Qwen3-4B text encoder (float32 data path for determinism).
        let (weights, _) = try WeightLoader.loadWeights(
            submodelDir: TestFixtures.root.appendingPathComponent("text_encoder"))
        let encoder = Qwen3TextEncoder(precision: .float32)
        encoder.loadWeights(weights)

        let ctx = encoder.promptEmbeds(MLXArray(ids, [1, ids.count]))
        eval(ctx)

        #expect(ctx.shape == [1, 512, 7680], "ctx shape \(ctx.shape)")   // 3 × 2560
        #expect(ctx.max().item(Float.self).isFinite)
        #expect(ctx.min().item(Float.self).isFinite)
    }
}
