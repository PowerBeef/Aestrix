import Foundation
import Testing
@testable import AestrixEngine

// Shared helpers (TestFixtures, heavyEnabled) live in TestSupport.swift.

// MARK: - ModelDownloader

@Suite("ModelDownloader")
struct DownloaderTests {

    @Test("downloads a small file, verifies size, and is idempotent")
    func downloadSmallFile() async throws {
        let file = TestFixtures.manifestFile("tokenizer/tokenizer_config.json") // 703 B
        let root = URL(fileURLWithPath: ".build/fixtures-test/downloader")
        let downloader = ModelDownloader(files: [file], destinationRoot: root)

        try await downloader.ensureDownloaded()
        #expect(await downloader.isComplete(file))

        // Second call must be a no-op (skip-if-complete).
        try await downloader.ensureDownloaded()
        #expect(await downloader.isComplete(file))

        // A bogus destination reports the file as missing.
        let bogus = ModelDownloader(files: [file], destinationRoot: URL(fileURLWithPath: "/nonexistent/aestrix"))
        #expect(await bogus.missingFiles().count == 1)
    }
}

// MARK: - AestrixTokenizer

@Suite("AestrixTokenizer")
struct TokenizerTests {

    @Test("encodes a prompt to 512 padded tokens using the Qwen3 chat template")
    func encodePrompt() async throws {
        try await TestFixtures.ensure(Flux2Klein4B4Bit.tokenizerFiles)
        let tokenizer = try await AestrixTokenizer(modelDir: TestFixtures.root)

        let ids = tokenizer.encode(prompt: "a beautiful mountain landscape")

        // FLUX.2 Klein always feeds a fixed 512-token context.
        #expect(ids.count == AestrixTokenizer.maxLength)

        // Exact parity vs Python `tokenizers` (HF) on the same tokenizer.json + chat template:
        // <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n  → 12 content tokens.
        let reference = [151644, 872, 198, 64, 6233, 16301, 18414, 151645, 198, 151644, 77091, 198]
        #expect(Array(ids.prefix(reference.count)) == reference)
        #expect(tokenizer.padId == 151643)              // <|endoftext|>
        #expect(ids.first == 151644)                    // <|im_start|>

        // The remaining 500 slots are padding.
        #expect(ids.dropFirst(reference.count).allSatisfy { $0 == tokenizer.padId })
    }
}

// MARK: - WeightLoader

@Suite("WeightLoader")
struct WeightLoaderTests {

    @Test("parses transformer index.json: 4-bit quantization, group_size 64")
    func parseTransformerIndex() async throws {
        try await TestFixtures.ensure([TestFixtures.manifestFile("transformer/model.safetensors.index.json")])

        let (map, quant) = try WeightLoader.parseIndex(
            submodelDir: TestFixtures.root.appendingPathComponent("transformer"))

        #expect(quant?.bits == 4)
        #expect(quant?.groupSize == 64)

        // Anchor tensors observed in the real index.json:
        #expect(map["x_embedder.weight"] != nil)                                    // image patch embed
        #expect(map["context_embedder.weight"] != nil)                              // text context proj
        #expect(map.keys.contains { $0.hasPrefix("transformer_blocks.4.") })        // 5 double-stream blocks (0…4)
        #expect(map.keys.contains { $0.hasPrefix("single_transformer_blocks.19.") }) // 20 single-stream blocks (0…19)
        #expect(map.keys.contains { $0 == "proj_out.weight" })                      // final projection
    }

    @Test("loads VAE weights from safetensors", .enabled(if: heavyEnabled))
    func loadVAEWeights() async throws {
        try await TestFixtures.ensure(Flux2Klein4B4Bit.vaeFiles)
        let (weights, _) = try WeightLoader.loadWeights(
            submodelDir: TestFixtures.root.appendingPathComponent("vae"))
        // The FLUX VAE has dozens of conv/norm/attention tensors.
        #expect(weights.count > 50)
        // Every loaded tensor has a concrete shape.
        #expect(weights.values.allSatisfy { $0.ndim >= 1 })
    }
}
