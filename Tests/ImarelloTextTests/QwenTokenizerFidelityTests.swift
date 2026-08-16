import Testing
import Foundation
import ImarelloCore
@testable import ImarelloText

/// Differential test against HF `tokenizers` reference ids
/// (Fixtures/qwen-tokenizer-cases.json, generated from the pinned snapshot's
/// tokenizer.json). Needs the local snapshot; skipped on CI without weights.
@Suite("Qwen tokenizer fidelity")
struct QwenTokenizerFidelityTests {
    private struct Case: Decodable {
        let text: String
        let ids: [Int]
    }

    private static func tokenizerDirectory() -> URL? {
        let models = AppCache.resolvedDirectory("models")
        let dir = models
            .appendingPathComponent("mlx-community--FLUX.2-Klein-4B-4bit")
            .appendingPathComponent("tokenizer")
        let json = dir.appendingPathComponent("tokenizer.json")
        return FileManager.default.fileExists(atPath: json.path) ? dir : nil
    }

    @Test("encode matches HF tokenizers on the reference corpus")
    func matchesReference() throws {
        guard let dir = Self.tokenizerDirectory() else {
            // No local snapshot (CI): fidelity is covered on dev machines.
            return
        }
        let fixtures = try #require(
            Bundle.module.url(forResource: "qwen-tokenizer-cases", withExtension: "json", subdirectory: "Fixtures")
        )
        let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: fixtures))
        #expect(cases.count >= 20)
        let tokenizer = try QwenTokenizer.load(from: dir)
        for c in cases {
            let got = tokenizer.encode(c.text)
            #expect(got == c.ids, "ids diverge for: \(c.text.prefix(60))")
        }
    }

    @Test("pre-tokenizer isolates digits and double spaces")
    func preTokenizePieces() {
        #expect(QwenTokenizer.preTokenize("room 42") == ["room", " ", "4", "2"])
        #expect(QwenTokenizer.preTokenize("a  b") == ["a", " ", " b"])
        #expect(QwenTokenizer.preTokenize("It's") == ["It", "'s"])
    }
}
