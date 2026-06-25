import Foundation
import Tokenizers

/// Tokenizes prompts for the FLUX.2-klein text encoder (Qwen3-4B).
///
/// Uses HuggingFace's `swift-transformers` (`AutoTokenizer.from(modelFolder:)`) to load
/// `tokenizer.json` (a Qwen2 BPE tokenizer). The Qwen3 chat template (`enable_thinking=false`)
/// is applied manually — it is a fixed, trivial format and the model ships its template as a
/// separate `chat_template.jinja` (not embedded in `tokenizer_config.json`), so a manual
/// render is both robust and trivially parity-testable.
public final class AestrixTokenizer: @unchecked Sendable {

    /// FLUX.2 Klein always feeds a fixed 512-token text context to the transformer.
    public static let maxLength = 512

    /// Layers [9,18,27] of Qwen3-4B feed `context_in_dim = 7680 = 2560 × 3`.
    public static let contextDim = 7680

    private let tokenizer: any Tokenizer
    public let padId: Int
    public let eosId: Int

    /// - Parameter modelDir: the model root containing a `tokenizer/` subdirectory.
    public init(modelDir: URL) async throws {
        let tokenizerDir = modelDir.appendingPathComponent("tokenizer")
        guard FileManager.default.fileExists(atPath: tokenizerDir.appendingPathComponent("tokenizer.json").path) else {
            throw AestrixError.invalidInput("tokenizer.json not found under \(tokenizerDir.path)")
        }
        self.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)
        self.padId = tokenizer.convertTokenToId("<|endoftext|>") ?? 0
        self.eosId = tokenizer.convertTokenToId("<|im_end|>") ?? padId
    }

    /// Encode `prompt` with the Qwen3 chat template (`enable_thinking=false`), then pad/truncate
    /// to `maxLength` (512) so the transformer always sees a `(512, 7680)` context.
    public func encode(prompt: String, maxLength: Int = AestrixTokenizer.maxLength) -> [Int] {
        let formatted = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        var ids = tokenizer.encode(text: formatted, addSpecialTokens: false)
        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength))
        } else if ids.count < maxLength {
            ids.append(contentsOf: Array(repeating: padId, count: maxLength - ids.count))
        }
        return ids
    }

    /// Convenience: decode ids back to tokens (for debugging / parity checks).
    public func decode(ids: [Int]) -> String {
        tokenizer.decode(tokens: ids)
    }

    /// Id of a token string (e.g. `id(of: "<|im_start|>")`), if it is in the vocabulary.
    public func id(of token: String) -> Int? {
        tokenizer.convertTokenToId(token)
    }
}
