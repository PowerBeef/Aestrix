import Foundation
import CryptoKit
import MLX
import ImarelloCore
import ImarelloText

/// Disk cache for Qwen3 prompt embeddings (`[1, 512, 7680]`, ~16 MB f32 per prompt).
///
/// On a hit the pipeline skips the entire TE stage (~2.26 GB weight load + encode).
/// Entries are keyed by prompt text, model id, TE quant bits, sequence length, and
/// the chat-template version, so template or weight changes invalidate old entries.
public enum PromptEmbedCache {
    /// Bump when `QwenChatTemplate` or tap/concat behavior changes.
    /// v2: tokenizer gained NFC + the reference pre-tokenizer (2026-08-16) —
    /// ids changed for digit/punctuation prompts, so v1 embeds must miss.
    /// v3: pad-content mode joined the key (TE-splice, 2026-08-18) — spliced
    /// and full-window entries must never alias.
    private static let formatVersion = "v3"

    public struct Entry {
        public let embeds: MLXArray
        public let realTokens: Int
    }

    static func cacheDirectory() -> URL {
        AppCache.directory("embeds")
    }

    static func entryFilename(
        prompt: String,
        modelID: String,
        bits: Int = TextEncoderWeights.defaultBits,
        maxLength: Int = ModelConstants.maxSequenceLength,
        padContent: String = "prompt"
    ) -> String {
        // The tap layers and concat width are part of the key so changing them
        // invalidates entries automatically instead of relying on a manual
        // formatVersion bump.
        let taps = ModelConstants.textEncoderLayers.map(String.init).joined(separator: ",")
        let keySource =
            "\(formatVersion)|\(modelID)|bits\(bits)|len\(maxLength)"
            + "|taps\(taps)|dim\(ModelConstants.jointAttentionDim)"
            + "|pad\(padContent)|\(prompt)"
        let digest = SHA256.hash(data: Data(keySource.utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return "\(hex).safetensors"
    }

    public static func entryURL(
        prompt: String,
        modelID: String,
        bits: Int = TextEncoderWeights.defaultBits,
        maxLength: Int = ModelConstants.maxSequenceLength,
        padContent: String = "prompt"
    ) -> URL {
        let name = entryFilename(
            prompt: prompt, modelID: modelID, bits: bits, maxLength: maxLength,
            padContent: padContent)
        return AppCache.resolvedItem(under: "embeds", item: name)
    }

    /// Returns nil on miss or any read error (cache is best-effort).
    /// A corrupt entry is deleted so it cannot poison every later run: MLX
    /// materializes safetensors lazily, so a truncated file with an intact
    /// header would otherwise only fail inside a non-throwing `eval()`.
    public static func load(url: URL) -> Entry? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        guard let (arrays, metadata) = try? loadArraysAndMetadata(url: url),
              let embeds = arrays["embeds"],
              let realTokens = metadata["real_tokens"].flatMap({ Int($0) }),
              validate(embeds: embeds, realTokens: realTokens, url: url)
        else {
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        return Entry(embeds: embeds, realTokens: realTokens)
    }

    private static func validate(embeds: MLXArray, realTokens: Int, url: URL) -> Bool {
        guard embeds.ndim == 3,
              embeds.dim(0) == 1,
              embeds.dim(1) == ModelConstants.maxSequenceLength,
              embeds.dim(2) == ModelConstants.jointAttentionDim,
              // The TE emits bf16 (f32 accepted for older entries). The audit-era
              // `.float32`-only check rejected every real entry, so the cache
              // silently never hit and each generate re-paid the full TE stage.
              embeds.dtype == .bfloat16 || embeds.dtype == .float32,
              realTokens > 0, realTokens <= embeds.dim(1)
        else { return false }
        // Truncation guard: the file must hold at least the tensor payload.
        let expectedBytes = embeds.shape.reduce(embeds.itemSize) { $0 * $1 }
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size])
            .flatMap { $0 as? Int }
        return (size ?? 0) >= expectedBytes
    }

    /// Best-effort write; failures are silent (cache is an optimization only).
    /// Writes go to a temp file first so a crash mid-save can never leave a
    /// half-written entry at the final path.
    public static func store(embeds: MLXArray, realTokens: Int, url: URL) {
        let fm = FileManager.default
        let dir = url.deletingLastPathComponent()
        let tmp = dir.appendingPathComponent(".tmp-\(UUID().uuidString).safetensors")
        do {
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            try save(
                arrays: ["embeds": embeds],
                metadata: ["real_tokens": String(realTokens)],
                url: tmp
            )
            try? fm.removeItem(at: url)
            try fm.moveItem(at: tmp, to: url)
        } catch {
            try? fm.removeItem(at: tmp)
            try? fm.removeItem(at: url)
        }
    }
}
