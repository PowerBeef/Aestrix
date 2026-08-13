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
enum PromptEmbedCache {
    /// Bump when `QwenChatTemplate` or tap/concat behavior changes.
    private static let formatVersion = "v1"

    struct Entry {
        let embeds: MLXArray
        let realTokens: Int
    }

    static func cacheDirectory() -> URL {
        AppCache.directory("embeds")
    }

    static func entryFilename(
        prompt: String,
        modelID: String,
        bits: Int = TextEncoderWeights.defaultBits,
        maxLength: Int = ModelConstants.maxSequenceLength
    ) -> String {
        let keySource = "\(formatVersion)|\(modelID)|bits\(bits)|len\(maxLength)|\(prompt)"
        let digest = SHA256.hash(data: Data(keySource.utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return "\(hex).safetensors"
    }

    static func entryURL(
        prompt: String,
        modelID: String,
        bits: Int = TextEncoderWeights.defaultBits,
        maxLength: Int = ModelConstants.maxSequenceLength
    ) -> URL {
        let name = entryFilename(
            prompt: prompt, modelID: modelID, bits: bits, maxLength: maxLength)
        return AppCache.resolvedItem(under: "embeds", item: name)
    }

    /// Returns nil on miss or any read error (cache is best-effort).
    static func load(url: URL) -> Entry? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        guard let (arrays, metadata) = try? loadArraysAndMetadata(url: url),
              let embeds = arrays["embeds"],
              let realTokens = metadata["real_tokens"].flatMap({ Int($0) })
        else { return nil }
        return Entry(embeds: embeds, realTokens: realTokens)
    }

    /// Best-effort write; failures are silent (cache is an optimization only).
    static func store(embeds: MLXArray, realTokens: Int, url: URL) {
        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try save(
                arrays: ["embeds": embeds],
                metadata: ["real_tokens": String(realTokens)],
                url: url
            )
        } catch {
            try? FileManager.default.removeItem(at: url)
        }
    }
}
