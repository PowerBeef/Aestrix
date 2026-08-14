import Foundation
import ImarelloCore

/// Resolved local paths for a pre-quant Hugging Face snapshot.
public struct ModelSnapshot: Sendable {
    public var modelID: String
    public var root: URL
    public var textEncoderDirectory: URL
    public var transformerDirectory: URL
    public var vaeDirectory: URL
    public var tokenizerDirectory: URL

    public init(modelID: String, root: URL) {
        self.modelID = modelID
        self.root = root
        self.textEncoderDirectory = root.appendingPathComponent("text_encoder", isDirectory: true)
        self.transformerDirectory = root.appendingPathComponent("transformer", isDirectory: true)
        self.vaeDirectory = root.appendingPathComponent("vae", isDirectory: true)
        self.tokenizerDirectory = root.appendingPathComponent("tokenizer", isDirectory: true)
    }

    public func validateLayout() throws {
        let fm = FileManager.default
        for (_, url) in [
            ("text_encoder", textEncoderDirectory),
            ("transformer", transformerDirectory),
            ("vae", vaeDirectory),
            ("tokenizer", tokenizerDirectory),
        ] {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
                throw ImarelloError.weightsNotFound(modelID: modelID, path: url.path)
            }
        }
    }

    public var isPresent: Bool {
        (try? validateLayout()) != nil
    }

    /// Commit SHA from `hf download` metadata, if the snapshot was fetched that way.
    public var detectedRevision: String? {
        Self.revisionFromHuggingFaceCache(root: root)
    }

    /// First line of Hugging Face `*.metadata` files is the git commit SHA.
    public static func revisionFromHuggingFaceCache(root: URL) -> String? {
        let download = root.appendingPathComponent(".cache/huggingface/download", isDirectory: true)
        guard let enumerator = FileManager.default.enumerator(
            at: download,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return nil }
        while let item = enumerator.nextObject() as? URL {
            guard item.pathExtension == "metadata" else { continue }
            if let sha = parseRevision(fromMetadataFile: item) {
                return sha
            }
        }
        return nil
    }

    public static func parseRevision(fromMetadataFile url: URL) -> String? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        let first = text.split(whereSeparator: \.isNewline).first.map(String.init) ?? ""
        guard first.count == 40,
              first.unicodeScalars.allSatisfy({ CharacterSet(charactersIn: "0123456789abcdefABCDEF").contains($0) })
        else { return nil }
        return first.lowercased()
    }
}

public enum ModelPaths {
    public static func defaultCacheRoot() -> URL {
        AppCache.resolvedDirectory("models")
    }

    /// Maps `org/name` → local cache directory (snapshot download target).
    public static func snapshotRoot(modelID: String, modelsDirectory: URL?) -> URL {
        let safe = modelID.replacingOccurrences(of: "/", with: "--")
        if let modelsDirectory {
            return modelsDirectory.appendingPathComponent(safe, isDirectory: true)
        }
        return AppCache.resolvedItem(under: "models", item: safe)
    }

    public static func resolveIfPresent(config: ImarelloConfig) -> ModelSnapshot? {
        let url = snapshotRoot(modelID: config.modelID, modelsDirectory: config.modelsDirectory)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            return nil
        }
        let snap = ModelSnapshot(modelID: config.modelID, root: url)
        return snap.isPresent ? snap : nil
    }

    public static func resolveOrThrow(config: ImarelloConfig) throws -> ModelSnapshot {
        if let snap = resolveIfPresent(config: config) {
            return snap
        }
        let expected = snapshotRoot(modelID: config.modelID, modelsDirectory: config.modelsDirectory)
        throw ImarelloError.weightsNotFound(modelID: config.modelID, path: expected.path)
    }

    public static func smallDecoderSnapshotRoot(modelsDirectory: URL?) -> URL {
        snapshotRoot(
            modelID: VAEDecoderVariant.smallDecoderPin.modelID,
            modelsDirectory: modelsDirectory)
    }

    /// Directory that contains `small_decoder.safetensors`, if present.
    public static func resolveSmallDecoderIfPresent(config: ImarelloConfig) -> URL? {
        let root = smallDecoderSnapshotRoot(modelsDirectory: config.modelsDirectory)
        let file = root.appendingPathComponent(VAEDecoderVariant.smallDecoderFileName)
        return FileManager.default.fileExists(atPath: file.path) ? root : nil
    }
}
