import Foundation
import AestrixCore

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
                throw AestrixError.weightsNotFound(modelID: modelID, path: url.path)
            }
        }
    }

    public var isPresent: Bool {
        (try? validateLayout()) != nil
    }
}

public enum ModelPaths {
    public static func defaultCacheRoot() -> URL {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return base.appendingPathComponent("Aestrix/models", isDirectory: true)
    }

    /// Maps `org/name` → local cache directory (snapshot download target).
    public static func snapshotRoot(modelID: String, modelsDirectory: URL?) -> URL {
        let root = modelsDirectory ?? defaultCacheRoot()
        let safe = modelID.replacingOccurrences(of: "/", with: "--")
        return root.appendingPathComponent(safe, isDirectory: true)
    }

    public static func resolveIfPresent(config: AestrixConfig) -> ModelSnapshot? {
        let url = snapshotRoot(modelID: config.modelID, modelsDirectory: config.modelsDirectory)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            return nil
        }
        let snap = ModelSnapshot(modelID: config.modelID, root: url)
        return snap.isPresent ? snap : nil
    }

    public static func resolveOrThrow(config: AestrixConfig) throws -> ModelSnapshot {
        if let snap = resolveIfPresent(config: config) {
            return snap
        }
        let expected = snapshotRoot(modelID: config.modelID, modelsDirectory: config.modelsDirectory)
        throw AestrixError.weightsNotFound(modelID: config.modelID, path: expected.path)
    }
}
