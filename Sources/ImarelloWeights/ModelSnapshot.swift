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
    public var expectedRevision: String?

    public init(modelID: String, root: URL, expectedRevision: String? = nil) {
        self.modelID = modelID
        self.root = root
        self.textEncoderDirectory = root.appendingPathComponent("text_encoder", isDirectory: true)
        self.transformerDirectory = root.appendingPathComponent("transformer", isDirectory: true)
        self.vaeDirectory = root.appendingPathComponent("vae", isDirectory: true)
        self.tokenizerDirectory = root.appendingPathComponent("tokenizer", isDirectory: true)
        self.expectedRevision = expectedRevision?.lowercased()
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

        let requiredFiles = [
            tokenizerDirectory.appendingPathComponent("tokenizer.json"),
            tokenizerDirectory.appendingPathComponent("tokenizer_config.json"),
            tokenizerDirectory.appendingPathComponent("chat_template.jinja"),
            textEncoderDirectory.appendingPathComponent("model.safetensors.index.json"),
            transformerDirectory.appendingPathComponent("model.safetensors.index.json"),
            vaeDirectory.appendingPathComponent("model.safetensors.index.json"),
        ]
        for url in requiredFiles {
            try Self.requireNonemptyFile(url, modelID: modelID)
        }
        for directory in [textEncoderDirectory, transformerDirectory, vaeDirectory] {
            try Self.validateShardIndex(directory: directory, modelID: modelID)
        }

        let revisions = detectedRevisions
        guard revisions.count <= 1 else {
            throw ImarelloError.unsupportedWeightFormat(
                "mixed snapshot revisions for \(modelID): \(revisions.sorted().joined(separator: ", "))")
        }
        if let expectedRevision {
            guard revisions == [expectedRevision] else {
                let found = revisions.isEmpty ? "none" : revisions.sorted().joined(separator: ", ")
                throw ImarelloError.unsupportedWeightFormat(
                    "snapshot revision mismatch for \(modelID): expected \(expectedRevision), found \(found)")
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

    /// Every commit SHA recorded by Hugging Face metadata. More than one value
    /// means files from different snapshots were mixed in the same directory.
    public var detectedRevisions: Set<String> {
        Self.revisionsFromHuggingFaceCache(root: root)
    }

    /// First line of Hugging Face `*.metadata` files is the git commit SHA.
    public static func revisionFromHuggingFaceCache(root: URL) -> String? {
        let revisions = revisionsFromHuggingFaceCache(root: root)
        return revisions.count == 1 ? revisions.first : nil
    }

    public static func revisionsFromHuggingFaceCache(root: URL) -> Set<String> {
        let download = root.appendingPathComponent(".cache/huggingface/download", isDirectory: true)
        guard let enumerator = FileManager.default.enumerator(
            at: download,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: []
        ) else { return [] }
        var revisions = Set<String>()
        while let item = enumerator.nextObject() as? URL {
            guard item.pathExtension == "metadata" else { continue }
            if let sha = parseRevision(fromMetadataFile: item) {
                revisions.insert(sha)
            }
        }
        return revisions
    }

    public static func parseRevision(fromMetadataFile url: URL) -> String? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        let first = text.split(whereSeparator: \.isNewline).first.map(String.init) ?? ""
        guard first.count == 40,
              first.unicodeScalars.allSatisfy({ CharacterSet(charactersIn: "0123456789abcdefABCDEF").contains($0) })
        else { return nil }
        return first.lowercased()
    }

    private struct SafetensorIndex: Decodable {
        var weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    private static func requireNonemptyFile(_ url: URL, modelID: String) throws {
        let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values?.isRegularFile == true, (values?.fileSize ?? 0) > 0 else {
            throw ImarelloError.weightsNotFound(modelID: modelID, path: url.path)
        }
    }

    private static func validateShardIndex(directory: URL, modelID: String) throws {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let data: Data
        do {
            data = try Data(contentsOf: indexURL)
        } catch {
            throw ImarelloError.weightsNotFound(modelID: modelID, path: indexURL.path)
        }
        let index: SafetensorIndex
        do {
            index = try JSONDecoder().decode(SafetensorIndex.self, from: data)
        } catch {
            throw ImarelloError.unsupportedWeightFormat(
                "invalid shard index at \(indexURL.path): \(error.localizedDescription)")
        }
        guard !index.weightMap.isEmpty else {
            throw ImarelloError.unsupportedWeightFormat("empty shard index at \(indexURL.path)")
        }
        for shard in Set(index.weightMap.values) {
            guard URL(fileURLWithPath: shard).lastPathComponent == shard else {
                throw ImarelloError.unsupportedWeightFormat(
                    "unsafe shard path '\(shard)' in \(indexURL.path)")
            }
            try requireNonemptyFile(directory.appendingPathComponent(shard), modelID: modelID)
        }
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
        let snap = ModelSnapshot(
            modelID: config.modelID, root: url, expectedRevision: config.revision)
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
        let configURL = root.appendingPathComponent("config.json")
        guard (try? file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])).map({
            $0.isRegularFile == true && ($0.fileSize ?? 0) > 0
        }) == true,
        (try? configURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])).map({
            $0.isRegularFile == true && ($0.fileSize ?? 0) > 0
        }) == true,
        ModelSnapshot.revisionsFromHuggingFaceCache(root: root)
            == Set([VAEDecoderVariant.smallDecoderPin.revision.lowercased()])
        else { return nil }
        return root
    }
}
