import CryptoKit
import Foundation
import ImarelloCore

public struct DirectPromptCacheIdentity: Codable, Sendable, Equatable {
    public static let formatVersion = 1

    public var modelRevision: String
    public var snapshotRevision: String
    public var tokenSemantics: String
    public var backendIdentifier: String
    public var planDigest: String
    public var shaderManifestHash: String

    public init(
        modelRevision: String,
        snapshotRevision: String,
        tokenSemantics: String,
        backendIdentifier: String,
        planDigest: String,
        shaderManifestHash: String
    ) {
        self.modelRevision = modelRevision
        self.snapshotRevision = snapshotRevision
        self.tokenSemantics = tokenSemantics
        self.backendIdentifier = backendIdentifier
        self.planDigest = planDigest
        self.shaderManifestHash = shaderManifestHash
    }

    public var fingerprint: String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let data = (try? encoder.encode(self)) ?? Data()
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    public static func tokenSemantics(for request: T2IRequest) -> String {
        "tokens=\(request.textTokens.rawValue)|pad=\(request.padContent.rawValue)"
            + "|keep=\(request.padKeep.map(String.init) ?? "none")"
            + "|bias=\(request.padBias)"
    }
}

/// MLX-independent BF16 bridge cache for `[1, 512, 7680]` TE output.
/// This namespace never resolves legacy `embeds/` entries.
public enum DirectPromptBridgeCache {
    public static let tensorName = "embeds"
    public static let shape = [1, ModelConstants.maxSequenceLength, ModelConstants.jointAttentionDim]
    public static let byteCount = shape.reduce(2, *)

    public struct Entry: Sendable, Equatable {
        public var bytes: Data
        public var realTokens: Int
    }

    public static func entryURL(prompt: String, identity: DirectPromptCacheIdentity) -> URL {
        let key = "v\(DirectPromptCacheIdentity.formatVersion)|\(identity.fingerprint)|\(prompt)"
        let digest = SHA256.hash(data: Data(key.utf8))
            .map { String(format: "%02x", $0) }.joined()
        return AppCache.directory("direct-embeds-v2")
            .appendingPathComponent("\(digest).safetensors")
    }

    public static func load(
        url: URL, identity: DirectPromptCacheIdentity
    ) -> Entry? {
        guard let data = try? Data(contentsOf: url, options: [.mappedIfSafe]),
              let index = try? SafetensorIndex.parse(
                data: data,
                allowedTensorNames: [tensorName],
                requiredTensorNames: [tensorName]),
              let descriptor = index.tensorsByName[tensorName],
              descriptor.dataType == .bfloat16,
              descriptor.shape == shape,
              let metadata = metadata(in: data, payloadOffset: index.payloadOffset),
              metadata["fingerprint"] == identity.fingerprint,
              let tokenString = metadata["real_tokens"],
              let realTokens = Int(tokenString),
              (1 ... ModelConstants.maxSequenceLength).contains(realTokens)
        else { return nil }
        return Entry(
            bytes: data.subdata(in: descriptor.byteRange),
            realTokens: realTokens)
    }

    public static func store(
        bytes: Data,
        realTokens: Int,
        url: URL,
        identity: DirectPromptCacheIdentity
    ) throws {
        guard bytes.count == byteCount,
              (1 ... ModelConstants.maxSequenceLength).contains(realTokens)
        else { throw SafetensorError.byteCountMismatch(
            name: tensorName, expected: byteCount, actual: bytes.count) }

        let header: [String: Any] = [
            "__metadata__": [
                "backend": identity.backendIdentifier,
                "fingerprint": identity.fingerprint,
                "real_tokens": String(realTokens),
            ],
            tensorName: [
                "data_offsets": [0, byteCount],
                "dtype": SafetensorDataType.bfloat16.rawValue,
                "shape": shape,
            ],
        ]
        var headerData = try JSONSerialization.data(
            withJSONObject: header, options: [.sortedKeys])
        while !headerData.count.isMultiple(of: 8) { headerData.append(0x20) }
        var headerLength = UInt64(headerData.count).littleEndian
        var file = Data(bytes: &headerLength, count: MemoryLayout<UInt64>.size)
        file.append(headerData)
        file.append(bytes)

        let manager = FileManager.default
        let directory = url.deletingLastPathComponent()
        try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        let temporary = directory.appendingPathComponent(".tmp-\(UUID().uuidString).safetensors")
        do {
            try file.write(to: temporary, options: [.atomic])
            if manager.fileExists(atPath: url.path) {
                _ = try manager.replaceItemAt(url, withItemAt: temporary)
            } else {
                try manager.moveItem(at: temporary, to: url)
            }
        } catch {
            try? manager.removeItem(at: temporary)
            throw error
        }
    }

    private static func metadata(
        in data: Data, payloadOffset: Int
    ) -> [String: String]? {
        guard payloadOffset >= 8,
              let object = try? JSONSerialization.jsonObject(
                with: data.subdata(in: 8 ..< payloadOffset)) as? [String: Any],
              let metadata = object["__metadata__"] as? [String: String]
        else { return nil }
        return metadata
    }
}
