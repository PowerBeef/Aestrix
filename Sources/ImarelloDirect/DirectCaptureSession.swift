import Foundation
import Metal
import MLX

public struct DirectCaptureRunMetadata: Codable, Sendable, Equatable {
    public var prompt: String
    public var seed: UInt64
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidance: Float
    public var tokenMode: String
    public var modelRevision: String
    public var snapshotRevision: String
    public var weightMode: String
    public var backendIdentifier: String
    public var mlxMetallibPath: String
    public var directShaderSHA256: String
    public var directMetallibSHA256: String
    public var deviceName: String
    public var operatingSystem: String
    public var thermalState: String
    public var outputPath: String
    public var planDigest: String?
}

public struct DirectCaptureEntry: Codable, Sendable, Equatable {
    public var id: String
    public var file: String
    public var dataType: String
    public var shape: [Int]
    public var byteCount: Int
}

public struct DirectCaptureManifest: Codable, Sendable, Equatable {
    public var schemaVersion = 1
    public var run: DirectCaptureRunMetadata
    public var entries: [DirectCaptureEntry]
}

/// Explicit, opt-in tensor capture. The product path does not construct this
/// object, so disabled instrumentation has no dispatch or allocation cost.
public final class DirectCaptureSession {
    public static let environmentKey = "IMARELLO_DIRECT_CAPTURE_DIR"

    public let directory: URL
    private var manifest: DirectCaptureManifest

    public init(directory: URL, metadata: DirectCaptureRunMetadata) throws {
        self.directory = directory
        self.manifest = DirectCaptureManifest(run: metadata, entries: [])
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
    }

    public static func fromEnvironment(
        metadata: DirectCaptureRunMetadata,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> DirectCaptureSession? {
        guard let path = environment[environmentKey], !path.isEmpty else { return nil }
        return try DirectCaptureSession(
            directory: URL(fileURLWithPath: path, isDirectory: true),
            metadata: metadata)
    }

    public func setPlanDigest(_ digest: String) {
        manifest.run.planDigest = digest
    }

    public func capture(
        id: String, data: Data, dataType: String, shape: [Int]
    ) throws {
        let stem = id.map { character -> Character in
            character.isLetter || character.isNumber || character == "-" ? character : "_"
        }
        let fileName = String(stem) + ".bin"
        let url = directory.appendingPathComponent(fileName)
        try data.write(to: url, options: [.atomic])
        manifest.entries.removeAll { $0.id == id }
        manifest.entries.append(
            DirectCaptureEntry(
                id: id, file: fileName, dataType: dataType,
                shape: shape, byteCount: data.count))
    }

    public func capture(id: String, array: MLXArray) throws {
        eval(array)
        try capture(
            id: id,
            data: array.asData(access: .copy).data,
            dataType: String(describing: array.dtype),
            shape: array.shape)
    }

    public func capture(
        id: String, buffer: MTLBuffer, byteCount: Int,
        dataType: String, shape: [Int]
    ) throws {
        guard byteCount >= 0, byteCount <= buffer.length else {
            throw DirectQmmSpike.SpikeError.invalidTensor(
                "capture \(id) requests \(byteCount) bytes from \(buffer.length)-byte buffer")
        }
        try capture(
            id: id,
            data: Data(bytes: buffer.contents(), count: byteCount),
            dataType: dataType,
            shape: shape)
    }

    public func finalize() throws {
        manifest.entries.sort { $0.id < $1.id }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(manifest).write(
            to: directory.appendingPathComponent("capture-manifest.json"),
            options: [.atomic])
    }
}
