import CryptoKit
import Foundation
import ImarelloCore

public struct DirectArtifactManifest: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var abiVersion: Int
    public var platform: String
    public var sdkVersion: String
    public var minimumOS: String
    public var sourceRevision: String
    public var shaderSHA256: String
    public var metallibSHA256: String
    public var requiredSymbols: [String]
    public var functionConstants: [String: Int]

    public init(
        schemaVersion: Int = 1,
        abiVersion: Int = DirectKernelABI.version,
        platform: String,
        sdkVersion: String,
        minimumOS: String = "26.2",
        sourceRevision: String,
        shaderSHA256: String,
        metallibSHA256: String,
        requiredSymbols: [String] = DirectKernelABI.requiredDirectSymbols,
        functionConstants: [String: Int] = DirectKernelABI.requiredFunctionConstants
    ) {
        self.schemaVersion = schemaVersion
        self.abiVersion = abiVersion
        self.platform = platform
        self.sdkVersion = sdkVersion
        self.minimumOS = minimumOS
        self.sourceRevision = sourceRevision
        self.shaderSHA256 = shaderSHA256.lowercased()
        self.metallibSHA256 = metallibSHA256.lowercased()
        self.requiredSymbols = requiredSymbols.sorted()
        self.functionConstants = functionConstants
    }

    public static func load(url: URL) throws -> DirectArtifactManifest {
        do {
            return try JSONDecoder().decode(
                DirectArtifactManifest.self, from: Data(contentsOf: url))
        } catch {
            throw ImarelloError.metallibNotReady(
                "invalid Direct manifest at \(url.path): \(error.localizedDescription)")
        }
    }
}

public struct DirectArtifactVerification: Sendable, Equatable {
    public var byteCount: Int
    public var sha256: String
    public var symbolsPresent: [String]
    public var symbolsMissing: [String]
    public var ready: Bool
    public var note: String

    public static func verify(
        data: Data, manifest: DirectArtifactManifest
    ) -> DirectArtifactVerification {
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let present = manifest.requiredSymbols.filter { symbol in
            data.range(of: Data(symbol.utf8)) != nil
        }
        let missing = manifest.requiredSymbols.filter { !present.contains($0) }
        let manifestCoversABI = Set(manifest.requiredSymbols)
            .isSuperset(of: DirectKernelABI.requiredDirectSymbols)
        let constantsMatchABI = manifest.functionConstants == DirectKernelABI.requiredFunctionConstants
        let sourceHashIsValid = manifest.shaderSHA256.count == 64
            && manifest.shaderSHA256.allSatisfy(\.isHexDigit)
        let ready = manifest.schemaVersion == 1
            && manifest.abiVersion == DirectKernelABI.version
            && manifestCoversABI
            && constantsMatchABI
            && sourceHashIsValid
            && !data.isEmpty
            && digest == manifest.metallibSHA256
            && missing.isEmpty
        let note: String
        if manifest.abiVersion != DirectKernelABI.version {
            note = "Direct ABI mismatch: manifest \(manifest.abiVersion), runtime \(DirectKernelABI.version)"
        } else if !manifestCoversABI {
            note = "Direct manifest omits required ABI symbols"
        } else if !constantsMatchABI {
            note = "Direct manifest function constants do not match the runtime ABI"
        } else if !sourceHashIsValid {
            note = "Direct shader source SHA-256 is invalid"
        } else if digest != manifest.metallibSHA256 {
            note = "Direct metallib SHA-256 mismatch"
        } else if !missing.isEmpty {
            note = "Direct symbols missing: \(missing.joined(separator: ", "))"
        } else if data.isEmpty {
            note = "Direct metallib is empty"
        } else {
            note = "Direct metallib and ABI manifest are ready"
        }
        return DirectArtifactVerification(
            byteCount: data.count,
            sha256: digest,
            symbolsPresent: present,
            symbolsMissing: missing,
            ready: ready,
            note: note)
    }

    public static func verify(
        metallibURL: URL, manifestURL: URL
    ) -> DirectArtifactVerification {
        guard let data = try? Data(contentsOf: metallibURL),
              let manifest = try? DirectArtifactManifest.load(url: manifestURL)
        else {
            return DirectArtifactVerification(
                byteCount: 0, sha256: "", symbolsPresent: [],
                symbolsMissing: DirectKernelABI.requiredDirectSymbols,
                ready: false,
                note: "Direct metallib or manifest is missing")
        }
        return verify(data: data, manifest: manifest)
    }
}

public struct DirectEngineArtifacts: Sendable, Equatable {
    public var mlxMetallibURL: URL
    public var directMetallibURL: URL
    public var directManifestURL: URL
    public var directManifest: DirectArtifactManifest

    public init(
        mlxMetallibURL: URL,
        directMetallibURL: URL,
        directManifestURL: URL
    ) throws {
        guard let mlxData = try? Data(contentsOf: mlxMetallibURL, options: [.mappedIfSafe]) else {
            throw ImarelloError.metallibNotReady("full MLX metallib is unreadable")
        }
        let mlx = MetallibVerification.verify(data: mlxData, path: mlxMetallibURL.path)
        guard mlx.productReady else {
            throw ImarelloError.metallibNotReady(mlx.note)
        }
        let mlxInventory = DirectDeviceCapabilities.inventory(mlxMetallibData: mlxData)
        guard mlxInventory.requiredCompatibilitySymbolsPresent else {
            throw ImarelloError.metallibNotReady(
                "full MLX metallib is missing Direct compatibility symbols: "
                    + mlxInventory.missingCompatibilitySymbols.joined(separator: ", "))
        }
        let manifest = try DirectArtifactManifest.load(url: directManifestURL)
        let direct = DirectArtifactVerification.verify(
            metallibURL: directMetallibURL, manifestURL: directManifestURL)
        guard direct.ready else {
            throw ImarelloError.metallibNotReady(direct.note)
        }
        self.mlxMetallibURL = mlxMetallibURL
        self.directMetallibURL = directMetallibURL
        self.directManifestURL = directManifestURL
        self.directManifest = manifest
    }

    public static func resolve(relativeTo executable: URL) throws -> DirectEngineArtifacts {
        let directory = executable.resolvingSymlinksInPath().deletingLastPathComponent()
        guard let mlx = MetallibVerification.resolveExisting(relativeTo: executable) else {
            throw ImarelloError.metallibNotReady(
                "full no-JIT MLX metallib not found next to \(executable.path)")
        }
        return try DirectEngineArtifacts(
            mlxMetallibURL: mlx,
            directMetallibURL: directory.appendingPathComponent("imarello-direct.metallib"),
            directManifestURL: directory.appendingPathComponent(
                "imarello-direct-manifest.json"))
    }

    static func directMetallibURL(beside mlxMetallibURL: URL) -> URL {
        mlxMetallibURL.deletingLastPathComponent()
            .appendingPathComponent("imarello-direct.metallib")
    }
}
