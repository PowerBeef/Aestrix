import CryptoKit
import Foundation
import Testing
@testable import ImarelloDirect

@Suite("Direct shader artifact contract")
struct DirectArtifactTests {
    @Test("matching hash, ABI, and complete symbol set is ready")
    func completeArtifact() {
        let data = artifactData(symbols: DirectKernelABI.requiredDirectSymbols)
        let manifest = manifest(for: data)
        let result = DirectArtifactVerification.verify(data: data, manifest: manifest)
        #expect(result.ready)
        #expect(result.symbolsMissing.isEmpty)
    }

    @Test("missing symbol and stale hash fail closed")
    func incompleteArtifact() {
        let symbols = Array(DirectKernelABI.requiredDirectSymbols.dropLast())
        let data = artifactData(symbols: symbols)
        var manifest = manifest(for: data)
        let missing = DirectKernelABI.requiredDirectSymbols.last!
        let incomplete = DirectArtifactVerification.verify(data: data, manifest: manifest)
        #expect(!incomplete.ready)
        #expect(incomplete.symbolsMissing == [missing])

        manifest.metallibSHA256 = String(repeating: "0", count: 64)
        let stale = DirectArtifactVerification.verify(
            data: artifactData(symbols: DirectKernelABI.requiredDirectSymbols),
            manifest: manifest)
        #expect(!stale.ready)
        #expect(stale.note.contains("SHA-256"))
    }

    @Test("manifest must cover the runtime ABI even when its own list is present")
    func manifestCoverage() {
        let data = artifactData(symbols: ["custom_only"])
        let manifest = DirectArtifactManifest(
            platform: "macosx", sdkVersion: "26.5", sourceRevision: "test",
            shaderSHA256: digest(Data("shader".utf8)),
            metallibSHA256: digest(data), requiredSymbols: ["custom_only"])
        let result = DirectArtifactVerification.verify(data: data, manifest: manifest)
        #expect(!result.ready)
        #expect(result.note.contains("omits"))
    }

    @Test("NAX attention inventory is optional and exact-shape specific")
    func naxInventory() {
        let compatibility = DirectKernelABI.requiredMLXSymbols.joined(separator: "\0")
        let withoutNAX = DirectDeviceCapabilities.inventory(
            mlxMetallibData: Data(compatibility.utf8))
        #expect(withoutNAX.requiredCompatibilitySymbolsPresent)
        #expect(withoutNAX.naxAttentionSymbolsPresent.isEmpty)

        let all = compatibility + "\0" + DirectKernelABI.naxAttentionSymbols.joined(separator: "\0")
        let withNAX = DirectDeviceCapabilities.inventory(mlxMetallibData: Data(all.utf8))
        #expect(withNAX.naxAttentionSymbolsMissing.isEmpty)
    }

    private func artifactData(symbols: [String]) -> Data {
        Data(("metallib\u{0}" + symbols.joined(separator: "\u{0}")).utf8)
    }

    private func manifest(for data: Data) -> DirectArtifactManifest {
        DirectArtifactManifest(
            platform: "macosx",
            sdkVersion: "26.5",
            sourceRevision: "test",
            shaderSHA256: digest(Data("shader".utf8)),
            metallibSHA256: digest(data))
    }

    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
