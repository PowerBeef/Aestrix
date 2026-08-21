import Foundation
import Testing
@testable import ImarelloDirect

@Suite("Direct V2 prompt bridge cache")
struct DirectPromptBridgeCacheTests {
    @Test("identity changes for every execution fingerprint field")
    func identityFingerprint() {
        let base = identity()
        let variants = [
            identity(model: "other"),
            identity(snapshot: "other"),
            identity(tokens: "other"),
            identity(backend: "other"),
            identity(plan: "other"),
            identity(shader: "other"),
        ]
        #expect(variants.allSatisfy { $0.fingerprint != base.fingerprint })
    }

    @Test("native BF16 bridge writes atomically and rejects another fingerprint")
    func bridgeRoundTrip() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("imarello-direct-cache-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let base = identity()
        let url = directory.appendingPathComponent("entry.safetensors")
        let bytes = Data(repeating: 0xA5, count: DirectPromptBridgeCache.byteCount)
        try DirectPromptBridgeCache.store(
            bytes: bytes, realTokens: 37, url: url, identity: base)
        let loaded = DirectPromptBridgeCache.load(url: url, identity: base)
        #expect(loaded?.realTokens == 37)
        #expect(loaded?.bytes == bytes)
        #expect(DirectPromptBridgeCache.load(
            url: url, identity: identity(plan: "different")) == nil)
    }

    private func identity(
        model: String = "model", snapshot: String = "snapshot",
        tokens: String = "tokens", backend: String = "backend",
        plan: String = "plan", shader: String = "shader"
    ) -> DirectPromptCacheIdentity {
        DirectPromptCacheIdentity(
            modelRevision: model,
            snapshotRevision: snapshot,
            tokenSemantics: tokens,
            backendIdentifier: backend,
            planDigest: plan,
            shaderManifestHash: shader)
    }
}
