import Testing
import Foundation
@testable import ImarelloRuntime

@Suite("Prompt embed cache keys")
struct PromptEmbedCacheKeyTests {
    @Test("different prompt, model, bits, or length produce different URLs")
    func keyChangesWithInputs() {
        let a = PromptEmbedCache.entryURL(prompt: "fox", modelID: "m", bits: 4, maxLength: 512)
        let b = PromptEmbedCache.entryURL(prompt: "mug", modelID: "m", bits: 4, maxLength: 512)
        let c = PromptEmbedCache.entryURL(prompt: "fox", modelID: "other", bits: 4, maxLength: 512)
        let d = PromptEmbedCache.entryURL(prompt: "fox", modelID: "m", bits: 3, maxLength: 512)
        let e = PromptEmbedCache.entryURL(prompt: "fox", modelID: "m", bits: 4, maxLength: 256)
        #expect(a != b)
        #expect(a != c)
        #expect(a != d)
        #expect(a != e)
        #expect(a.pathExtension == "safetensors")
    }

    @Test("same inputs produce the same URL")
    func keyStable() {
        let a = PromptEmbedCache.entryURL(prompt: "fox", modelID: "m", bits: 4, maxLength: 512)
        let b = PromptEmbedCache.entryURL(prompt: "fox", modelID: "m", bits: 4, maxLength: 512)
        #expect(a == b)
    }
}
