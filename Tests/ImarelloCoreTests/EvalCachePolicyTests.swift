import Testing
@testable import ImarelloCore

@Suite("Eval cache policy")
struct EvalCachePolicyTests {
    @Test("product is the 8 GB-safe default")
    func productDefaults() {
        let p = EvalCachePolicy.product
        #expect(p.blockCacheClearInterval == 1)
        #expect(p.clearCacheAfterDenoiseStep)
        #expect(!p.requiresHighRAM)
        #expect(EvalCachePolicy.named("product") == p)
        #expect(EvalCachePolicy.named("high") == nil)
    }

    @Test("mid is a high-RAM bench profile")
    func midRequiresHighRAM() {
        #expect(EvalCachePolicy.mid.requiresHighRAM)
        #expect(EvalCachePolicy.mid.blockCacheClearInterval == 2)
        #expect(EvalCachePolicy.mid.denoiseCacheLimitBytes == 512 * 1_024 * 1_024)
        #expect(EvalCachePolicy.named("mid")?.profileName == "mid")
        #expect(EvalCachePolicy.named("low") == .product)
        #expect(EvalCachePolicy.named("bogus") == nil)
    }

    @Test("mid is refused on 8 GB unless --force")
    func midRefusedOnLowTier() {
        #expect(EvalCachePolicy.mid.refusalReason(tier: .low, force: false) != nil)
        #expect(EvalCachePolicy.mid.refusalReason(tier: .low, force: true) == nil)
        #expect(EvalCachePolicy.mid.refusalReason(tier: .mid, force: false) == nil)
        #expect(EvalCachePolicy.product.refusalReason(tier: .low, force: false) == nil)
    }
}
