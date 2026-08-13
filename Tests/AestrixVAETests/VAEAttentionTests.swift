import Testing
import Foundation
import MLX
@testable import AestrixVAE
import AestrixCore

@Suite("VAE D=512 attention")
struct VAEAttentionTests {
    @Test("product defaults never eval or clear per chunk")
    func defaultsAreSafe() {
        let cfg = VAEAttentionConfig()
        #expect(cfg.queryChunkSize == 64)
        #expect(cfg.evalEachChunk == false)
        #expect(cfg.clearCacheEachChunk == false)
        #expect(cfg.useMLXFast == false)
        #expect(VAEAttention.scoreBytes(queryChunk: 64, keySeq: 4096) == 64 * 4096 * 4)
    }
}

@Suite("VAE attention numerics", .enabled(if: ProcessInfo.processInfo.environment["AESTRIX_MLX_TESTS"] == "1"))
struct VAEAttentionNumericTests {
    @Test("chunked SDPA matches one-shot oracle on a tiny tensor")
    func chunkedMatchesOracle() {
        let saved = VAEAttentionConfig.current
        defer { VAEAttentionConfig.current = saved }
        VAEAttentionConfig.current = VAEAttentionConfig(queryChunkSize: 3, evalEachChunk: false)

        let q = MLXRandom.normal([1, 1, 8, 4])
        let k = MLXRandom.normal([1, 1, 8, 4])
        let v = MLXRandom.normal([1, 1, 8, 4])
        let scale: Float = 0.5
        let chunked = VAEAttention.scaledDotProductAttention(query: q, key: k, value: v, scale: scale)
        let oracle = VAEAttention.referenceSDPA(query: q, key: k, value: v, scale: scale)
        eval(chunked, oracle)
        let diff = abs(chunked - oracle)
        eval(diff)
        #expect(diff.max().item(Float.self) < 1e-4)
    }
}
