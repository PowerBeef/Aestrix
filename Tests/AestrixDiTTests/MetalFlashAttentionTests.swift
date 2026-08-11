import Testing
import Foundation
import MLX
@testable import AestrixDiT

@Suite("Metal FlashAttention")
struct MetalFlashAttentionTests {

    /// Reference SDPA: softmax(scale * Q K^T) V for shapes [B,H,S,D].
    private func referenceSDPA(
        q: MLXArray, k: MLXArray, v: MLXArray, scale: Float
    ) -> MLXArray {
        // scores: [B,H,Sq,Sk]
        var scores = matmul(q.asType(.float32) * scale, k.asType(.float32).transposed(0, 1, 3, 2))
        scores = softmax(scores, axis: -1)
        return matmul(scores, v.asType(.float32))
    }

    @Test("Metal FA matches reference SDPA (small f32)")
    func paritySmallF32() {
        MLXRandom.seed(0)
        let B = 1, H = 2, S = 48, D = 32
        let q = MLXRandom.normal([B, H, S, D]).asType(.float32)
        let k = MLXRandom.normal([B, H, S, D]).asType(.float32)
        let v = MLXRandom.normal([B, H, S, D]).asType(.float32)
        let scale = 1.0 / sqrt(Float(D))
        eval(q, k, v)

        let ref = referenceSDPA(q: q, k: k, v: v, scale: scale)
        let fa = MetalFlashAttention.scaledDotProductAttention(
            query: q, key: k, value: v, scale: scale
        )
        eval(ref, fa)

        let diff = abs(ref - fa.asType(.float32))
        let maxDiff = diff.max().item(Float.self)
        let meanDiff = mean(diff).item(Float.self)
        #expect(maxDiff < 2e-3, "max abs err \(maxDiff)")
        #expect(meanDiff < 2e-4, "mean abs err \(meanDiff)")
    }

    @Test("Metal FA matches reference for FLUX head dim D=128")
    func parityFluxHeadDim() {
        MLXRandom.seed(1)
        let B = 1, H = 2, S = 64, D = 128
        let q = MLXRandom.normal([B, H, S, D]).asType(.float32) * 0.1
        let k = MLXRandom.normal([B, H, S, D]).asType(.float32) * 0.1
        let v = MLXRandom.normal([B, H, S, D]).asType(.float32) * 0.1
        let scale = 1.0 / sqrt(Float(D))
        eval(q, k, v)

        let ref = referenceSDPA(q: q, k: k, v: v, scale: scale)
        let fa = MetalFlashAttention.scaledDotProductAttention(
            query: q, key: k, value: v, scale: scale
        )
        eval(ref, fa)

        let maxDiff = abs(ref - fa).max().item(Float.self)
        #expect(maxDiff < 5e-3, "max abs err \(maxDiff) at D=128")
    }

    @Test("Metal FA f16 path runs and is finite")
    func f16Path() {
        MLXRandom.seed(2)
        let B = 1, H = 1, S = 32, D = 64
        let q = MLXRandom.normal([B, H, S, D]).asType(.float16)
        let k = MLXRandom.normal([B, H, S, D]).asType(.float16)
        let v = MLXRandom.normal([B, H, S, D]).asType(.float16)
        let scale = 1.0 / sqrt(Float(D))
        eval(q, k, v)

        let fa = MetalFlashAttention.scaledDotProductAttention(
            query: q, key: k, value: v, scale: scale
        )
        eval(fa)
        let maxAbs = abs(fa.asType(.float32)).max().item(Float.self)
        #expect(maxAbs.isFinite)
        #expect(fa.dtype == .float16)
    }

    @Test("backend metal-fa path via computeAttention")
    func backendMetalFA() {
        let prev = AttentionTuning.current
        defer { AttentionTuning.current = prev }

        var t = AttentionTuning.default
        t.backend = .metalFA
        AttentionTuning.current = t

        MLXRandom.seed(3)
        let B = 1, H = 1, S = 128, D = 32
        let q = MLXRandom.normal([B, H, S, D]).asType(.float32)
        let k = MLXRandom.normal([B, H, S, D]).asType(.float32)
        let v = MLXRandom.normal([B, H, S, D]).asType(.float32)
        eval(q, k, v)

        let out = AttentionUtils.computeAttention(
            query: q, key: k, value: v,
            batchSize: B, numHeads: H, headDim: D
        )
        eval(out)
        #expect(out.shape == [B, S, H * D])
    }

    @Test("default backend is mlx (safe product path)")
    func defaultBackendIsMLX() {
        #expect(AttentionTuning.default.backend == .mlx)
    }

    @Test("multi-tile hybrid FA2 matches reference")
    func multiTileParity() {
        MLXRandom.seed(4)
        // D=32 is outside Steel set → hybrid path.
        let B = 1, H = 2, S = 96, D = 32
        let q = MLXRandom.normal([B, H, S, D]).asType(.float32) * 0.05
        let k = MLXRandom.normal([B, H, S, D]).asType(.float32) * 0.05
        let v = MLXRandom.normal([B, H, S, D]).asType(.float32) * 0.05
        let scale = 1.0 / sqrt(Float(D))
        eval(q, k, v)

        let ref = referenceSDPA(q: q, k: k, v: v, scale: scale)
        let fa = MetalFlashAttention.hybridBlockGEMM(
            query: q, key: k, value: v, scale: scale, blockC: 32
        )
        eval(ref, fa)
        let maxDiff = abs(ref - fa).max().item(Float.self)
        #expect(maxDiff < 5e-3, "multi-tile max abs err \(maxDiff)")
    }

    @Test("fused Metal float4 FA matches reference")
    func fusedMetalParity() {
        MLXRandom.seed(5)
        let B = 1, H = 2, S = 48, D = 32
        let q = MLXRandom.normal([B, H, S, D]).asType(.float32) * 0.1
        let k = MLXRandom.normal([B, H, S, D]).asType(.float32) * 0.1
        let v = MLXRandom.normal([B, H, S, D]).asType(.float32) * 0.1
        let scale = 1.0 / sqrt(Float(D))
        eval(q, k, v)
        let ref = referenceSDPA(q: q, k: k, v: v, scale: scale)
        let fa = MetalFlashAttention.scaledDotProductAttentionFusedMetal(
            query: q, key: k, value: v, scale: scale
        )
        eval(ref, fa)
        let maxDiff = abs(ref - fa).max().item(Float.self)
        #expect(maxDiff < 5e-3, "fused metal max abs err \(maxDiff)")
    }

    @Test("Steel head dim 128 path matches reference")
    func steelPathD128() {
        MLXRandom.seed(6)
        let B = 1, H = 2, S = 64, D = 128
        let q = MLXRandom.normal([B, H, S, D]).asType(.float32) * 0.05
        let k = MLXRandom.normal([B, H, S, D]).asType(.float32) * 0.05
        let v = MLXRandom.normal([B, H, S, D]).asType(.float32) * 0.05
        let scale = 1.0 / sqrt(Float(D))
        eval(q, k, v)
        let ref = referenceSDPA(q: q, k: k, v: v, scale: scale)
        let fa = MetalFlashAttention.scaledDotProductAttention(
            query: q, key: k, value: v, scale: scale
        )
        eval(ref, fa)
        let maxDiff = abs(ref - fa).max().item(Float.self)
        #expect(maxDiff < 1e-2, "steel path max abs err \(maxDiff)")
    }
}
