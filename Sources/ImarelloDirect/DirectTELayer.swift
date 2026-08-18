import Foundation
import Metal
import MLX
import MLXNN
import MLXFast

/// Stage-2 Milestone B: one full Qwen3 decoder layer (layer 0, M=512) executed
/// as a SINGLE Metal command buffer — qmm + steel attention from MLX's
/// metallib, glue from `DirectGlueKernels` — verified stage-by-stage against
/// an MLX oracle running identical f16 math.
public enum DirectTELayerSpike {

    static let L = 512
    static let hidden = 2560
    static let nHeads = 32
    static let nKV = 8
    static let headDim = 128
    static let inter = 9728
    static let eps: Float = 1e-6
    static let ropeBase: Float = 1_000_000

    struct QuantWeights {
        let w: MLXArray
        let scales: MLXArray
        let biases: MLXArray
        let n: Int
        let k: Int
    }

    // MARK: - Context

    final class Ctx {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let qmmPSO: MTLComputePipelineState
        let attnPSO: MTLComputePipelineState
        let rmsPSO: MTLComputePipelineState
        let ropePSO: MTLComputePipelineState
        let siluMulPSO: MTLComputePipelineState
        let addPSO: MTLComputePipelineState

        init(metallibURL: URL) throws {
            guard let dev = MTLCreateSystemDefaultDevice() else {
                throw DirectQmmSpike.SpikeError.metal("no device")
            }
            device = dev
            guard let q = dev.makeCommandQueue() else {
                throw DirectQmmSpike.SpikeError.metal("no queue")
            }
            queue = q
            let mlxLib = try dev.makeLibrary(URL: metallibURL)
            func fn(_ name: String) throws -> MTLFunction {
                guard let f = mlxLib.makeFunction(name: name) else {
                    throw DirectQmmSpike.SpikeError.metal("missing kernel \(name)")
                }
                return f
            }
            qmmPSO = try dev.makeComputePipelineState(
                function: fn("affine_qmm_t_float16_t_gs_64_b_4_alN_true_batch_0"))

            // Steel attention: specialized via function constants.
            let consts = MTLFunctionConstantValues()
            var t = true, f = false
            consts.setConstantValue(&t, type: .bool, index: 200)  // align_Q
            consts.setConstantValue(&t, type: .bool, index: 201)  // align_K
            consts.setConstantValue(&f, type: .bool, index: 300)  // has_mask
            consts.setConstantValue(&t, type: .bool, index: 301)  // do_causal
            consts.setConstantValue(&f, type: .bool, index: 302)  // has_sinks
            let attnFn = try mlxLib.makeFunction(
                name: "steel_attention_float16_bq32_bk16_bd128_wm4_wn1_maskfloat16",
                constantValues: consts)
            attnPSO = try dev.makeComputePipelineState(function: attnFn)

            let glue = try DirectGlueKernels.makeLibrary(device: dev)
            func gfn(_ name: String) throws -> MTLComputePipelineState {
                guard let f = glue.makeFunction(name: name) else {
                    throw DirectQmmSpike.SpikeError.metal("missing glue \(name)")
                }
                return try dev.makeComputePipelineState(function: f)
            }
            rmsPSO = try gfn("dq_rmsnorm")
            ropePSO = try gfn("dq_rope")
            siluMulPSO = try gfn("dq_silu_mul")
            addPSO = try gfn("dq_add")
        }

        func upload(_ array: MLXArray, _ label: String) throws -> MTLBuffer {
            let data = array.asData(noCopy: false)
            return try data.withUnsafeBytes { raw -> MTLBuffer in
                guard let base = raw.baseAddress,
                    let b = device.makeBuffer(bytes: base, length: raw.count)
                else { throw DirectQmmSpike.SpikeError.metal("upload \(label)") }
                b.label = label
                return b
            }
        }

        func scratch(_ halfCount: Int, _ label: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: halfCount * 2) else {
                throw DirectQmmSpike.SpikeError.metal("scratch \(label)")
            }
            b.label = label
            return b
        }
    }

    // MARK: - Encoding helpers

    static func encodeQmm(
        _ enc: MTLComputeCommandEncoder, _ ctx: Ctx,
        w: MTLBuffer, s: MTLBuffer, b: MTLBuffer,
        x: MTLBuffer, y: MTLBuffer, m: Int, n: Int, k: Int,
        pso: MTLComputePipelineState? = nil
    ) {
        enc.setComputePipelineState(pso ?? ctx.qmmPSO)
        enc.setBuffer(w, offset: 0, index: 0)
        enc.setBuffer(s, offset: 0, index: 1)
        enc.setBuffer(b, offset: 0, index: 2)
        enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(y, offset: 0, index: 4)
        var k32 = Int32(k), n32 = Int32(n), m32 = Int32(m)
        enc.setBytes(&k32, length: 4, index: 5)
        enc.setBytes(&n32, length: 4, index: 6)
        enc.setBytes(&m32, length: 4, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: (n + 31) / 32, height: (m + 31) / 32, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
    }

    static func encodeRms(
        _ enc: MTLComputeCommandEncoder, _ ctx: Ctx,
        x: MTLBuffer, w: MTLBuffer, y: MTLBuffer, rows: Int, d: Int,
        pso: MTLComputePipelineState? = nil
    ) {
        enc.setComputePipelineState(pso ?? ctx.rmsPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(w, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var d32 = Int32(d)
        var e = eps
        enc.setBytes(&d32, length: 4, index: 3)
        enc.setBytes(&e, length: 4, index: 4)
        let tp = min(256, max(32, (d + 3) / 4))
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tp, height: 1, depth: 1))
    }

    static func encodeRopeL(
        _ enc: MTLComputeCommandEncoder, _ ctx: Ctx,
        x: MTLBuffer, y: MTLBuffer, heads: Int, seqLen: Int,
        pso: MTLComputePipelineState? = nil
    ) {
        enc.setComputePipelineState(pso ?? ctx.ropePSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(y, offset: 0, index: 1)
        var h32 = Int32(heads), d32 = Int32(headDim)
        var base = ropeBase
        enc.setBytes(&h32, length: 4, index: 2)
        enc.setBytes(&d32, length: 4, index: 3)
        enc.setBytes(&base, length: 4, index: 4)
        enc.dispatchThreads(
            MTLSize(width: headDim / 2, height: heads, depth: seqLen),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    static func encodeRope(
        _ enc: MTLComputeCommandEncoder, _ ctx: Ctx,
        x: MTLBuffer, y: MTLBuffer, heads: Int
    ) {
        enc.setComputePipelineState(ctx.ropePSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(y, offset: 0, index: 1)
        var h32 = Int32(heads), d32 = Int32(headDim)
        var base = ropeBase
        enc.setBytes(&h32, length: 4, index: 2)
        enc.setBytes(&d32, length: 4, index: 3)
        enc.setBytes(&base, length: 4, index: 4)
        enc.dispatchThreads(
            MTLSize(width: headDim / 2, height: heads, depth: L),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    static func encodeAttention(
        _ enc: MTLComputeCommandEncoder, _ ctx: Ctx,
        q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, o: MTLBuffer
    ) {
        enc.setComputePipelineState(ctx.attnPSO)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(k, offset: 0, index: 1)
        enc.setBuffer(v, offset: 0, index: 2)
        enc.setBuffer(o, offset: 0, index: 3)

        // AttnParams, exact C++ layout: 14 int32 + float, pad to 64, 4×3 int64.
        let bq = 32, bk = 16
        let nq = (L + bq - 1) / bq, nk = (L + bk - 1) / bk
        var data = Data(capacity: 160)
        func i32(_ v: Int) { var x = Int32(v); withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func f32(_ v: Float) { var x = v; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func i64x3(_ a: Int, _ b: Int, _ c: Int) {
            for v in [Int64(a), Int64(b), Int64(c)] {
                var x = v; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
            }
        }
        i32(1)                    // B
        i32(nHeads)               // H
        i32(headDim)              // D
        i32(L)                    // qL
        i32(L)                    // kL
        i32(nHeads / nKV)         // gqa_factor
        f32(1.0 / Float(Double(headDim).squareRoot()))  // scale
        i32(nq)                   // NQ
        i32(nk)                   // NK
        i32(L / bq)               // NQ_aligned
        i32(L / bk)               // NK_aligned
        i32(L - (L / bq) * bq)    // qL_rem
        i32(L - (L / bk) * bk)    // kL_rem
        i32(0)                    // qL_off = kL - qL (56 bytes so far — already 8-aligned)
        // Views of [L, H*D] as [B, H, L, D]:
        i64x3(L * nHeads * headDim, headDim, nHeads * headDim)  // Q
        i64x3(L * nKV * headDim, headDim, nKV * headDim)        // K
        i64x3(L * nKV * headDim, headDim, nKV * headDim)        // V
        i64x3(L * nHeads * headDim, headDim, nHeads * headDim)  // O
        data.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: data.count, index: 4) }

        enc.dispatchThreadgroups(
            MTLSize(width: nq, height: nHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
    }

    static func encodeSiluMul(
        _ enc: MTLComputeCommandEncoder, _ ctx: Ctx,
        g: MTLBuffer, u: MTLBuffer, y: MTLBuffer, n: Int,
        pso: MTLComputePipelineState? = nil
    ) {
        enc.setComputePipelineState(pso ?? ctx.siluMulPSO)
        enc.setBuffer(g, offset: 0, index: 0)
        enc.setBuffer(u, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var n32 = UInt32(n)
        enc.setBytes(&n32, length: 4, index: 3)
        enc.dispatchThreads(
            MTLSize(width: n, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    static func encodeAdd(
        _ enc: MTLComputeCommandEncoder, _ ctx: Ctx,
        a: MTLBuffer, b: MTLBuffer, y: MTLBuffer, n: Int,
        pso: MTLComputePipelineState? = nil
    ) {
        enc.setComputePipelineState(pso ?? ctx.addPSO)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var n32 = UInt32(n)
        enc.setBytes(&n32, length: 4, index: 3)
        enc.dispatchThreads(
            MTLSize(width: n, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    // MARK: - Run

    public static func run(teDirectory: URL, metallibURL: URL) throws -> String {
        let shard = teDirectory.appendingPathComponent("0.safetensors")
        let arrays = try MLX.loadArrays(url: shard)
        func quant(_ name: String, n: Int, k: Int) throws -> QuantWeights {
            guard let w = arrays["layers.0.\(name).weight"],
                let s = arrays["layers.0.\(name).scales"],
                let b = arrays["layers.0.\(name).biases"]
            else { throw DirectQmmSpike.SpikeError.missingTensor("layers.0.\(name)") }
            return QuantWeights(
                w: w, scales: s.asType(.float16), biases: b.asType(.float16), n: n, k: k)
        }
        func normW(_ name: String) throws -> MLXArray {
            guard let w = arrays["layers.0.\(name).weight"] else {
                throw DirectQmmSpike.SpikeError.missingTensor("layers.0.\(name)")
            }
            return w.asType(.float32)
        }
        let qP = try quant("self_attn.q_proj", n: nHeads * headDim, k: hidden)
        let kP = try quant("self_attn.k_proj", n: nKV * headDim, k: hidden)
        let vP = try quant("self_attn.v_proj", n: nKV * headDim, k: hidden)
        let oP = try quant("self_attn.o_proj", n: hidden, k: nHeads * headDim)
        let gP = try quant("mlp.gate_proj", n: inter, k: hidden)
        let uP = try quant("mlp.up_proj", n: inter, k: hidden)
        let dP = try quant("mlp.down_proj", n: hidden, k: inter)
        let wIn = try normW("input_layernorm")
        let wPost = try normW("post_attention_layernorm")
        let wQn = try normW("self_attn.q_norm")
        let wKn = try normW("self_attn.k_norm")

        MLXRandom.seed(7)
        let x = (MLXRandom.normal([L, hidden]) * 0.5).asType(.float16)
        eval(x)

        // -- MLX oracle, identical f16 math, staged checkpoints ---------------
        func rmsF32(_ t: MLXArray, _ w: MLXArray) -> MLXArray {
            MLXFast.rmsNorm(t.asType(.float32), weight: w, eps: eps).asType(.float16)
        }
        func mm(_ t: MLXArray, _ p: QuantWeights) -> MLXArray {
            quantizedMM(
                t, p.w, scales: p.scales, biases: p.biases,
                transpose: true, groupSize: 64, bits: 4)
        }
        let rope = RoPE(dimensions: headDim, traditional: false, base: ropeBase)

        let oH1 = rmsF32(x, wIn)
        let oQ = mm(oH1, qP)
        let oK = mm(oH1, kP)
        let oV = mm(oH1, vP)
        let oQn = rmsF32(oQ.reshaped([L, nHeads, headDim]), wQn)
        let oKn = rmsF32(oK.reshaped([L, nKV, headDim]), wKn)
        let oQr = rope(oQn.reshaped([1, L, nHeads, headDim]).transposed(0, 2, 1, 3))
        let oKr = rope(oKn.reshaped([1, L, nKV, headDim]).transposed(0, 2, 1, 3))
        let oVt = oV.reshaped([1, L, nKV, headDim]).transposed(0, 2, 1, 3)
        let oAttn = MLXFast.scaledDotProductAttention(
            queries: oQr, keys: oKr, values: oVt,
            scale: 1.0 / Float(Double(headDim).squareRoot()), mask: .causal)
        let oAttnFlat = oAttn.transposed(0, 2, 1, 3).reshaped([L, nHeads * headDim])
        let oAttnProj = mm(oAttnFlat, oP)
        let oResid1 = x + oAttnProj
        let oH2 = rmsF32(oResid1, wPost)
        let oGate = mm(oH2, gP)
        let oUp = mm(oH2, uP)
        let oMul = silu(oGate) * oUp
        let oDown = mm(oMul, dP)
        let oOut = oResid1 + oDown
        eval(oH1, oQ, oQr, oAttnFlat, oResid1, oOut)

        // Oracle wall per layer (warm, full graph per call).
        var oracleMS = 0.0
        do {
            func once() -> MLXArray {
                let h1 = rmsF32(x, wIn)
                let q = rope(rmsF32(mm(h1, qP).reshaped([L, nHeads, headDim]), wQn)
                    .reshaped([1, L, nHeads, headDim]).transposed(0, 2, 1, 3))
                let k = rope(rmsF32(mm(h1, kP).reshaped([L, nKV, headDim]), wKn)
                    .reshaped([1, L, nKV, headDim]).transposed(0, 2, 1, 3))
                let v = mm(h1, vP).reshaped([1, L, nKV, headDim]).transposed(0, 2, 1, 3)
                let a = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v,
                    scale: 1.0 / Float(Double(headDim).squareRoot()), mask: .causal)
                let r1 = x + mm(a.transposed(0, 2, 1, 3).reshaped([L, nHeads * headDim]), oP)
                let h2 = rmsF32(r1, wPost)
                return r1 + mm(silu(mm(h2, gP)) * mm(h2, uP), dP)
            }
            for _ in 0 ..< 3 { eval(once()) }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< 10 { eval(once()) }
            oracleMS = (CFAbsoluteTimeGetCurrent() - t0) * 100
        }

        // -- Direct path ------------------------------------------------------
        let ctx = try Ctx(metallibURL: metallibURL)
        func up3(_ p: QuantWeights, _ tag: String) throws -> (MTLBuffer, MTLBuffer, MTLBuffer) {
            (try ctx.upload(p.w, "\(tag).w"),
             try ctx.upload(p.scales, "\(tag).s"),
             try ctx.upload(p.biases, "\(tag).b"))
        }
        let qB = try up3(qP, "q"), kB = try up3(kP, "k"), vB = try up3(vP, "v")
        let oB = try up3(oP, "o"), gB = try up3(gP, "g"), uB = try up3(uP, "u")
        let dB = try up3(dP, "d")
        let wInB = try ctx.upload(wIn, "wIn")
        let wPostB = try ctx.upload(wPost, "wPost")
        let wQnB = try ctx.upload(wQn, "wQn")
        let wKnB = try ctx.upload(wKn, "wKn")
        let xB = try ctx.upload(x, "x")

        let h1B = try ctx.scratch(L * hidden, "h1")
        let qOut = try ctx.scratch(L * nHeads * headDim, "qOut")
        let kOut = try ctx.scratch(L * nKV * headDim, "kOut")
        let vOut = try ctx.scratch(L * nKV * headDim, "vOut")
        let qNorm = try ctx.scratch(L * nHeads * headDim, "qNorm")
        let kNorm = try ctx.scratch(L * nKV * headDim, "kNorm")
        let qRope = try ctx.scratch(L * nHeads * headDim, "qRope")
        let kRope = try ctx.scratch(L * nKV * headDim, "kRope")
        let attnO = try ctx.scratch(L * nHeads * headDim, "attnO")
        let attnP = try ctx.scratch(L * hidden, "attnP")
        let resid1 = try ctx.scratch(L * hidden, "resid1")
        let h2B = try ctx.scratch(L * hidden, "h2")
        let gOut = try ctx.scratch(L * inter, "gOut")
        let uOut = try ctx.scratch(L * inter, "uOut")
        let mOut = try ctx.scratch(L * inter, "mOut")
        let dOut = try ctx.scratch(L * hidden, "dOut")
        let yB = try ctx.scratch(L * hidden, "y")

        func encodeLayer() throws {
            guard let cb = ctx.queue.makeCommandBuffer(),
                let enc = cb.makeComputeCommandEncoder()
            else { throw DirectQmmSpike.SpikeError.metal("cb") }
            encodeRms(enc, ctx, x: xB, w: wInB, y: h1B, rows: L, d: hidden)
            encodeQmm(enc, ctx, w: qB.0, s: qB.1, b: qB.2, x: h1B, y: qOut,
                      m: L, n: nHeads * headDim, k: hidden)
            encodeQmm(enc, ctx, w: kB.0, s: kB.1, b: kB.2, x: h1B, y: kOut,
                      m: L, n: nKV * headDim, k: hidden)
            encodeQmm(enc, ctx, w: vB.0, s: vB.1, b: vB.2, x: h1B, y: vOut,
                      m: L, n: nKV * headDim, k: hidden)
            encodeRms(enc, ctx, x: qOut, w: wQnB, y: qNorm, rows: L * nHeads, d: headDim)
            encodeRms(enc, ctx, x: kOut, w: wKnB, y: kNorm, rows: L * nKV, d: headDim)
            encodeRope(enc, ctx, x: qNorm, y: qRope, heads: nHeads)
            encodeRope(enc, ctx, x: kNorm, y: kRope, heads: nKV)
            encodeAttention(enc, ctx, q: qRope, k: kRope, v: vOut, o: attnO)
            encodeQmm(enc, ctx, w: oB.0, s: oB.1, b: oB.2, x: attnO, y: attnP,
                      m: L, n: hidden, k: nHeads * headDim)
            encodeAdd(enc, ctx, a: xB, b: attnP, y: resid1, n: L * hidden)
            encodeRms(enc, ctx, x: resid1, w: wPostB, y: h2B, rows: L, d: hidden)
            encodeQmm(enc, ctx, w: gB.0, s: gB.1, b: gB.2, x: h2B, y: gOut,
                      m: L, n: inter, k: hidden)
            encodeQmm(enc, ctx, w: uB.0, s: uB.1, b: uB.2, x: h2B, y: uOut,
                      m: L, n: inter, k: hidden)
            encodeSiluMul(enc, ctx, g: gOut, u: uOut, y: mOut, n: L * inter)
            encodeQmm(enc, ctx, w: dB.0, s: dB.1, b: dB.2, x: mOut, y: dOut,
                      m: L, n: hidden, k: inter)
            encodeAdd(enc, ctx, a: resid1, b: dOut, y: yB, n: L * hidden)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw DirectQmmSpike.SpikeError.metal("exec: \(e)") }
        }

        try encodeLayer()  // warm
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< 10 { try encodeLayer() }
        let directMS = (CFAbsoluteTimeGetCurrent() - t0) * 100

        // -- Verification battery ---------------------------------------------
        func compare(_ buf: MTLBuffer, _ oracle: MLXArray, _ count: Int, _ tag: String) -> String {
            let ptr = buf.contents().bindMemory(to: Float16.self, capacity: count)
            let ref = oracle.asType(.float32).asArray(Float.self)
            var dot = 0.0, na = 0.0, nb = 0.0, maxDiff = 0.0
            for i in 0 ..< count {
                let a = Double(Float(ptr[i])), b = Double(ref[i])
                dot += a * b; na += a * a; nb += b * b
                maxDiff = max(maxDiff, abs(a - b))
            }
            let cos = dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
            return String(format: "  %-10s cosine=%.7f maxAbs=%.5f", (tag as NSString).utf8String!, cos, maxDiff)
        }
        var lines: [String] = []
        lines.append(compare(h1B, oH1, L * hidden, "rmsnorm"))
        lines.append(compare(qOut, oQ, L * nHeads * headDim, "q_proj"))
        lines.append(compare(qRope, oQr.transposed(0, 2, 1, 3).reshaped([L, nHeads * headDim]),
                             L * nHeads * headDim, "q_rope"))
        lines.append(compare(attnO, oAttnFlat, L * nHeads * headDim, "attention"))
        lines.append(compare(resid1, oResid1, L * hidden, "resid1"))
        lines.append(compare(mOut, oMul, L * inter, "silu_mul"))
        lines.append(compare(yB, oOut, L * hidden, "layer_out"))

        return """
        direct-te-layer spike (milestone B) — Qwen3 layer 0, M=\(L), one command buffer, 18 dispatches
        \(lines.joined(separator: "\n"))
          oracle_per_layer: \(String(format: "%.2f", oracleMS)) ms (MLX runtime, warm)
          direct_per_layer: \(String(format: "%.2f", directMS)) ms (single CB, blocking wait)
        """
    }
}
