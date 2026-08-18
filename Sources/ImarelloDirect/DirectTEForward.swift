import Foundation
import Metal
import MLX
import MLXNN
import MLXFast

/// Stage-2 Milestone C: the full 27-layer Qwen3 TE forward as ONE recorded
/// command buffer per encode (486 dispatches + tap blits), A/B'd against an
/// identical-math MLX loop at L=512 and the splice regime L=30.
public enum DirectTEForward {

    static let hidden = 2560
    static let nHeads = 32
    static let nKV = 8
    static let headDim = 128
    static let inter = 9728
    static let nLayers = 27
    static let taps = [9, 18, 27]  // hidden-state indices (after k layers)
    static let eps: Float = 1e-6
    static let ropeBase: Float = 1_000_000

    struct LayerBuffers {
        var qmm: [(w: MTLBuffer, s: MTLBuffer, b: MTLBuffer, n: Int, k: Int)]  // q,k,v,o,g,u,d
        var wIn: MTLBuffer
        var wPost: MTLBuffer
        var wQn: MTLBuffer
        var wKn: MTLBuffer
    }

    final class Engine {
        let ctx: DirectTELayerSpike.Ctx
        let metallibURL: URL
        var layers: [LayerBuffers] = []
        private var attnPSOs: [String: MTLComputePipelineState] = [:]

        init(metallibURL: URL) throws {
            self.metallibURL = metallibURL
            ctx = try DirectTELayerSpike.Ctx(metallibURL: metallibURL)
        }

        func attnPSO(alignQ: Bool, alignK: Bool) throws -> MTLComputePipelineState {
            let key = "\(alignQ)-\(alignK)"
            if let p = attnPSOs[key] { return p }
            let lib = try ctx.device.makeLibrary(URL: metallibURL)
            let consts = MTLFunctionConstantValues()
            var aQ = alignQ, aK = alignK, t = true, f = false
            consts.setConstantValue(&aQ, type: .bool, index: 200)
            consts.setConstantValue(&aK, type: .bool, index: 201)
            consts.setConstantValue(&f, type: .bool, index: 300)  // has_mask
            consts.setConstantValue(&t, type: .bool, index: 301)  // do_causal
            consts.setConstantValue(&f, type: .bool, index: 302)  // has_sinks
            let fn = try lib.makeFunction(
                name: "steel_attention_float16_bq32_bk16_bd128_wm4_wn1_maskfloat16",
                constantValues: consts)
            let p = try ctx.device.makeComputePipelineState(function: fn)
            attnPSOs[key] = p
            return p
        }
    }

    // MARK: - Scratch (sized per L, reused across all layers — the static plan)

    final class Scratch {
        let L: Int
        let h1, q, k, v, qn, kn, qr, kr, attnO, attnP, resid1, h2, g, u, m, d: MTLBuffer
        var x, y: MTLBuffer  // ping-pong hidden states
        let tapBufs: [MTLBuffer]

        init(_ ctx: DirectTELayerSpike.Ctx, L: Int) throws {
            self.L = L
            h1 = try ctx.scratch(L * hidden, "h1")
            q = try ctx.scratch(L * nHeads * headDim, "q")
            k = try ctx.scratch(L * nKV * headDim, "k")
            v = try ctx.scratch(L * nKV * headDim, "v")
            qn = try ctx.scratch(L * nHeads * headDim, "qn")
            kn = try ctx.scratch(L * nKV * headDim, "kn")
            qr = try ctx.scratch(L * nHeads * headDim, "qr")
            kr = try ctx.scratch(L * nKV * headDim, "kr")
            attnO = try ctx.scratch(L * nHeads * headDim, "attnO")
            attnP = try ctx.scratch(L * hidden, "attnP")
            resid1 = try ctx.scratch(L * hidden, "resid1")
            h2 = try ctx.scratch(L * hidden, "h2")
            g = try ctx.scratch(L * inter, "g")
            u = try ctx.scratch(L * inter, "u")
            m = try ctx.scratch(L * inter, "m")
            d = try ctx.scratch(L * hidden, "d")
            x = try ctx.scratch(L * hidden, "x")
            y = try ctx.scratch(L * hidden, "y")
            tapBufs = try (0 ..< 3).map { try ctx.scratch(L * hidden, "tap\($0)") }
        }
    }

    // MARK: - Weight loading

    static func loadLayers(_ engine: Engine, arrays: [String: MLXArray], rawScales: Bool = false) throws {
        let dims: [(String, Int, Int)] = [
            ("self_attn.q_proj", nHeads * headDim, hidden),
            ("self_attn.k_proj", nKV * headDim, hidden),
            ("self_attn.v_proj", nKV * headDim, hidden),
            ("self_attn.o_proj", hidden, nHeads * headDim),
            ("mlp.gate_proj", inter, hidden),
            ("mlp.up_proj", inter, hidden),
            ("mlp.down_proj", hidden, inter),
        ]
        for li in 0 ..< nLayers {
            var qmm: [(MTLBuffer, MTLBuffer, MTLBuffer, Int, Int)] = []
            for (name, n, k) in dims {
                guard let w = arrays["layers.\(li).\(name).weight"],
                    let sc = arrays["layers.\(li).\(name).scales"],
                    let bi = arrays["layers.\(li).\(name).biases"]
                else { throw DirectQmmSpike.SpikeError.missingTensor("layers.\(li).\(name)") }
                let s16 = rawScales ? sc : sc.asType(.float16)
                let b16 = rawScales ? bi : bi.asType(.float16)
                eval(s16, b16)
                qmm.append((
                    try engine.ctx.upload(w, "L\(li).\(name).w"),
                    try engine.ctx.upload(s16, "L\(li).\(name).s"),
                    try engine.ctx.upload(b16, "L\(li).\(name).b"),
                    n, k))
            }
            func norm(_ name: String) throws -> MTLBuffer {
                guard let w = arrays["layers.\(li).\(name).weight"] else {
                    throw DirectQmmSpike.SpikeError.missingTensor("layers.\(li).\(name)")
                }
                let f = w.asType(.float32)
                eval(f)
                return try engine.ctx.upload(f, "L\(li).\(name)")
            }
            engine.layers.append(LayerBuffers(
                qmm: qmm.map { (w: $0.0, s: $0.1, b: $0.2, n: $0.3, k: $0.4) },
                wIn: try norm("input_layernorm"),
                wPost: try norm("post_attention_layernorm"),
                wQn: try norm("self_attn.q_norm"),
                wKn: try norm("self_attn.k_norm")))
        }
    }

    // MARK: - Attention encode (parameterized L)

    static func encodeAttention(
        _ enc: MTLComputeCommandEncoder, _ engine: Engine, L: Int,
        q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, o: MTLBuffer
    ) throws {
        let bq = 32, bk = 16
        let pso = try engine.attnPSO(alignQ: L % bq == 0, alignK: L % bk == 0)
        enc.setComputePipelineState(pso)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(k, offset: 0, index: 1)
        enc.setBuffer(v, offset: 0, index: 2)
        enc.setBuffer(o, offset: 0, index: 3)
        let nq = (L + bq - 1) / bq, nk = (L + bk - 1) / bk
        var data = Data(capacity: 152)
        func i32(_ v: Int) { var x = Int32(v); withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func f32v(_ v: Float) { var x = v; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
        func i64x3(_ a: Int, _ b: Int, _ c: Int) {
            for v in [Int64(a), Int64(b), Int64(c)] {
                var x = v; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) }
            }
        }
        i32(1); i32(nHeads); i32(headDim); i32(L); i32(L)
        i32(nHeads / nKV)
        f32v(1.0 / Float(Double(headDim).squareRoot()))
        i32(nq); i32(nk)
        i32(L / bq); i32(L / bk)
        i32(L - (L / bq) * bq); i32(L - (L / bk) * bk)
        i32(0)  // qL_off = kL - qL; 56 bytes of scalars — already 8-aligned
        i64x3(L * nHeads * headDim, headDim, nHeads * headDim)
        i64x3(L * nKV * headDim, headDim, nKV * headDim)
        i64x3(L * nKV * headDim, headDim, nKV * headDim)
        i64x3(L * nHeads * headDim, headDim, nHeads * headDim)
        data.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: data.count, index: 4) }
        enc.dispatchThreadgroups(
            MTLSize(width: nq, height: nHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
    }

    // MARK: - One layer into the encoder

    static func encodeLayer(
        _ enc: MTLComputeCommandEncoder, _ engine: Engine, _ lb: LayerBuffers,
        _ s: Scratch, L: Int
    ) throws {
        let ctx = engine.ctx
        typealias LS = DirectTELayerSpike
        LS.encodeRms(enc, ctx, x: s.x, w: lb.wIn, y: s.h1, rows: L, d: hidden)
        LS.encodeQmm(enc, ctx, w: lb.qmm[0].w, s: lb.qmm[0].s, b: lb.qmm[0].b,
                     x: s.h1, y: s.q, m: L, n: lb.qmm[0].n, k: lb.qmm[0].k)
        LS.encodeQmm(enc, ctx, w: lb.qmm[1].w, s: lb.qmm[1].s, b: lb.qmm[1].b,
                     x: s.h1, y: s.k, m: L, n: lb.qmm[1].n, k: lb.qmm[1].k)
        LS.encodeQmm(enc, ctx, w: lb.qmm[2].w, s: lb.qmm[2].s, b: lb.qmm[2].b,
                     x: s.h1, y: s.v, m: L, n: lb.qmm[2].n, k: lb.qmm[2].k)
        LS.encodeRms(enc, ctx, x: s.q, w: lb.wQn, y: s.qn, rows: L * nHeads, d: headDim)
        LS.encodeRms(enc, ctx, x: s.k, w: lb.wKn, y: s.kn, rows: L * nKV, d: headDim)
        LS.encodeRopeL(enc, ctx, x: s.qn, y: s.qr, heads: nHeads, seqLen: L)
        LS.encodeRopeL(enc, ctx, x: s.kn, y: s.kr, heads: nKV, seqLen: L)
        try encodeAttention(enc, engine, L: L, q: s.qr, k: s.kr, v: s.v, o: s.attnO)
        LS.encodeQmm(enc, ctx, w: lb.qmm[3].w, s: lb.qmm[3].s, b: lb.qmm[3].b,
                     x: s.attnO, y: s.attnP, m: L, n: lb.qmm[3].n, k: lb.qmm[3].k)
        LS.encodeAdd(enc, ctx, a: s.x, b: s.attnP, y: s.resid1, n: L * hidden)
        LS.encodeRms(enc, ctx, x: s.resid1, w: lb.wPost, y: s.h2, rows: L, d: hidden)
        LS.encodeQmm(enc, ctx, w: lb.qmm[4].w, s: lb.qmm[4].s, b: lb.qmm[4].b,
                     x: s.h2, y: s.g, m: L, n: lb.qmm[4].n, k: lb.qmm[4].k)
        LS.encodeQmm(enc, ctx, w: lb.qmm[5].w, s: lb.qmm[5].s, b: lb.qmm[5].b,
                     x: s.h2, y: s.u, m: L, n: lb.qmm[5].n, k: lb.qmm[5].k)
        LS.encodeSiluMul(enc, ctx, g: s.g, u: s.u, y: s.m, n: L * inter)
        LS.encodeQmm(enc, ctx, w: lb.qmm[6].w, s: lb.qmm[6].s, b: lb.qmm[6].b,
                     x: s.m, y: s.d, m: L, n: lb.qmm[6].n, k: lb.qmm[6].k)
        LS.encodeAdd(enc, ctx, a: s.resid1, b: s.d, y: s.y, n: L * hidden)
    }

    // MARK: - Full forward (one command buffer)

    static func forward(_ engine: Engine, _ s: Scratch, L: Int) throws {
        guard let cb = engine.ctx.queue.makeCommandBuffer() else {
            throw DirectQmmSpike.SpikeError.metal("cb")
        }
        var tapIdx = 0
        for (li, lb) in engine.layers.enumerated() {
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw DirectQmmSpike.SpikeError.metal("encoder")
            }
            try encodeLayer(enc, engine, lb, s, L: L)
            enc.endEncoding()
            swap(&s.x, &s.y)
            if tapIdx < taps.count, li + 1 == taps[tapIdx] {
                guard let blit = cb.makeBlitCommandEncoder() else {
                    throw DirectQmmSpike.SpikeError.metal("blit")
                }
                blit.copy(
                    from: s.x, sourceOffset: 0,
                    to: s.tapBufs[tapIdx], destinationOffset: 0,
                    size: L * hidden * 2)
                blit.endEncoding()
                tapIdx += 1
            }
        }
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw DirectQmmSpike.SpikeError.metal("exec: \(e)") }
    }

    // MARK: - Run

    public static func run(teDirectory: URL, metallibURL: URL, seqLen: Int) throws -> String {
        let L = seqLen
        let shard = teDirectory.appendingPathComponent("0.safetensors")
        let raw = try MLX.loadArrays(url: shard)
        // Pre-converted oracle weights (f16 scales/biases, f32 norms) so the
        // MLX baseline does no per-call dtype conversion.
        var W: [String: MLXArray] = [:]
        for li in 0 ..< nLayers {
            for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
                let p = "layers.\(li).\(name)"
                W["\(p).weight"] = raw["\(p).weight"]
                W["\(p).scales"] = raw["\(p).scales"]!.asType(.float16)
                W["\(p).biases"] = raw["\(p).biases"]!.asType(.float16)
            }
            for name in ["input_layernorm", "post_attention_layernorm",
                         "self_attn.q_norm", "self_attn.k_norm"] {
                let p = "layers.\(li).\(name)"
                W["\(p).weight"] = raw["\(p).weight"]!.asType(.float32)
            }
        }
        eval(Array(W.values))

        let engine = try Engine(metallibURL: metallibURL)
        let tLoad0 = CFAbsoluteTimeGetCurrent()
        try loadLayers(engine, arrays: raw)
        let loadMS = (CFAbsoluteTimeGetCurrent() - tLoad0) * 1000

        MLXRandom.seed(3)
        let x0 = (MLXRandom.normal([L, hidden]) * 0.5).asType(.float16)
        eval(x0)
        let s = try Scratch(engine.ctx, L: L)
        let x0Data = x0.asData(noCopy: false)
        func seedX() {
            x0Data.withUnsafeBytes { rawB in
                s.x.contents().copyMemory(from: rawB.baseAddress!, byteCount: rawB.count)
            }
        }

        seedX()
        try forward(engine, s, L: L)  // warm; leaves taps for verification
        let reps = 5
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< reps {
            seedX()
            try forward(engine, s, L: L)
        }
        let directMS = (CFAbsoluteTimeGetCurrent() - t0) * 1000 / Double(reps)
        // Keep the warm-run taps: redo one verification forward after timing.
        seedX()
        try forward(engine, s, L: L)

        // -- Oracle -----------------------------------------------------------
        let rope = RoPE(dimensions: headDim, traditional: false, base: ropeBase)
        func layerOracle(_ x: MLXArray, _ li: Int) -> MLXArray {
            func mm(_ t: MLXArray, _ name: String) -> MLXArray {
                let p = "layers.\(li).\(name)"
                return quantizedMM(
                    t, W["\(p).weight"]!, scales: W["\(p).scales"]!, biases: W["\(p).biases"]!,
                    transpose: true, groupSize: 64, bits: 4)
            }
            func rms(_ t: MLXArray, _ name: String) -> MLXArray {
                MLXFast.rmsNorm(
                    t.asType(.float32),
                    weight: W["layers.\(li).\(name).weight"]!, eps: eps
                ).asType(.float16)
            }
            let h1 = rms(x, "input_layernorm")
            let q = rope(rms(mm(h1, "self_attn.q_proj").reshaped([L, nHeads, headDim]), "self_attn.q_norm")
                .reshaped([1, L, nHeads, headDim]).transposed(0, 2, 1, 3))
            let k = rope(rms(mm(h1, "self_attn.k_proj").reshaped([L, nKV, headDim]), "self_attn.k_norm")
                .reshaped([1, L, nKV, headDim]).transposed(0, 2, 1, 3))
            let v = mm(h1, "self_attn.v_proj")
                .reshaped([1, L, nKV, headDim]).transposed(0, 2, 1, 3)
            let a = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v,
                scale: 1.0 / Float(Double(headDim).squareRoot()), mask: .causal)
            let r1 = x + mm(a.transposed(0, 2, 1, 3).reshaped([L, nHeads * headDim]), "self_attn.o_proj")
            let h2 = rms(r1, "post_attention_layernorm")
            return r1 + mm(silu(mm(h2, "mlp.gate_proj")) * mm(h2, "mlp.up_proj"), "mlp.down_proj")
        }
        func oracleForward() -> [MLXArray] {
            var h = x0
            var outTaps: [MLXArray] = []
            var tapIdx = 0
            for li in 0 ..< nLayers {
                h = layerOracle(h, li)
                if tapIdx < taps.count, li + 1 == taps[tapIdx] {
                    outTaps.append(h)
                    tapIdx += 1
                }
            }
            eval(outTaps)
            return outTaps
        }
        let oTaps = oracleForward()
        _ = oracleForward()  // warm
        let t1 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< reps { _ = oracleForward() }
        let oracleMS = (CFAbsoluteTimeGetCurrent() - t1) * 1000 / Double(reps)

        // -- Verify taps ------------------------------------------------------
        var lines: [String] = []
        var worst = 1.0
        for (i, tap) in oTaps.enumerated() {
            let ptr = s.tapBufs[i].contents().bindMemory(to: Float16.self, capacity: L * hidden)
            let ref = tap.asType(.float32).asArray(Float.self)
            var dot = 0.0, na = 0.0, nb = 0.0, maxDiff = 0.0
            for j in 0 ..< L * hidden {
                let a = Double(Float(ptr[j])), b = Double(ref[j])
                dot += a * b; na += a * a; nb += b * b
                maxDiff = max(maxDiff, abs(a - b))
            }
            let cos = dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
            worst = min(worst, cos)
            lines.append(String(
                format: "  tap[%d] (after layer %d): cosine=%.7f maxAbs=%.4f",
                i, taps[i], cos, maxDiff))
        }
        let speedup = oracleMS / directMS
        let gate = speedup >= 1.3 && worst >= 0.9999
        return """
        direct-te-forward (milestone C) — \(nLayers) layers, L=\(L), one command buffer, \(nLayers * 18) dispatches + 3 tap blits
        \(lines.joined(separator: "\n"))
          weight_upload:    \(String(format: "%.0f", loadMS)) ms (one-time, \(nLayers) layers)
          oracle_forward:   \(String(format: "%.1f", oracleMS)) ms (identical-math MLX loop, warm)
          direct_forward:   \(String(format: "%.1f", directMS)) ms (single CB, blocking wait)
          speedup:          \(String(format: "%.2f", speedup))×
          gate(≥1.3× & cos≥0.9999): \(gate ? "PASS" : "not yet")
        """
    }
}
