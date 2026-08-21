import Foundation
import Metal
import MLX
import ImarelloText

/// Stage-2 Milestone D: a complete, product-shaped TE encode on the direct
/// engine — chat-templated tokens, CPU embedding dequant, 27 layers with the
/// production attention semantics (causal ∧ pad-keys-blocked, additive −1e4),
/// taps concatenated to `[1, 512, 7680]`.
///
/// Runs **bf16** end-to-end like the production TE: real Qwen activations
/// exceed f16 range (taps alone reach |1.5e4|; MLP intermediates overflow),
/// which is why the product path is bf16 and the engine must be too.
public final class DirectTEEncoder {

    static let hidden = DirectTEForward.hidden
    static let maxLen = 512
    static let vocabGroup = 64

    let engine: DirectTEForward.Engine
    let scratch: DirectTEForward.Scratch
    let tokenizer: QwenTokenizer
    // Embedding tables on CPU (bf16 bit patterns for scales/biases).
    let embedPacked: [UInt32]   // [vocab, hidden/8]
    let embedScales: [UInt16]   // [vocab, hidden/64], bf16 bits
    let embedBiases: [UInt16]
    let embedOracle: (packed: MLXArray, scales: MLXArray, biases: MLXArray)
    // bf16 PSO set.
    let qmmPSO: MTLComputePipelineState
    let maskAttnPSO: MTLComputePipelineState
    let rmsPSO: MTLComputePipelineState
    let ropePSO: MTLComputePipelineState
    let siluMulPSO: MTLComputePipelineState
    let addPSO: MTLComputePipelineState
    let maskBuf: MTLBuffer

    public private(set) var lastRealTokens = 0

    func capturedTaps() -> [(layer: Int, data: Data)] {
        zip(DirectTEForward.taps, scratch.tapBufs).map { layer, buffer in
            (layer, Data(
                bytes: buffer.contents(),
                count: Self.maxLen * Self.hidden * MemoryLayout<UInt16>.size))
        }
    }

    @inline(__always) static func f32ToBF16(_ v: Float) -> UInt16 {
        let bits = v.bitPattern
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }
    @inline(__always) static func bf16ToF32(_ u: UInt16) -> Float {
        Float(bitPattern: UInt32(u) << 16)
    }

    public init(teDirectory: URL, tokenizerDirectory: URL, metallibURL: URL) throws {
        let shard = teDirectory.appendingPathComponent("0.safetensors")
        let arrays = try MLX.loadArrays(url: shard)
        engine = try DirectTEForward.Engine(metallibURL: metallibURL)
        try DirectTEForward.loadLayers(engine, arrays: arrays, rawScales: true)
        scratch = try DirectTEForward.Scratch(engine.ctx, L: Self.maxLen)
        tokenizer = try QwenTokenizer.load(from: tokenizerDirectory)

        guard let ew = arrays["embed_tokens.weight"],
            let es = arrays["embed_tokens.scales"],
            let eb = arrays["embed_tokens.biases"]
        else { throw DirectQmmSpike.SpikeError.missingTensor("embed_tokens") }
        guard ew.ndim == 2 else {
            throw DirectQmmSpike.SpikeError.invalidTensor(
                "embed_tokens.weight expected rank 2, got shape \(ew.shape)")
        }
        let vocabularySize = ew.dim(0)
        try DirectTensorValidation.requireQuantized(
            weight: ew, scales: es, biases: eb,
            n: vocabularySize, k: Self.hidden, name: "embed_tokens")
        eval(ew, es, eb)
        embedPacked = ew.asArray(UInt32.self)
        // Raw bf16 bit patterns (asData preserves bytes; asArray would convert values).
        embedScales = es.asData(access: .copy).data.withUnsafeBytes {
            Array($0.bindMemory(to: UInt16.self))
        }
        embedBiases = eb.asData(access: .copy).data.withUnsafeBytes {
            Array($0.bindMemory(to: UInt16.self))
        }
        embedOracle = (ew, es, eb)

        let dev = engine.ctx.device
        let mlxLib = try dev.makeLibrary(URL: metallibURL)
        guard let qfn = mlxLib.makeFunction(
            name: "affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0")
        else { throw DirectQmmSpike.SpikeError.metal("bf16 qmm kernel missing") }
        qmmPSO = try dev.makeComputePipelineState(function: qfn)

        let consts = MTLFunctionConstantValues()
        var t = true, f = false
        consts.setConstantValue(&t, type: .bool, index: 200)
        consts.setConstantValue(&t, type: .bool, index: 201)
        consts.setConstantValue(&t, type: .bool, index: 300)  // has_mask
        consts.setConstantValue(&f, type: .bool, index: 301)  // do_causal (mask carries it)
        consts.setConstantValue(&f, type: .bool, index: 302)
        let afn = try mlxLib.makeFunction(
            name: "steel_attention_bfloat16_bq32_bk16_bd128_wm4_wn1_maskbfloat16",
            constantValues: consts)
        maskAttnPSO = try dev.makeComputePipelineState(function: afn)

        let glue = try DirectGlueKernels.makeLibrary(
            device: dev,
            directMetallibURL: DirectEngineArtifacts.directMetallibURL(
                beside: metallibURL))
        func gfn(_ name: String) throws -> MTLComputePipelineState {
            let typedName = DirectGlueKernels.functionName(name, dtypeName: "bfloat")
            guard let f = glue.makeFunction(name: typedName) else {
                throw DirectQmmSpike.SpikeError.metal("missing glue \(typedName)")
            }
            return try dev.makeComputePipelineState(function: f)
        }
        rmsPSO = try gfn("dq_rmsnorm")
        ropePSO = try gfn("dq_rope")
        siluMulPSO = try gfn("dq_silu_mul")
        addPSO = try gfn("dq_add")

        guard let mb = dev.makeBuffer(length: Self.maxLen * Self.maxLen * 2) else {
            throw DirectQmmSpike.SpikeError.metal("mask buffer")
        }
        maskBuf = mb
    }

    // MARK: - CPU embedding dequant (affine 4-bit, LSB-first nibbles, bf16 out)

    func embedRow(_ token: Int, into out: UnsafeMutablePointer<UInt16>) {
        let perRowPacked = Self.hidden / 8
        let perRowGroups = Self.hidden / Self.vocabGroup
        let pBase = token * perRowPacked
        let gBase = token * perRowGroups
        for g in 0 ..< perRowGroups {
            let scale = Self.bf16ToF32(embedScales[gBase + g])
            let bias = Self.bf16ToF32(embedBiases[gBase + g])
            let colBase = g * Self.vocabGroup
            for w in 0 ..< Self.vocabGroup / 8 {
                let packed = embedPacked[pBase + colBase / 8 + w]
                for n in 0 ..< 8 {
                    let q = Float((packed >> (4 * UInt32(n))) & 0xF)
                    out[colBase + w * 8 + n] = Self.f32ToBF16(scale * q + bias)
                }
            }
        }
    }

    /// Verify the CPU dequant against MLX's `dequantized` for one row.
    public func verifyEmbedding(token: Int) -> (cosine: Double, maxAbs: Double) {
        var row = [UInt16](repeating: 0, count: Self.hidden)
        row.withUnsafeMutableBufferPointer { embedRow(token, into: $0.baseAddress!) }
        let ref = dequantized(
            embedOracle.packed[token ..< token + 1],
            scales: embedOracle.scales[token ..< token + 1],
            biases: embedOracle.biases[token ..< token + 1],
            groupSize: Self.vocabGroup, bits: 4
        ).asType(.float32).asArray(Float.self)
        var dot = 0.0, na = 0.0, nb = 0.0, maxDiff = 0.0
        for i in 0 ..< Self.hidden {
            let a = Double(Self.bf16ToF32(row[i])), b = Double(ref[i])
            dot += a * b; na += a * a; nb += b * b
            maxDiff = max(maxDiff, abs(a - b))
        }
        return (dot / (na.squareRoot() * nb.squareRoot() + 1e-30), maxDiff)
    }

    // MARK: - Mask (production semantics: causal ∧ key < realTokens, additive −1e4)

    func fillMask(realTokens: Int) {
        let L = Self.maxLen
        let ptr = maskBuf.contents().bindMemory(to: UInt16.self, capacity: L * L)
        let neg = Self.f32ToBF16(-10_000.0)
        let zero = Self.f32ToBF16(0)
        for i in 0 ..< L {
            let rowLimit = min(i, realTokens - 1)
            for j in 0 ..< L {
                ptr[i * L + j] = j <= rowLimit ? zero : neg
            }
        }
    }

    func encodeMaskedAttention(
        _ enc: MTLComputeCommandEncoder,
        q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, o: MTLBuffer
    ) {
        let L = Self.maxLen
        let nHeads = DirectTEForward.nHeads
        let nKV = DirectTEForward.nKV
        let headDim = DirectTEForward.headDim
        let bq = 32, bk = 16
        enc.setComputePipelineState(maskAttnPSO)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(k, offset: 0, index: 1)
        enc.setBuffer(v, offset: 0, index: 2)
        enc.setBuffer(o, offset: 0, index: 3)
        let nq = L / bq, nk = L / bk
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
        i32(nq); i32(nk); i32(nq); i32(nk); i32(0); i32(0); i32(0)
        i64x3(L * nHeads * headDim, headDim, nHeads * headDim)
        i64x3(L * nKV * headDim, headDim, nKV * headDim)
        i64x3(L * nKV * headDim, headDim, nKV * headDim)
        i64x3(L * nHeads * headDim, headDim, nHeads * headDim)
        data.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: data.count, index: 4) }
        // Mask is [1, 1, L, L] broadcast over batch and heads → strides 0, 0, L
        // (the kernel adds head × M_strides[1]; a nonzero stride reads OOB).
        var mask = Data(capacity: 24)
        for v in [Int64(0), Int64(0), Int64(L)] {
            var x = v; withUnsafeBytes(of: &x) { mask.append(contentsOf: $0) }
        }
        mask.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: mask.count, index: 5) }
        enc.setBuffer(maskBuf, offset: 0, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: nq, height: nHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
    }

    // MARK: - Causal attention (splice path: no pads in window)

    func encodeCausalAttention(
        _ enc: MTLComputeCommandEncoder, L: Int,
        q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, o: MTLBuffer
    ) throws {
        let nHeads = DirectTEForward.nHeads
        let nKV = DirectTEForward.nKV
        let headDim = DirectTEForward.headDim
        let bq = 32, bk = 16
        let pso = try engine.attnPSO(
            alignQ: L % bq == 0, alignK: L % bk == 0, dtypeName: "bfloat16")
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
        i32(0)
        i64x3(L * nHeads * headDim, headDim, nHeads * headDim)
        i64x3(L * nKV * headDim, headDim, nKV * headDim)
        i64x3(L * nKV * headDim, headDim, nKV * headDim)
        i64x3(L * nHeads * headDim, headDim, nHeads * headDim)
        data.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: data.count, index: 4) }
        enc.dispatchThreadgroups(
            MTLSize(width: nq, height: nHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
    }

    /// Splice path: encode ONLY the real (chat-templated) tokens — pure causal,
    /// work scales with the prompt, not with 512. Returns `[1, r, 7680]` bf16.
    public func encodeRealOnly(_ prompt: String) throws -> (embeds: MLXArray, realTokens: Int, ms: Double) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let ids = tokenizer.encodePromptUnpadded(prompt, maxLength: Self.maxLen)
        let r = ids.count
        lastRealTokens = r

        let xPtr = scratch.x.contents().bindMemory(to: UInt16.self, capacity: r * Self.hidden)
        for (pos, token) in ids.enumerated() {
            embedRow(token, into: xPtr + pos * Self.hidden)
        }

        guard let cb = engine.ctx.queue.makeCommandBuffer() else {
            throw DirectQmmSpike.SpikeError.metal("cb")
        }
        var tapIdx = 0
        for (li, lb) in engine.layers.enumerated() {
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw DirectQmmSpike.SpikeError.metal("encoder")
            }
            try encodeLayer(enc, lb, L: r, kind: .causal)
            enc.endEncoding()
            swap(&scratch.x, &scratch.y)
            if tapIdx < DirectTEForward.taps.count, li + 1 == DirectTEForward.taps[tapIdx] {
                guard let blit = cb.makeBlitCommandEncoder() else {
                    throw DirectQmmSpike.SpikeError.metal("blit")
                }
                blit.copy(
                    from: scratch.x, sourceOffset: 0,
                    to: scratch.tapBufs[tapIdx], destinationOffset: 0,
                    size: r * Self.hidden * 2)
                blit.endEncoding()
                tapIdx += 1
            }
        }
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw DirectQmmSpike.SpikeError.metal("exec: \(e)") }

        var out = [Float](repeating: 0, count: r * Self.hidden * 3)
        for (t, buf) in scratch.tapBufs.enumerated() {
            let p = buf.contents().bindMemory(to: UInt16.self, capacity: r * Self.hidden)
            out.withUnsafeMutableBufferPointer { ob in
                for l in 0 ..< r {
                    let dst = ob.baseAddress! + l * Self.hidden * 3 + t * Self.hidden
                    let src = p + l * Self.hidden
                    for c in 0 ..< Self.hidden {
                        dst[c] = Self.bf16ToF32(src[c])
                    }
                }
            }
        }
        let embeds = MLXArray(out, [1, r, Self.hidden * 3]).asType(.bfloat16)
        eval(embeds)
        let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        return (embeds, r, ms)
    }

    // MARK: - Full encode

    /// Chat-templated, padded-window encode. Returns `[1, 512, 7680]` bf16.
    public func encode(_ prompt: String) throws -> (embeds: MLXArray, realTokens: Int, ms: Double) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let L = Self.maxLen
        let (ids, attn) = tokenizer.encodePrompt(prompt, maxLength: L)
        let realTokens = attn.reduce(0, +)
        lastRealTokens = realTokens

        let xPtr = scratch.x.contents().bindMemory(to: UInt16.self, capacity: L * Self.hidden)
        for (pos, token) in ids.enumerated() {
            embedRow(token, into: xPtr + pos * Self.hidden)
        }
        fillMask(realTokens: realTokens)

        guard let cb = engine.ctx.queue.makeCommandBuffer() else {
            throw DirectQmmSpike.SpikeError.metal("cb")
        }
        var tapIdx = 0
        for (li, lb) in engine.layers.enumerated() {
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw DirectQmmSpike.SpikeError.metal("encoder")
            }
            try encodeLayer(enc, lb, L: L, kind: .maskedFullWindow)
            enc.endEncoding()
            swap(&scratch.x, &scratch.y)
            if tapIdx < DirectTEForward.taps.count, li + 1 == DirectTEForward.taps[tapIdx] {
                guard let blit = cb.makeBlitCommandEncoder() else {
                    throw DirectQmmSpike.SpikeError.metal("blit")
                }
                blit.copy(
                    from: scratch.x, sourceOffset: 0,
                    to: scratch.tapBufs[tapIdx], destinationOffset: 0,
                    size: L * Self.hidden * 2)
                blit.endEncoding()
                tapIdx += 1
            }
        }
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw DirectQmmSpike.SpikeError.metal("exec: \(e)") }

        // Assemble taps → [1, L, 3*hidden] bf16 (via f32 staging).
        var out = [Float](repeating: 0, count: L * Self.hidden * 3)
        for (t, buf) in scratch.tapBufs.enumerated() {
            let p = buf.contents().bindMemory(to: UInt16.self, capacity: L * Self.hidden)
            out.withUnsafeMutableBufferPointer { ob in
                for l in 0 ..< L {
                    let dst = ob.baseAddress! + l * Self.hidden * 3 + t * Self.hidden
                    let src = p + l * Self.hidden
                    for c in 0 ..< Self.hidden {
                        dst[c] = Self.bf16ToF32(src[c])
                    }
                }
            }
        }
        let embeds = MLXArray(out, [1, L, Self.hidden * 3]).asType(.bfloat16)
        eval(embeds)
        let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        return (embeds, realTokens, ms)
    }

    enum AttentionKind { case maskedFullWindow, causal }

    private func encodeLayer(
        _ enc: MTLComputeCommandEncoder, _ lb: DirectTEForward.LayerBuffers,
        L: Int, kind: AttentionKind
    ) throws {
        let ctx = engine.ctx
        let s = scratch
        let nHeads = DirectTEForward.nHeads
        let nKV = DirectTEForward.nKV
        let headDim = DirectTEForward.headDim
        let hidden = Self.hidden
        let inter = DirectTEForward.inter
        typealias LS = DirectTELayerSpike
        LS.encodeRms(enc, ctx, x: s.x, w: lb.wIn, y: s.h1, rows: L, d: hidden, pso: rmsPSO)
        LS.encodeQmm(enc, ctx, w: lb.qmm[0].w, s: lb.qmm[0].s, b: lb.qmm[0].b,
                     x: s.h1, y: s.q, m: L, n: lb.qmm[0].n, k: lb.qmm[0].k, pso: qmmPSO)
        LS.encodeQmm(enc, ctx, w: lb.qmm[1].w, s: lb.qmm[1].s, b: lb.qmm[1].b,
                     x: s.h1, y: s.k, m: L, n: lb.qmm[1].n, k: lb.qmm[1].k, pso: qmmPSO)
        LS.encodeQmm(enc, ctx, w: lb.qmm[2].w, s: lb.qmm[2].s, b: lb.qmm[2].b,
                     x: s.h1, y: s.v, m: L, n: lb.qmm[2].n, k: lb.qmm[2].k, pso: qmmPSO)
        LS.encodeRms(enc, ctx, x: s.q, w: lb.wQn, y: s.qn, rows: L * nHeads, d: headDim, pso: rmsPSO)
        LS.encodeRms(enc, ctx, x: s.k, w: lb.wKn, y: s.kn, rows: L * nKV, d: headDim, pso: rmsPSO)
        LS.encodeRopeL(enc, ctx, x: s.qn, y: s.qr, heads: nHeads, seqLen: L, pso: ropePSO)
        LS.encodeRopeL(enc, ctx, x: s.kn, y: s.kr, heads: nKV, seqLen: L, pso: ropePSO)
        switch kind {
        case .maskedFullWindow:
            encodeMaskedAttention(enc, q: s.qr, k: s.kr, v: s.v, o: s.attnO)
        case .causal:
            try encodeCausalAttention(enc, L: L, q: s.qr, k: s.kr, v: s.v, o: s.attnO)
        }
        LS.encodeQmm(enc, ctx, w: lb.qmm[3].w, s: lb.qmm[3].s, b: lb.qmm[3].b,
                     x: s.attnO, y: s.attnP, m: L, n: lb.qmm[3].n, k: lb.qmm[3].k, pso: qmmPSO)
        LS.encodeAdd(enc, ctx, a: s.x, b: s.attnP, y: s.resid1, n: L * hidden, pso: addPSO)
        LS.encodeRms(enc, ctx, x: s.resid1, w: lb.wPost, y: s.h2, rows: L, d: hidden, pso: rmsPSO)
        LS.encodeQmm(enc, ctx, w: lb.qmm[4].w, s: lb.qmm[4].s, b: lb.qmm[4].b,
                     x: s.h2, y: s.g, m: L, n: lb.qmm[4].n, k: lb.qmm[4].k, pso: qmmPSO)
        LS.encodeQmm(enc, ctx, w: lb.qmm[5].w, s: lb.qmm[5].s, b: lb.qmm[5].b,
                     x: s.h2, y: s.u, m: L, n: lb.qmm[5].n, k: lb.qmm[5].k, pso: qmmPSO)
        LS.encodeSiluMul(enc, ctx, g: s.g, u: s.u, y: s.m, n: L * inter, pso: siluMulPSO)
        LS.encodeQmm(enc, ctx, w: lb.qmm[6].w, s: lb.qmm[6].s, b: lb.qmm[6].b,
                     x: s.m, y: s.d, m: L, n: lb.qmm[6].n, k: lb.qmm[6].k, pso: qmmPSO)
        LS.encodeAdd(enc, ctx, a: s.resid1, b: s.d, y: s.y, n: L * hidden, pso: addPSO)
    }
}
