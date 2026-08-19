import Foundation
import Metal
import MLX
import MLXNN
import ImarelloDiT

/// Stage-2 F1: one FLUX single-stream DiT block (block 0, joint L=1536 ≙ 512²)
/// executed as a single command buffer — metallib qmm ×2 + steel full
/// attention + seven fused glue kernels — verified against the REAL product
/// block (`Flux2SingleTransformerBlock` with the shipped 4-bit weights and
/// the production ÷16/×16 f16-qmm protocol).
public enum DirectDiTBlockSpike {

    static let L = 1536
    static let dim = 3072
    static let heads = 24
    static let headDim = 128
    static let qkvWidth = 9216      // 3 × dim
    static let mlpWidth = 18432     // 2 × 9216
    static let projWidth = 27648    // qkv + mlp
    static let concatWidth = 12288  // attn 3072 + mlpOut 9216

    public static func run(transformerDirectory: URL, metallibURL: URL) throws -> String {
        // -- Weights ----------------------------------------------------------
        var arrays: [String: MLXArray] = [:]
        for shard in ["0.safetensors", "1.safetensors"] {
            let url = transformerDirectory.appendingPathComponent(shard)
            if FileManager.default.fileExists(atPath: url.path) {
                for (k, v) in try MLX.loadArrays(url: url) { arrays[k] = v }
            }
        }
        let prefix = "single_transformer_blocks.0."
        var sub: [String: MLXArray] = [:]
        for (k, v) in arrays where k.hasPrefix(prefix) {
            let key = String(k.dropFirst(prefix.count))
            // Product loader pre-casts qmm scales/biases to f16.
            if key.hasSuffix(".scales") || key.hasSuffix(".biases") {
                sub[key] = v.asType(.float16)
            } else {
                sub[key] = v
            }
        }
        guard !sub.isEmpty else {
            throw DirectQmmSpike.SpikeError.missingTensor("single_transformer_blocks.0.*")
        }

        // -- Oracle: the real product block -----------------------------------
        let block = Flux2SingleTransformerBlock(
            dim: dim, numAttentionHeads: heads, attentionHeadDim: headDim)
        quantize(model: block, groupSize: 64, bits: 4) { _, _ in true }
        try block.update(parameters: ModuleParameters.unflattened(sub), verify: [.all])

        MLXRandom.seed(11)
        let x = (MLXRandom.normal([1, L, dim]) * 0.5).asType(.float32)
        let modShift = (MLXRandom.normal([1, 1, dim]) * 0.1).asType(.float32)
        let modScale = (MLXRandom.normal([1, 1, dim]) * 0.1).asType(.float32)
        let modGate = (MLXRandom.normal([1, 1, dim]) * 0.1).asType(.float32)
        let theta = MLXRandom.uniform(low: 0.0, high: 6.2831853, [L, headDim / 2])
        let cosT = cos(theta).asType(.float32)
        let sinT = sin(theta).asType(.float32)
        eval(x, modShift, modScale, modGate, cosT, sinT)

        let yOracle = block(
            x, tembModParams: (modShift, modScale, modGate),
            imageRotaryEmb: (cosT, sinT))
        eval(yOracle)
        var oracleMS = 0.0
        do {
            for _ in 0 ..< 3 {
                eval(block(x, tembModParams: (modShift, modScale, modGate),
                           imageRotaryEmb: (cosT, sinT)))
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< 10 {
                eval(block(x, tembModParams: (modShift, modScale, modGate),
                           imageRotaryEmb: (cosT, sinT)))
            }
            oracleMS = (CFAbsoluteTimeGetCurrent() - t0) * 100
        }

        // -- Direct -----------------------------------------------------------
        guard let device = MTLCreateSystemDefaultDevice(),
            let queue = device.makeCommandQueue()
        else { throw DirectQmmSpike.SpikeError.metal("device/queue") }
        let mlxLib = try device.makeLibrary(URL: metallibURL)
        guard let qmmFn = mlxLib.makeFunction(
            name: "affine_qmm_t_float16_t_gs_64_b_4_alN_true_batch_0")
        else { throw DirectQmmSpike.SpikeError.metal("qmm kernel") }
        let qmmPSO = try device.makeComputePipelineState(function: qmmFn)

        let consts = MTLFunctionConstantValues()
        var t = true, f = false
        consts.setConstantValue(&t, type: .bool, index: 200)  // align_Q (1536 % 32)
        consts.setConstantValue(&t, type: .bool, index: 201)  // align_K (1536 % 16)
        consts.setConstantValue(&f, type: .bool, index: 300)  // has_mask
        consts.setConstantValue(&f, type: .bool, index: 301)  // do_causal — FULL attention
        consts.setConstantValue(&f, type: .bool, index: 302)
        let attnFn = try mlxLib.makeFunction(
            name: "steel_attention_float16_bq32_bk16_bd128_wm4_wn1_maskfloat16",
            constantValues: consts)
        let attnPSO = try device.makeComputePipelineState(function: attnFn)

        let glue = try DirectDiTKernels.makeLibrary(device: device)
        func pso(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = glue.makeFunction(name: name) else {
                throw DirectQmmSpike.SpikeError.metal("glue \(name)")
            }
            return try device.makeComputePipelineState(function: fn)
        }
        let lnModPSO = try pso("dd_ln_mod_prescale")
        let rmsPSO = try pso("dd_rmsnorm_pitched")
        let ropePSO = try pso("dd_rope_interleaved")
        let scaleCastPSO = try pso("dd_scale_cast_pitched")
        let swigluPSO = try pso("dd_swiglu_pitched")
        let scaleInPSO = try pso("dd_scale_inplace")
        let gateAddPSO = try pso("dd_gate_add")

        func upload(_ a: MLXArray, _ label: String) throws -> MTLBuffer {
            let d = a.asData(noCopy: false)
            return try d.withUnsafeBytes { raw -> MTLBuffer in
                guard let base = raw.baseAddress,
                    let b = device.makeBuffer(bytes: base, length: raw.count)
                else { throw DirectQmmSpike.SpikeError.metal("upload \(label)") }
                b.label = label
                return b
            }
        }
        func scratch(_ bytes: Int, _ label: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: bytes) else {
                throw DirectQmmSpike.SpikeError.metal("scratch \(label)")
            }
            b.label = label
            return b
        }

        let projW = try upload(sub["attn.to_qkv_mlp_proj.weight"]!, "proj.w")
        let projS = try upload(sub["attn.to_qkv_mlp_proj.scales"]!, "proj.s")
        let projB = try upload(sub["attn.to_qkv_mlp_proj.biases"]!, "proj.b")
        let outW = try upload(sub["attn.to_out.weight"]!, "out.w")
        let outS = try upload(sub["attn.to_out.scales"]!, "out.s")
        let outB = try upload(sub["attn.to_out.biases"]!, "out.b")
        let normQW = try upload(sub["attn.norm_q.weight"]!.asType(.float32), "normq")
        let normKW = try upload(sub["attn.norm_k.weight"]!.asType(.float32), "normk")

        let xBuf = try upload(x, "x")
        let shiftBuf = try upload(modShift, "shift")
        let scaleBuf = try upload(modScale, "scale")
        let gateBuf = try upload(modGate, "gate")
        let cosBuf = try upload(cosT, "cos")
        let sinBuf = try upload(sinT, "sin")

        let nhBuf = try scratch(L * dim * 2, "nh16")
        let projBuf = try scratch(L * projWidth * 2, "proj16")
        let qBuf = try scratch(L * qkvWidth / 3 * 2, "q")
        let kBuf = try scratch(L * qkvWidth / 3 * 2, "k")
        let qrBuf = try scratch(L * qkvWidth / 3 * 2, "qr")
        let krBuf = try scratch(L * qkvWidth / 3 * 2, "kr")
        let vBuf = try scratch(L * dim * 2, "v")
        let concatBuf = try scratch(L * concatWidth * 2, "concat")
        let outBuf = try scratch(L * dim * 2, "out16")
        let yBuf = try scratch(L * dim * 4, "y32")

        func encodeQmm(
            _ enc: MTLComputeCommandEncoder,
            w: MTLBuffer, s: MTLBuffer, b: MTLBuffer, x: MTLBuffer, y: MTLBuffer,
            m: Int, n: Int, k: Int
        ) {
            enc.setComputePipelineState(qmmPSO)
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

        func runOnce() throws {
            guard let cb = queue.makeCommandBuffer(),
                let enc = cb.makeComputeCommandEncoder()
            else { throw DirectQmmSpike.SpikeError.metal("cb") }

            // 1. LayerNorm + modulate + ÷16 → nh16
            enc.setComputePipelineState(lnModPSO)
            enc.setBuffer(xBuf, offset: 0, index: 0)
            enc.setBuffer(scaleBuf, offset: 0, index: 1)
            enc.setBuffer(shiftBuf, offset: 0, index: 2)
            enc.setBuffer(nhBuf, offset: 0, index: 3)
            var d32 = Int32(dim)
            enc.setBytes(&d32, length: 4, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: L, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            // 2. fused proj qmm
            encodeQmm(enc, w: projW, s: projS, b: projB, x: nhBuf, y: projBuf,
                      m: L, n: projWidth, k: dim)

            // 3/4. q,k RMSNorm (×16 cancels) → contiguous
            for (off, wBuf, dst) in [(0, normQW, qBuf), (dim, normKW, kBuf)] {
                enc.setComputePipelineState(rmsPSO)
                enc.setBuffer(projBuf, offset: 0, index: 0)
                enc.setBuffer(wBuf, offset: 0, index: 1)
                enc.setBuffer(dst, offset: 0, index: 2)
                var hd = Int32(headDim), pitch = Int32(projWidth)
                var sOff = Int32(off), hpr = Int32(heads)
                var eps = Float(1e-5)
                enc.setBytes(&hd, length: 4, index: 3)
                enc.setBytes(&pitch, length: 4, index: 4)
                enc.setBytes(&sOff, length: 4, index: 5)
                enc.setBytes(&hpr, length: 4, index: 6)
                enc.setBytes(&eps, length: 4, index: 7)
                enc.dispatchThreadgroups(
                    MTLSize(width: L * heads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }

            // 5. RoPE q, k
            for (src, dst) in [(qBuf, qrBuf), (kBuf, krBuf)] {
                enc.setComputePipelineState(ropePSO)
                enc.setBuffer(src, offset: 0, index: 0)
                enc.setBuffer(dst, offset: 0, index: 1)
                enc.setBuffer(cosBuf, offset: 0, index: 2)
                enc.setBuffer(sinBuf, offset: 0, index: 3)
                var h32 = Int32(heads), hd = Int32(headDim)
                enc.setBytes(&h32, length: 4, index: 4)
                enc.setBytes(&hd, length: 4, index: 5)
                enc.dispatchThreads(
                    MTLSize(width: headDim / 2, height: heads, depth: L),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }

            // 6. V extraction ×16
            enc.setComputePipelineState(scaleCastPSO)
            enc.setBuffer(projBuf, offset: 0, index: 0)
            enc.setBuffer(vBuf, offset: 0, index: 1)
            var pitch = Int32(projWidth), vOff = Int32(2 * dim)
            var vW = Int32(dim)
            var sixteen = Float(16)
            enc.setBytes(&pitch, length: 4, index: 2)
            enc.setBytes(&vOff, length: 4, index: 3)
            enc.setBytes(&vW, length: 4, index: 4)
            enc.setBytes(&sixteen, length: 4, index: 5)
            enc.dispatchThreads(
                MTLSize(width: dim, height: L, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            // 7. steel full attention → concat left (via O strides)
            enc.setComputePipelineState(attnPSO)
            enc.setBuffer(qrBuf, offset: 0, index: 0)
            enc.setBuffer(krBuf, offset: 0, index: 1)
            enc.setBuffer(vBuf, offset: 0, index: 2)
            enc.setBuffer(concatBuf, offset: 0, index: 3)
            var params = Data(capacity: 152)
            func i32p(_ v: Int) { var x = Int32(v); withUnsafeBytes(of: &x) { params.append(contentsOf: $0) } }
            func f32p(_ v: Float) { var x = v; withUnsafeBytes(of: &x) { params.append(contentsOf: $0) } }
            func i64x3p(_ a: Int, _ b: Int, _ c: Int) {
                for v in [Int64(a), Int64(b), Int64(c)] {
                    var x = v; withUnsafeBytes(of: &x) { params.append(contentsOf: $0) }
                }
            }
            let bq = 32, bk = 16
            i32p(1); i32p(heads); i32p(headDim); i32p(L); i32p(L)
            i32p(1)  // gqa_factor
            f32p(1.0 / Float(Double(headDim).squareRoot()))
            i32p(L / bq); i32p(L / bk)
            i32p(L / bq); i32p(L / bk)
            i32p(0); i32p(0); i32p(0)
            i64x3p(L * heads * headDim, headDim, heads * headDim)  // Q
            i64x3p(L * heads * headDim, headDim, heads * headDim)  // K
            i64x3p(L * heads * headDim, headDim, heads * headDim)  // V
            i64x3p(L * concatWidth, headDim, concatWidth)          // O → concat left
            params.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: params.count, index: 4) }
            enc.dispatchThreadgroups(
                MTLSize(width: L / bq, height: heads, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))

            // 8. SwiGLU → concat right (in ×16, out ÷16)
            enc.setComputePipelineState(swigluPSO)
            enc.setBuffer(projBuf, offset: 0, index: 0)
            enc.setBuffer(concatBuf, offset: 0, index: 1)
            var gOff = Int32(qkvWidth), uOff = Int32(qkvWidth + mlpWidth / 2)
            var sw = Int32(mlpWidth / 2), oPitch = Int32(concatWidth)
            var oOff = Int32(dim)
            var swInS = Float(16), swOutS = Float(1.0 / 16.0)
            enc.setBytes(&pitch, length: 4, index: 2)
            enc.setBytes(&gOff, length: 4, index: 3)
            enc.setBytes(&uOff, length: 4, index: 4)
            enc.setBytes(&sw, length: 4, index: 5)
            enc.setBytes(&oPitch, length: 4, index: 6)
            enc.setBytes(&oOff, length: 4, index: 7)
            enc.setBytes(&swInS, length: 4, index: 8)
            enc.setBytes(&swOutS, length: 4, index: 9)
            enc.dispatchThreads(
                MTLSize(width: mlpWidth / 2, height: L, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            // 9. attn half ÷16 (uniform to_out input scaling)
            enc.setComputePipelineState(scaleInPSO)
            enc.setBuffer(concatBuf, offset: 0, index: 0)
            var cPitch = Int32(concatWidth), zero = Int32(0)
            var aW = Int32(dim)
            var inv16 = Float(1.0 / 16.0)
            enc.setBytes(&cPitch, length: 4, index: 1)
            enc.setBytes(&zero, length: 4, index: 2)
            enc.setBytes(&aW, length: 4, index: 3)
            enc.setBytes(&inv16, length: 4, index: 4)
            enc.dispatchThreads(
                MTLSize(width: dim, height: L, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            // 10. to_out qmm
            encodeQmm(enc, w: outW, s: outS, b: outB, x: concatBuf, y: outBuf,
                      m: L, n: dim, k: concatWidth)

            // 11. gated residual (×16 restores scale)
            enc.setComputePipelineState(gateAddPSO)
            enc.setBuffer(xBuf, offset: 0, index: 0)
            enc.setBuffer(gateBuf, offset: 0, index: 1)
            enc.setBuffer(outBuf, offset: 0, index: 2)
            enc.setBuffer(yBuf, offset: 0, index: 3)
            enc.setBytes(&d32, length: 4, index: 4)
            enc.setBytes(&sixteen, length: 4, index: 5)
            enc.dispatchThreads(
                MTLSize(width: dim, height: L, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw DirectQmmSpike.SpikeError.metal("exec: \(e)") }
        }

        try runOnce()  // warm
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< 10 { try runOnce() }
        let directMS = (CFAbsoluteTimeGetCurrent() - t0) * 100

        // -- Compare ----------------------------------------------------------
        let yPtr = yBuf.contents().bindMemory(to: Float.self, capacity: L * dim)
        let ref = yOracle.reshaped([L, dim]).asType(.float32).asArray(Float.self)
        var dot = 0.0, na = 0.0, nb = 0.0, maxDiff = 0.0
        for i in 0 ..< L * dim {
            let a = Double(yPtr[i]), b = Double(ref[i])
            dot += a * b; na += a * a; nb += b * b
            maxDiff = max(maxDiff, abs(a - b))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
        return """
        direct-dit-block (F1) — single-stream block 0, joint L=\(L), one command buffer, 11 dispatches
          cosine_vs_product_block: \(String(format: "%.7f", cosine))
          max_abs_diff:            \(String(format: "%.5f", maxDiff))
          oracle_per_block:        \(String(format: "%.2f", oracleMS)) ms (MLX product path, warm)
          direct_per_block:        \(String(format: "%.2f", directMS)) ms (single CB, blocking wait)
          verdict:                 \(cosine >= 0.9999 ? "PASS" : "investigate")
        """
    }
}
