import Foundation
import Metal
import MLX
import MLXNN
import ImarelloDiT

/// Stage-2 F2: one FLUX double-stream block (block 0) at 512² shapes
/// (L_img=1024, L_txt=512, joint 1536) as a single command buffer — 10 qmm +
/// steel joint attention + the F1 glue kernels — verified against the REAL
/// product `Flux2TransformerBlock` with shipped weights.
public enum DirectDiTDoubleBlockSpike {

    static let lImg = 1024
    static let lTxt = 512
    static let J = 1536
    static let dim = 3072
    static let heads = 24
    static let headDim = 128
    static let ffInner = 9216
    static let ffWide = 18432

    struct Q {
        let w: MTLBuffer
        let s: MTLBuffer
        let b: MTLBuffer
        let n: Int
        let k: Int
    }

    public static func run(transformerDirectory: URL, metallibURL: URL) throws -> String {
        var arrays: [String: MLXArray] = [:]
        for shard in ["0.safetensors", "1.safetensors"] {
            let url = transformerDirectory.appendingPathComponent(shard)
            if FileManager.default.fileExists(atPath: url.path) {
                for (k, v) in try MLX.loadArrays(url: url) { arrays[k] = v }
            }
        }
        let prefix = "transformer_blocks.0."
        var sub: [String: MLXArray] = [:]
        for (k, v) in arrays where k.hasPrefix(prefix) {
            let key = String(k.dropFirst(prefix.count))
            if key.hasSuffix(".scales") || key.hasSuffix(".biases") {
                sub[key] = v.asType(.float16)
            } else {
                sub[key] = v
            }
        }
        guard !sub.isEmpty else {
            throw DirectQmmSpike.SpikeError.missingTensor("transformer_blocks.0.*")
        }

        // -- Oracle -----------------------------------------------------------
        let block = Flux2TransformerBlock(
            dim: dim, numAttentionHeads: heads, attentionHeadDim: headDim)
        quantize(model: block, groupSize: 64, bits: 4) { _, _ in true }
        try block.update(parameters: ModuleParameters.unflattened(sub), verify: [.all])

        MLXRandom.seed(13)
        let h = (MLXRandom.normal([1, lImg, dim]) * 0.5).asType(.float32)
        let e = (MLXRandom.normal([1, lTxt, dim]) * 0.5).asType(.float32)
        func triple() -> (MLXArray, MLXArray, MLXArray) {
            ((MLXRandom.normal([1, 1, dim]) * 0.1).asType(.float32),
             (MLXRandom.normal([1, 1, dim]) * 0.1).asType(.float32),
             (MLXRandom.normal([1, 1, dim]) * 0.1).asType(.float32))
        }
        let mImg = [triple(), triple()]
        let mTxt = [triple(), triple()]
        let theta = MLXRandom.uniform(low: 0.0, high: 6.2831853, [J, headDim / 2])
        let cosT = cos(theta).asType(.float32)
        let sinT = sin(theta).asType(.float32)
        eval(h, e, cosT, sinT)

        let (oE, oH) = block(
            hiddenStates: h, encoderHiddenStates: e,
            tembModParamsImg: mImg, tembModParamsTxt: mTxt,
            imageRotaryEmb: (cosT, sinT))
        eval(oE, oH)
        var oracleMS = 0.0
        do {
            for _ in 0 ..< 3 {
                let (a, b) = block(
                    hiddenStates: h, encoderHiddenStates: e,
                    tembModParamsImg: mImg, tembModParamsTxt: mTxt,
                    imageRotaryEmb: (cosT, sinT))
                eval(a, b)
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< 10 {
                let (a, b) = block(
                    hiddenStates: h, encoderHiddenStates: e,
                    tembModParamsImg: mImg, tembModParamsTxt: mTxt,
                    imageRotaryEmb: (cosT, sinT))
                eval(a, b)
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
        var tt = true, ff = false
        consts.setConstantValue(&tt, type: .bool, index: 200)
        consts.setConstantValue(&tt, type: .bool, index: 201)
        consts.setConstantValue(&ff, type: .bool, index: 300)
        consts.setConstantValue(&ff, type: .bool, index: 301)
        consts.setConstantValue(&ff, type: .bool, index: 302)
        let attnPSO = try device.makeComputePipelineState(
            function: try mlxLib.makeFunction(
                name: "steel_attention_float16_bq32_bk16_bd128_wm4_wn1_maskfloat16",
                constantValues: consts))
        let glue = try DirectDiTKernels.makeLibrary(device: device)
        func pso(_ n: String) throws -> MTLComputePipelineState {
            guard let f = glue.makeFunction(name: n) else {
                throw DirectQmmSpike.SpikeError.metal("glue \(n)")
            }
            return try device.makeComputePipelineState(function: f)
        }
        let lnModPSO = try pso("dd_ln_mod_prescale")
        let rmsPSO = try pso("dd_rmsnorm_pitched")
        let ropePSO = try pso("dd_rope_interleaved")
        let scaleCastPSO = try pso("dd_scale_cast_pitched")
        let swigluPSO = try pso("dd_swiglu_pitched")
        let scaleInPSO = try pso("dd_scale_inplace")
        let gateAddPSO = try pso("dd_gate_add")

        func upload(_ a: MLXArray, _ l: String) throws -> MTLBuffer {
            let d = a.asData(noCopy: false)
            return try d.withUnsafeBytes { raw -> MTLBuffer in
                guard let base = raw.baseAddress,
                    let b = device.makeBuffer(bytes: base, length: raw.count)
                else { throw DirectQmmSpike.SpikeError.metal("upload \(l)") }
                b.label = l
                return b
            }
        }
        func scratch(_ bytes: Int, _ l: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: bytes) else {
                throw DirectQmmSpike.SpikeError.metal("scratch \(l)")
            }
            b.label = l
            return b
        }
        func quant(_ name: String, n: Int, k: Int) throws -> Q {
            Q(w: try upload(sub["\(name).weight"]!, "\(name).w"),
              s: try upload(sub["\(name).scales"]!, "\(name).s"),
              b: try upload(sub["\(name).biases"]!, "\(name).b"),
              n: n, k: k)
        }

        let toQ = try quant("attn.to_q", n: dim, k: dim)
        let toK = try quant("attn.to_k", n: dim, k: dim)
        let toV = try quant("attn.to_v", n: dim, k: dim)
        let addQ = try quant("attn.add_q_proj", n: dim, k: dim)
        let addK = try quant("attn.add_k_proj", n: dim, k: dim)
        let addV = try quant("attn.add_v_proj", n: dim, k: dim)
        let toOut = try quant("attn.to_out", n: dim, k: dim)
        let toAddOut = try quant("attn.to_add_out", n: dim, k: dim)
        let ffIn = try quant("ff.linear_in", n: ffWide, k: dim)
        let ffOut = try quant("ff.linear_out", n: dim, k: ffInner)
        let ffcIn = try quant("ff_context.linear_in", n: ffWide, k: dim)
        let ffcOut = try quant("ff_context.linear_out", n: dim, k: ffInner)
        let normQW = try upload(sub["attn.norm_q.weight"]!.asType(.float32), "nq")
        let normKW = try upload(sub["attn.norm_k.weight"]!.asType(.float32), "nk")
        let normAQW = try upload(sub["attn.norm_added_q.weight"]!.asType(.float32), "naq")
        let normAKW = try upload(sub["attn.norm_added_k.weight"]!.asType(.float32), "nak")

        let hBuf = try upload(h, "h")
        let eBuf = try upload(e, "e")
        let cosBuf = try upload(cosT, "cos")
        let sinBuf = try upload(sinT, "sin")
        func vecs(_ t: (MLXArray, MLXArray, MLXArray), _ tag: String) throws -> (MTLBuffer, MTLBuffer, MTLBuffer) {
            (try upload(t.0, "\(tag).shift"), try upload(t.1, "\(tag).scale"), try upload(t.2, "\(tag).gate"))
        }
        let mImgMsa = try vecs(mImg[0], "img.msa")
        let mImgMlp = try vecs(mImg[1], "img.mlp")
        let mTxtMsa = try vecs(mTxt[0], "txt.msa")
        let mTxtMlp = try vecs(mTxt[1], "txt.mlp")

        let nhImg = try scratch(lImg * dim * 2, "nhImg")
        let neTxt = try scratch(lTxt * dim * 2, "neTxt")
        let projImg = try scratch(lImg * dim * 2, "projImg")   // reused per q/k/v
        let projTxt = try scratch(lTxt * dim * 2, "projTxt")
        let jointQ = try scratch(J * dim * 2, "jointQ")
        let jointK = try scratch(J * dim * 2, "jointK")
        let jointV = try scratch(J * dim * 2, "jointV")
        let jointQr = try scratch(J * dim * 2, "jointQr")
        let jointKr = try scratch(J * dim * 2, "jointKr")
        let jointO = try scratch(J * dim * 2, "jointO")
        let outImg = try scratch(lImg * dim * 2, "outImg")
        let outTxt = try scratch(lTxt * dim * 2, "outTxt")
        let y1Img = try scratch(lImg * dim * 4, "y1Img")
        let y1Txt = try scratch(lTxt * dim * 4, "y1Txt")
        let ffWideImg = try scratch(lImg * ffWide * 2, "ffWideImg")
        let ffWideTxt = try scratch(lTxt * ffWide * 2, "ffWideTxt")
        let swImg = try scratch(lImg * ffInner * 2, "swImg")
        let swTxt = try scratch(lTxt * ffInner * 2, "swTxt")
        let ffOutImg = try scratch(lImg * dim * 2, "ffOutImg")
        let ffOutTxt = try scratch(lTxt * dim * 2, "ffOutTxt")
        let y2Img = try scratch(lImg * dim * 4, "y2Img")
        let y2Txt = try scratch(lTxt * dim * 4, "y2Txt")

        let txtByteOff = lTxt * dim * 2  // image-part offset inside joint buffers

        func runOnce() throws {
            guard let cb = queue.makeCommandBuffer(),
                let enc = cb.makeComputeCommandEncoder()
            else { throw DirectQmmSpike.SpikeError.metal("cb") }

            func lnMod(_ x: MTLBuffer, _ sc: MTLBuffer, _ sh: MTLBuffer, _ y: MTLBuffer, rows: Int) {
                enc.setComputePipelineState(lnModPSO)
                enc.setBuffer(x, offset: 0, index: 0)
                enc.setBuffer(sc, offset: 0, index: 1)
                enc.setBuffer(sh, offset: 0, index: 2)
                enc.setBuffer(y, offset: 0, index: 3)
                var d32 = Int32(dim)
                enc.setBytes(&d32, length: 4, index: 4)
                enc.dispatchThreadgroups(
                    MTLSize(width: rows, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            func qmm(_ q: Q, x: MTLBuffer, xOff: Int, y: MTLBuffer, yOff: Int, m: Int) {
                enc.setComputePipelineState(qmmPSO)
                enc.setBuffer(q.w, offset: 0, index: 0)
                enc.setBuffer(q.s, offset: 0, index: 1)
                enc.setBuffer(q.b, offset: 0, index: 2)
                enc.setBuffer(x, offset: xOff, index: 3)
                enc.setBuffer(y, offset: yOff, index: 4)
                var k32 = Int32(q.k), n32 = Int32(q.n), m32 = Int32(m)
                enc.setBytes(&k32, length: 4, index: 5)
                enc.setBytes(&n32, length: 4, index: 6)
                enc.setBytes(&m32, length: 4, index: 7)
                enc.dispatchThreadgroups(
                    MTLSize(width: (q.n + 31) / 32, height: (m + 31) / 32, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
            }
            func rms(_ x: MTLBuffer, _ w: MTLBuffer, _ y: MTLBuffer, yOff: Int, rows: Int) {
                enc.setComputePipelineState(rmsPSO)
                enc.setBuffer(x, offset: 0, index: 0)
                enc.setBuffer(w, offset: 0, index: 1)
                enc.setBuffer(y, offset: yOff, index: 2)
                var hd = Int32(headDim), pitch = Int32(dim), off0 = Int32(0)
                var hpr = Int32(heads)
                var eps = Float(1e-5)
                enc.setBytes(&hd, length: 4, index: 3)
                enc.setBytes(&pitch, length: 4, index: 4)
                enc.setBytes(&off0, length: 4, index: 5)
                enc.setBytes(&hpr, length: 4, index: 6)
                enc.setBytes(&eps, length: 4, index: 7)
                enc.dispatchThreadgroups(
                    MTLSize(width: rows * heads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }
            func scaleCast(_ x: MTLBuffer, _ y: MTLBuffer, yOff: Int, rows: Int, scale: Float) {
                enc.setComputePipelineState(scaleCastPSO)
                enc.setBuffer(x, offset: 0, index: 0)
                enc.setBuffer(y, offset: yOff, index: 1)
                var pitch = Int32(dim), off0 = Int32(0), w32 = Int32(dim)
                var s = scale
                enc.setBytes(&pitch, length: 4, index: 2)
                enc.setBytes(&off0, length: 4, index: 3)
                enc.setBytes(&w32, length: 4, index: 4)
                enc.setBytes(&s, length: 4, index: 5)
                enc.dispatchThreads(
                    MTLSize(width: dim, height: rows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            func gateAdd(_ x: MTLBuffer, _ g: MTLBuffer, _ v: MTLBuffer, _ y: MTLBuffer, rows: Int) {
                enc.setComputePipelineState(gateAddPSO)
                enc.setBuffer(x, offset: 0, index: 0)
                enc.setBuffer(g, offset: 0, index: 1)
                enc.setBuffer(v, offset: 0, index: 2)
                enc.setBuffer(y, offset: 0, index: 3)
                var d32 = Int32(dim)
                var sixteen = Float(16)
                enc.setBytes(&d32, length: 4, index: 4)
                enc.setBytes(&sixteen, length: 4, index: 5)
                enc.dispatchThreads(
                    MTLSize(width: dim, height: rows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            func scaleInPlace(_ x: MTLBuffer, rows: Int, width: Int, pitch: Int, scale: Float) {
                enc.setComputePipelineState(scaleInPSO)
                enc.setBuffer(x, offset: 0, index: 0)
                var p = Int32(pitch), o = Int32(0), w32 = Int32(width)
                var s = scale
                enc.setBytes(&p, length: 4, index: 1)
                enc.setBytes(&o, length: 4, index: 2)
                enc.setBytes(&w32, length: 4, index: 3)
                enc.setBytes(&s, length: 4, index: 4)
                enc.dispatchThreads(
                    MTLSize(width: width, height: rows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            func swiglu(_ x: MTLBuffer, _ y: MTLBuffer, rows: Int) {
                enc.setComputePipelineState(swigluPSO)
                enc.setBuffer(x, offset: 0, index: 0)
                enc.setBuffer(y, offset: 0, index: 1)
                var pitch = Int32(ffWide), gOff = Int32(0), uOff = Int32(ffInner)
                var w32 = Int32(ffInner), oPitch = Int32(ffInner), oOff = Int32(0)
                var inS = Float(16), outS = Float(1.0 / 16.0)
                enc.setBytes(&pitch, length: 4, index: 2)
                enc.setBytes(&gOff, length: 4, index: 3)
                enc.setBytes(&uOff, length: 4, index: 4)
                enc.setBytes(&w32, length: 4, index: 5)
                enc.setBytes(&oPitch, length: 4, index: 6)
                enc.setBytes(&oOff, length: 4, index: 7)
                enc.setBytes(&inS, length: 4, index: 8)
                enc.setBytes(&outS, length: 4, index: 9)
                enc.dispatchThreads(
                    MTLSize(width: ffInner, height: rows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }

            // Attention halves: modulated norms, projections, joint assembly.
            lnMod(hBuf, mImgMsa.1, mImgMsa.0, nhImg, rows: lImg)
            lnMod(eBuf, mTxtMsa.1, mTxtMsa.0, neTxt, rows: lTxt)
            // Text stream → joint rows 0..<512
            qmm(addQ, x: neTxt, xOff: 0, y: projTxt, yOff: 0, m: lTxt)
            rms(projTxt, normAQW, jointQ, yOff: 0, rows: lTxt)
            qmm(addK, x: neTxt, xOff: 0, y: projTxt, yOff: 0, m: lTxt)
            rms(projTxt, normAKW, jointK, yOff: 0, rows: lTxt)
            qmm(addV, x: neTxt, xOff: 0, y: projTxt, yOff: 0, m: lTxt)
            scaleCast(projTxt, jointV, yOff: 0, rows: lTxt, scale: 16)
            // Image stream → joint rows 512...
            qmm(toQ, x: nhImg, xOff: 0, y: projImg, yOff: 0, m: lImg)
            rms(projImg, normQW, jointQ, yOff: txtByteOff, rows: lImg)
            qmm(toK, x: nhImg, xOff: 0, y: projImg, yOff: 0, m: lImg)
            rms(projImg, normKW, jointK, yOff: txtByteOff, rows: lImg)
            qmm(toV, x: nhImg, xOff: 0, y: projImg, yOff: 0, m: lImg)
            scaleCast(projImg, jointV, yOff: txtByteOff, rows: lImg, scale: 16)

            // Joint RoPE
            for (src, dst) in [(jointQ, jointQr), (jointK, jointKr)] {
                enc.setComputePipelineState(ropePSO)
                enc.setBuffer(src, offset: 0, index: 0)
                enc.setBuffer(dst, offset: 0, index: 1)
                enc.setBuffer(cosBuf, offset: 0, index: 2)
                enc.setBuffer(sinBuf, offset: 0, index: 3)
                var h32 = Int32(heads), hd = Int32(headDim)
                enc.setBytes(&h32, length: 4, index: 4)
                enc.setBytes(&hd, length: 4, index: 5)
                enc.dispatchThreads(
                    MTLSize(width: headDim / 2, height: heads, depth: J),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            }

            // Steel joint attention
            enc.setComputePipelineState(attnPSO)
            enc.setBuffer(jointQr, offset: 0, index: 0)
            enc.setBuffer(jointKr, offset: 0, index: 1)
            enc.setBuffer(jointV, offset: 0, index: 2)
            enc.setBuffer(jointO, offset: 0, index: 3)
            var params = Data(capacity: 152)
            func i32p(_ v: Int) { var x = Int32(v); withUnsafeBytes(of: &x) { params.append(contentsOf: $0) } }
            func f32p(_ v: Float) { var x = v; withUnsafeBytes(of: &x) { params.append(contentsOf: $0) } }
            func i64x3p(_ a: Int, _ b: Int, _ c: Int) {
                for v in [Int64(a), Int64(b), Int64(c)] {
                    var x = v; withUnsafeBytes(of: &x) { params.append(contentsOf: $0) }
                }
            }
            let bq = 32, bk = 16
            i32p(1); i32p(heads); i32p(headDim); i32p(J); i32p(J)
            i32p(1)
            f32p(1.0 / Float(Double(headDim).squareRoot()))
            i32p(J / bq); i32p(J / bk)
            i32p(J / bq); i32p(J / bk)
            i32p(0); i32p(0); i32p(0)
            i64x3p(J * dim, headDim, dim)
            i64x3p(J * dim, headDim, dim)
            i64x3p(J * dim, headDim, dim)
            i64x3p(J * dim, headDim, dim)
            params.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: params.count, index: 4) }
            enc.dispatchThreadgroups(
                MTLSize(width: J / bq, height: heads, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))

            // Out projections (÷16 whole joint, then per-stream qmm, then gates)
            scaleInPlace(jointO, rows: J, width: dim, pitch: dim, scale: 1.0 / 16.0)
            qmm(toAddOut, x: jointO, xOff: 0, y: outTxt, yOff: 0, m: lTxt)
            qmm(toOut, x: jointO, xOff: txtByteOff, y: outImg, yOff: 0, m: lImg)
            gateAdd(hBuf, mImgMsa.2, outImg, y1Img, rows: lImg)
            gateAdd(eBuf, mTxtMsa.2, outTxt, y1Txt, rows: lTxt)

            // FFN halves
            lnMod(y1Img, mImgMlp.1, mImgMlp.0, nhImg, rows: lImg)
            qmm(ffIn, x: nhImg, xOff: 0, y: ffWideImg, yOff: 0, m: lImg)
            swiglu(ffWideImg, swImg, rows: lImg)
            qmm(ffOut, x: swImg, xOff: 0, y: ffOutImg, yOff: 0, m: lImg)
            gateAdd(y1Img, mImgMlp.2, ffOutImg, y2Img, rows: lImg)

            lnMod(y1Txt, mTxtMlp.1, mTxtMlp.0, neTxt, rows: lTxt)
            qmm(ffcIn, x: neTxt, xOff: 0, y: ffWideTxt, yOff: 0, m: lTxt)
            swiglu(ffWideTxt, swTxt, rows: lTxt)
            qmm(ffcOut, x: swTxt, xOff: 0, y: ffOutTxt, yOff: 0, m: lTxt)
            gateAdd(y1Txt, mTxtMlp.2, ffOutTxt, y2Txt, rows: lTxt)

            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let err = cb.error { throw DirectQmmSpike.SpikeError.metal("exec: \(err)") }
        }

        try runOnce()
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< 10 { try runOnce() }
        let directMS = (CFAbsoluteTimeGetCurrent() - t0) * 100

        func compare(_ buf: MTLBuffer, _ oracle: MLXArray, _ count: Int, _ tag: String) -> String {
            let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
            let ref = oracle.asType(.float32).asArray(Float.self)
            var dot = 0.0, na = 0.0, nb = 0.0, maxDiff = 0.0
            for i in 0 ..< count {
                let a = Double(ptr[i]), b = Double(ref[i])
                dot += a * b; na += a * a; nb += b * b
                maxDiff = max(maxDiff, abs(a - b))
            }
            let cos = dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
            return String(format: "  %-6s cosine=%.7f maxAbs=%.5f", (tag as NSString).utf8String!, cos, maxDiff)
        }
        let lines = [
            compare(y2Img, oH.reshaped([lImg, dim]), lImg * dim, "image"),
            compare(y2Txt, oE.reshaped([lTxt, dim]), lTxt * dim, "text"),
        ]
        return """
        direct-dit-double (F2) — double block 0, L_img=\(lImg) L_txt=\(lTxt), one command buffer, 26 dispatches
        \(lines.joined(separator: "\n"))
          oracle_per_block: \(String(format: "%.2f", oracleMS)) ms (MLX product path, warm)
          direct_per_block: \(String(format: "%.2f", directMS)) ms (single CB, blocking wait)
        """
    }
}
