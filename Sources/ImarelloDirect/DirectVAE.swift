import Foundation
import Metal
import MLX
import MLXNN
import ImarelloVAE
import ImarelloCore
import ImarelloWeights
import ImarelloRuntime

/// Direct VAE decoder engine (Stage 2, V-track). Convs run on the metallib's
/// f32 implicit-gemm kernels; everything else on `DirectVAEKernels`. Weights
/// load through the product's own `CompVisVAEMapper` (name + layout truth).
public final class DirectVAE {

    let device: MTLDevice
    let queue: MTLCommandQueue
    let mlxLib: MTLLibrary
    let glue: MTLLibrary
    var convPSOs: [String: MTLComputePipelineState] = [:]
    let gnPartialPSO, gnFinalizePSO, gnApplyPSO: MTLComputePipelineState
    let biasPSO, upPSO, addPSO, mmNTPSO, mmNNPSO, softmaxPSO: MTLComputePipelineState
    var gnPartial: MTLBuffer?
    var gnStats: MTLBuffer?

    /// Module-named f32 weights (mapper output), uploaded once.
    var weights: [String: MTLBuffer] = [:]
    var bnMean: MLXArray?
    var bnStd: MLXArray?
    var shapes: [String: [Int]] = [:]
    public private(set) var ownedBytes = 0

    public init(smallDecoderFile: URL, metallibURL: URL) throws {
        guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else {
            throw DirectQmmSpike.SpikeError.metal("device/queue")
        }
        device = dev
        queue = q
        mlxLib = try dev.makeLibrary(URL: metallibURL)
        let glueLib = try DirectVAEKernels.makeLibrary(device: dev)
        glue = glueLib
        func gpso(_ n: String) throws -> MTLComputePipelineState {
            guard let f = glueLib.makeFunction(name: n) else {
                throw DirectQmmSpike.SpikeError.metal("glue \(n)")
            }
            return try dev.makeComputePipelineState(function: f)
        }
        gnPartialPSO = try gpso("dv_gn_partial")
        gnFinalizePSO = try gpso("dv_gn_finalize")
        gnApplyPSO = try gpso("dv_gn_apply")
        biasPSO = try gpso("dv_bias_act")
        upPSO = try gpso("dv_upsample2")
        addPSO = try gpso("dv_add")
        mmNTPSO = try gpso("dv_matmul_nt")
        mmNNPSO = try gpso("dv_matmul_nn")
        softmaxPSO = try gpso("dv_softmax_rows")

        let raw = try MLX.loadArrays(url: smallDecoderFile)
        let mapped = try CompVisVAEMapper.remap(raw)
        for (k, v) in mapped {
            let f = v.asType(.float32)
            eval(f)
            let d = f.asData(noCopy: false)
            let buf = try d.withUnsafeBytes { rawB -> MTLBuffer in
                guard let base = rawB.baseAddress,
                    let b = dev.makeBuffer(bytes: base, length: rawB.count)
                else { throw DirectQmmSpike.SpikeError.metal("upload \(k)") }
                b.label = k
                return b
            }
            weights[k] = buf
            shapes[k] = f.shape
            ownedBytes += d.count
        }
    }

    func w(_ key: String) throws -> MTLBuffer {
        guard let b = weights[key] else {
            throw DirectQmmSpike.SpikeError.missingTensor(key)
        }
        return b
    }

    public func scratch(_ bytes: Int, _ l: String) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: bytes) else {
            throw DirectQmmSpike.SpikeError.metal("scratch \(l)")
        }
        b.label = l
        ownedBytes += bytes
        return b
    }

    // MARK: - Conv (metallib implicit-gemm, f32, stride 1, pad k/2)

    func convPSO(bm: Int, bn: Int, wm: Int, wn: Int, channelSpec: Int, smallFilter: Bool) throws -> MTLComputePipelineState {
        let chan = channelSpec > 0 ? String(channelSpec) : "l"
        let name = "implicit_gemm_conv_2d_float32_bm\(bm)_bn\(bn)_bk16_wm\(wm)_wn\(wn)_channel_\(chan)_filter_\(smallFilter ? "s" : "l")"
        if let p = convPSOs[name] { return p }
        guard let fn = mlxLib.makeFunction(name: name) else {
            throw DirectQmmSpike.SpikeError.metal("conv kernel \(name)")
        }
        let p = try device.makeComputePipelineState(function: fn)
        convPSOs[name] = p
        return p
    }

    func encodeConv(
        _ enc: MTLComputeCommandEncoder,
        x: MTLBuffer, wt: MTLBuffer, y: MTLBuffer,
        H: Int, W: Int, C: Int, O: Int, k: Int
    ) throws {
        let pad = k / 2
        let implicitM = H * W
        let implicitN = O
        let implicitK = k * k * C
        var wm = 2, wn = 2
        let bm = implicitM >= 8192 && C >= 64 ? 64 : 32
        var bn = (bm == 64 || implicitN >= 64) ? 64 : 32
        if implicitN <= 16 { bn = 8; wm = 4; wn = 1 }
        let bk = 16
        let tn = (implicitN + bn - 1) / bn
        let tm = (implicitM + bm - 1) / bm
        var channelSpec = 0
        var gemmKIters = k * k * ((C + bk - 1) / bk)
        if C <= 2 {
            gemmKIters = (implicitK + bk - 1) / bk
            channelSpec = C
        } else if C <= 4 {
            gemmKIters = ((k * k * 4) + bk - 1) / bk
            channelSpec = C
        }
        let smallFilter = channelSpec == 0 && k <= 16
        let ijw = C, ijh = W * C
        let inpJumpW = ijw
        let inpJumpH = ijh - (k - 1) * ijw
        let inpJumpC = bk - (k - 1) * ijh - (k - 1) * ijw

        let pso = try convPSO(bm: bm, bn: bn, wm: wm, wn: wn, channelSpec: channelSpec, smallFilter: smallFilter)
        enc.setComputePipelineState(pso)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(wt, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)

        var params = Data()
        func i32(_ v: Int) { var t = Int32(v); withUnsafeBytes(of: &t) { params.append(contentsOf: $0) } }
        func i64(_ v: Int) { var t = Int64(v); withUnsafeBytes(of: &t) { params.append(contentsOf: $0) } }
        i32(1); i32(C); i32(O)
        i32(H); i32(W)
        i32(k); i32(k)
        i32(H); i32(W)
        i32(1); i32(1)
        i32(pad); i32(pad)
        i32(1); i32(1)
        i32(1); i32(1)
        params.append(Data(repeating: 0, count: (8 - params.count % 8) % 8))
        i64(H * W * C); i64(W * C); i64(C); i64(1)
        i64(k * k * C); i64(k * C); i64(C); i64(1)
        i64(H * W * O); i64(W * O); i64(O); i64(1)
        i32(1)
        var flip = false
        withUnsafeBytes(of: &flip) { params.append(contentsOf: $0.prefix(1)) }
        params.append(Data(repeating: 0, count: (8 - params.count % 8) % 8))
        params.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: params.count, index: 3) }

        var gemm = Data()
        for v in [implicitM, implicitN, implicitK, gemmKIters,
                  inpJumpW, inpJumpH, inpJumpC, tn, tm, 0] {
            var t = Int32(v)
            withUnsafeBytes(of: &t) { gemm.append(contentsOf: $0) }
        }
        gemm.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: gemm.count, index: 4) }
        enc.dispatchThreadgroups(
            MTLSize(width: tn, height: tm, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: wn, depth: wm))
    }

    // MARK: - Glue encoders

    func encodeGroupNormAct(
        _ enc: MTLComputeCommandEncoder,
        x: MTLBuffer, gamma: MTLBuffer, beta: MTLBuffer, y: MTLBuffer,
        HW: Int, C: Int, silu: Bool, groups: Int = 32, eps: Float = 1e-6
    ) {
        let chunks = 256
        if gnPartial == nil {
            // Sized for the widest channel count (384); reused across calls —
            // hazard tracking serializes successive GNs in one command buffer.
            gnPartial = device.makeBuffer(length: chunks * 384 * 2 * 4)
            gnStats = device.makeBuffer(length: 32 * 2 * 4)
        }
        guard let partial = gnPartial, let stats = gnStats else { return }
        var hw32 = Int32(HW), c32 = Int32(C), g32 = Int32(groups), ch32 = Int32(chunks)
        var e = eps
        var s32 = Int32(silu ? 1 : 0)

        enc.setComputePipelineState(gnPartialPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(partial, offset: 0, index: 1)
        enc.setBytes(&hw32, length: 4, index: 2)
        enc.setBytes(&c32, length: 4, index: 3)
        enc.setBytes(&ch32, length: 4, index: 4)
        enc.dispatchThreads(
            MTLSize(width: C, height: chunks, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(C, 64), height: 4, depth: 1))

        enc.setComputePipelineState(gnFinalizePSO)
        enc.setBuffer(partial, offset: 0, index: 0)
        enc.setBuffer(stats, offset: 0, index: 1)
        enc.setBytes(&hw32, length: 4, index: 2)
        enc.setBytes(&c32, length: 4, index: 3)
        enc.setBytes(&g32, length: 4, index: 4)
        enc.setBytes(&ch32, length: 4, index: 5)
        enc.setBytes(&e, length: 4, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))

        enc.setComputePipelineState(gnApplyPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(gamma, offset: 0, index: 1)
        enc.setBuffer(beta, offset: 0, index: 2)
        enc.setBuffer(stats, offset: 0, index: 3)
        enc.setBuffer(y, offset: 0, index: 4)
        enc.setBytes(&c32, length: 4, index: 5)
        enc.setBytes(&g32, length: 4, index: 6)
        enc.setBytes(&s32, length: 4, index: 7)
        enc.dispatchThreads(
            MTLSize(width: C, height: HW, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(C, 64), height: 4, depth: 1))
    }

    func encodeBiasAct(
        _ enc: MTLComputeCommandEncoder, x: MTLBuffer, bias: MTLBuffer,
        HW: Int, C: Int, silu: Bool
    ) {
        enc.setComputePipelineState(biasPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(bias, offset: 0, index: 1)
        var c32 = Int32(C)
        var s32 = Int32(silu ? 1 : 0)
        enc.setBytes(&c32, length: 4, index: 2)
        enc.setBytes(&s32, length: 4, index: 3)
        enc.dispatchThreads(
            MTLSize(width: C, height: HW, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(C, 256), height: 1, depth: 1))
    }

    func encodeAdd(_ enc: MTLComputeCommandEncoder, a: MTLBuffer, b: MTLBuffer, y: MTLBuffer, n: Int) {
        enc.setComputePipelineState(addPSO)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var n32 = UInt32(n)
        enc.setBytes(&n32, length: 4, index: 3)
        enc.dispatchThreads(
            MTLSize(width: n, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    // MARK: - Resnet block

    /// GN+SiLU → conv1+bias → GN+SiLU → conv2+bias → (+shortcut) → add.
    /// `t1`/`t2` are scratch sized H·W·max(Cin,Cout)·4.
    func encodeResnet(
        _ enc: MTLComputeCommandEncoder, prefix: String,
        x: MTLBuffer, y: MTLBuffer, t1: MTLBuffer, t2: MTLBuffer,
        H: Int, W: Int, cIn: Int, cOut: Int
    ) throws {
        let hw = H * W
        encodeGroupNormAct(enc, x: x, gamma: try w("\(prefix).norm1.weight"),
                           beta: try w("\(prefix).norm1.bias"), y: t1,
                           HW: hw, C: cIn, silu: true)
        try encodeConv(enc, x: t1, wt: try w("\(prefix).conv1.weight"), y: t2,
                       H: H, W: W, C: cIn, O: cOut, k: 3)
        encodeBiasAct(enc, x: t2, bias: try w("\(prefix).conv1.bias"), HW: hw, C: cOut, silu: false)
        encodeGroupNormAct(enc, x: t2, gamma: try w("\(prefix).norm2.weight"),
                           beta: try w("\(prefix).norm2.bias"), y: t1,
                           HW: hw, C: cOut, silu: true)
        try encodeConv(enc, x: t1, wt: try w("\(prefix).conv2.weight"), y: t2,
                       H: H, W: W, C: cOut, O: cOut, k: 3)
        encodeBiasAct(enc, x: t2, bias: try w("\(prefix).conv2.bias"), HW: hw, C: cOut, silu: false)
        if cIn != cOut {
            // shortcut into y, then add
            try encodeConv(enc, x: x, wt: try w("\(prefix).conv_shortcut.weight"), y: y,
                           H: H, W: W, C: cIn, O: cOut, k: 1)
            encodeBiasAct(enc, x: y, bias: try w("\(prefix).conv_shortcut.bias"), HW: hw, C: cOut, silu: false)
            encodeAdd(enc, a: t2, b: y, y: y, n: hw * cOut)
        } else {
            encodeAdd(enc, a: t2, b: x, y: y, n: hw * cOut)
        }
    }
}

/// V2 spike: one mid-block resnet with real Small Decoder weights vs an
/// exact-math MLX oracle.
public enum DirectVAEResnetSpike {
    public static func run(smallDecoderFile: URL, metallibURL: URL) throws -> String {
        let vae = try DirectVAE(smallDecoderFile: smallDecoderFile, metallibURL: metallibURL)
        let H = 64, W = 64, C = 384
        let prefix = "decoder.mid_block.resnets.0"

        MLXRandom.seed(31)
        let x = (MLXRandom.normal([1, H, W, C]) * 1.0).asType(.float32)
        eval(x)

        // Oracle: raw MLX ops, exact component math.
        let raw = try MLX.loadArrays(url: smallDecoderFile)
        let m = try CompVisVAEMapper.remap(raw)
        func gnOracle(_ t: MLXArray, _ p: String) -> MLXArray {
            let gn = GroupNorm(groupCount: 32, dimensions: C, eps: 1e-6, affine: true, pytorchCompatible: true)
            try! gn.update(parameters: ModuleParameters.unflattened(
                ["weight": m["\(p).weight"]!, "bias": m["\(p).bias"]!]), verify: [.all])
            return gn(t)
        }
        var h = gnOracle(x, "\(prefix).norm1")
        h = silu(h)
        h = conv2d(h, m["\(prefix).conv1.weight"]!, stride: .init(1), padding: .init(1))
            + m["\(prefix).conv1.bias"]!.reshaped([1, 1, 1, C])
        h = gnOracle(h, "\(prefix).norm2")
        h = silu(h)
        h = conv2d(h, m["\(prefix).conv2.weight"]!, stride: .init(1), padding: .init(1))
            + m["\(prefix).conv2.bias"]!.reshaped([1, 1, 1, C])
        let oracle = h + x
        eval(oracle)

        // Direct
        let xData = x.asData(noCopy: false)
        let xBuf = try xData.withUnsafeBytes { rawB -> MTLBuffer in
            guard let base = rawB.baseAddress,
                let b = vae.device.makeBuffer(bytes: base, length: rawB.count)
            else { throw DirectQmmSpike.SpikeError.metal("x") }
            return b
        }
        let bytes = H * W * C * 4
        let yBuf = try vae.scratch(bytes, "y")
        let t1 = try vae.scratch(bytes, "t1")
        let t2 = try vae.scratch(bytes, "t2")
        guard let cb = vae.queue.makeCommandBuffer(),
            let enc = cb.makeComputeCommandEncoder()
        else { throw DirectQmmSpike.SpikeError.metal("cb") }
        try vae.encodeResnet(enc, prefix: prefix, x: xBuf, y: yBuf, t1: t1, t2: t2,
                             H: H, W: W, cIn: C, cOut: C)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw DirectQmmSpike.SpikeError.metal("exec: \(e)") }

        let n = H * W * C
        let ptr = yBuf.contents().bindMemory(to: Float.self, capacity: n)
        let ref = oracle.reshaped([n]).asArray(Float.self)
        var dot = 0.0, na = 0.0, nb = 0.0, maxDiff = 0.0
        for i in 0 ..< n {
            let a = Double(ptr[i]), b = Double(ref[i])
            dot += a * b; na += a * a; nb += b * b
            maxDiff = max(maxDiff, abs(a - b))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
        return """
        direct-vae V2 — mid resnet 0 (\(C)ch @\(H)×\(W)), real Small Decoder weights
          cosine_vs_oracle: \(String(format: "%.7f", cosine))
          max_abs_diff:     \(String(format: "%.6f", maxDiff))
          verdict:          \(cosine >= 0.99999 ? "PASS" : "investigate")
        """
    }
}

// MARK: - Attention + mid block (V3)

extension DirectVAE {

    func encodeMatmulNT(_ enc: MTLComputeCommandEncoder, a: MTLBuffer, b: MTLBuffer, y: MTLBuffer, m: Int, n: Int, k: Int) {
        enc.setComputePipelineState(mmNTPSO)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var m32 = Int32(m), n32 = Int32(n), k32 = Int32(k)
        enc.setBytes(&m32, length: 4, index: 3)
        enc.setBytes(&n32, length: 4, index: 4)
        enc.setBytes(&k32, length: 4, index: 5)
        enc.dispatchThreads(
            MTLSize(width: n, height: m, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }

    func encodeMatmulNN(_ enc: MTLComputeCommandEncoder, p: MTLBuffer, v: MTLBuffer, y: MTLBuffer, m: Int, n: Int, k: Int) {
        enc.setComputePipelineState(mmNNPSO)
        enc.setBuffer(p, offset: 0, index: 0)
        enc.setBuffer(v, offset: 0, index: 1)
        enc.setBuffer(y, offset: 0, index: 2)
        var m32 = Int32(m), n32 = Int32(n), k32 = Int32(k)
        enc.setBytes(&m32, length: 4, index: 3)
        enc.setBytes(&n32, length: 4, index: 4)
        enc.setBytes(&k32, length: 4, index: 5)
        enc.dispatchThreads(
            MTLSize(width: n, height: m, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }

    /// Single-head D=C attention block: GN → q/k/v (dense) → softmax(q·kᵀ/√C)·v
    /// → out proj → +residual. Needs scores scratch [HW, HW].
    func encodeAttentionBlock(
        _ enc: MTLComputeCommandEncoder, prefix: String,
        x: MTLBuffer, y: MTLBuffer,
        t1: MTLBuffer, q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, scores: MTLBuffer,
        HW: Int, C: Int
    ) throws {
        encodeGroupNormAct(enc, x: x, gamma: try w("\(prefix).group_norm.weight"),
                           beta: try w("\(prefix).group_norm.bias"), y: t1,
                           HW: HW, C: C, silu: false)
        try encodeSteelGemm(enc, a: t1, b: try w("\(prefix).to_q.weight"), d: q, m: HW, n: C, k: C, transB: true)
        encodeBiasAct(enc, x: q, bias: try w("\(prefix).to_q.bias"), HW: HW, C: C, silu: false)
        try encodeSteelGemm(enc, a: t1, b: try w("\(prefix).to_k.weight"), d: k, m: HW, n: C, k: C, transB: true)
        encodeBiasAct(enc, x: k, bias: try w("\(prefix).to_k.bias"), HW: HW, C: C, silu: false)
        try encodeSteelGemm(enc, a: t1, b: try w("\(prefix).to_v.weight"), d: v, m: HW, n: C, k: C, transB: true)
        encodeBiasAct(enc, x: v, bias: try w("\(prefix).to_v.bias"), HW: HW, C: C, silu: false)
        try encodeSteelGemm(enc, a: q, b: k, d: scores, m: HW, n: HW, k: C, transB: true)
        // softmax rows with scale 1/√C
        enc.setComputePipelineState(softmaxPSO)
        enc.setBuffer(scores, offset: 0, index: 0)
        enc.setBuffer(scores, offset: 0, index: 1)
        var n32 = Int32(HW)
        var scale = Float(1.0 / Double(C).squareRoot())
        enc.setBytes(&n32, length: 4, index: 2)
        enc.setBytes(&scale, length: 4, index: 3)
        enc.dispatchThreadgroups(
            MTLSize(width: HW, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        try encodeSteelGemm(enc, a: scores, b: v, d: t1, m: HW, n: C, k: HW, transB: false)
        try encodeSteelGemm(enc, a: t1, b: try w("\(prefix).to_out.weight"), d: q, m: HW, n: C, k: C, transB: true)
        encodeBiasAct(enc, x: q, bias: try w("\(prefix).to_out.bias"), HW: HW, C: C, silu: false)
        encodeAdd(enc, a: x, b: q, y: y, n: HW * C)
    }
}

extension DirectVAE {
    /// Mid block: resnet0 → attention → resnet1 (all at `C` channels).
    func encodeMid(
        _ enc: MTLComputeCommandEncoder,
        x: MTLBuffer, y: MTLBuffer,
        t1: MTLBuffer, t2: MTLBuffer, t3: MTLBuffer, t4: MTLBuffer, scores: MTLBuffer,
        H: Int, W: Int, C: Int
    ) throws {
        let hw = H * W
        try encodeResnet(enc, prefix: "decoder.mid_block.resnets.0",
                         x: x, y: t3, t1: t1, t2: t2, H: H, W: W, cIn: C, cOut: C)
        try encodeAttentionBlock(enc, prefix: "decoder.mid_block.attentions.0",
                                 x: t3, y: t4, t1: t1, q: t2, k: x, v: y,
                                 scores: scores, HW: hw, C: C)
        try encodeResnet(enc, prefix: "decoder.mid_block.resnets.1",
                         x: t4, y: y, t1: t1, t2: t2, H: H, W: W, cIn: C, cOut: C)
    }
}

/// V3 spike: the full mid block (resnet0 → single-head attention → resnet1)
/// with real Small Decoder weights vs an exact-math MLX oracle.
public enum DirectVAEMidSpike {
    public static func run(smallDecoderFile: URL, metallibURL: URL) throws -> String {
        let vae = try DirectVAE(smallDecoderFile: smallDecoderFile, metallibURL: metallibURL)
        let H = 64, W = 64, C = 384
        let hw = H * W

        MLXRandom.seed(37)
        let x = MLXRandom.normal([1, H, W, C]).asType(.float32)
        eval(x)

        let raw = try MLX.loadArrays(url: smallDecoderFile)
        let m = try CompVisVAEMapper.remap(raw)
        func gnOracle(_ t: MLXArray, _ p: String) -> MLXArray {
            let gn = GroupNorm(groupCount: 32, dimensions: C, eps: 1e-6, affine: true, pytorchCompatible: true)
            try! gn.update(parameters: ModuleParameters.unflattened(
                ["weight": m["\(p).weight"]!, "bias": m["\(p).bias"]!]), verify: [.all])
            return gn(t)
        }
        func resnetOracle(_ input: MLXArray, _ p: String) -> MLXArray {
            var h = silu(gnOracle(input, "\(p).norm1"))
            h = conv2d(h, m["\(p).conv1.weight"]!, stride: .init(1), padding: .init(1))
                + m["\(p).conv1.bias"]!.reshaped([1, 1, 1, C])
            h = silu(gnOracle(h, "\(p).norm2"))
            h = conv2d(h, m["\(p).conv2.weight"]!, stride: .init(1), padding: .init(1))
                + m["\(p).conv2.bias"]!.reshaped([1, 1, 1, C])
            return h + input
        }
        func attnOracle(_ input: MLXArray) -> MLXArray {
            let p = "decoder.mid_block.attentions.0"
            let normed = gnOracle(input, "\(p).group_norm").reshaped([hw, C])
            let q = matmul(normed, m["\(p).to_q.weight"]!.transposed()) + m["\(p).to_q.bias"]!
            let k = matmul(normed, m["\(p).to_k.weight"]!.transposed()) + m["\(p).to_k.bias"]!
            let v = matmul(normed, m["\(p).to_v.weight"]!.transposed()) + m["\(p).to_v.bias"]!
            let scale = Float(1.0 / Double(C).squareRoot())
            let probs = softmax(matmul(q, k.transposed()) * scale, axis: -1)
            var out = matmul(probs, v)
            out = matmul(out, m["\(p).to_out.weight"]!.transposed()) + m["\(p).to_out.bias"]!
            return input + out.reshaped([1, H, W, C])
        }
        var h = resnetOracle(x, "decoder.mid_block.resnets.0")
        h = attnOracle(h)
        let oracle = resnetOracle(h, "decoder.mid_block.resnets.1")
        eval(oracle)

        let xData = x.asData(noCopy: false)
        let xBuf = try xData.withUnsafeBytes { rawB -> MTLBuffer in
            guard let base = rawB.baseAddress,
                let b = vae.device.makeBuffer(bytes: base, length: rawB.count)
            else { throw DirectQmmSpike.SpikeError.metal("x") }
            return b
        }
        let bytes = hw * C * 4
        let yBuf = try vae.scratch(bytes, "y")
        let t1 = try vae.scratch(bytes, "t1")
        let t2 = try vae.scratch(bytes, "t2")
        let t3 = try vae.scratch(bytes, "t3")
        let t4 = try vae.scratch(bytes, "t4")
        let scores = try vae.scratch(hw * hw * 4, "scores")
        guard let cb = vae.queue.makeCommandBuffer(),
            let enc = cb.makeComputeCommandEncoder()
        else { throw DirectQmmSpike.SpikeError.metal("cb") }
        try vae.encodeMid(enc, x: xBuf, y: yBuf, t1: t1, t2: t2, t3: t3, t4: t4,
                          scores: scores, H: H, W: W, C: C)
        enc.endEncoding()
        let t0 = CFAbsoluteTimeGetCurrent()
        cb.commit()
        cb.waitUntilCompleted()
        let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        if let e = cb.error { throw DirectQmmSpike.SpikeError.metal("exec: \(e)") }

        let n = hw * C
        let ptr = yBuf.contents().bindMemory(to: Float.self, capacity: n)
        let ref = oracle.reshaped([n]).asArray(Float.self)
        var dot = 0.0, na = 0.0, nb = 0.0, maxDiff = 0.0
        for i in 0 ..< n {
            let a = Double(ptr[i]), b = Double(ref[i])
            dot += a * b; na += a * a; nb += b * b
            maxDiff = max(maxDiff, abs(a - b))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
        return """
        direct-vae V3 — full mid block (resnet·attn·resnet, \(C)ch @\(H)×\(W))
          cosine_vs_oracle: \(String(format: "%.7f", cosine))
          max_abs_diff:     \(String(format: "%.6f", maxDiff))
          gpu_wall:         \(String(format: "%.1f", ms)) ms
          verdict:          \(cosine >= 0.9999 ? "PASS" : "investigate")
        """
    }
}

// MARK: - Full decoder (V4)

extension DirectVAE {

    struct UpSpec {
        let cIn: Int
        let cOut: Int
        let upsample: Bool
    }
    /// Small Decoder up path: reversed [96,192,384,384] → 3 resnets each.
    static let upSpecs: [UpSpec] = [
        UpSpec(cIn: 384, cOut: 384, upsample: true),
        UpSpec(cIn: 384, cOut: 384, upsample: true),
        UpSpec(cIn: 384, cOut: 192, upsample: true),
        UpSpec(cIn: 192, cOut: 96, upsample: false),
    ]

    func encodeUpsample(
        _ enc: MTLComputeCommandEncoder, prefix: String,
        x: MTLBuffer, y: MTLBuffer, t: MTLBuffer, H: Int, W: Int, C: Int
    ) throws {
        enc.setComputePipelineState(upPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(t, offset: 0, index: 1)
        var h32 = Int32(H), w32 = Int32(W), c32 = Int32(C)
        enc.setBytes(&h32, length: 4, index: 2)
        enc.setBytes(&w32, length: 4, index: 3)
        enc.setBytes(&c32, length: 4, index: 4)
        enc.dispatchThreads(
            MTLSize(width: C, height: 2 * W, depth: 2 * H),
            threadsPerThreadgroup: MTLSize(width: min(C, 64), height: 4, depth: 1))
        try encodeConv(enc, x: t, wt: try w("\(prefix).conv.weight"), y: y,
                       H: 2 * H, W: 2 * W, C: C, O: C, k: 3)
        encodeBiasAct(enc, x: y, bias: try w("\(prefix).conv.bias"), HW: 4 * H * W, C: C, silu: false)
    }

    /// post_quant_conv → conv_in → mid → 4 up blocks → GN+SiLU → conv_out.
    /// `z` NHWC [1,H8,W8,32] → `rgb` NHWC [1,8·H8,8·W8,3].
    /// `a`/`b`/`t1`/`t2` sized for the largest stage (8·H8 × 8·W8 × 192 f32);
    /// `t3`/`t4` mid-sized (H8·W8·384); `scores` H8²·W8² f32.
    func encodeDecoder(
        _ enc: MTLComputeCommandEncoder,
        z: MTLBuffer, rgb: MTLBuffer,
        a: MTLBuffer, b: MTLBuffer, t1: MTLBuffer, t2: MTLBuffer,
        t3: MTLBuffer, t4: MTLBuffer, scores: MTLBuffer,
        H8: Int, W8: Int
    ) throws {
        var H = H8, W = W8
        try encodeConv(enc, x: z, wt: try w("post_quant_conv.weight"), y: a,
                       H: H, W: W, C: 32, O: 32, k: 1)
        encodeBiasAct(enc, x: a, bias: try w("post_quant_conv.bias"), HW: H * W, C: 32, silu: false)
        try encodeConv(enc, x: a, wt: try w("decoder.conv_in.weight"), y: b,
                       H: H, W: W, C: 32, O: 384, k: 3)
        encodeBiasAct(enc, x: b, bias: try w("decoder.conv_in.bias"), HW: H * W, C: 384, silu: false)
        try encodeMid(enc, x: b, y: a, t1: t1, t2: t2, t3: t3, t4: t4,
                      scores: scores, H: H, W: W, C: 384)
        var cur = a, alt = b
        for (i, spec) in Self.upSpecs.enumerated() {
            for r in 0 ..< 3 {
                let cIn = r == 0 ? spec.cIn : spec.cOut
                try encodeResnet(enc, prefix: "decoder.up_blocks.\(i).resnets.\(r)",
                                 x: cur, y: alt, t1: t1, t2: t2,
                                 H: H, W: W, cIn: cIn, cOut: spec.cOut)
                swap(&cur, &alt)
            }
            if spec.upsample {
                try encodeUpsample(enc, prefix: "decoder.up_blocks.\(i).upsamplers.0",
                                   x: cur, y: alt, t: t1, H: H, W: W, C: spec.cOut)
                swap(&cur, &alt)
                H *= 2
                W *= 2
            }
        }
        encodeGroupNormAct(enc, x: cur, gamma: try w("decoder.conv_norm_out.weight"),
                           beta: try w("decoder.conv_norm_out.bias"), y: alt,
                           HW: H * W, C: 96, silu: true)
        try encodeConv(enc, x: alt, wt: try w("decoder.conv_out.weight"), y: rgb,
                       H: H, W: W, C: 96, O: 3, k: 3)
        encodeBiasAct(enc, x: rgb, bias: try w("decoder.conv_out.bias"), HW: H * W, C: 3, silu: false)
    }
}

/// V4 spike: the whole Small Decoder untiled @512² on the direct engine vs
/// the product `VAEModule.decodePacked` on the same packed latent.
public enum DirectVAEDecodeSpike {
    public static func run(
        snapshot: ModelSnapshot, smallDecoderDirectory: URL, metallibURL: URL,
        config: ImarelloConfig
    ) async throws -> String {
        let width = 512, height = 512
        let h16 = height / 16, w16 = width / 16
        let H8 = 2 * h16, W8 = 2 * w16

        let packed = LatentOps.samplePackedNoise(width: width, height: height, seed: 42)
        let spatial = LatentOps.unpackSequence(packed, height: h16, width: w16)
        eval(spatial)

        // Oracle: the product decode path (untiled at 512²), then unload.
        let module = VAEModule(
            snapshot: snapshot, decoderVariant: .smallDecoder,
            smallDecoderDirectory: smallDecoderDirectory)
        try await module.load(mode: .decodeOnly)
        let oracleNCHW = try await module.decodePacked(spatial)
        eval(oracleNCHW)
        let oracle = oracleNCHW.transposed(0, 2, 3, 1)  // NHWC for comparison
        eval(oracle)
        await module.unload()
        Memory.clearCache()

        // Direct entry: BN denorm + unpatchify on MLX (tiny), engine from conv on.
        let vaeArrays = try MLX.loadArrays(
            url: snapshot.vaeDirectory.appendingPathComponent("0.safetensors"))
        guard let bnMean = vaeArrays["bn.running_mean"],
            let bnVar = vaeArrays["bn.running_var"]
        else { throw DirectQmmSpike.SpikeError.missingTensor("bn stats") }
        let bnStd = sqrt(bnVar.asType(.float32).reshaped([1, -1, 1, 1]) + 1e-4)
        let denormed = spatial.asType(.float32) * bnStd
            + bnMean.asType(.float32).reshaped([1, -1, 1, 1])
        let zNHWC = Flux2VAE.unpatchify(denormed).transposed(0, 2, 3, 1).asType(.float32)
        eval(zNHWC)

        let vae = try DirectVAE(
            smallDecoderFile: smallDecoderDirectory.appendingPathComponent("small_decoder.safetensors"),
            metallibURL: metallibURL)
        let zData = zNHWC.asData(noCopy: false)
        let zBuf = try zData.withUnsafeBytes { rawB -> MTLBuffer in
            guard let base = rawB.baseAddress,
                let bb = vae.device.makeBuffer(bytes: base, length: rawB.count)
            else { throw DirectQmmSpike.SpikeError.metal("z") }
            return bb
        }
        let outH = 8 * H8, outW = 8 * W8
        let bigBytes = outH * outW * 192 * 4
        let midBytes = H8 * W8 * 384 * 4
        let a = try vae.scratch(bigBytes, "a")
        let b = try vae.scratch(bigBytes, "b")
        let t1 = try vae.scratch(bigBytes, "t1")
        let t2 = try vae.scratch(bigBytes, "t2")
        let t3 = try vae.scratch(midBytes, "t3")
        let t4 = try vae.scratch(midBytes, "t4")
        let scores = try vae.scratch(H8 * W8 * H8 * W8 * 4, "scores")
        let rgbBuf = try vae.scratch(outH * outW * 3 * 4, "rgb")

        guard let cb = vae.queue.makeCommandBuffer(),
            let enc = cb.makeComputeCommandEncoder()
        else { throw DirectQmmSpike.SpikeError.metal("cb") }
        try vae.encodeDecoder(enc, z: zBuf, rgb: rgbBuf, a: a, b: b,
                              t1: t1, t2: t2, t3: t3, t4: t4, scores: scores,
                              H8: H8, W8: W8)
        enc.endEncoding()
        let t0 = CFAbsoluteTimeGetCurrent()
        cb.commit()
        await cb.completed()
        let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        if let e = cb.error { throw DirectQmmSpike.SpikeError.metal("exec: \(e)") }

        let n = outH * outW * 3
        let ptr = rgbBuf.contents().bindMemory(to: Float.self, capacity: n)
        let ref = oracle.reshaped([n]).asArray(Float.self)
        var dot = 0.0, na = 0.0, nb = 0.0, maxDiff = 0.0
        for i in 0 ..< n {
            let x = Double(ptr[i]), y = Double(ref[i])
            dot += x * y; na += x * x; nb += y * y
            maxDiff = max(maxDiff, abs(x - y))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
        return """
        direct-vae V4 — full Small Decoder untiled @\(width)² vs product decodePacked
          cosine_vs_oracle: \(String(format: "%.7f", cosine))
          max_abs_diff:     \(String(format: "%.6f", maxDiff))
          gpu_wall:         \(String(format: "%.1f", ms)) ms
          engine_owned:     \(String(format: "%.2f", Double(vae.ownedBytes) / 1_073_741_824)) GiB
          verdict:          \(cosine >= 0.9999 ? "PASS" : "investigate")
        """
    }
}

// MARK: - Tiled decode (V5)

extension DirectVAE {

    /// Decode one NCHW latent tile [1,32,th,tw] → NCHW RGB, on the engine.
    /// Sized-per-call scratch; sync (used as the product stitcher's closure).
    func decodeTileNCHW(_ tile: MLXArray) throws -> MLXArray {
        let H8 = tile.dim(2), W8 = tile.dim(3)
        let zNHWC = tile.transposed(0, 2, 3, 1).asType(.float32)
        eval(zNHWC)
        let zData = zNHWC.asData(noCopy: false)
        let zBuf = try zData.withUnsafeBytes { rawB -> MTLBuffer in
            guard let base = rawB.baseAddress,
                let bb = device.makeBuffer(bytes: base, length: rawB.count)
            else { throw DirectQmmSpike.SpikeError.metal("z tile") }
            return bb
        }
        let outH = 8 * H8, outW = 8 * W8
        let bigBytes = outH * outW * 192 * 4
        let midBytes = H8 * W8 * 384 * 4
        func mk(_ n: Int, _ l: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: n) else {
                throw DirectQmmSpike.SpikeError.metal(l)
            }
            return b
        }
        let a = try mk(bigBytes, "a"), b = try mk(bigBytes, "b")
        let t1 = try mk(bigBytes, "t1"), t2 = try mk(bigBytes, "t2")
        let t3 = try mk(midBytes, "t3"), t4 = try mk(midBytes, "t4")
        let scores = try mk(H8 * W8 * H8 * W8 * 4, "scores")
        let rgbBuf = try mk(outH * outW * 3 * 4, "rgb")
        guard let cb = queue.makeCommandBuffer(),
            let enc = cb.makeComputeCommandEncoder()
        else { throw DirectQmmSpike.SpikeError.metal("cb") }
        try encodeDecoder(enc, z: zBuf, rgb: rgbBuf, a: a, b: b,
                          t1: t1, t2: t2, t3: t3, t4: t4, scores: scores,
                          H8: H8, W8: W8)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw DirectQmmSpike.SpikeError.metal("tile exec: \(e)") }
        let n = outH * outW * 3
        var host = [Float](repeating: 0, count: n)
        host.withUnsafeMutableBufferPointer { p in
            p.baseAddress!.update(
                from: rgbBuf.contents().bindMemory(to: Float.self, capacity: n), count: n)
        }
        return MLXArray(host, [1, outH, outW, 3]).transposed(0, 3, 1, 2)
    }
}

/// V5 spike: tiled 1024² decode — the product's cosine-blend stitcher driving
/// the direct engine per tile, vs the product decodePacked end to end.
public enum DirectVAETiledSpike {
    public static func run(
        snapshot: ModelSnapshot, smallDecoderDirectory: URL, metallibURL: URL
    ) async throws -> String {
        let width = 1024, height = 1024
        let h16 = height / 16, w16 = width / 16

        let packed = LatentOps.samplePackedNoise(width: width, height: height, seed: 42)
        let spatial = LatentOps.unpackSequence(packed, height: h16, width: w16)
        eval(spatial)

        let module = VAEModule(
            snapshot: snapshot, decoderVariant: .smallDecoder,
            smallDecoderDirectory: smallDecoderDirectory)
        try await module.load(mode: .decodeOnly)
        let oracleNCHW = try await module.decodePacked(spatial)
        eval(oracleNCHW)
        await module.unload()
        Memory.clearCache()

        let vaeArrays = try MLX.loadArrays(
            url: snapshot.vaeDirectory.appendingPathComponent("0.safetensors"))
        guard let bnMean = vaeArrays["bn.running_mean"],
            let bnVar = vaeArrays["bn.running_var"]
        else { throw DirectQmmSpike.SpikeError.missingTensor("bn stats") }
        let bnStd = sqrt(bnVar.asType(.float32).reshaped([1, -1, 1, 1]) + 1e-4)
        let denormed = spatial.asType(.float32) * bnStd
            + bnMean.asType(.float32).reshaped([1, -1, 1, 1])
        let latents = Flux2VAE.unpatchify(denormed)
        eval(latents)

        let vae = try DirectVAE(
            smallDecoderFile: smallDecoderDirectory.appendingPathComponent("small_decoder.safetensors"),
            metallibURL: metallibURL)
        let t0 = CFAbsoluteTimeGetCurrent()
        let direct = Flux2VAE.decodeLatentsTiled(
            latents,
            decode: { tile in try! vae.decodeTileNCHW(tile) },
            config: .current)
        eval(direct)
        let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        let n = 3 * height * width
        let a = direct.reshaped([n]).asArray(Float.self)
        let b = oracleNCHW.asType(.float32).reshaped([n]).asArray(Float.self)
        var dot = 0.0, na = 0.0, nb = 0.0, maxDiff = 0.0
        for i in 0 ..< n {
            let x = Double(a[i]), y = Double(b[i])
            dot += x * y; na += x * x; nb += y * y
            maxDiff = max(maxDiff, abs(x - y))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
        return """
        direct-vae V5 — tiled 1024² (product cosine stitcher, direct tile decode)
          cosine_vs_oracle: \(String(format: "%.7f", cosine))
          max_abs_diff:     \(String(format: "%.6f", maxDiff))
          tiled_wall:       \(String(format: "%.1f", ms)) ms
          verdict:          \(cosine >= 0.9999 ? "PASS" : "investigate")
        """
    }
}

// MARK: - Steel GEMM attention (V6)

extension DirectVAE {

    /// PSO for `steel_gemm_fused` with function constants applied.
    /// M2 'g'-class tiles: f32 NT → bm64/bn32/bk32/wm2/wn2, f32 NN → bm64/bn64/bk16/wm2/wn2.
    func steelGemmPSO(
        transB: Bool, bm: Int, bn: Int, bk: Int, wm: Int, wn: Int,
        alignM: Bool, alignN: Bool, alignK: Bool
    ) throws -> MTLComputePipelineState {
        let base = "steel_gemm_fused_n\(transB ? "t" : "n")_float32_float32_bm\(bm)_bn\(bn)_bk\(bk)_wm\(wm)_wn\(wn)"
        let key = "\(base)_\(alignM)_\(alignN)_\(alignK)"
        if let p = convPSOs[key] { return p }
        let consts = MTLFunctionConstantValues()
        var f = false
        var aM = alignM, aN = alignN, aK = alignK
        consts.setConstantValue(&f, type: .bool, index: 10)   // has_batch
        consts.setConstantValue(&f, type: .bool, index: 100)  // use_out_source
        consts.setConstantValue(&f, type: .bool, index: 110)  // do_axpby
        consts.setConstantValue(&aM, type: .bool, index: 200)
        consts.setConstantValue(&aN, type: .bool, index: 201)
        consts.setConstantValue(&aK, type: .bool, index: 202)
        let fn = try mlxLib.makeFunction(name: base, constantValues: consts)
        let p = try device.makeComputePipelineState(function: fn)
        convPSOs[key] = p
        return p
    }

    /// D[M,N] = A[M,K] · op(B) on the metallib's fused steel GEMM (f32, B=1).
    /// `transB: true` → B is [N,K] (NT: projections, Q·Kᵀ); false → B is [K,N] (NN: P·V).
    func encodeSteelGemm(
        _ enc: MTLComputeCommandEncoder,
        a: MTLBuffer, b: MTLBuffer, d: MTLBuffer,
        m: Int, n: Int, k: Int, transB: Bool
    ) throws {
        let bm = 64
        let bn = transB ? 32 : 64
        let bk = transB ? 32 : 16
        let pso = try steelGemmPSO(
            transB: transB, bm: bm, bn: bn, bk: bk, wm: 2, wn: 2,
            alignM: m % bm == 0, alignN: n % bn == 0, alignK: k % bk == 0)
        enc.setComputePipelineState(pso)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(d, offset: 0, index: 3)
        let tn = (n + bn - 1) / bn
        let tm = (m + bm - 1) / bm
        var params = Data()
        func i32(_ v: Int) { var t = Int32(v); withUnsafeBytes(of: &t) { params.append(contentsOf: $0) } }
        func i64(_ v: Int) { var t = Int64(v); withUnsafeBytes(of: &t) { params.append(contentsOf: $0) } }
        i32(m); i32(n); i32(k)
        i32(k)                    // lda: A row-major [M,K]
        i32(transB ? k : n)       // ldb: [N,K] or [K,N]
        i32(n)                    // ldd
        i32(tn); i32(tm)
        i64(0); i64(0); i64(0)    // batch strides (grid depth 1)
        i32(0)                    // swizzle_log
        i32(k / bk)               // gemm_k_iterations_aligned
        i32(1)                    // batch_ndim
        params.append(Data(repeating: 0, count: (8 - params.count % 8) % 8))
        params.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: params.count, index: 4) }
        enc.dispatchThreadgroups(
            MTLSize(width: tn, height: tm, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
    }
}

/// Per-stage GPU timing of the direct decoder walk @512² (one command buffer
/// per stage, so waits mark stage boundaries). Correctness lives in V4.
public enum DirectVAEProfileSpike {
    public static func run(smallDecoderDirectory: URL, metallibURL: URL) throws -> String {
        let H8 = 64, W8 = 64
        let vae = try DirectVAE(
            smallDecoderFile: smallDecoderDirectory.appendingPathComponent("small_decoder.safetensors"),
            metallibURL: metallibURL)
        MLXRandom.seed(11)
        let z = MLXRandom.normal([1, H8, W8, 32]).asType(.float32)
        eval(z)
        let zData = z.asData(noCopy: false)
        let zBuf = try zData.withUnsafeBytes { rawB -> MTLBuffer in
            guard let base = rawB.baseAddress,
                let bb = vae.device.makeBuffer(bytes: base, length: rawB.count)
            else { throw DirectQmmSpike.SpikeError.metal("z") }
            return bb
        }
        let outH = 8 * H8, outW = 8 * W8
        let bigBytes = outH * outW * 192 * 4
        let midBytes = H8 * W8 * 384 * 4
        let a = try vae.scratch(bigBytes, "a")
        let b = try vae.scratch(bigBytes, "b")
        let t1 = try vae.scratch(bigBytes, "t1")
        let t2 = try vae.scratch(bigBytes, "t2")
        let t3 = try vae.scratch(midBytes, "t3")
        let t4 = try vae.scratch(midBytes, "t4")
        let scores = try vae.scratch(H8 * W8 * H8 * W8 * 4, "scores")
        let rgbBuf = try vae.scratch(outH * outW * 3 * 4, "rgb")

        var report = "direct-vae profile @512² (per-stage command buffers)\n"
        func stage(_ name: String, _ body: (MTLComputeCommandEncoder) throws -> Void) throws {
            guard let cb = vae.queue.makeCommandBuffer(),
                let enc = cb.makeComputeCommandEncoder()
            else { throw DirectQmmSpike.SpikeError.metal("cb") }
            try body(enc)
            enc.endEncoding()
            let t0 = CFAbsoluteTimeGetCurrent()
            cb.commit()
            cb.waitUntilCompleted()
            let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            if let e = cb.error { throw DirectQmmSpike.SpikeError.metal("\(name): \(e)") }
            report += String(format: "  %-22s %7.1f ms\n", (name as NSString).utf8String!, ms)
        }

        var H = H8, W = W8
        try stage("post_quant+conv_in") { enc in
            try vae.encodeConv(enc, x: zBuf, wt: try vae.w("post_quant_conv.weight"), y: a,
                               H: H, W: W, C: 32, O: 32, k: 1)
            vae.encodeBiasAct(enc, x: a, bias: try vae.w("post_quant_conv.bias"), HW: H * W, C: 32, silu: false)
            try vae.encodeConv(enc, x: a, wt: try vae.w("decoder.conv_in.weight"), y: b,
                               H: H, W: W, C: 32, O: 384, k: 3)
            vae.encodeBiasAct(enc, x: b, bias: try vae.w("decoder.conv_in.bias"), HW: H * W, C: 384, silu: false)
        }
        try stage("mid (r·attn·r)") { enc in
            try vae.encodeMid(enc, x: b, y: a, t1: t1, t2: t2, t3: t3, t4: t4,
                              scores: scores, H: H, W: W, C: 384)
        }
        var cur = a, alt = b
        for (i, spec) in DirectVAE.upSpecs.enumerated() {
            let hNow = H, wNow = W
            try stage("up\(i) resnets ×3 @\(hNow)²") { enc in
                for r in 0 ..< 3 {
                    let cIn = r == 0 ? spec.cIn : spec.cOut
                    try vae.encodeResnet(enc, prefix: "decoder.up_blocks.\(i).resnets.\(r)",
                                         x: cur, y: alt, t1: t1, t2: t2,
                                         H: hNow, W: wNow, cIn: cIn, cOut: spec.cOut)
                    swap(&cur, &alt)
                }
            }
            if spec.upsample {
                try stage("up\(i) upsample → \(2 * hNow)²") { enc in
                    try vae.encodeUpsample(enc, prefix: "decoder.up_blocks.\(i).upsamplers.0",
                                           x: cur, y: alt, t: t1, H: hNow, W: wNow, C: spec.cOut)
                }
                swap(&cur, &alt)
                H *= 2
                W *= 2
            }
        }
        try stage("tail GN+conv_out") { enc in
            vae.encodeGroupNormAct(enc, x: cur, gamma: try vae.w("decoder.conv_norm_out.weight"),
                                   beta: try vae.w("decoder.conv_norm_out.bias"), y: alt,
                                   HW: H * W, C: 96, silu: true)
            try vae.encodeConv(enc, x: alt, wt: try vae.w("decoder.conv_out.weight"), y: rgbBuf,
                               H: H, W: W, C: 96, O: 3, k: 3)
            vae.encodeBiasAct(enc, x: rgbBuf, bias: try vae.w("decoder.conv_out.bias"), HW: H * W, C: 3, silu: false)
        }
        return report
    }
}

/// Micro-profile of the up3-stage components @512² (GN vs conv vs bias vs add).
public enum DirectVAEMicroSpike {
    public static func run(smallDecoderDirectory: URL, metallibURL: URL) throws -> String {
        let H = 512, W = 512
        let vae = try DirectVAE(
            smallDecoderFile: smallDecoderDirectory.appendingPathComponent("small_decoder.safetensors"),
            metallibURL: metallibURL)
        let hw = H * W
        let a = try vae.scratch(hw * 192 * 4, "a")
        let b = try vae.scratch(hw * 192 * 4, "b")
        memset(a.contents(), 0, hw * 192 * 4)

        var report = "direct-vae micro @512² (each ×6, per-call ms)\n"
        func time(_ name: String, reps: Int = 6, _ body: (MTLComputeCommandEncoder) throws -> Void) throws {
            guard let cb = vae.queue.makeCommandBuffer(),
                let enc = cb.makeComputeCommandEncoder()
            else { throw DirectQmmSpike.SpikeError.metal("cb") }
            for _ in 0 ..< reps { try body(enc) }
            enc.endEncoding()
            let t0 = CFAbsoluteTimeGetCurrent()
            cb.commit()
            cb.waitUntilCompleted()
            let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000 / Double(reps)
            if let e = cb.error { throw DirectQmmSpike.SpikeError.metal("\(name): \(e)") }
            report += String(format: "  %-26s %7.2f ms\n", (name as NSString).utf8String!, ms)
        }

        let g0 = try vae.w("decoder.up_blocks.3.resnets.0.norm1.weight")
        let b0 = try vae.w("decoder.up_blocks.3.resnets.0.norm1.bias")
        try time("groupnorm+silu 192ch") { enc in
            vae.encodeGroupNormAct(enc, x: a, gamma: g0, beta: b0, y: b, HW: hw, C: 192, silu: true)
        }
        let cw = try vae.w("decoder.up_blocks.3.resnets.1.conv1.weight")  // 96→96
        try time("conv3x3 96→96") { enc in
            try vae.encodeConv(enc, x: a, wt: cw, y: b, H: H, W: W, C: 96, O: 96, k: 3)
        }
        let cw0 = try vae.w("decoder.up_blocks.3.resnets.0.conv1.weight")  // 192→96
        try time("conv3x3 192→96") { enc in
            try vae.encodeConv(enc, x: a, wt: cw0, y: b, H: H, W: W, C: 192, O: 96, k: 3)
        }
        let bias = try vae.w("decoder.up_blocks.3.resnets.1.conv1.bias")
        try time("bias+silu 96ch") { enc in
            vae.encodeBiasAct(enc, x: a, bias: bias, HW: hw, C: 96, silu: true)
        }
        try time("add 96ch") { enc in
            vae.encodeAdd(enc, a: a, b: a, y: b, n: hw * 96)
        }
        return report
    }
}

// MARK: - Production decode API

extension DirectVAE {

    /// Load the BN denorm stats (klein `vae/` pack — never the Small Decoder file).
    public func loadBNStats(vaeDirectory: URL) throws {
        let arrays = try MLX.loadArrays(
            url: vaeDirectory.appendingPathComponent("0.safetensors"))
        guard let mean = arrays["bn.running_mean"], let v = arrays["bn.running_var"] else {
            throw DirectQmmSpike.SpikeError.missingTensor("bn stats")
        }
        bnMean = mean.asType(.float32).reshaped([1, -1, 1, 1])
        bnStd = sqrt(v.asType(.float32).reshaped([1, -1, 1, 1]) + 1e-4)
        eval(bnMean!, bnStd!)
    }

    /// Product-shaped entry: packed spatial latents NCHW [1,128,h16,w16] → RGB
    /// NCHW. BN denorm + unpatchify are trivial MLX ops; the UNet runs on the
    /// engine, tiled per `VAETileConfig` (same stitcher as the product).
    public func decodePacked(_ spatial: MLXArray, tileConfig: VAETileConfig = .current) throws -> MLXArray {
        guard let mean = bnMean, let std = bnStd else {
            throw DirectQmmSpike.SpikeError.missingTensor("bn stats not loaded")
        }
        let denormed = spatial.asType(.float32) * std + mean
        let latents = Flux2VAE.unpatchify(denormed)
        eval(latents)
        if tileConfig.shouldTile(height: latents.dim(2), width: latents.dim(3)) {
            return Flux2VAE.decodeLatentsTiled(
                latents, decode: { tile in try! self.decodeTileNCHW(tile) },
                config: tileConfig)
        }
        return try decodeTileNCHW(latents)
    }
}

// MARK: - Product-pipeline adapter

/// Lazy `PackedLatentDecoding` for `ImarelloPipeline`: the engine (and its
/// ~97 MB of f32 weights) builds on the first decode call, preserving staged
/// residency — nothing VAE-related is resident during the TE/DiT stages.
public final class DirectVAEPackedDecoder: PackedLatentDecoding {
    let smallDecoderFile: URL
    let vaeDirectory: URL
    let metallibURL: URL
    var engine: DirectVAE?

    public init(smallDecoderDirectory: URL, vaeDirectory: URL, metallibURL: URL) {
        self.smallDecoderFile = smallDecoderDirectory.appendingPathComponent("small_decoder.safetensors")
        self.vaeDirectory = vaeDirectory
        self.metallibURL = metallibURL
    }

    public func decodePacked(_ spatial: MLXArray) throws -> MLXArray {
        if engine == nil {
            let v = try DirectVAE(smallDecoderFile: smallDecoderFile, metallibURL: metallibURL)
            try v.loadBNStats(vaeDirectory: vaeDirectory)
            engine = v
        }
        return try engine!.decodePacked(spatial)
    }
}
