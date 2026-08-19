import Foundation
import Metal
import MLX
import MLXNN
import ImarelloVAE
import ImarelloCore

/// Direct VAE decoder engine (Stage 2, V-track). Convs run on the metallib's
/// f32 implicit-gemm kernels; everything else on `DirectVAEKernels`. Weights
/// load through the product's own `CompVisVAEMapper` (name + layout truth).
public final class DirectVAE {

    let device: MTLDevice
    let queue: MTLCommandQueue
    let mlxLib: MTLLibrary
    let glue: MTLLibrary
    var convPSOs: [String: MTLComputePipelineState] = [:]
    let gnPSO, biasPSO, upPSO, addPSO, mmNTPSO, mmNNPSO, softmaxPSO: MTLComputePipelineState

    /// Module-named f32 weights (mapper output), uploaded once.
    var weights: [String: MTLBuffer] = [:]
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
        gnPSO = try gpso("dv_groupnorm_act")
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
        enc.setComputePipelineState(gnPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(gamma, offset: 0, index: 1)
        enc.setBuffer(beta, offset: 0, index: 2)
        enc.setBuffer(y, offset: 0, index: 3)
        var hw = Int32(HW), c32 = Int32(C), g32 = Int32(groups)
        var e = eps
        var s32 = Int32(silu ? 1 : 0)
        enc.setBytes(&hw, length: 4, index: 4)
        enc.setBytes(&c32, length: 4, index: 5)
        enc.setBytes(&g32, length: 4, index: 6)
        enc.setBytes(&e, length: 4, index: 7)
        enc.setBytes(&s32, length: 4, index: 8)
        enc.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
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
        encodeMatmulNT(enc, a: t1, b: try w("\(prefix).to_q.weight"), y: q, m: HW, n: C, k: C)
        encodeBiasAct(enc, x: q, bias: try w("\(prefix).to_q.bias"), HW: HW, C: C, silu: false)
        encodeMatmulNT(enc, a: t1, b: try w("\(prefix).to_k.weight"), y: k, m: HW, n: C, k: C)
        encodeBiasAct(enc, x: k, bias: try w("\(prefix).to_k.bias"), HW: HW, C: C, silu: false)
        encodeMatmulNT(enc, a: t1, b: try w("\(prefix).to_v.weight"), y: v, m: HW, n: C, k: C)
        encodeBiasAct(enc, x: v, bias: try w("\(prefix).to_v.bias"), HW: HW, C: C, silu: false)
        encodeMatmulNT(enc, a: q, b: k, y: scores, m: HW, n: HW, k: C)
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
        encodeMatmulNN(enc, p: scores, v: v, y: t1, m: HW, n: C, k: HW)
        encodeMatmulNT(enc, a: t1, b: try w("\(prefix).to_out.weight"), y: q, m: HW, n: C, k: C)
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
