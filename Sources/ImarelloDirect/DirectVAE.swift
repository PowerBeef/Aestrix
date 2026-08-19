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
