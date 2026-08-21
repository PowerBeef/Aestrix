import Foundation
import Metal
import MLX

/// Direct-VAE V1: one decoder-class conv (3×3, stride 1, pad 1, NHWC) driven
/// through the metallib's steel implicit-gemm conv kernel, verified against
/// the MLX `conv2d` oracle. Proves the conv ABI (MLXConvParams<2> +
/// ImplicitGemmConv2DParams) so the decoder can be ported stage by stage.
public enum DirectVAESpike {

    public static func run(metallibURL: URL) throws -> String {
        let N = 1, H = 64, W = 64, C = 256, O = 256
        let kH = 3, kW = 3

        MLXRandom.seed(21)
        let input = (MLXRandom.normal([N, H, W, C]) * 0.5).asType(.float16)
        let weight = (MLXRandom.normal([O, kH, kW, C]) * 0.05).asType(.float16)
        eval(input, weight)

        let oracle = conv2d(input, weight, stride: .init(1), padding: .init(1))
        eval(oracle)
        var oracleMS = 0.0
        do {
            for _ in 0 ..< 3 { eval(conv2d(input, weight, stride: .init(1), padding: .init(1))) }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< 20 { eval(conv2d(input, weight, stride: .init(1), padding: .init(1))) }
            oracleMS = (CFAbsoluteTimeGetCurrent() - t0) * 50
        }

        guard let device = MTLCreateSystemDefaultDevice(),
            let queue = device.makeCommandQueue()
        else { throw DirectQmmSpike.SpikeError.metal("device") }
        let lib = try device.makeLibrary(URL: metallibURL)

        // Host dispatch mirror (conv.cpp implicit_gemm_conv_2D_gpu).
        let implicitM = N * H * W
        let implicitN = O
        let implicitK = kH * kW * C
        let wm = 2, wn = 2
        let bm = implicitM >= 8192 && C >= 64 ? 64 : 32
        let bn = (bm == 64 || implicitN >= 64) ? 64 : 32
        let bk = 16
        let tn = (implicitN + bn - 1) / bn
        let tm = (implicitM + bm - 1) / bm
        let channelKIters = (C + bk - 1) / bk
        let gemmKIters = kH * kW * channelKIters
        let inStrideW = C, inStrideH = W * C
        let ijw = inStrideW * 1, ijh = inStrideH * 1
        let inpJumpW = ijw
        let inpJumpH = ijh - (kW - 1) * ijw
        let inpJumpC = bk - (kH - 1) * ijh - (kW - 1) * ijw

        let kname = "implicit_gemm_conv_2d_float16_bm\(bm)_bn\(bn)_bk\(bk)_wm\(wm)_wn\(wn)_channel_l_filter_s"
        guard let fn = lib.makeFunction(name: kname) else {
            throw DirectQmmSpike.SpikeError.metal("kernel not found: \(kname)")
        }
        let pso = try device.makeComputePipelineState(function: fn)

        func upload(_ a: MLXArray, _ l: String) throws -> MTLBuffer {
            let d = a.asData(access: .copy).data
            return try d.withUnsafeBytes { raw -> MTLBuffer in
                guard let base = raw.baseAddress,
                    let b = device.makeBuffer(bytes: base, length: raw.count)
                else { throw DirectQmmSpike.SpikeError.metal(l) }
                return b
            }
        }
        let inBuf = try upload(input, "in")
        let wtBuf = try upload(weight, "wt")
        guard let outBuf = device.makeBuffer(length: N * H * W * O * 2) else {
            throw DirectQmmSpike.SpikeError.metal("out")
        }

        // MLXConvParams<2>: 17 int32 (68 B), pad to 72, 3×4 int64 strides, groups, flip.
        var params = Data()
        func i32(_ v: Int) { var x = Int32(v); withUnsafeBytes(of: &x) { params.append(contentsOf: $0) } }
        func i64(_ v: Int) { var x = Int64(v); withUnsafeBytes(of: &x) { params.append(contentsOf: $0) } }
        i32(N); i32(C); i32(O)
        i32(H); i32(W)          // iS
        i32(kH); i32(kW)        // wS
        i32(H); i32(W)          // oS
        i32(1); i32(1)          // str
        i32(1); i32(1)          // pad
        i32(1); i32(1)          // kdil
        i32(1); i32(1)          // idil
        params.append(Data(repeating: 0, count: (8 - params.count % 8) % 8))
        i64(H * W * C); i64(W * C); i64(C); i64(1)          // in_strides
        i64(kH * kW * C); i64(kW * C); i64(C); i64(1)       // wt_strides
        i64(H * W * O); i64(W * O); i64(O); i64(1)          // out_strides
        i32(1)                   // groups
        var flip = false
        withUnsafeBytes(of: &flip) { params.append(contentsOf: $0.prefix(1)) }
        params.append(Data(repeating: 0, count: (8 - params.count % 8) % 8))

        var gemm = Data()
        for v in [implicitM, implicitN, implicitK, gemmKIters,
                  inpJumpW, inpJumpH, inpJumpC, tn, tm, 0] {
            var x = Int32(v)
            withUnsafeBytes(of: &x) { gemm.append(contentsOf: $0) }
        }

        func runOnce() throws {
            guard let cb = queue.makeCommandBuffer(),
                let enc = cb.makeComputeCommandEncoder()
            else { throw DirectQmmSpike.SpikeError.metal("cb") }
            enc.setComputePipelineState(pso)
            enc.setBuffer(inBuf, offset: 0, index: 0)
            enc.setBuffer(wtBuf, offset: 0, index: 1)
            enc.setBuffer(outBuf, offset: 0, index: 2)
            params.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: params.count, index: 3) }
            gemm.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: gemm.count, index: 4) }
            enc.dispatchThreadgroups(
                MTLSize(width: tn, height: tm, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: wn, depth: wm))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw DirectQmmSpike.SpikeError.metal("exec: \(e)") }
        }
        try runOnce()
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< 20 { try runOnce() }
        let directMS = (CFAbsoluteTimeGetCurrent() - t0) * 50

        let n = N * H * W * O
        let ptr = outBuf.contents().bindMemory(to: UInt16.self, capacity: n)
        let ref = oracle.asType(.float32).asArray(Float.self)
        var dot = 0.0, na = 0.0, nb = 0.0, maxDiff = 0.0
        var exact = 0
        let refF16 = oracle.asData(access: .copy).data
        try refF16.withUnsafeBytes { raw in
            guard let base = raw.baseAddress,
                  raw.count == n * MemoryLayout<UInt16>.size
            else { throw DirectQmmSpike.SpikeError.invalidTensor("VAE oracle f16 byte count") }
            let refBits = base.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< n {
                let a = Double(Float(Float16(bitPattern: ptr[i]))), b = Double(ref[i])
                dot += a * b; na += a * a; nb += b * b
                maxDiff = max(maxDiff, abs(a - b))
                if ptr[i] == refBits[i] { exact += 1 }
            }
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
        return """
        direct-vae V1 — conv 3×3 \(C)→\(O) @\(H)×\(W) NHWC, \(kname)
          cosine_vs_oracle: \(String(format: "%.7f", cosine))
          max_abs_diff:     \(String(format: "%.6f", maxDiff))
          bit_exact:        \(exact)/\(n) (\(String(format: "%.1f", 100.0 * Double(exact) / Double(n)))%)
          oracle_per_call:  \(String(format: "%.3f", oracleMS)) ms · direct: \(String(format: "%.3f", directMS)) ms
          verdict:          \(cosine >= 0.9999 ? "PASS — conv ABI proven" : "investigate")
        """
    }
}
