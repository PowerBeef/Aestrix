import Foundation
import Metal
import MLX

/// Stage-2 direct-dispatch research (bare-metal ladder, `research/bare-metal`).
///
/// Milestone A: run ONE real TE projection (layer-0 `q_proj`, M=512×K=2560→N=4096,
/// 4-bit affine gs=64) by encoding MLX's own `affine_qmm_t` metallib kernel from a
/// hand-built Metal command buffer, and verify the output against the MLX oracle
/// (`quantizedMM`) executing the same kernel through the MLX runtime.
///
/// The point is not speed of one GEMM — it is proving the kernel-ABI contract
/// (names, buffer indices, scalar widths, grid shape) so the full TE forward can
/// be recorded once and replayed without MLX's per-op machinery.
public enum DirectQmmSpike {

    public enum SpikeError: Error, CustomStringConvertible {
        case missingTensor(String)
        case invalidTensor(String)
        case metal(String)

        public var description: String {
            switch self {
            case .missingTensor(let name): return "missing tensor: \(name)"
            case .invalidTensor(let what): return "invalid tensor: \(what)"
            case .metal(let what): return "Metal setup failed: \(what)"
            }
        }
    }

    static let M = 512
    static let K = 2560
    static let N = 4096
    static let groupSize = 64
    static let bits = 4

    public static func run(teDirectory: URL, metallibURL: URL) throws -> String {
        // -- Tensors ---------------------------------------------------------
        let shard = teDirectory.appendingPathComponent("0.safetensors")
        let arrays = try MLX.loadArrays(url: shard)
        let prefix = "layers.0.self_attn.q_proj"
        guard let w = arrays["\(prefix).weight"] else {
            throw SpikeError.missingTensor("\(prefix).weight")
        }
        guard let scalesRaw = arrays["\(prefix).scales"],
            let biasesRaw = arrays["\(prefix).biases"]
        else {
            throw SpikeError.missingTensor("\(prefix).scales/.biases")
        }
        let scales = scalesRaw.asType(.float16)
        let biases = biasesRaw.asType(.float16)
        MLXRandom.seed(42)
        let x = (MLXRandom.normal([M, K]) * 0.05).asType(.float16)
        eval(w, scales, biases, x)

        // -- Oracle (MLX runtime, same underlying kernel) --------------------
        let yRef = quantizedMM(
            x, w, scales: scales, biases: biases,
            transpose: true, groupSize: groupSize, bits: bits)
        eval(yRef)
        let yRefF = yRef.asType(.float32).asArray(Float.self)

        // Oracle per-call wall (graph build + eval + dispatch), warm.
        var oracleMS = 0.0
        do {
            for _ in 0 ..< 3 { eval(quantizedMM(
                x, w, scales: scales, biases: biases,
                transpose: true, groupSize: groupSize, bits: bits)) }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< 10 {
                let y = quantizedMM(
                    x, w, scales: scales, biases: biases,
                    transpose: true, groupSize: groupSize, bits: bits)
                eval(y)
            }
            oracleMS = (CFAbsoluteTimeGetCurrent() - t0) * 100
        }

        // -- Direct dispatch -------------------------------------------------
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SpikeError.metal("no default device")
        }
        let library = try device.makeLibrary(URL: metallibURL)
        // Host names use the Metal type spelling ("float16_t"), matching the
        // .metal instantiation macros the nojit runtime resolves against.
        let kname = "affine_qmm_t_float16_t_gs_\(groupSize)_b_\(bits)_alN_true_batch_0"
        guard let fn = library.makeFunction(name: kname) else {
            throw SpikeError.metal("kernel not found in metallib: \(kname)")
        }
        let pso = try device.makeComputePipelineState(function: fn)

        func upload(_ array: MLXArray, label: String) throws -> MTLBuffer {
            let data = array.asData(access: .copy).data
            return try data.withUnsafeBytes { raw -> MTLBuffer in
                guard let base = raw.baseAddress,
                    let b = device.makeBuffer(bytes: base, length: raw.count)
                else { throw SpikeError.metal("buffer upload failed: \(label)") }
                b.label = label
                return b
            }
        }
        let wBuf = try upload(w, label: "w")
        let sBuf = try upload(scales, label: "scales")
        let bBuf = try upload(biases, label: "biases")
        let xBuf = try upload(x, label: "x")
        guard let yBuf = device.makeBuffer(length: M * N * 2),
            let queue = device.makeCommandQueue()
        else { throw SpikeError.metal("output buffer / queue") }
        yBuf.label = "y"

        // Kernel scalar args are `constant int&` — 32-bit.
        var k32 = Int32(K)
        var n32 = Int32(N)
        var m32 = Int32(M)
        let grid = MTLSize(width: (N + 31) / 32, height: (M + 31) / 32, depth: 1)
        let tpg = MTLSize(width: 32, height: 2, depth: 2)

        func dispatchOnce() throws {
            guard let cb = queue.makeCommandBuffer(),
                let enc = cb.makeComputeCommandEncoder()
            else { throw SpikeError.metal("command buffer") }
            enc.setComputePipelineState(pso)
            enc.setBuffer(wBuf, offset: 0, index: 0)
            enc.setBuffer(sBuf, offset: 0, index: 1)
            enc.setBuffer(bBuf, offset: 0, index: 2)
            enc.setBuffer(xBuf, offset: 0, index: 3)
            enc.setBuffer(yBuf, offset: 0, index: 4)
            enc.setBytes(&k32, length: 4, index: 5)
            enc.setBytes(&n32, length: 4, index: 6)
            enc.setBytes(&m32, length: 4, index: 7)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tpg)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let err = cb.error { throw SpikeError.metal("execution: \(err)") }
        }
        try dispatchOnce()  // PSO/JIT warm
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0 ..< 10 { try dispatchOnce() }
        let directMS = (CFAbsoluteTimeGetCurrent() - t0) * 100

        // -- Verify ----------------------------------------------------------
        let yPtr = yBuf.contents().bindMemory(to: UInt16.self, capacity: M * N)
        var dot = 0.0, na = 0.0, nb = 0.0, maxDiff = 0.0
        var exact = 0
        let refF16 = yRef.asData(access: .copy).data
        try refF16.withUnsafeBytes { raw in
            guard let base = raw.baseAddress,
                  raw.count == M * N * MemoryLayout<UInt16>.size
            else { throw SpikeError.invalidTensor("oracle f16 byte count") }
            let refBits = base.assumingMemoryBound(to: UInt16.self)
            for i in 0 ..< M * N {
                let a = Double(Float(Float16(bitPattern: yPtr[i])))
                let b = Double(yRefF[i])
                dot += a * b; na += a * a; nb += b * b
                maxDiff = max(maxDiff, abs(a - b))
                if yPtr[i] == refBits[i] { exact += 1 }
            }
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
        let pass = cosine >= 0.99999 && maxDiff < 0.01

        return """
        direct-qmm spike (milestone A) — layer-0 q_proj, M=\(M) K=\(K) N=\(N), \(kname)
          cosine_vs_oracle: \(String(format: "%.7f", cosine))
          max_abs_diff:     \(String(format: "%.6f", maxDiff))
          bit_exact:        \(exact)/\(M * N) (\(String(format: "%.1f", 100.0 * Double(exact) / Double(M * N)))%)
          oracle_per_call:  \(String(format: "%.3f", oracleMS)) ms (MLX runtime, warm)
          direct_per_call:  \(String(format: "%.3f", directMS)) ms (raw dispatch incl. blocking wait)
          verdict:          \(pass ? "PASS — ABI contract proven" : "FAIL — investigate ABI mismatch")
        """
    }
}
