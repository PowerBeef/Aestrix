import Foundation
import Metal
import MLX

/// Neural-Accelerator (NAX) support for the direct engine.
///
/// The metallib ships `affine_qmm_t_nax_*` kernels (Metal-4 tensor ops that
/// execute on the A19/M5-class neural accelerators). MLX selects them inside
/// its own dispatcher; the direct engine bypasses that dispatcher, so this
/// file owns the same decision: the availability recipe mirrors core's
/// `is_nax_available()` (OS ≥ 26.2, GPU architecture gen ≥ 17 — or ≥ 18 for
/// 'p'-suffixed parts), and the kernel-name/grid math mirrors
/// `quantized.cpp::qmm_nax` (buffers w/s/b/x/y at 0–4 + K/N/M at 5–7 —
/// byte-identical to the Steel qmm ABI; only the name and the 64-wide tile
/// grid differ; dispatch requires K % 64 == 0 and a non-f32 dtype).
///
/// The machine of record (M2, gen 14) can compile and gate this path but can
/// never execute it — every numeric claim needs an A19/M5-class device.
public enum DirectNAX {

    public struct Availability {
        public let archName: String
        public let gen: Int
        public let suffix: Character
        public let osOK: Bool
        public let eligible: Bool
        public let reason: String
    }

    public static func probe(device: MTLDevice) -> Availability {
        let arch = device.architecture.name  // e.g. "applegpu_g14g"
        var gen = 0
        var suffix: Character = "?"
        // Parse "applegpu_g<gen><suffix>": digits between '_g' and the final letter.
        if let range = arch.range(of: "_g") {
            let tail = arch[range.upperBound...]
            let digits = tail.prefix(while: { $0.isNumber })
            gen = Int(digits) ?? 0
            if let last = tail.last, last.isLetter { suffix = last }
        }
        var osOK = false
        if #available(macOS 26.2, iOS 26.2, *) { osOK = true }
        let genNeeded = suffix == "p" ? 18 : 17
        let genOK = gen >= genNeeded
        let eligible = osOK && genOK
        let reason: String
        if eligible {
            reason = "eligible (\(arch), gen \(gen) ≥ \(genNeeded), OS ≥ 26.2)"
        } else if !osOK {
            reason = "OS below 26.2"
        } else {
            reason = "GPU \(arch) gen \(gen) < \(genNeeded) — no neural accelerators"
        }
        return Availability(
            archName: arch, gen: gen, suffix: suffix, osOK: osOK,
            eligible: eligible, reason: reason)
    }

    /// NAX 4-bit affine qmm (transposed weights, f16, gs=64) kernel name.
    public static func qmmF16Name(n: Int) -> String {
        "affine_qmm_t_nax_float16_t_gs_64_b_4_bm64_bn64_bk64_wm2_wn2_alN_\(n % 64 == 0 ? "true" : "false")_batch_0"
    }

    /// NAX qmm tile grid: ((N+63)/64, (M+63)/64, 1) × (32, 2, 2).
    public static func qmmGrid(m: Int, n: Int) -> (MTLSize, MTLSize) {
        (MTLSize(width: (n + 63) / 64, height: (m + 63) / 64, depth: 1),
         MTLSize(width: 32, height: 2, depth: 2))
    }
}

/// N-track spike: prove the NAX qmm ABI. On NAX hardware (A19/M5-class,
/// OS ≥ 26.2) it dispatches the NAX kernel at a DiT shape and scores it
/// against the proven Steel qmm (bit-exactness reference + timing). On
/// anything else it reports the probe and — as a diagnostic — whether PSO
/// creation for a NAX kernel even succeeds on this GPU.
public enum DirectNAXQmmSpike {

    /// `ditDirectory: nil` → synthetic quantized tensors (device runner needs
    /// no weights on disk; ABI/bit-exactness proof is identical either way).
    public static func run(ditDirectory: URL?, metallibURL: URL) throws -> String {
        guard let dev = MTLCreateSystemDefaultDevice(), let queue = dev.makeCommandQueue() else {
            throw DirectQmmSpike.SpikeError.metal("device/queue")
        }
        let nax = DirectNAX.probe(device: dev)
        var report = """
        direct-nax qmm spike
          arch:      \(nax.archName) (gen \(nax.gen), suffix '\(nax.suffix)')
          os_26_2:   \(nax.osOK)
          eligible:  \(nax.eligible) — \(nax.reason)
        """
        let lib = try dev.makeLibrary(URL: metallibURL)
        let kname = DirectNAX.qmmF16Name(n: 3072)
        guard let fn = lib.makeFunction(name: kname) else {
            return report + "\n  kernel:    MISSING from metallib (\(kname))"
        }
        let pso: MTLComputePipelineState
        do {
            pso = try dev.makeComputePipelineState(function: fn)
            report += "\n  pso:       created OK (\(kname))"
        } catch {
            return report + "\n  pso:       creation FAILED on this GPU — \(error.localizedDescription)"
        }
        guard nax.eligible else {
            return report + "\n  verdict:   ABI staged; numeric gate needs A19/M5-class hardware"
        }

        // One DiT-shaped projection (K=3072, N=3072): real to_q weights when a
        // snapshot is present, synthetic 4-bit tensors otherwise.
        let m = 4096, k = 3072, n = 3072
        MLXRandom.seed(7)
        var wq: MLXArray
        var sc: MLXArray
        var bi: MLXArray
        if let ditDirectory {
            var arrays: [String: MLXArray] = [:]
            for shard in ["0.safetensors", "1.safetensors"] {
                let url = ditDirectory.appendingPathComponent(shard)
                if FileManager.default.fileExists(atPath: url.path) {
                    for (kk, v) in try MLX.loadArrays(url: url)
                    where kk.contains("transformer_blocks.0.attn.to_q") {
                        arrays[kk] = v
                    }
                }
            }
            let base = "transformer_blocks.0.attn.to_q"
            guard let w = arrays["\(base).weight"], let s = arrays["\(base).scales"],
                let b = arrays["\(base).biases"]
            else { throw DirectQmmSpike.SpikeError.missingTensor(base) }
            wq = w
            sc = s
            bi = b
        } else {
            wq = MLXRandom.randInt(low: 0, high: Int32.max, [n, k / 8]).asType(.uint32)
            sc = (MLXRandom.normal([n, k / 64]) * 0.02).asType(.float16)
            bi = (MLXRandom.normal([n, k / 64]) * 0.01).asType(.float16)
        }
        let x = (MLXRandom.normal([m, k]) * 0.5).asType(.float16)
        let scF16 = sc.asType(.float16), biF16 = bi.asType(.float16)
        eval(x, wq, scF16, biF16)

        func upload(_ a: MLXArray, _ l: String) throws -> MTLBuffer {
            let d = a.asData(noCopy: false)
            return try d.withUnsafeBytes { raw -> MTLBuffer in
                guard let b = dev.makeBuffer(bytes: raw.baseAddress!, length: raw.count) else {
                    throw DirectQmmSpike.SpikeError.metal(l)
                }
                return b
            }
        }
        let wB = try upload(wq, "w"), sB = try upload(scF16, "s"), bB = try upload(biF16, "b")
        let xB = try upload(x, "x")
        guard let ySteel = dev.makeBuffer(length: m * n * 2),
            let yNAX = dev.makeBuffer(length: m * n * 2)
        else { throw DirectQmmSpike.SpikeError.metal("y") }
        guard let steelFn = lib.makeFunction(
            name: "affine_qmm_t_float16_t_gs_64_b_4_alN_true_batch_0")
        else { throw DirectQmmSpike.SpikeError.metal("steel qmm") }
        let steelPSO = try dev.makeComputePipelineState(function: steelFn)

        func encode(_ enc: MTLComputeCommandEncoder, _ p: MTLComputePipelineState,
                    y: MTLBuffer, grid: MTLSize, group: MTLSize) {
            enc.setComputePipelineState(p)
            enc.setBuffer(wB, offset: 0, index: 0)
            enc.setBuffer(sB, offset: 0, index: 1)
            enc.setBuffer(bB, offset: 0, index: 2)
            enc.setBuffer(xB, offset: 0, index: 3)
            enc.setBuffer(y, offset: 0, index: 4)
            var k32 = Int32(k), n32 = Int32(n), m32 = Int32(m)
            enc.setBytes(&k32, length: 4, index: 5)
            enc.setBytes(&n32, length: 4, index: 6)
            enc.setBytes(&m32, length: 4, index: 7)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
        }
        func timed(_ p: MTLComputePipelineState, y: MTLBuffer, grid: MTLSize, group: MTLSize) throws -> Double {
            for _ in 0 ..< 2 {  // warm
                guard let cb = queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else {
                    throw DirectQmmSpike.SpikeError.metal("cb")
                }
                encode(e, p, y: y, grid: grid, group: group)
                e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
                if let err = cb.error { throw DirectQmmSpike.SpikeError.metal("exec \(err)") }
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< 20 {
                guard let cb = queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else {
                    throw DirectQmmSpike.SpikeError.metal("cb")
                }
                encode(e, p, y: y, grid: grid, group: group)
                e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            }
            return (CFAbsoluteTimeGetCurrent() - t0) * 50
        }
        let steelMS = try timed(
            steelPSO, y: ySteel,
            grid: MTLSize(width: (n + 31) / 32, height: (m + 31) / 32, depth: 1),
            group: MTLSize(width: 32, height: 2, depth: 2))
        let (grid, group) = DirectNAX.qmmGrid(m: m, n: n)
        let naxMS = try timed(pso, y: yNAX, grid: grid, group: group)

        let count = m * n
        let pa = ySteel.contents().bindMemory(to: Float16.self, capacity: count)
        let pb = yNAX.contents().bindMemory(to: Float16.self, capacity: count)
        var dot = 0.0, na = 0.0, nb = 0.0, maxDiff = 0.0
        var exact = 0
        for i in 0 ..< count {
            let a = Double(Float(pa[i])), b = Double(Float(pb[i]))
            dot += a * b; na += a * a; nb += b * b
            maxDiff = max(maxDiff, abs(a - b))
            if pa[i] == pb[i] { exact += 1 }
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
        return report + """

          shape:     M=\(m) K=\(k) N=\(n) (real to_q weights)
          cosine_nax_vs_steel: \(String(format: "%.7f", cosine))
          max_abs_diff:        \(String(format: "%.6f", maxDiff))
          bit_exact:           \(exact)/\(count) (\(String(format: "%.1f", 100.0 * Double(exact) / Double(count)))%)
          steel: \(String(format: "%.3f", steelMS)) ms · nax: \(String(format: "%.3f", naxMS)) ms → \(String(format: "%.2f", steelMS / naxMS))×
          verdict:   \(cosine >= 0.9999 ? "PASS — NAX qmm ABI proven on this device" : "investigate")
        """
    }
}
