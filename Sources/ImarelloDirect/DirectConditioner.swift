import Foundation
import Metal
import MLX
import ImarelloDiT
import ImarelloWeights
import ImarelloCore
import ImarelloRuntime

/// MLX-module-free conditioning for the direct engine: timestep embedding +
/// the three modulation heads + AdaLN-out on the metallib's f32 qmm (M=1,
/// exact product parity), context projection on the bf16 qmm, rope tables via
/// the weight-free `Flux2PosEmbed`. Removes the 1.9 GiB product-module load
/// from `direct-generate`.
public final class DirectConditioner {

    static let dim = 3072
    static let sinDim = 256

    struct Q {
        let w: MTLBuffer
        let s: MTLBuffer
        let b: MTLBuffer
        let n: Int
        let k: Int
    }

    let device: MTLDevice
    let queue: MTLCommandQueue
    let qmmF32PSO: MTLComputePipelineState
    let qmmBF16PSO: MTLComputePipelineState
    let siluPSO: MTLComputePipelineState
    let timeL1, timeL2, modImg, modTxt, modSingle, normOut, contextEmb: Q
    let posEmbed = Flux2PosEmbed()

    public init(transformerDirectory: URL, metallibURL: URL) throws {
        guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else {
            throw DirectQmmSpike.SpikeError.metal("device/queue")
        }
        device = dev
        queue = q
        let lib = try dev.makeLibrary(URL: metallibURL)
        func pso(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = lib.makeFunction(name: name) else {
                throw DirectQmmSpike.SpikeError.metal("kernel \(name)")
            }
            return try dev.makeComputePipelineState(function: fn)
        }
        qmmF32PSO = try pso("affine_qmm_t_float_gs_64_b_4_alN_true_batch_0")
        qmmBF16PSO = try pso("affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0")
        let glue = try DirectDiTKernels.makeLibrary(
            device: dev,
            directMetallibURL: DirectEngineArtifacts.directMetallibURL(
                beside: metallibURL))
        guard let sfn = glue.makeFunction(name: "dd_silu_f32") else {
            throw DirectQmmSpike.SpikeError.metal("dd_silu_f32")
        }
        siluPSO = try dev.makeComputePipelineState(function: sfn)

        var arrays: [String: MLXArray] = [:]
        for shard in ["0.safetensors", "1.safetensors"] {
            let url = transformerDirectory.appendingPathComponent(shard)
            if FileManager.default.fileExists(atPath: url.path) {
                for (k, v) in try MLX.loadArrays(url: url) { arrays[k] = v }
            }
        }
        func upload(_ a: MLXArray, _ l: String) throws -> MTLBuffer {
            eval(a)
            let d = a.asData(access: .copy).data
            return try d.withUnsafeBytes { raw -> MTLBuffer in
                guard let base = raw.baseAddress,
                    let b = dev.makeBuffer(bytes: base, length: raw.count)
                else { throw DirectQmmSpike.SpikeError.metal("upload \(l)") }
                b.label = l
                return b
            }
        }
        /// f32 qmm kernel wants T = float scales/biases; bf16 context stays raw.
        func quant(_ name: String, n: Int, k: Int, scalesF32: Bool) throws -> Q {
            guard let w = arrays["\(name).weight"], let sc = arrays["\(name).scales"],
                let bi = arrays["\(name).biases"]
            else { throw DirectQmmSpike.SpikeError.missingTensor(name) }
            try DirectTensorValidation.requireQuantized(
                weight: w, scales: sc, biases: bi, n: n, k: k, name: name)
            return Q(
                w: try upload(w, "\(name).w"),
                s: try upload(scalesF32 ? sc.asType(.float32) : sc, "\(name).s"),
                b: try upload(scalesF32 ? bi.asType(.float32) : bi, "\(name).b"),
                n: n, k: k)
        }
        timeL1 = try quant("time_guidance_embed.linear_1", n: Self.dim, k: Self.sinDim, scalesF32: true)
        timeL2 = try quant("time_guidance_embed.linear_2", n: Self.dim, k: Self.dim, scalesF32: true)
        modImg = try quant("double_stream_modulation_img.linear", n: 6 * Self.dim, k: Self.dim, scalesF32: true)
        modTxt = try quant("double_stream_modulation_txt.linear", n: 6 * Self.dim, k: Self.dim, scalesF32: true)
        modSingle = try quant("single_stream_modulation.linear", n: 3 * Self.dim, k: Self.dim, scalesF32: true)
        normOut = try quant("norm_out.linear", n: 2 * Self.dim, k: Self.dim, scalesF32: true)
        contextEmb = try quant("context_embedder", n: Self.dim, k: 7680, scalesF32: false)
    }

    // MARK: - GPU helpers (M-row f32/bf16 qmm + silu, blocking — runs once per generate)

    private func qmm(_ q: Q, x: MTLBuffer, y: MTLBuffer, m: Int, pso: MTLComputePipelineState) throws {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw DirectQmmSpike.SpikeError.metal("cb")
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(q.w, offset: 0, index: 0)
        enc.setBuffer(q.s, offset: 0, index: 1)
        enc.setBuffer(q.b, offset: 0, index: 2)
        enc.setBuffer(x, offset: 0, index: 3)
        enc.setBuffer(y, offset: 0, index: 4)
        var k32 = Int32(q.k), n32 = Int32(q.n), m32 = Int32(m)
        enc.setBytes(&k32, length: 4, index: 5)
        enc.setBytes(&n32, length: 4, index: 6)
        enc.setBytes(&m32, length: 4, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: (q.n + 31) / 32, height: (m + 31) / 32, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw DirectQmmSpike.SpikeError.metal("qmm: \(e)") }
    }

    private func encodeQMM(
        _ q: Q, x: MTLBuffer, y: MTLBuffer, rows: Int,
        pso: MTLComputePipelineState, encoder: MTLComputeCommandEncoder
    ) {
        encoder.setComputePipelineState(pso)
        encoder.setBuffer(q.w, offset: 0, index: 0)
        encoder.setBuffer(q.s, offset: 0, index: 1)
        encoder.setBuffer(q.b, offset: 0, index: 2)
        encoder.setBuffer(x, offset: 0, index: 3)
        encoder.setBuffer(y, offset: 0, index: 4)
        var k32 = Int32(q.k), n32 = Int32(q.n), m32 = Int32(rows)
        encoder.setBytes(&k32, length: 4, index: 5)
        encoder.setBytes(&n32, length: 4, index: 6)
        encoder.setBytes(&m32, length: 4, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: (q.n + 31) / 32, height: (rows + 31) / 32, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 2))
    }

    private func encodeSilu(
        _ x: MTLBuffer, _ y: MTLBuffer, count: Int,
        encoder: MTLComputeCommandEncoder
    ) {
        encoder.setComputePipelineState(siluPSO)
        encoder.setBuffer(x, offset: 0, index: 0)
        encoder.setBuffer(y, offset: 0, index: 1)
        var n32 = UInt32(count)
        encoder.setBytes(&n32, length: 4, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func buffer(_ floats: [Float], _ l: String) throws -> MTLBuffer {
        try floats.withUnsafeBufferPointer { p -> MTLBuffer in
            guard let b = device.makeBuffer(bytes: p.baseAddress!, length: floats.count * 4) else {
                throw DirectQmmSpike.SpikeError.metal(l)
            }
            return b
        }
    }
    private func empty(_ count: Int, _ l: String) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: count * 4) else {
            throw DirectQmmSpike.SpikeError.metal(l)
        }
        return b
    }
    private func readF32(_ b: MTLBuffer, _ count: Int) -> [Float] {
        Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: Float.self, capacity: count), count: count))
    }
    private func silu(_ x: MTLBuffer, _ y: MTLBuffer, _ n: Int) throws {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else {
            throw DirectQmmSpike.SpikeError.metal("cb")
        }
        enc.setComputePipelineState(siluPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(y, offset: 0, index: 1)
        var n32 = UInt32(n)
        enc.setBytes(&n32, length: 4, index: 2)
        enc.dispatchThreads(
            MTLSize(width: n, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    // MARK: - Public API

    /// Sinusoidal embedding, exact product math (flipSinToCos: [cos | sin]).
    static func sinusoid(_ t: Float) -> [Float] {
        let half = sinDim / 2
        var out = [Float](repeating: 0, count: sinDim)
        for i in 0 ..< half {
            let freq = expf(-logf(10000) * Float(i) / Float(half))
            let arg = t * freq
            out[i] = cosf(arg)
            out[half + i] = sinf(arg)
        }
        return out
    }

    public func stepConditioning(timesteps: [Float]) throws -> [Flux2StepConditioning] {
        guard !timesteps.isEmpty else { return [] }
        let d = Self.dim
        let rows = timesteps.count
        let sinusoidal = timesteps.flatMap(Self.sinusoid)
        let sinBuf = try buffer(sinusoidal, "sin")
        let t1 = try empty(rows * d, "t1")
        let t1s = try empty(rows * d, "t1s")
        let temb = try empty(rows * d, "temb")
        let tembS = try empty(rows * d, "tembS")
        let mImg = try empty(rows * 6 * d, "mImg")
        let mTxt = try empty(rows * 6 * d, "mTxt")
        let mSingle = try empty(rows * 3 * d, "mSingle")
        let outC = try empty(rows * 2 * d, "outC")
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { throw DirectQmmSpike.SpikeError.metal("conditioner command buffer") }
        encodeQMM(timeL1, x: sinBuf, y: t1, rows: rows, pso: qmmF32PSO, encoder: encoder)
        encodeSilu(t1, t1s, count: rows * d, encoder: encoder)
        encodeQMM(timeL2, x: t1s, y: temb, rows: rows, pso: qmmF32PSO, encoder: encoder)
        encodeSilu(temb, tembS, count: rows * d, encoder: encoder)
        encodeQMM(modImg, x: tembS, y: mImg, rows: rows, pso: qmmF32PSO, encoder: encoder)
        encodeQMM(modTxt, x: tembS, y: mTxt, rows: rows, pso: qmmF32PSO, encoder: encoder)
        encodeQMM(modSingle, x: tembS, y: mSingle, rows: rows, pso: qmmF32PSO, encoder: encoder)
        encodeQMM(normOut, x: tembS, y: outC, rows: rows, pso: qmmF32PSO, encoder: encoder)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw DirectQmmSpike.SpikeError.metal("conditioner batch: \(error)")
        }

        let tembValues = readF32(temb, rows * d)
        let imgValues = readF32(mImg, rows * 6 * d)
        let txtValues = readF32(mTxt, rows * 6 * d)
        let singleValues = readF32(mSingle, rows * 3 * d)
        let outValues = readF32(outC, rows * 2 * d)
        return (0 ..< rows).map { row in
            func triples(
                _ values: [Float], sets: Int
            ) -> [(MLXArray, MLXArray, MLXArray)] {
                return (0 ..< sets).map { i in
                    let base = row * 3 * sets * d + 3 * i * d
                    return (
                        MLXArray(Array(values[base ..< base + d]), [1, 1, d]),
                        MLXArray(Array(values[base + d ..< base + 2 * d]), [1, 1, d]),
                        MLXArray(Array(values[base + 2 * d ..< base + 3 * d]), [1, 1, d]))
                }
            }
            let tembBase = row * d
            let outBase = row * 2 * d
            return Flux2StepConditioning(
                temb: MLXArray(Array(tembValues[tembBase ..< tembBase + d]), [1, d]),
                doubleImg: triples(imgValues, sets: 2),
                doubleTxt: triples(txtValues, sets: 2),
                single: triples(singleValues, sets: 1)[0],
                outConditioning: (
                    scale: MLXArray(Array(outValues[outBase ..< outBase + d]), [1, d]),
                    shift: MLXArray(Array(outValues[outBase + d ..< outBase + 2 * d]), [1, d])))
        }
    }

    /// Context projection `[1, 512, 7680] bf16 → [1, 512, 3072] f32` on the
    /// bf16 qmm (bit-exact with the product's contextEmbedder call).
    public func projectContext(_ embeds: MLXArray) throws -> MLXArray {
        let L = embeds.dim(1)
        let e16 = embeds.asType(.bfloat16)
        eval(e16)
        let data = e16.asData(access: .copy).data
        let xBuf = try data.withUnsafeBytes { raw -> MTLBuffer in
            guard let base = raw.baseAddress,
                let b = device.makeBuffer(bytes: base, length: raw.count)
            else { throw DirectQmmSpike.SpikeError.metal("ctx x") }
            return b
        }
        guard let yBuf = device.makeBuffer(length: L * Self.dim * 2) else {
            throw DirectQmmSpike.SpikeError.metal("ctx y")
        }
        try qmm(contextEmb, x: xBuf, y: yBuf, m: L, pso: qmmBF16PSO)
        let u = yBuf.contents().bindMemory(to: UInt16.self, capacity: L * Self.dim)
        var f = [Float](repeating: 0, count: L * Self.dim)
        for i in 0 ..< L * Self.dim {
            f[i] = Float(bitPattern: UInt32(u[i]) << 16)
        }
        let out = MLXArray(f, [1, L, Self.dim])
        eval(out)
        return out
    }

    /// Joint rope tables (txt ∥ img) via the weight-free product PosEmbed.
    public func rope(imgIds: MLXArray, txtIds: MLXArray) -> (MLXArray, MLXArray) {
        var iIds = imgIds
        var tIds = txtIds
        if iIds.ndim == 3 { iIds = iIds[0] }
        if tIds.ndim == 3 { tIds = tIds[0] }
        let img = posEmbed(iIds)
        let txt = posEmbed(tIds)
        let c = concatenated([txt.0, img.0], axis: 0)
        let s = concatenated([txt.1, img.1], axis: 0)
        eval(c, s)
        return (c, s)
    }
}

/// Verification: the conditioner vs the product module's precompute.
public enum DirectConditionerSpike {
    public static func run(snapshot: ModelSnapshot, metallibURL: URL) async throws -> String {
        let cond = try DirectConditioner(
            transformerDirectory: snapshot.root.appendingPathComponent("transformer", isDirectory: true),
            metallibURL: metallibURL)

        let dit = DiTModule(snapshot: snapshot)
        try await dit.load()
        let scheduler = Flux2Scheduler(numInferenceSteps: 4, imageSeqLen: 1024)
        let stepTimesteps: [MLXArray] = scheduler.timesteps.map { MLXArray([$0]).asType(.float32) }
        let oracle = try dit.precomputeStepConditioning(
            timesteps: stepTimesteps, batch: 1, dtype: .float32, guidance: MLXArray(1.0))
        MLXRandom.seed(5)
        let embeds = (MLXRandom.normal([1, 512, 7680]) * 0.5).asType(.bfloat16)
        eval(embeds)
        let oCtx = try dit.projectContext(embeds).asType(.float32)
        let imgIds = LatentOps.imageIds(width: 512, height: 512)
        let txtIds = LatentOps.textIds()
        let oRope = try dit.prepareRotaryEmbeddings(imgIds: imgIds, txtIds: txtIds)
        eval(oCtx, oRope.0, oRope.1)

        let mine = try cond.stepConditioning(timesteps: scheduler.timesteps)
        let mCtx = try cond.projectContext(embeds)
        let mRope = cond.rope(imgIds: imgIds, txtIds: txtIds)

        func cos(_ a: MLXArray, _ b: MLXArray) -> Double {
            let x = a.asType(.float32).reshaped([-1]).asArray(Float.self)
            let y = b.asType(.float32).reshaped([-1]).asArray(Float.self)
            var dot = 0.0, na = 0.0, nb = 0.0
            for i in 0 ..< x.count {
                let p = Double(x[i]), q = Double(y[i])
                dot += p * q; na += p * p; nb += q * q
            }
            return dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
        }
        var lines: [String] = []
        for (i, (m, o)) in zip(mine, oracle).enumerated() {
            let c = [
                cos(m.temb, o.temb),
                cos(m.doubleImg[0].0, o.doubleImg[0].0), cos(m.doubleImg[1].2, o.doubleImg[1].2),
                cos(m.doubleTxt[0].1, o.doubleTxt[0].1),
                cos(m.single.0, o.single.0),
                cos(m.outConditioning.scale, o.outConditioning.scale),
                cos(m.outConditioning.shift, o.outConditioning.shift),
            ].min()!
            lines.append(String(format: "  step %d worst-field cosine: %.7f", i, c))
        }
        lines.append(String(format: "  context cosine: %.7f", cos(mCtx, oCtx)))
        lines.append(String(format: "  rope cos/sin cosine: %.7f / %.7f",
                            cos(mRope.0, oRope.0), cos(mRope.1, oRope.1)))
        return "direct-conditioner vs product module:\n" + lines.joined(separator: "\n")
    }
}
