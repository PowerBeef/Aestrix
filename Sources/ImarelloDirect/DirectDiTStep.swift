import Foundation
import Metal
import MLX
import MLXNN
import ImarelloDiT
import ImarelloPlan

/// Stage-2 F3: the full 25-block FLUX denoise step (5 double + 20 single) as
/// ONE command buffer over a static buffer plan — the direct engine's DiT core.
/// Klein shares one modulation set across all doubles and one across all
/// singles, so 15 conditioning vectors drive the whole step.
public final class DirectDiTStep {

    let lImg: Int
    let lTxt = 512
    var J: Int { lImg + lTxt }
    static let dim = 3072
    static let heads = 24
    static let headDim = 128
    static let ffInner = 9216

    /// At most two scratch-sized passes while still covering every row. Kept
    /// as a testable integer plan so odd token counts cannot regress silently.
    static func chunkRanges(totalRows: Int) -> [Range<Int>] {
        guard totalRows > 0 else { return [] }
        let chunkRows = (totalRows + 1) / 2
        return stride(from: 0, to: totalRows, by: chunkRows).map {
            $0 ..< min(totalRows, $0 + chunkRows)
        }
    }
    static let ffWide = 18432
    static let qkvWidth = 9216
    static let projWidth = 27648
    static let concatWidth = 12288

    struct Q {
        let w: MTLBuffer
        let s: MTLBuffer
        let b: MTLBuffer
        let n: Int
        let k: Int
    }
    struct DoubleWeights {
        let toQ, toK, toV, addQ, addK, addV, toOut, toAddOut, ffIn, ffOut, ffcIn, ffcOut: Q
        let normQ, normK, normAQ, normAK: MTLBuffer
    }
    struct SingleWeights {
        let proj, toOut: Q
        let normQ, normK: MTLBuffer
    }

    let device: MTLDevice
    let queue: MTLCommandQueue
    let directLib: MTLLibrary
    let qmmPSO: MTLComputePipelineState
    /// NAX qmm PSOs (alN_true / alN_false), built only when `useNAXQmm` was
    /// requested AND the GPU is NAX-eligible. Nil ⇒ every qmm rides Steel.
    let qmmNAXTruePSO: MTLComputePipelineState?
    let qmmNAXFalsePSO: MTLComputePipelineState?
    let attnPSO: MTLComputePipelineState
    let attentionProfile: DirectAttentionProfile
    let lnModPSO, rmsPSO, ropePSO, scaleCastPSO, swigluPSO, scaleInPSO, gateAddPSO: MTLComputePipelineState

    var doubles: [DoubleWeights] = []
    var singles: [SingleWeights] = []
    var scratchHeap: MTLHeap!
    var heapComputeBase = 0
    var heapComputeSize = 0
    private var headCursor = 0
    public private(set) var scratchPlanDigest = ""
    lazy var fence: MTLFence = device.makeFence()!

    // Conditioning (uploaded once per generate/step set)
    struct BufferBinding {
        let buffer: MTLBuffer
        let offset: Int
    }
    var modBufs: [String: BufferBinding] = [:]
    private var stepModBindings: [[String: BufferBinding]] = []
    private var conditionerPlanBuffer: MTLBuffer?
    var cosBuf: MTLBuffer!
    var sinBuf: MTLBuffer!

    // Static scratch plan (shared across every block)
    var hA, hB, eA, eB: MTLBuffer!      // f32 double-phase ping-pong
    var jointXA, jointXB: MTLBuffer!    // f32 single-phase ping-pong
    var nhImg, neTxt, projImg, projTxt: MTLBuffer!
    var jointQ, jointK, jointV, jointO: MTLBuffer!
    var outImg, outTxt: MTLBuffer!
    var y1Img, y1Txt: MTLBuffer!
    var ffWideImg, ffWideTxt, swImg, swTxt, ffOutImg, ffOutTxt: MTLBuffer!
    var nhJoint, projJoint, sQ, sK, sV, sConcat, sOut: MTLBuffer!

    public init(
        lImg: Int,
        metallibURL: URL,
        useNAXQmm: Bool = false,
        attentionBackend: DirectAttentionBackend = .steel
    ) throws {
        self.lImg = lImg
        guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else {
            throw DirectQmmSpike.SpikeError.metal("device/queue")
        }
        device = dev
        queue = q
        let mlxLib = try dev.makeLibrary(URL: metallibURL)
        guard let qmmFn = mlxLib.makeFunction(
            name: "affine_qmm_t_float16_t_gs_64_b_4_alN_true_batch_0")
        else { throw DirectQmmSpike.SpikeError.metal("qmm kernel") }
        qmmPSO = try dev.makeComputePipelineState(function: qmmFn)
        if useNAXQmm {
            let nax = DirectNAX.probe(device: dev)
            guard nax.eligible else {
                throw DirectQmmSpike.SpikeError.metal("NAX qmm requested but \(nax.reason)")
            }
            guard let fnT = mlxLib.makeFunction(name: DirectNAX.qmmF16Name(n: 64)),
                let fnF = mlxLib.makeFunction(name: DirectNAX.qmmF16Name(n: 1))
            else { throw DirectQmmSpike.SpikeError.metal("NAX qmm kernels missing from metallib") }
            qmmNAXTruePSO = try dev.makeComputePipelineState(function: fnT)
            qmmNAXFalsePSO = try dev.makeComputePipelineState(function: fnF)
        } else {
            qmmNAXTruePSO = nil
            qmmNAXFalsePSO = nil
        }
        let selectedAttention: DirectAttentionProfile
        switch attentionBackend {
        case .steel:
            selectedAttention = .steelF16
        case .nax:
            let nax = DirectNAX.probe(device: dev)
            guard nax.eligible else {
                throw DirectQmmSpike.SpikeError.metal(
                    "NAX attention requested but \(nax.reason)")
            }
            selectedAttention = .naxF16
        }
        attentionProfile = selectedAttention
        let consts = MTLFunctionConstantValues()
        var f = false
        let jointLength = lImg + lTxt
        var aQ = jointLength % selectedAttention.blockQueries == 0
        var aK = jointLength % selectedAttention.blockKeys == 0
        consts.setConstantValue(&aQ, type: .bool, index: 200)
        consts.setConstantValue(&aK, type: .bool, index: 201)
        consts.setConstantValue(&f, type: .bool, index: 300)
        consts.setConstantValue(&f, type: .bool, index: 301)
        consts.setConstantValue(&f, type: .bool, index: 302)
        attnPSO = try dev.makeComputePipelineState(
            function: try mlxLib.makeFunction(
                name: selectedAttention.functionName,
                constantValues: consts))
        let glue = try DirectDiTKernels.makeLibrary(
            device: dev,
            directMetallibURL: DirectEngineArtifacts.directMetallibURL(
                beside: metallibURL))
        directLib = glue
        func pso(_ n: String) throws -> MTLComputePipelineState {
            guard let fn = glue.makeFunction(name: n) else {
                throw DirectQmmSpike.SpikeError.metal("glue \(n)")
            }
            return try dev.makeComputePipelineState(function: fn)
        }
        lnModPSO = try pso("dd_ln_mod_prescale")
        rmsPSO = try pso("dd_rmsnorm_pitched")
        ropePSO = try pso("dd_rope_interleaved")
        scaleCastPSO = try pso("dd_scale_cast_pitched")
        swigluPSO = try pso("dd_swiglu_pitched")
        scaleInPSO = try pso("dd_scale_inplace")
        gateAddPSO = try pso("dd_gate_add")
        try allocateScratch()
    }

    public private(set) var memoryLedger = DirectMemoryLedger()

    /// Compatibility projection for existing benchmark output. Unlike the old
    /// monotonically increasing counter, this is current engine-owned memory.
    public var ownedBytes: Int { memoryLedger.liveEngineOwnedBytes }

    public func upload(_ a: MLXArray, _ l: String) throws -> MTLBuffer {
        try upload(a, l, retention: .bridge)
    }

    private enum UploadRetention { case persistentWeight, conditioning, bridge }

    private func upload(
        _ a: MLXArray, _ l: String, retention: UploadRetention
    ) throws -> MTLBuffer {
        let d = a.asData(access: .copy).data
        return try d.withUnsafeBytes { raw -> MTLBuffer in
            guard let base = raw.baseAddress,
                let b = device.makeBuffer(bytes: base, length: raw.count)
            else { throw DirectQmmSpike.SpikeError.metal("upload \(l)") }
            b.label = l
            switch retention {
            case .persistentWeight:
                memoryLedger.recordPersistentUpload(bytes: raw.count)
            case .conditioning:
                memoryLedger.recordConditioningUpload(bytes: raw.count)
            case .bridge:
                memoryLedger.recordBridgeUpload(bytes: raw.count)
            }
            return b
        }
    }
    func scratch(_ bytes: Int, _ l: String) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: bytes) else {
            throw DirectQmmSpike.SpikeError.metal("scratch \(l)")
        }
        b.label = l
        memoryLedger.recordScratch(bytes: bytes)
        return b
    }

    private func allocateScratch() throws {
        let d = Self.dim
        // Placement heap with two zones. Persistent zone: the f32 residual
        // ping-pongs, where jointXA/B alias the halves of h/e that are
        // provably dead at the phase boundary (5 doubles -> cur lands on
        // hB/eB, so hA/eA are free for jointXA; hB/eB free after the blit
        // for jointXB). Compute zone: double-phase and single-phase scratch
        // are disjoint in time, so they overlay -> max instead of sum.
        // The heap is UNTRACKED; every encoder joins the MTLFence chain.
        func aligned(_ bytes: Int) -> Int {
            let a = device.heapBufferSizeAndAlign(length: bytes, options: [.storageModeShared])
            return (bytes + a.align - 1) / a.align * a.align
        }
        let hBytes = aligned(lImg * d * 4)
        let eBytes = aligned(lTxt * d * 4)
        let jxBytes = aligned(J * d * 4)
        // Persistent zone layout: [hA][hB][eA][eB]; jointXA at hA, jointXB at hB.
        let pZone = hBytes * 2 + eBytes * 2
        precondition(jxBytes <= hBytes + eBytes, "jointX must fit the aliased half")

        // Compute-zone layouts.
        var dOff = 0
        func dPlace(_ bytes: Int) -> Int { let o = dOff; dOff += aligned(bytes); return o }
        var doubleOffsets: [String: Int] = [:]
        doubleOffsets["nhImg"] = dPlace(lImg * d * 2)
        doubleOffsets["neTxt"] = dPlace(lTxt * d * 2)
        doubleOffsets["projImg"] = dPlace(lImg * d * 2)
        doubleOffsets["projTxt"] = dPlace(lTxt * d * 2)
        doubleOffsets["jointQ"] = dPlace(J * d * 2)
        doubleOffsets["jointK"] = dPlace(J * d * 2)
        doubleOffsets["jointV"] = dPlace(J * d * 2)
        doubleOffsets["jointO"] = dPlace(J * d * 2)
        doubleOffsets["outImg"] = dPlace(lImg * d * 2)
        doubleOffsets["outTxt"] = dPlace(lTxt * d * 2)
        doubleOffsets["y1Img"] = dPlace(lImg * d * 4)
        doubleOffsets["y1Txt"] = dPlace(lTxt * d * 4)
        let imageChunkRows = (lImg + 1) / 2
        doubleOffsets["ffWideImg"] = dPlace(imageChunkRows * Self.ffWide * 2)
        doubleOffsets["ffWideTxt"] = dPlace(lTxt * Self.ffWide * 2)
        doubleOffsets["swImg"] = dPlace(imageChunkRows * Self.ffInner * 2)
        doubleOffsets["swTxt"] = dPlace(lTxt * Self.ffInner * 2)
        doubleOffsets["ffOutImg"] = dPlace(lImg * d * 2)
        doubleOffsets["ffOutTxt"] = dPlace(lTxt * d * 2)
        let doubleZone = dOff

        var sOff = 0
        func sPlace(_ bytes: Int) -> Int { let o = sOff; sOff += aligned(bytes); return o }
        var singleOffsets: [String: Int] = [:]
        singleOffsets["nhJoint"] = sPlace(J * d * 2)
        let jointChunkRows = (J + 1) / 2
        singleOffsets["projJoint"] = sPlace(jointChunkRows * Self.projWidth * 2)
        singleOffsets["sQ"] = sPlace(J * d * 2)
        singleOffsets["sK"] = sPlace(J * d * 2)
        singleOffsets["sV"] = sPlace(J * d * 2)
        singleOffsets["sConcat"] = sPlace(J * Self.concatWidth * 2)
        singleOffsets["sOut"] = sPlace(J * d * 2)
        let singleZone = sOff

        let cZone = max(doubleZone, singleZone)
        let deviceAlignment = device.heapBufferSizeAndAlign(
            length: 1, options: [.storageModeShared]).align
        let placementPlan = try DirectLegacyPlanFactory.make(
            imageTokens: lImg, alignment: deviceAlignment)
        guard placementPlan.placement.peakBytes == pZone + cZone else {
            throw DirectQmmSpike.SpikeError.metal(
                "scratch plan/runtime mismatch: plan \(placementPlan.placement.peakBytes), runtime \(pZone + cZone)")
        }
        scratchPlanDigest = placementPlan.digest
        let heapDesc = MTLHeapDescriptor()
        heapDesc.type = .placement
        heapDesc.storageMode = .shared
        heapDesc.hazardTrackingMode = .untracked
        heapDesc.size = pZone + cZone
        guard let heap = device.makeHeap(descriptor: heapDesc) else {
            throw DirectQmmSpike.SpikeError.metal("heap \(pZone + cZone) bytes")
        }
        scratchHeap = heap
        heapComputeBase = pZone
        heapComputeSize = cZone
        memoryLedger.recordScratch(bytes: pZone + cZone)

        func hb(_ bytes: Int, _ offset: Int, _ label: String) throws -> MTLBuffer {
            guard let b = heap.makeBuffer(
                length: bytes, options: [.storageModeShared], offset: offset)
            else { throw DirectQmmSpike.SpikeError.metal("place \(label)") }
            b.label = label
            return b
        }
        // Persistent zone, ordered so the dead-at-blit halves are CONTIGUOUS:
        // [hA][eA][hB][eB]. After 5 doubles cur = hB/eB, so hA+eA (offset 0,
        // exactly jxBytes) back jointXA; hB+eB back jointXB once the blit is
        // done. The single-phase ping-pong walks those two aliases only.
        hA = try hb(lImg * d * 4, 0, "hA")
        eA = try hb(lTxt * d * 4, hBytes, "eA")
        hB = try hb(lImg * d * 4, hBytes + eBytes, "hB")
        eB = try hb(lTxt * d * 4, hBytes + eBytes + hBytes, "eB")
        jointXA = try hb(J * d * 4, 0, "jointXA")                    // over hA+eA
        jointXB = try hb(J * d * 4, hBytes + eBytes, "jointXB")      // over hB+eB
        // Compute zone (base pZone).
        func dz(_ name: String, _ bytes: Int) throws -> MTLBuffer {
            try hb(bytes, pZone + doubleOffsets[name]!, name)
        }
        func sz(_ name: String, _ bytes: Int) throws -> MTLBuffer {
            try hb(bytes, pZone + singleOffsets[name]!, name)
        }
        nhImg = try dz("nhImg", lImg * d * 2)
        neTxt = try dz("neTxt", lTxt * d * 2)
        projImg = try dz("projImg", lImg * d * 2)
        projTxt = try dz("projTxt", lTxt * d * 2)
        jointQ = try dz("jointQ", J * d * 2)
        jointK = try dz("jointK", J * d * 2)
        jointV = try dz("jointV", J * d * 2)
        jointO = try dz("jointO", J * d * 2)
        outImg = try dz("outImg", lImg * d * 2)
        outTxt = try dz("outTxt", lTxt * d * 2)
        y1Img = try dz("y1Img", lImg * d * 4)
        y1Txt = try dz("y1Txt", lTxt * d * 4)
        ffWideImg = try dz("ffWideImg", imageChunkRows * Self.ffWide * 2)
        ffWideTxt = try dz("ffWideTxt", lTxt * Self.ffWide * 2)
        swImg = try dz("swImg", imageChunkRows * Self.ffInner * 2)
        swTxt = try dz("swTxt", lTxt * Self.ffInner * 2)
        ffOutImg = try dz("ffOutImg", lImg * d * 2)
        ffOutTxt = try dz("ffOutTxt", lTxt * d * 2)
        nhJoint = try sz("nhJoint", J * d * 2)
        projJoint = try sz("projJoint", (J + 1) / 2 * Self.projWidth * 2)
        sQ = try sz("sQ", J * d * 2)
        sK = try sz("sK", J * d * 2)
        sV = try sz("sV", J * d * 2)
        sConcat = try sz("sConcat", J * Self.concatWidth * 2)
        sOut = try sz("sOut", J * d * 2)
    }

    // MARK: - Weight loading

    func quant(_ sub: [String: MLXArray], _ name: String, n: Int, k: Int) throws -> Q {
        guard let w = sub["\(name).weight"], let s = sub["\(name).scales"],
            let b = sub["\(name).biases"]
        else { throw DirectQmmSpike.SpikeError.missingTensor(name) }
        try DirectTensorValidation.requireQuantized(
            weight: w, scales: s, biases: b, n: n, k: k, name: name)
        return Q(
            w: try upload(w, "\(name).w", retention: .persistentWeight),
            s: try upload(s, "\(name).s", retention: .persistentWeight),
            b: try upload(b, "\(name).b", retention: .persistentWeight),
            n: n, k: k)
    }
    func normW(_ sub: [String: MLXArray], _ name: String, count: Int) throws -> MTLBuffer {
        guard let w = sub["\(name).weight"] else {
            throw DirectQmmSpike.SpikeError.missingTensor(name)
        }
        guard w.shape.reduce(1, *) == count else {
            throw DirectQmmSpike.SpikeError.invalidTensor(
                "\(name).weight expected \(count) values, got shape \(w.shape)")
        }
        let f = w.asType(.float32)
        eval(f)
        return try upload(f, name, retention: .persistentWeight)
    }

    public func loadBlocks(arrays: [String: MLXArray], nDouble: Int, nSingle: Int) throws {
        let d = Self.dim
        func sub(_ prefix: String) -> [String: MLXArray] {
            var out: [String: MLXArray] = [:]
            for (k, v) in arrays where k.hasPrefix(prefix) {
                let key = String(k.dropFirst(prefix.count))
                if key.hasSuffix(".scales") || key.hasSuffix(".biases") {
                    out[key] = v.asType(.float16)
                } else {
                    out[key] = v
                }
            }
            return out
        }
        for i in 0 ..< nDouble {
            let s = sub("transformer_blocks.\(i).")
            doubles.append(DoubleWeights(
                toQ: try quant(s, "attn.to_q", n: d, k: d),
                toK: try quant(s, "attn.to_k", n: d, k: d),
                toV: try quant(s, "attn.to_v", n: d, k: d),
                addQ: try quant(s, "attn.add_q_proj", n: d, k: d),
                addK: try quant(s, "attn.add_k_proj", n: d, k: d),
                addV: try quant(s, "attn.add_v_proj", n: d, k: d),
                toOut: try quant(s, "attn.to_out", n: d, k: d),
                toAddOut: try quant(s, "attn.to_add_out", n: d, k: d),
                ffIn: try quant(s, "ff.linear_in", n: Self.ffWide, k: d),
                ffOut: try quant(s, "ff.linear_out", n: d, k: Self.ffInner),
                ffcIn: try quant(s, "ff_context.linear_in", n: Self.ffWide, k: d),
                ffcOut: try quant(s, "ff_context.linear_out", n: d, k: Self.ffInner),
                normQ: try normW(s, "attn.norm_q", count: Self.headDim),
                normK: try normW(s, "attn.norm_k", count: Self.headDim),
                normAQ: try normW(s, "attn.norm_added_q", count: Self.headDim),
                normAK: try normW(s, "attn.norm_added_k", count: Self.headDim)))
        }
        for i in 0 ..< nSingle {
            let s = sub("single_transformer_blocks.\(i).")
            singles.append(SingleWeights(
                proj: try quant(s, "attn.to_qkv_mlp_proj", n: Self.projWidth, k: d),
                toOut: try quant(s, "attn.to_out", n: d, k: Self.concatWidth),
                normQ: try normW(s, "attn.norm_q", count: Self.headDim),
                normK: try normW(s, "attn.norm_k", count: Self.headDim)))
        }
    }

    public func setConditioning(
        imgMsa: (MLXArray, MLXArray, MLXArray), imgMlp: (MLXArray, MLXArray, MLXArray),
        txtMsa: (MLXArray, MLXArray, MLXArray), txtMlp: (MLXArray, MLXArray, MLXArray),
        single: (MLXArray, MLXArray, MLXArray),
        cos: MLXArray, sin: MLXArray
    ) throws {
        func put(_ t: (MLXArray, MLXArray, MLXArray), _ tag: String) throws {
            modBufs["\(tag).shift"] = BufferBinding(
                buffer: try upload(t.0, "\(tag).shift", retention: .conditioning), offset: 0)
            modBufs["\(tag).scale"] = BufferBinding(
                buffer: try upload(t.1, "\(tag).scale", retention: .conditioning), offset: 0)
            modBufs["\(tag).gate"] = BufferBinding(
                buffer: try upload(t.2, "\(tag).gate", retention: .conditioning), offset: 0)
        }
        try put(imgMsa, "img.msa")
        try put(imgMlp, "img.mlp")
        try put(txtMsa, "txt.msa")
        try put(txtMlp, "txt.mlp")
        try put(single, "single")
        cosBuf = try upload(cos, "cos", retention: .conditioning)
        sinBuf = try upload(sin, "sin", retention: .conditioning)
        memoryLedger.replaceConditioning(
            bytes: Set(modBufs.values.map { ObjectIdentifier($0.buffer) })
                .compactMap { id in
                    modBufs.values.first { ObjectIdentifier($0.buffer) == id }?.buffer.length
                }.reduce(cosBuf.length + sinBuf.length, +))
    }

    // MARK: - Encode primitives

    private func lnMod(_ enc: MTLComputeCommandEncoder, _ x: MTLBuffer, xOff: Int,
                       _ tag: String, _ y: MTLBuffer, rows: Int) {
        enc.setComputePipelineState(lnModPSO)
        enc.setBuffer(x, offset: xOff, index: 0)
        let scale = modBufs["\(tag).scale"]!
        let shift = modBufs["\(tag).shift"]!
        enc.setBuffer(scale.buffer, offset: scale.offset, index: 1)
        enc.setBuffer(shift.buffer, offset: shift.offset, index: 2)
        enc.setBuffer(y, offset: 0, index: 3)
        var d32 = Int32(Self.dim)
        enc.setBytes(&d32, length: 4, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func qmm(_ enc: MTLComputeCommandEncoder, _ q: Q,
                     x: MTLBuffer, xOff: Int, y: MTLBuffer, yOff: Int, m: Int) {
        // NAX path: same ABI, 64-wide tiles; requires K % 64 == 0 (all DiT
        // projections satisfy it — the guard is belt-and-braces).
        if let naxT = qmmNAXTruePSO, let naxF = qmmNAXFalsePSO, q.k % 64 == 0 {
            enc.setComputePipelineState(q.n % 64 == 0 ? naxT : naxF)
            enc.setBuffer(q.w, offset: 0, index: 0)
            enc.setBuffer(q.s, offset: 0, index: 1)
            enc.setBuffer(q.b, offset: 0, index: 2)
            enc.setBuffer(x, offset: xOff, index: 3)
            enc.setBuffer(y, offset: yOff, index: 4)
            var k32 = Int32(q.k), n32 = Int32(q.n), m32 = Int32(m)
            enc.setBytes(&k32, length: 4, index: 5)
            enc.setBytes(&n32, length: 4, index: 6)
            enc.setBytes(&m32, length: 4, index: 7)
            let (grid, group) = DirectNAX.qmmGrid(m: m, n: q.n)
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: group)
            return
        }
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

    private func rms(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, pitch: Int, off: Int,
                     w: MTLBuffer, y: MTLBuffer, yOff: Int, rows: Int) {
        enc.setComputePipelineState(rmsPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(w, offset: 0, index: 1)
        enc.setBuffer(y, offset: yOff, index: 2)
        var hd = Int32(Self.headDim), p = Int32(pitch), o = Int32(off)
        var hpr = Int32(Self.heads)
        var eps = Float(1e-5)
        enc.setBytes(&hd, length: 4, index: 3)
        enc.setBytes(&p, length: 4, index: 4)
        enc.setBytes(&o, length: 4, index: 5)
        enc.setBytes(&hpr, length: 4, index: 6)
        enc.setBytes(&eps, length: 4, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: rows * Self.heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    private func scaleCast(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, pitch: Int, off: Int,
                           width: Int, y: MTLBuffer, yOff: Int, rows: Int, scale: Float) {
        enc.setComputePipelineState(scaleCastPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(y, offset: yOff, index: 1)
        var p = Int32(pitch), o = Int32(off), w32 = Int32(width)
        var s = scale
        enc.setBytes(&p, length: 4, index: 2)
        enc.setBytes(&o, length: 4, index: 3)
        enc.setBytes(&w32, length: 4, index: 4)
        enc.setBytes(&s, length: 4, index: 5)
        enc.dispatchThreads(
            MTLSize(width: width, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func rope(_ enc: MTLComputeCommandEncoder, _ src: MTLBuffer, _ dst: MTLBuffer, rows: Int) {
        enc.setComputePipelineState(ropePSO)
        enc.setBuffer(src, offset: 0, index: 0)
        enc.setBuffer(dst, offset: 0, index: 1)
        enc.setBuffer(cosBuf, offset: 0, index: 2)
        enc.setBuffer(sinBuf, offset: 0, index: 3)
        var h32 = Int32(Self.heads), hd = Int32(Self.headDim)
        enc.setBytes(&h32, length: 4, index: 4)
        enc.setBytes(&hd, length: 4, index: 5)
        enc.dispatchThreads(
            MTLSize(width: Self.headDim / 2, height: Self.heads, depth: rows),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    private func attention(_ enc: MTLComputeCommandEncoder, q: MTLBuffer, k: MTLBuffer,
                           v: MTLBuffer, o: MTLBuffer, oPitch: Int) {
        enc.setComputePipelineState(attnPSO)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(k, offset: 0, index: 1)
        enc.setBuffer(v, offset: 0, index: 2)
        enc.setBuffer(o, offset: 0, index: 3)
        var params = Data(capacity: 152)
        func i32p(_ v: Int) { var x = Int32(v); withUnsafeBytes(of: &x) { params.append(contentsOf: $0) } }
        func f32p(_ v: Float) { var x = v; withUnsafeBytes(of: &x) { params.append(contentsOf: $0) } }
        func i64x3p(_ a: Int, _ b: Int, _ c: Int) {
            for v in [Int64(a), Int64(b), Int64(c)] {
                var x = v; withUnsafeBytes(of: &x) { params.append(contentsOf: $0) }
            }
        }
        let bq = attentionProfile.blockQueries
        let bk = attentionProfile.blockKeys
        i32p(1); i32p(Self.heads); i32p(Self.headDim); i32p(J); i32p(J)
        i32p(1)
        f32p(1.0 / Float(Double(Self.headDim).squareRoot()))
        i32p((J + bq - 1) / bq); i32p((J + bk - 1) / bk)
        i32p(J / bq); i32p(J / bk)
        i32p(J - (J / bq) * bq); i32p(J - (J / bk) * bk)
        i32p(0)
        i64x3p(J * Self.dim, Self.headDim, Self.dim)
        i64x3p(J * Self.dim, Self.headDim, Self.dim)
        i64x3p(J * Self.dim, Self.headDim, Self.dim)
        i64x3p(J * oPitch, Self.headDim, oPitch)
        params.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: params.count, index: 4) }
        enc.dispatchThreadgroups(
            MTLSize(width: (J + bq - 1) / bq, height: Self.heads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
    }

    private func scaleInPlace(_ enc: MTLComputeCommandEncoder, _ x: MTLBuffer,
                              pitch: Int, off: Int, width: Int, rows: Int, scale: Float) {
        enc.setComputePipelineState(scaleInPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        var p = Int32(pitch), o = Int32(off), w32 = Int32(width)
        var s = scale
        enc.setBytes(&p, length: 4, index: 1)
        enc.setBytes(&o, length: 4, index: 2)
        enc.setBytes(&w32, length: 4, index: 3)
        enc.setBytes(&s, length: 4, index: 4)
        enc.dispatchThreads(
            MTLSize(width: width, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func gateAdd(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, xOff: Int, _ tag: String,
                         v: MTLBuffer, y: MTLBuffer, yOff: Int, rows: Int) {
        enc.setComputePipelineState(gateAddPSO)
        enc.setBuffer(x, offset: xOff, index: 0)
        let gate = modBufs["\(tag).gate"]!
        enc.setBuffer(gate.buffer, offset: gate.offset, index: 1)
        enc.setBuffer(v, offset: 0, index: 2)
        enc.setBuffer(y, offset: yOff, index: 3)
        var d32 = Int32(Self.dim)
        var sixteen = Float(16)
        enc.setBytes(&d32, length: 4, index: 4)
        enc.setBytes(&sixteen, length: 4, index: 5)
        enc.dispatchThreads(
            MTLSize(width: Self.dim, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    private func swiglu(_ enc: MTLComputeCommandEncoder, _ x: MTLBuffer, xByteOff: Int = 0,
                        _ y: MTLBuffer, yByteOff: Int = 0,
                        pitch: Int, gOff: Int, uOff: Int, width: Int,
                        outPitch: Int, outOff: Int, rows: Int) {
        enc.setComputePipelineState(swigluPSO)
        enc.setBuffer(x, offset: xByteOff, index: 0)
        enc.setBuffer(y, offset: yByteOff, index: 1)
        var p = Int32(pitch), g = Int32(gOff), u = Int32(uOff)
        var w32 = Int32(width), op = Int32(outPitch), oo = Int32(outOff)
        var inS = Float(16), outS = Float(1.0 / 16.0)
        enc.setBytes(&p, length: 4, index: 2)
        enc.setBytes(&g, length: 4, index: 3)
        enc.setBytes(&u, length: 4, index: 4)
        enc.setBytes(&w32, length: 4, index: 5)
        enc.setBytes(&op, length: 4, index: 6)
        enc.setBytes(&oo, length: 4, index: 7)
        enc.setBytes(&inS, length: 4, index: 8)
        enc.setBytes(&outS, length: 4, index: 9)
        enc.dispatchThreads(
            MTLSize(width: width, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    // MARK: - Blocks

    private func encodeDouble(_ enc: MTLComputeCommandEncoder, _ w: DoubleWeights,
                              hIn: MTLBuffer, eIn: MTLBuffer, hOut: MTLBuffer, eOut: MTLBuffer) {
        let d = Self.dim
        let txtByteOff = lTxt * d * 2
        lnMod(enc, hIn, xOff: 0, "img.msa", nhImg, rows: lImg)
        lnMod(enc, eIn, xOff: 0, "txt.msa", neTxt, rows: lTxt)
        qmm(enc, w.addQ, x: neTxt, xOff: 0, y: projTxt, yOff: 0, m: lTxt)
        rms(enc, x: projTxt, pitch: d, off: 0, w: w.normAQ, y: jointQ, yOff: 0, rows: lTxt)
        qmm(enc, w.addK, x: neTxt, xOff: 0, y: projTxt, yOff: 0, m: lTxt)
        rms(enc, x: projTxt, pitch: d, off: 0, w: w.normAK, y: jointK, yOff: 0, rows: lTxt)
        qmm(enc, w.addV, x: neTxt, xOff: 0, y: projTxt, yOff: 0, m: lTxt)
        scaleCast(enc, x: projTxt, pitch: d, off: 0, width: d, y: jointV, yOff: 0, rows: lTxt, scale: 16)
        qmm(enc, w.toQ, x: nhImg, xOff: 0, y: projImg, yOff: 0, m: lImg)
        rms(enc, x: projImg, pitch: d, off: 0, w: w.normQ, y: jointQ, yOff: txtByteOff, rows: lImg)
        qmm(enc, w.toK, x: nhImg, xOff: 0, y: projImg, yOff: 0, m: lImg)
        rms(enc, x: projImg, pitch: d, off: 0, w: w.normK, y: jointK, yOff: txtByteOff, rows: lImg)
        qmm(enc, w.toV, x: nhImg, xOff: 0, y: projImg, yOff: 0, m: lImg)
        scaleCast(enc, x: projImg, pitch: d, off: 0, width: d, y: jointV, yOff: txtByteOff, rows: lImg, scale: 16)
        rope(enc, jointQ, jointQ, rows: J)  // in-place: each thread owns its pair
        rope(enc, jointK, jointK, rows: J)
        attention(enc, q: jointQ, k: jointK, v: jointV, o: jointO, oPitch: d)
        scaleInPlace(enc, jointO, pitch: d, off: 0, width: d, rows: J, scale: 1.0 / 16.0)
        qmm(enc, w.toAddOut, x: jointO, xOff: 0, y: outTxt, yOff: 0, m: lTxt)
        qmm(enc, w.toOut, x: jointO, xOff: txtByteOff, y: outImg, yOff: 0, m: lImg)
        gateAdd(enc, x: hIn, xOff: 0, "img.msa", v: outImg, y: y1Img, yOff: 0, rows: lImg)
        gateAdd(enc, x: eIn, xOff: 0, "txt.msa", v: outTxt, y: y1Txt, yOff: 0, rows: lTxt)
        lnMod(enc, y1Img, xOff: 0, "img.mlp", nhImg, rows: lImg)
        for range in Self.chunkRanges(totalRows: lImg) {
            let rowOff = range.lowerBound
            let rows = range.count
            qmm(enc, w.ffIn, x: nhImg, xOff: rowOff * Self.dim * 2, y: ffWideImg, yOff: 0, m: rows)
            swiglu(enc, ffWideImg, swImg, pitch: Self.ffWide, gOff: 0, uOff: Self.ffInner,
                   width: Self.ffInner, outPitch: Self.ffInner, outOff: 0, rows: rows)
            qmm(enc, w.ffOut, x: swImg, xOff: 0, y: ffOutImg, yOff: rowOff * Self.dim * 2, m: rows)
        }
        gateAdd(enc, x: y1Img, xOff: 0, "img.mlp", v: ffOutImg, y: hOut, yOff: 0, rows: lImg)
        lnMod(enc, y1Txt, xOff: 0, "txt.mlp", neTxt, rows: lTxt)
        qmm(enc, w.ffcIn, x: neTxt, xOff: 0, y: ffWideTxt, yOff: 0, m: lTxt)
        swiglu(enc, ffWideTxt, swTxt, pitch: Self.ffWide, gOff: 0, uOff: Self.ffInner,
               width: Self.ffInner, outPitch: Self.ffInner, outOff: 0, rows: lTxt)
        qmm(enc, w.ffcOut, x: swTxt, xOff: 0, y: ffOutTxt, yOff: 0, m: lTxt)
        gateAdd(enc, x: y1Txt, xOff: 0, "txt.mlp", v: ffOutTxt, y: eOut, yOff: 0, rows: lTxt)
    }

    private func encodeSingle(_ enc: MTLComputeCommandEncoder, _ w: SingleWeights,
                              xIn: MTLBuffer, xOut: MTLBuffer) {
        let d = Self.dim
        lnMod(enc, xIn, xOff: 0, "single", nhJoint, rows: J)
        // The fused 27648-wide proj runs in row-halves so its scratch is half
        // the sequence; every pitched consumer indexes chunk-locally with a
        // global byte offset on its output.
        for range in Self.chunkRanges(totalRows: J) {
            let rowOff = range.lowerBound
            let rows = range.count
            qmm(enc, w.proj, x: nhJoint, xOff: rowOff * d * 2, y: projJoint, yOff: 0, m: rows)
            rms(enc, x: projJoint, pitch: Self.projWidth, off: 0, w: w.normQ,
                y: sQ, yOff: rowOff * d * 2, rows: rows)
            rms(enc, x: projJoint, pitch: Self.projWidth, off: d, w: w.normK,
                y: sK, yOff: rowOff * d * 2, rows: rows)
            scaleCast(enc, x: projJoint, pitch: Self.projWidth, off: 2 * d, width: d,
                      y: sV, yOff: rowOff * d * 2, rows: rows, scale: 16)
            swiglu(enc, projJoint, sConcat, yByteOff: rowOff * Self.concatWidth * 2,
                   pitch: Self.projWidth, gOff: Self.qkvWidth,
                   uOff: Self.qkvWidth + Self.ffWide / 2, width: Self.ffWide / 2,
                   outPitch: Self.concatWidth, outOff: d, rows: rows)
        }
        rope(enc, sQ, sQ, rows: J)
        rope(enc, sK, sK, rows: J)
        attention(enc, q: sQ, k: sK, v: sV, o: sConcat, oPitch: Self.concatWidth)
        scaleInPlace(enc, sConcat, pitch: Self.concatWidth, off: 0, width: d, rows: J, scale: 1.0 / 16.0)
        qmm(enc, w.toOut, x: sConcat, xOff: 0, y: sOut, yOff: 0, m: J)
        gateAdd(enc, x: xIn, xOff: 0, "single", v: sOut, y: xOut, yOff: 0, rows: J)
    }

    // MARK: - Full step

    /// Encode the whole 25-block step. `hIn`/`eIn` are f32 device buffers; the
    /// result lands in the returned joint f32 buffer (text rows first).
    public func encodeStep(hIn: MTLBuffer, eIn: MTLBuffer) throws -> MTLBuffer {
        guard let cb = queue.makeCommandBuffer() else {
            throw DirectQmmSpike.SpikeError.metal("cb")
        }
        let out = try encodeBlocks(cb: cb, hIn: hIn, eIn: eIn)
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw DirectQmmSpike.SpikeError.metal("exec: \(e)") }
        return out
    }

    /// Encode the 25 blocks into an existing command buffer (no commit).
    func encodeBlocks(cb: MTLCommandBuffer, hIn: MTLBuffer, eIn: MTLBuffer) throws -> MTLBuffer {
        // Copy inputs into the internal ping-pong so callers' buffers are
        // never written (the step is re-runnable with pristine inputs).
        guard let inBlit = cb.makeBlitCommandEncoder() else {
            throw DirectQmmSpike.SpikeError.metal("blit-in")
        }
        inBlit.waitForFence(fence)
        inBlit.copy(from: hIn, sourceOffset: 0, to: hA, destinationOffset: 0,
                    size: lImg * Self.dim * 4)
        inBlit.copy(from: eIn, sourceOffset: 0, to: eA, destinationOffset: 0,
                    size: lTxt * Self.dim * 4)
        inBlit.updateFence(fence)
        inBlit.endEncoding()
        var curH = hA!, curE = eA!
        var altH = hB!, altE = eB!
        for w in doubles {
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw DirectQmmSpike.SpikeError.metal("enc")
            }
            enc.waitForFence(fence)
            encodeDouble(enc, w, hIn: curH, eIn: curE, hOut: altH, eOut: altE)
            enc.updateFence(fence)
            enc.endEncoding()
            swap(&curH, &altH)
            swap(&curE, &altE)
        }
        // Concat: text rows then image rows into jointXA (f32 blits).
        guard let blit = cb.makeBlitCommandEncoder() else {
            throw DirectQmmSpike.SpikeError.metal("blit")
        }
        blit.waitForFence(fence)
        blit.copy(from: curE, sourceOffset: 0, to: jointXA, destinationOffset: 0,
                  size: lTxt * Self.dim * 4)
        blit.copy(from: curH, sourceOffset: 0, to: jointXA, destinationOffset: lTxt * Self.dim * 4,
                  size: lImg * Self.dim * 4)
        blit.updateFence(fence)
        blit.endEncoding()
        var curX = jointXA!, altX = jointXB!
        for w in singles {
            guard let enc = cb.makeComputeCommandEncoder() else {
                throw DirectQmmSpike.SpikeError.metal("enc")
            }
            enc.waitForFence(fence)
            encodeSingle(enc, w, xIn: curX, xOut: altX)
            enc.updateFence(fence)
            enc.endEncoding()
            swap(&curX, &altX)
        }
        return curX
    }
}

/// F3 verification: the full step vs an oracle loop of product blocks.
public enum DirectDiTStepSpike {
    public static func run(transformerDirectory: URL, metallibURL: URL) throws -> String {
        let lImg = 1024, lTxt = 512, J = 1536, dim = 3072

        var arrays: [String: MLXArray] = [:]
        for shard in ["0.safetensors", "1.safetensors"] {
            let url = transformerDirectory.appendingPathComponent(shard)
            if FileManager.default.fileExists(atPath: url.path) {
                for (k, v) in try MLX.loadArrays(url: url) { arrays[k] = v }
            }
        }

        // Oracle blocks
        func subdict(_ prefix: String) -> [String: MLXArray] {
            var out: [String: MLXArray] = [:]
            for (k, v) in arrays where k.hasPrefix(prefix) {
                let key = String(k.dropFirst(prefix.count))
                out[key] = key.hasSuffix(".scales") || key.hasSuffix(".biases")
                    ? v.asType(.float16) : v
            }
            return out
        }
        var oDoubles: [Flux2TransformerBlock] = []
        for i in 0 ..< 5 {
            let b = Flux2TransformerBlock(dim: dim, numAttentionHeads: 24, attentionHeadDim: 128)
            quantize(model: b, groupSize: 64, bits: 4) { _, _ in true }
            try b.update(parameters: ModuleParameters.unflattened(subdict("transformer_blocks.\(i).")), verify: [.all])
            oDoubles.append(b)
        }
        var oSingles: [Flux2SingleTransformerBlock] = []
        for i in 0 ..< 20 {
            let b = Flux2SingleTransformerBlock(dim: dim, numAttentionHeads: 24, attentionHeadDim: 128)
            quantize(model: b, groupSize: 64, bits: 4) { _, _ in true }
            try b.update(parameters: ModuleParameters.unflattened(subdict("single_transformer_blocks.\(i).")), verify: [.all])
            oSingles.append(b)
        }

        MLXRandom.seed(17)
        let h0 = (MLXRandom.normal([1, lImg, dim]) * 0.5).asType(.float32)
        let e0 = (MLXRandom.normal([1, lTxt, dim]) * 0.5).asType(.float32)
        func triple() -> (MLXArray, MLXArray, MLXArray) {
            ((MLXRandom.normal([1, 1, dim]) * 0.1).asType(.float32),
             (MLXRandom.normal([1, 1, dim]) * 0.1).asType(.float32),
             (MLXRandom.normal([1, 1, dim]) * 0.1).asType(.float32))
        }
        let mImg = [triple(), triple()], mTxt = [triple(), triple()]
        let mSingle = triple()
        let theta = MLXRandom.uniform(low: 0.0, high: 6.2831853, [J, 64])
        let cosT = cos(theta).asType(.float32)
        let sinT = sin(theta).asType(.float32)
        eval(h0, e0, cosT, sinT)

        func oracleStep() -> MLXArray {
            var h = h0, e = e0
            for b in oDoubles {
                let out = b(hiddenStates: h, encoderHiddenStates: e,
                            tembModParamsImg: mImg, tembModParamsTxt: mTxt,
                            imageRotaryEmb: (cosT, sinT))
                e = out.encoder
                h = out.hidden
                eval(h, e)
            }
            var x = concatenated([e, h], axis: 1)
            eval(x)
            for b in oSingles {
                x = b(x, tembModParams: mSingle, imageRotaryEmb: (cosT, sinT))
                eval(x)
            }
            return x
        }
        let oOut = oracleStep()
        _ = oracleStep()
        var oracleMS = 0.0
        do {
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< 3 { _ = oracleStep() }
            oracleMS = (CFAbsoluteTimeGetCurrent() - t0) * 1000 / 3
        }

        // Direct engine
        let engine = try DirectDiTStep(lImg: lImg, metallibURL: metallibURL)
        let tLoad = CFAbsoluteTimeGetCurrent()
        try engine.loadBlocks(arrays: arrays, nDouble: 5, nSingle: 20)
        let loadMS = (CFAbsoluteTimeGetCurrent() - tLoad) * 1000
        try engine.setConditioning(
            imgMsa: mImg[0], imgMlp: mImg[1], txtMsa: mTxt[0], txtMlp: mTxt[1],
            single: mSingle, cos: cosT, sin: sinT)
        let hBuf = try engine.upload(h0, "h0")
        let eBuf = try engine.upload(e0, "e0")

        var final = try engine.encodeStep(hIn: hBuf, eIn: eBuf)
        var directMS = 0.0
        do {
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< 3 { final = try engine.encodeStep(hIn: hBuf, eIn: eBuf) }
            directMS = (CFAbsoluteTimeGetCurrent() - t0) * 1000 / 3
        }

        let ptr = final.contents().bindMemory(to: Float.self, capacity: J * dim)
        let ref = oOut.reshaped([J, dim]).asType(.float32).asArray(Float.self)
        var dot = 0.0, na = 0.0, nb = 0.0, maxDiff = 0.0
        for i in 0 ..< J * dim {
            let a = Double(ptr[i]), b = Double(ref[i])
            dot += a * b; na += a * a; nb += b * b
            maxDiff = max(maxDiff, abs(a - b))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
        return """
        direct-dit-step (F3) — 5 double + 20 single blocks, one command buffer, ~355 dispatches
          cosine_vs_product_step: \(String(format: "%.7f", cosine))
          max_abs_diff:           \(String(format: "%.4f", maxDiff))
          weight_upload:          \(String(format: "%.0f", loadMS)) ms (one-time)
          oracle_per_step:        \(String(format: "%.1f", oracleMS)) ms (MLX product blocks, warm)
          direct_per_step:        \(String(format: "%.1f", directMS)) ms (single CB, blocking wait)
          verdict:                \(cosine >= 0.9999 ? "PASS" : "investigate")
        """
    }
}

// MARK: - F4: the complete denoise stage

extension DirectDiTStep {

    /// Head + embedder state (loaded once).
    public struct Head {
        let xEmb: Q
        let projOut: Q
        let castPrePSO: MTLComputePipelineState
        let castPostPSO: MTLComputePipelineState
        let onesGate: MTLBuffer
        let lat16: MTLBuffer      // [lImg, 128] f16 (embedder input)
        let hEmb16: MTLBuffer     // [lImg, 3072] f16 (embedder output)
        let h0: MTLBuffer         // [lImg, 3072] f32 (step input)
        let nhHead: MTLBuffer     // [lImg, 3072] f16 (norm_out output, ÷16)
        let pred16: MTLBuffer     // [lImg, 128] f16 (proj_out output)
        public let latA: MTLBuffer  // [lImg, 128] f32 ping-pong
        public let latB: MTLBuffer
    }

    /// Place a head/embedder buffer inside the heap's compute zone: the
    /// pre-phase runs before any block scratch is live and the head phase
    /// after all of it is dead, so these overlay the block scratch for free
    /// (fence chain orders the phases).
    private func headPlace(_ bytes: Int, _ label: String) throws -> MTLBuffer {
        let a = device.heapBufferSizeAndAlign(length: bytes, options: [.storageModeShared])
        let aligned = (headCursor + a.align - 1) / a.align * a.align
        precondition(aligned + bytes <= heapComputeSize, "head overlay exceeds compute zone")
        guard let b = scratchHeap.makeBuffer(
            length: bytes, options: [.storageModeShared], offset: heapComputeBase + aligned)
        else { throw DirectQmmSpike.SpikeError.metal("headPlace \(label)") }
        b.label = label
        headCursor = aligned + bytes
        return b
    }

    public func loadHead(arrays: [String: MLXArray]) throws -> Head {
        func conv(_ name: String) throws -> Q {
            guard let w = arrays["\(name).weight"], let sc = arrays["\(name).scales"],
                let bi = arrays["\(name).biases"]
            else { throw DirectQmmSpike.SpikeError.missingTensor(name) }
            let expected: (n: Int, k: Int)
            switch name {
            case "x_embedder": expected = (Self.dim, 128)
            case "proj_out": expected = (128, Self.dim)
            default:
                throw DirectQmmSpike.SpikeError.invalidTensor("unsupported head tensor \(name)")
            }
            try DirectTensorValidation.requireQuantized(
                weight: w, scales: sc, biases: bi,
                n: expected.n, k: expected.k, name: name)
            let s16 = sc.asType(.float16)
            let b16 = bi.asType(.float16)
            eval(s16, b16)
            return Q(
                w: try upload(w, "\(name).w", retention: .persistentWeight),
                s: try upload(s16, "\(name).s", retention: .persistentWeight),
                b: try upload(b16, "\(name).b", retention: .persistentWeight),
                n: expected.n, k: expected.k)
        }
        func pso(_ n: String) throws -> MTLComputePipelineState {
            guard let fn = directLib.makeFunction(name: n) else {
                throw DirectQmmSpike.SpikeError.metal("glue \(n)")
            }
            return try device.makeComputePipelineState(function: fn)
        }
        let ones = [Float](repeating: 1, count: Self.dim)
        let onesBuf = try ones.withUnsafeBufferPointer { p -> MTLBuffer in
            guard let b = device.makeBuffer(bytes: p.baseAddress!, length: Self.dim * 4) else {
                throw DirectQmmSpike.SpikeError.metal("ones")
            }
            return b
        }
        memoryLedger.recordPersistentUpload(bytes: Self.dim * 4)
        return Head(
            xEmb: try conv("x_embedder"),
            projOut: try conv("proj_out"),
            castPrePSO: try pso("dd_cast_prescale"),
            castPostPSO: try pso("dd_cast_postscale"),
            onesGate: onesBuf,
            lat16: try headPlace(lImg * 128 * 2, "lat16"),
            hEmb16: try headPlace(lImg * Self.dim * 2, "hEmb16"),
            h0: try headPlace(lImg * Self.dim * 4, "h0"),
            nhHead: try headPlace(lImg * Self.dim * 2, "nhHead"),
            pred16: try headPlace(lImg * 128 * 2, "pred16"),
            latA: try scratch(lImg * 128 * 4, "latA"),   // persists across steps
            latB: try scratch(lImg * 128 * 4, "latB"))
    }

    /// Pack every denoising step's modulation fields into one aligned upload.
    /// The step loop only switches offsets; it performs no per-field uploads.
    public func setStepConditioningSequence(
        _ sequence: [Flux2StepConditioning], cos: MLXArray, sin: MLXArray
    ) throws {
        guard !sequence.isEmpty else {
            throw DirectQmmSpike.SpikeError.invalidTensor("empty conditioning sequence")
        }
        let alignment = 256
        var packed = Data()
        var offsets = [[String: Int]]()
        offsets.reserveCapacity(sequence.count)

        func fields(
            _ sc: Flux2StepConditioning
        ) -> [(String, MLXArray)] {
            [
                ("img.msa.shift", sc.doubleImg[0].0),
                ("img.msa.scale", sc.doubleImg[0].1),
                ("img.msa.gate", sc.doubleImg[0].2),
                ("img.mlp.shift", sc.doubleImg[1].0),
                ("img.mlp.scale", sc.doubleImg[1].1),
                ("img.mlp.gate", sc.doubleImg[1].2),
                ("txt.msa.shift", sc.doubleTxt[0].0),
                ("txt.msa.scale", sc.doubleTxt[0].1),
                ("txt.msa.gate", sc.doubleTxt[0].2),
                ("txt.mlp.shift", sc.doubleTxt[1].0),
                ("txt.mlp.scale", sc.doubleTxt[1].1),
                ("txt.mlp.gate", sc.doubleTxt[1].2),
                ("single.shift", sc.single.0),
                ("single.scale", sc.single.1),
                ("single.gate", sc.single.2),
                ("out.scale", sc.outConditioning.scale),
                ("out.shift", sc.outConditioning.shift),
            ]
        }

        for step in sequence {
            var stepOffsets = [String: Int]()
            for (name, value) in fields(step) {
                let padding = (alignment - packed.count % alignment) % alignment
                if padding > 0 { packed.append(Data(repeating: 0, count: padding)) }
                let array = value.asType(.float32)
                eval(array)
                let bytes = array.asData(access: .copy).data
                stepOffsets[name] = packed.count
                packed.append(bytes)
            }
            offsets.append(stepOffsets)
        }
        guard let buffer = packed.withUnsafeBytes({ raw -> MTLBuffer? in
            guard let base = raw.baseAddress else { return nil }
            return device.makeBuffer(bytes: base, length: raw.count)
        }) else {
            throw DirectQmmSpike.SpikeError.metal("conditioning plan upload")
        }
        buffer.label = "conditioning.plan"
        memoryLedger.recordConditioningUpload(bytes: buffer.length)
        conditionerPlanBuffer = buffer
        stepModBindings = offsets.map { step in
            Dictionary(uniqueKeysWithValues: step.map { key, offset in
                (key, BufferBinding(buffer: buffer, offset: offset))
            })
        }
        cosBuf = try upload(cos, "cos", retention: .conditioning)
        sinBuf = try upload(sin, "sin", retention: .conditioning)
        memoryLedger.replaceConditioning(
            bytes: buffer.length + cosBuf.length + sinBuf.length)
    }

    public func activateStepConditioning(_ step: Int) throws {
        guard stepModBindings.indices.contains(step) else {
            throw DirectQmmSpike.SpikeError.invalidTensor(
                "conditioning step \(step) is outside 0..<\(stepModBindings.count)")
        }
        modBufs = stepModBindings[step]
    }

    /// Compatibility entry point for spikes that execute one isolated step.
    public func setStepConditioning(_ sc: Flux2StepConditioning, cos: MLXArray, sin: MLXArray) throws {
        try setStepConditioningSequence([sc], cos: cos, sin: sin)
        try activateStepConditioning(0)
    }

    /// One full denoise step: x_embedder → 25 blocks → norm_out/proj_out →
    /// Euler, all in ONE command buffer. latIn/latOut are f32 [lImg, 128].
    public func encodeDenoiseStep(
        latIn: MTLBuffer, latOut: MTLBuffer, e0: MTLBuffer, head: Head, dt: Float
    ) throws {
        guard let cb = queue.makeCommandBuffer() else {
            throw DirectQmmSpike.SpikeError.metal("cb")
        }
        // Embedder
        guard let pre = cb.makeComputeCommandEncoder() else {
            throw DirectQmmSpike.SpikeError.metal("enc")
        }
        pre.waitForFence(fence)
        pre.setComputePipelineState(head.castPrePSO)
        pre.setBuffer(latIn, offset: 0, index: 0)
        pre.setBuffer(head.lat16, offset: 0, index: 1)
        var n1 = UInt32(lImg * 128)
        var inv16 = Float(1.0 / 16.0)
        pre.setBytes(&n1, length: 4, index: 2)
        pre.setBytes(&inv16, length: 4, index: 3)
        pre.dispatchThreads(
            MTLSize(width: lImg * 128, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        qmm(pre, head.xEmb, x: head.lat16, xOff: 0, y: head.hEmb16, yOff: 0, m: lImg)
        pre.setComputePipelineState(head.castPostPSO)
        pre.setBuffer(head.hEmb16, offset: 0, index: 0)
        pre.setBuffer(head.h0, offset: 0, index: 1)
        var n2 = UInt32(lImg * Self.dim)
        var sixteen = Float(16)
        pre.setBytes(&n2, length: 4, index: 2)
        pre.setBytes(&sixteen, length: 4, index: 3)
        pre.dispatchThreads(
            MTLSize(width: lImg * Self.dim, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        pre.updateFence(fence)
        pre.endEncoding()

        // 25 blocks (shared encoder machinery; input blit inside encodeBlocks)
        let finalJoint = try encodeBlocks(cb: cb, hIn: head.h0, eIn: e0)

        // Head: norm_out over image rows (+÷16), proj_out, Euler
        guard let henc = cb.makeComputeCommandEncoder() else {
            throw DirectQmmSpike.SpikeError.metal("enc")
        }
        henc.waitForFence(fence)
        lnMod(henc, finalJoint, xOff: lTxt * Self.dim * 4, "out", head.nhHead, rows: lImg)
        qmm(henc, head.projOut, x: head.nhHead, xOff: 0, y: head.pred16, yOff: 0, m: lImg)
        henc.setComputePipelineState(gateAddPSO)
        henc.setBuffer(latIn, offset: 0, index: 0)
        henc.setBuffer(head.onesGate, offset: 0, index: 1)
        henc.setBuffer(head.pred16, offset: 0, index: 2)
        henc.setBuffer(latOut, offset: 0, index: 3)
        var d128 = Int32(128)
        var eulerScale = Float(16.0 * dt)
        henc.setBytes(&d128, length: 4, index: 4)
        henc.setBytes(&eulerScale, length: 4, index: 5)
        henc.dispatchThreads(
            MTLSize(width: 128, height: lImg, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        henc.updateFence(fence)
        henc.endEncoding()

        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw DirectQmmSpike.SpikeError.metal("exec: \(e)") }
    }

    /// Total bytes of every MTLBuffer the engine holds (its full "watermark").
    public var allocatedBytes: Int {
        device.currentAllocatedSize
    }
}
