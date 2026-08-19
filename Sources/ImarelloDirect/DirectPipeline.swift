import Foundation
import Metal
import MLX
import ImarelloCore
import ImarelloWeights
import ImarelloDiT
import ImarelloVAE
import ImarelloRuntime

/// Bespoke pipeline with TIERED residency. On ≥12 GB hosts every engine stays
/// resident across generates (canvas state cached). On 8 GB the measured
/// truth mirrors the product's staged lock: ANY two engines co-resident page
/// each other out through compressed swap (full co-residency: ~7 s TE encodes;
/// DiT-resident + staged TE: 24–26 s denoises) — so both engines stage per
/// generate, and the pipeline amortizes only the cheap persistent parts
/// (process, PSO specialization, conditioner, VAE module).
public final class DirectPipeline {

    let snapshot: ModelSnapshot
    let metallibURL: URL
    let keepTE: Bool
    let keepDiT: Bool
    let keepVAE: Bool
    var te: DirectTEEncoder?
    let conditioner: DirectConditioner
    let config: ImarelloConfig
    var vae: VAEModule?

    struct CanvasState {
        let width: Int
        let height: Int
        let engine: DirectDiTStep
        let head: DirectDiTStep.Head
        let rope: (MLXArray, MLXArray)
        let conditioning: [Flux2StepConditioning]
        let dts: [Float]
    }
    var canvas: CanvasState?

    public init(snapshot: ModelSnapshot, metallibURL: URL, config: ImarelloConfig) async throws {
        self.snapshot = snapshot
        self.metallibURL = metallibURL
        self.config = config
        let bigHost = ProcessInfo.processInfo.physicalMemory >= 12 * 1_073_741_824
        keepTE = bigHost
        keepDiT = bigHost
        keepVAE = bigHost
        conditioner = try DirectConditioner(
            transformerDirectory: snapshot.root.appendingPathComponent("transformer", isDirectory: true),
            metallibURL: metallibURL)
        if keepVAE {
            let v = VAEModule(
                snapshot: snapshot,
                decoderVariant: .smallDecoder,
                smallDecoderDirectory: ModelPaths.resolveSmallDecoderIfPresent(config: config))
            try await v.load(mode: .decodeOnly)
            vae = v
        }
    }

    func canvasState(width: Int, height: Int) throws -> CanvasState {
        if keepDiT, let c = canvas, c.width == width, c.height == height { return c }
        canvas = nil  // free the previous engine before building the next
        Memory.clearCache()
        let lImg = (width / 16) * (height / 16)
        var arrays: [String: MLXArray] = [:]
        let tDir = snapshot.root.appendingPathComponent("transformer", isDirectory: true)
        for shard in ["0.safetensors", "1.safetensors"] {
            let url = tDir.appendingPathComponent(shard)
            if FileManager.default.fileExists(atPath: url.path) {
                for (k, v) in try MLX.loadArrays(url: url) { arrays[k] = v }
            }
        }
        let engine = try DirectDiTStep(lImg: lImg, metallibURL: metallibURL)
        try engine.loadBlocks(arrays: arrays, nDouble: 5, nSingle: 20)
        let head = try engine.loadHead(arrays: arrays)
        arrays.removeAll()
        Memory.clearCache()

        let imgIds = LatentOps.imageIds(width: width, height: height)
        let txtIds = LatentOps.textIds()
        let rope = conditioner.rope(imgIds: imgIds, txtIds: txtIds)
        let scheduler = Flux2Scheduler(numInferenceSteps: 4, imageSeqLen: lImg)
        let conditioning = try conditioner.stepConditioning(timesteps: scheduler.timesteps)
        let dts: [Float] = (0 ..< scheduler.sigmas.count - 1).map {
            scheduler.sigmas[$0 + 1] - scheduler.sigmas[$0]
        }
        let state = CanvasState(
            width: width, height: height, engine: engine, head: head,
            rope: rope, conditioning: conditioning, dts: dts)
        if keepDiT { canvas = state }
        return state
    }

    public struct Timings {
        public var teMS = 0.0
        public var ditMS = 0.0
        public var vaeMS = 0.0
        public var totalMS = 0.0
    }

    @discardableResult
    public func generate(
        prompt: String, width: Int, height: Int, seed: UInt64, outputURL: URL
    ) async throws -> Timings {
        var tm = Timings()
        let t0 = CFAbsoluteTimeGetCurrent()
        Memory.clearCache()  // 8 GB co-residency: drop MLX pool before the Metal-heavy stages

        let tTE = CFAbsoluteTimeGetCurrent()
        let teLocal: DirectTEEncoder
        if let existing = te {
            teLocal = existing
        } else {
            teLocal = try DirectTEEncoder(
                teDirectory: snapshot.textEncoderDirectory,
                tokenizerDirectory: snapshot.tokenizerDirectory,
                metallibURL: metallibURL)
            if keepTE { te = teLocal }
        }
        let (embeds, _, _) = try teLocal.encode(prompt)
        let e0 = try conditioner.projectContext(embeds)
        if !keepTE { te = nil }  // teLocal frees at scope end; keep pressure low for DiT+VAE
        tm.teMS = (CFAbsoluteTimeGetCurrent() - tTE) * 1000

        var cOpt: CanvasState? = try canvasState(width: width, height: height)
        let c = cOpt!
        let lImg = (width / 16) * (height / 16)

        let tDiT = CFAbsoluteTimeGetCurrent()
        let e0Buf = try c.engine.upload(e0, "e0")
        let noise = LatentOps.samplePackedNoise(width: width, height: height, seed: seed)
        eval(noise)
        let noiseData = noise.asData(noCopy: false)
        noiseData.withUnsafeBytes { raw in
            c.head.latA.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }
        var cur = c.head.latA, alt = c.head.latB
        for step in 0 ..< 4 {
            try c.engine.setStepConditioning(c.conditioning[step], cos: c.rope.0, sin: c.rope.1)
            try c.engine.encodeDenoiseStep(
                latIn: cur, latOut: alt, e0: e0Buf, head: c.head, dt: c.dts[step])
            swap(&cur, &alt)
        }
        var lat = [Float](repeating: 0, count: lImg * 128)
        lat.withUnsafeMutableBufferPointer { p in
            p.baseAddress!.update(
                from: cur.contents().bindMemory(to: Float.self, capacity: lImg * 128),
                count: lImg * 128)
        }
        let latents = MLXArray(lat, [1, lImg, 128])
        eval(latents)
        tm.ditMS = (CFAbsoluteTimeGetCurrent() - tDiT) * 1000
        if !keepDiT {
            cOpt = nil  // stage out the DiT engine before the VAE decode
            Memory.clearCache()
        }
        _ = cOpt

        let tVAE = CFAbsoluteTimeGetCurrent()
        let vaeLocal: VAEModule
        if let v = vae {
            vaeLocal = v
        } else {
            vaeLocal = VAEModule(
                snapshot: snapshot,
                decoderVariant: .smallDecoder,
                smallDecoderDirectory: ModelPaths.resolveSmallDecoderIfPresent(config: config))
            try await vaeLocal.load(mode: .decodeOnly)
        }
        let spatial = LatentOps.unpackSequence(latents, height: height / 16, width: width / 16)
        let rgb = try vaeLocal.decodePacked(spatial)
        eval(rgb)
        if !keepVAE { await vaeLocal.unload() }
        try ImageExport.writePNG(rgb, to: outputURL)
        Memory.clearCache()
        tm.vaeMS = (CFAbsoluteTimeGetCurrent() - tVAE) * 1000
        tm.totalMS = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        return tm
    }
}
