import Foundation
import Metal
import MLX
import os
import ImarelloCore
import ImarelloWeights
import ImarelloDiT
import ImarelloVAE
import ImarelloRuntime

/// Bespoke pipeline with product-safe staged residency on every memory tier.
/// The pipeline amortizes only cheap persistent infrastructure; TE, DiT, and
/// VAE weights never remain co-resident between generations.
public final class DirectPipeline {

    let snapshot: ModelSnapshot
    let metallibURL: URL
    let artifacts: DirectEngineArtifacts
    let keepTE: Bool
    let keepDiT: Bool
    let keepVAE: Bool
    private let naxSetting = OSAllocatedUnfairLock(initialState: false)
    public var useNAXQmm: Bool {
        get { naxSetting.withLock { $0 } }
        set { naxSetting.withLock { $0 = newValue } }
    }
    var te: DirectTEEncoder?
    let config: ImarelloConfig
    var vae: DirectVAE?

    struct CanvasState {
        let width: Int
        let height: Int
        let steps: Int
        let engine: DirectDiTStep
        let head: DirectDiTStep.Head
        let rope: (MLXArray, MLXArray)
        let conditioning: [Flux2StepConditioning]
        let dts: [Float]
    }
    var canvas: CanvasState?

    /// Construction retains paths and immutable configuration only. Model
    /// weights are loaded inside `generate` and released at stage boundaries.
    public init(
        snapshot: ModelSnapshot,
        artifacts: DirectEngineArtifacts,
        config: ImarelloConfig
    ) {
        self.snapshot = snapshot
        self.artifacts = artifacts
        self.metallibURL = artifacts.mlxMetallibURL
        self.config = config
        keepTE = false
        keepDiT = false
        keepVAE = false
    }

    func canvasState(
        width: Int, height: Int, steps: Int, conditioner: DirectConditioner
    ) throws -> CanvasState {
        if keepDiT, let c = canvas,
           c.width == width, c.height == height, c.steps == steps
        {
            return c
        }
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
        let engine = try DirectDiTStep(lImg: lImg, metallibURL: metallibURL, useNAXQmm: useNAXQmm)
        try engine.loadBlocks(arrays: arrays, nDouble: 5, nSingle: 20)
        let head = try engine.loadHead(arrays: arrays)
        arrays.removeAll()
        Memory.clearCache()

        let imgIds = LatentOps.imageIds(width: width, height: height)
        let txtIds = LatentOps.textIds()
        let rope = conditioner.rope(imgIds: imgIds, txtIds: txtIds)
        let scheduler = Flux2Scheduler(numInferenceSteps: steps, imageSeqLen: lImg)
        let conditioning = try conditioner.stepConditioning(timesteps: scheduler.timesteps)
        let dts: [Float] = (0 ..< scheduler.sigmas.count - 1).map {
            scheduler.sigmas[$0 + 1] - scheduler.sigmas[$0]
        }
        let state = CanvasState(
            width: width, height: height, steps: steps, engine: engine, head: head,
            rope: rope, conditioning: conditioning, dts: dts)
        if keepDiT { canvas = state }
        return state
    }

    static func makeVAE(
        snapshot: ModelSnapshot, metallibURL: URL, config: ImarelloConfig
    ) throws -> DirectVAE {
        guard let smallDir = ModelPaths.resolveSmallDecoderIfPresent(config: config) else {
            throw DirectQmmSpike.SpikeError.missingTensor("small decoder snapshot")
        }
        let v = try DirectVAE(
            smallDecoderFile: smallDir.appendingPathComponent("small_decoder.safetensors"),
            metallibURL: metallibURL)
        try v.loadBNStats(vaeDirectory: snapshot.vaeDirectory)
        return v
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
        try generateSynchronously(
            prompt: prompt, width: width, height: height, steps: 4,
            seed: seed, outputURL: outputURL, onProgress: nil, trace: nil)
    }

    func generateSynchronously(
        prompt: String,
        width: Int,
        height: Int,
        steps: Int,
        seed: UInt64,
        outputURL: URL,
        onProgress: (@Sendable (PipelineProgress) -> Void)?,
        trace: PipelineTrace?
    ) throws -> Timings {
        try MetalWorkLease.withLease {
            try generateUnderLease(
                prompt: prompt, width: width, height: height, steps: steps,
                seed: seed, outputURL: outputURL, onProgress: onProgress, trace: trace)
        }
    }

    private func generateUnderLease(
        prompt: String,
        width: Int,
        height: Int,
        steps: Int,
        seed: UInt64,
        outputURL: URL,
        onProgress: (@Sendable (PipelineProgress) -> Void)?,
        trace: PipelineTrace?
    ) throws -> Timings {
        try DimensionValidation.validate(
            width: width, height: height, maxSide: config.maxSide, tier: config.tier)
        guard steps > 0 else { throw ImarelloError.invalidSteps(steps) }
        let deviceName = MTLCreateSystemDefaultDevice()?.name ?? "unavailable"
        var capture = try DirectCaptureSession.fromEnvironment(
            metadata: DirectCaptureRunMetadata(
                prompt: prompt,
                seed: seed,
                width: width,
                height: height,
                steps: steps,
                guidance: 1.0,
                tokenMode: TextTokenMode.full512.rawValue,
                modelRevision: config.revision,
                snapshotRevision: snapshot.detectedRevision ?? "undetected",
                weightMode: "prequantized-4bit",
                backendIdentifier: "direct-v2-shell",
                mlxMetallibPath: artifacts.mlxMetallibURL.path,
                directShaderSHA256: artifacts.directManifest.shaderSHA256,
                directMetallibSHA256: artifacts.directManifest.metallibSHA256,
                deviceName: deviceName,
                operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
                thermalState: Self.thermalStateName,
                outputPath: outputURL.path))
        defer { try? capture?.finalize() }
        var tm = Timings()
        let t0 = CFAbsoluteTimeGetCurrent()
        onProgress?(PipelineProgress(phase: .preparing))
        trace?.emit(.memorySample(label: "direct_prepare"))
        try Task.checkCancellation()
        Memory.clearCache()  // 8 GB co-residency: drop MLX pool before the Metal-heavy stages

        let tTE = CFAbsoluteTimeGetCurrent()
        onProgress?(PipelineProgress(phase: .encodingText))
        trace?.emit(.stageBegin("load_te"))
        var teLocal: DirectTEEncoder?
        if let existing = te {
            teLocal = existing
        } else {
            teLocal = try DirectTEEncoder(
                teDirectory: snapshot.textEncoderDirectory,
                tokenizerDirectory: snapshot.tokenizerDirectory,
                metallibURL: metallibURL)
            if keepTE { te = teLocal }
        }
        trace?.emit(.stageEnd("load_te"))
        trace?.emit(.stageBegin("encode_te"))
        let (embeds, _, _) = try teLocal!.encode(prompt)
        if let capture {
            try capture.capture(id: "te.concatenated-taps", array: embeds)
            for tap in teLocal!.capturedTaps() {
                try capture.capture(
                    id: "te.tap-\(tap.layer)", data: tap.data,
                    dataType: "bfloat16", shape: [1, 512, 2_560])
            }
        }
        trace?.emit(.stageEnd("encode_te"))
        trace?.emit(.stageBegin("unload_te"))
        if !keepTE {
            te = nil
            teLocal = nil
            Memory.clearCache()
        }
        trace?.emit(.stageEnd("unload_te"))
        try Task.checkCancellation()

        // Conditioner weights belong to the DiT stage. Construct only after
        // TE has been released and let the local die before VAE construction.
        trace?.emit(.stageBegin("load_dit"))
        var conditioner: DirectConditioner? = try DirectConditioner(
            transformerDirectory: snapshot.transformerDirectory,
            metallibURL: metallibURL)
        let e0 = try conditioner!.projectContext(embeds)
        tm.teMS = (CFAbsoluteTimeGetCurrent() - tTE) * 1000
        trace?.emit(.memorySample(label: "direct_after_te"))

        var cOpt: CanvasState? = try canvasState(
            width: width, height: height, steps: steps, conditioner: conditioner!)
        trace?.emit(.stageEnd("load_dit"))
        let c = cOpt!
        capture?.setPlanDigest(c.engine.scratchPlanDigest)
        if let capture {
            try capture.capture(id: "dit.projected-context", array: e0)
            try capture.capture(id: "dit.rope-cos", array: c.rope.0)
            try capture.capture(id: "dit.rope-sin", array: c.rope.1)
            for (step, conditioning) in c.conditioning.enumerated() {
                try Self.captureConditioning(
                    conditioning, step: step, capture: capture)
            }
        }
        let lImg = (width / 16) * (height / 16)

        let tDiT = CFAbsoluteTimeGetCurrent()
        trace?.emit(.stageBegin("denoise"))
        let e0Buf = try c.engine.upload(e0, "e0")
        let noise = LatentOps.samplePackedNoise(width: width, height: height, seed: seed)
        eval(noise)
        let noiseData = noise.asData(access: .copy).data
        noiseData.withUnsafeBytes { raw in
            c.head.latA.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }
        var cur = c.head.latA, alt = c.head.latB
        try c.engine.setStepConditioningSequence(
            c.conditioning, cos: c.rope.0, sin: c.rope.1)
        for step in 0 ..< steps {
            try Task.checkCancellation()
            onProgress?(PipelineProgress(phase: .denoising, step: step, totalSteps: steps))
            trace?.emit(.denoiseStepBegin(index: step, total: steps))
            try c.engine.activateStepConditioning(step)
            try c.engine.encodeDenoiseStep(
                latIn: cur, latOut: alt, e0: e0Buf, head: c.head, dt: c.dts[step])
            swap(&cur, &alt)
            if let capture {
                try capture.capture(
                    id: "dit.latent-step-\(step)", buffer: cur,
                    byteCount: lImg * 128 * MemoryLayout<Float>.size,
                    dataType: "float32", shape: [1, lImg, 128])
            }
            trace?.emit(.denoiseStepEnd(index: step, total: steps))
        }
        var lat = [Float](repeating: 0, count: lImg * 128)
        lat.withUnsafeMutableBufferPointer { p in
            p.baseAddress!.update(
                from: cur.contents().bindMemory(to: Float.self, capacity: lImg * 128),
                count: lImg * 128)
        }
        let latents = MLXArray(lat, [1, lImg, 128])
        eval(latents)
        if let capture { try capture.capture(id: "dit.final-latent", array: latents) }
        trace?.emit(.stageEnd("denoise"))
        tm.ditMS = (CFAbsoluteTimeGetCurrent() - tDiT) * 1000
        trace?.emit(.stageBegin("unload_dit"))
        if !keepDiT {
            cOpt = nil  // stage out the DiT engine before the VAE decode
            conditioner = nil
            Memory.clearCache()
        }
        _ = cOpt
        trace?.emit(.stageEnd("unload_dit"))
        trace?.emit(.memorySample(label: "direct_after_dit"))

        let tVAE = CFAbsoluteTimeGetCurrent()
        try Task.checkCancellation()
        onProgress?(PipelineProgress(phase: .decoding))
        trace?.emit(.stageBegin("load_vae"))
        var vaeLocal: DirectVAE?
        if let v = vae {
            vaeLocal = v
        } else {
            vaeLocal = try Self.makeVAE(snapshot: snapshot, metallibURL: metallibURL, config: config)
        }
        trace?.emit(.stageEnd("load_vae"))
        trace?.emit(.stageBegin("decode_vae"))
        let spatial = LatentOps.unpackSequence(latents, height: height / 16, width: width / 16)
        let rgb = try vaeLocal!.decodePacked(spatial)
        eval(rgb)
        if let capture { try capture.capture(id: "vae.final-rgb", array: rgb) }
        trace?.emit(.stageEnd("decode_vae"))
        trace?.emit(.stageBegin("unload_vae"))
        if !keepVAE {
            vaeLocal = nil
            Memory.clearCache()
        }
        _ = vaeLocal
        trace?.emit(.stageEnd("unload_vae"))
        trace?.emit(.stageBegin("export_png"))
        try ImageExport.writePNG(rgb, to: outputURL)
        if let activeCapture = capture {
            try activeCapture.finalize()
            capture = nil
        }
        trace?.emit(.stageEnd("export_png"))
        Memory.clearCache()
        tm.vaeMS = (CFAbsoluteTimeGetCurrent() - tVAE) * 1000
        tm.totalMS = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        onProgress?(PipelineProgress(phase: .finished, step: steps, totalSteps: steps))
        return tm
    }

    private static var thermalStateName: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    private static func captureConditioning(
        _ value: Flux2StepConditioning,
        step: Int,
        capture: DirectCaptureSession
    ) throws {
        let prefix = "conditioner.step-\(step)"
        try capture.capture(id: "\(prefix).temb", array: value.temb)
        let streams: [(String, [(MLXArray, MLXArray, MLXArray)])] = [
            ("img", value.doubleImg),
            ("txt", value.doubleTxt),
        ]
        for (stream, sets) in streams {
            for (index, field) in sets.enumerated() {
                let kind = index == 0 ? "msa" : "mlp"
                try capture.capture(
                    id: "\(prefix).\(stream).\(kind).shift", array: field.0)
                try capture.capture(
                    id: "\(prefix).\(stream).\(kind).scale", array: field.1)
                try capture.capture(
                    id: "\(prefix).\(stream).\(kind).gate", array: field.2)
            }
        }
        try capture.capture(id: "\(prefix).single.shift", array: value.single.0)
        try capture.capture(id: "\(prefix).single.scale", array: value.single.1)
        try capture.capture(id: "\(prefix).single.gate", array: value.single.2)
        try capture.capture(
            id: "\(prefix).out.scale", array: value.outConditioning.scale)
        try capture.capture(
            id: "\(prefix).out.shift", array: value.outConditioning.shift)
    }
}
