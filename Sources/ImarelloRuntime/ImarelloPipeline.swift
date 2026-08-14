import Foundation
import MLX
import ImarelloCore
import ImarelloWeights
import ImarelloText
import ImarelloDiT
import ImarelloVAE

/// Public entry point. All MLX work is isolated on this actor.
public actor ImarelloPipeline {
    public let config: ImarelloConfig
    /// Local HF snapshot when present under the Imarello models cache.
    public let snapshot: ModelSnapshot?
    private let orchestrator: StageOrchestrator
    private var generationInFlight = false

    public init(config: ImarelloConfig = .autoDetectingTier()) {
        self.config = config
        let snap = ModelPaths.resolveIfPresent(config: config)
        self.snapshot = snap
        let bits = Self.bits(for: config.weightPreset)
        self.orchestrator = StageOrchestrator(
            textEncoder: TextEncoderModule(snapshot: snap),
            dit: DiTModule(snapshot: snap, bits: bits),
            vae: VAEModule(snapshot: snap),
            memoryPolicy: config.memoryPolicy
        )
    }

    private static func bits(for preset: WeightPreset) -> Int {
        switch preset {
        case .bits3: return 3
        case .bits4: return 4
        case .bits6: return 6
        case .bits8: return 8
        }
    }

    /// Product default clamps at 768². `mid` keeps that floor; `.high` is not a product type.
    private static func isLargeCanvasForCache(maxSide: Int, forceLarge: Bool = false) -> Bool {
        forceLarge || maxSide >= EvalCachePolicy.current.pipelineCacheClampMinSide
    }

    private static func applyPreDenoiseCacheClamp() {
        let policy = EvalCachePolicy.current
        if let limit = policy.denoiseCacheLimitBytes {
            Memory.cacheLimit = Int(limit)
        }
        Memory.clearCache()
    }

    public var hasLocalSnapshot: Bool { snapshot != nil }
    public var snapshotPath: String? { snapshot?.root.path }
    /// Hugging Face commit recorded by `hf download`, when metadata is present.
    public var snapshotRevision: String? { snapshot?.detectedRevision }

    /// Ensure snapshot exists and directory layout is valid.
    public func loadMetadata() async throws {
        _ = try ModelPaths.resolveOrThrow(config: config)
    }

    /// Load DiT quant weights from snapshot, return leaf parameter count, unload if staged.
    public func loadDiT() async throws -> Int {
        try await orchestrator.loadDiTAndCountParameters()
    }

    /// Load DiT, run one synthetic 512-class forward, print activation/weight dtypes, unload.
    /// Does not change product numerics (probe is off unless this method enables it).
    public func probeComputeDtypes(width: Int, height: Int) async throws -> String {
        ComputeDTypeProbe.reset()
        ComputeDTypeProbe.enabled = true
        defer { ComputeDTypeProbe.reset() }

        try await orchestrator.loadDiTExclusive()
        do {
            if let model = orchestrator.dit.model {
                ComputeDTypeProbe.recordWeights(from: model)
            }
            let noise = LatentOps.samplePackedNoise(width: width, height: height, seed: 42)
            let ctx = MLXArray.zeros(
                [1, ModelConstants.maxSequenceLength, ModelConstants.innerDim],
                dtype: .float32
            )
            let imgIds = LatentOps.imageIds(width: width, height: height)
            let txtIds = LatentOps.textIds()
            let t = MLXArray([Float(1000)])
            eval(noise, ctx, imgIds, txtIds, t)
            let rope = try orchestrator.dit.prepareRotaryEmbeddings(imgIds: imgIds, txtIds: txtIds)
            _ = try orchestrator.dit.forward(
                hiddenStates: noise,
                encoderHiddenStates: ctx,
                timestep: t,
                imgIds: imgIds,
                txtIds: txtIds,
                imageRotaryEmb: rope,
                contextIsProjected: true
            )
            try await orchestrator.unloadDiTIfStaged()
            return ComputeDTypeProbe.report()
        } catch {
            try? await orchestrator.unloadDiTIfStaged()
            throw error
        }
    }

    /// Load VAE weights from snapshot, return leaf parameter count, unload if staged.
    public func loadVAE() async throws -> Int {
        try await orchestrator.loadVAEAndCountParameters()
    }

    /// Decode-only packed noise at `width`×`height` (bench `vae-decode`).
    public func decodePackedNoise(width: Int, height: Int, seed: UInt64 = 42) async throws {
        try DimensionValidation.validate(
            width: width, height: height, maxSide: config.maxSide, tier: config.tier)
        let noise = LatentOps.samplePackedNoise(width: width, height: height, seed: seed)
        let (ph, pw) = LatentOps.packedSpatial(width: width, height: height)
        try await orchestrator.loadVAEExclusive(mode: .decodeOnly)
        do {
            let spatial = LatentOps.unpackSequence(noise, height: ph, width: pw)
            let decoded = try orchestrator.vae.decodePacked(spatial)
            eval(decoded)
            try await orchestrator.unloadVAEIfStaged()
        } catch {
            try? await orchestrator.unloadVAEIfStaged()
            throw error
        }
    }

    /// Load Qwen3 TE quant weights from snapshot, return leaf parameter count, unload if staged.
    public func loadTextEncoder() async throws -> Int {
        try await orchestrator.loadTextEncoderAndCountParameters()
    }

    /// Load TE, encode prompt to embeddings, unload TE. Returns shape description.
    public func encodePrompt(_ prompt: String) async throws -> (shape: [Int], realTokens: Int) {
        try await orchestrator.loadTextEncoderExclusive()
        do {
            let (embeds, real) = try orchestrator.textEncoder.encode(prompt)
            let shape = embeds.shape
            try await orchestrator.unloadTextEncoderIfStaged()
            return (shape, real)
        } catch {
            try? await orchestrator.unloadTextEncoderIfStaged()
            throw error
        }
    }

    public func purge() async {
        await orchestrator.purge()
    }

    public func memorySelfTest() async throws -> [MemorySample] {
        try await orchestrator.runMemorySelfTest()
    }

    /// Load TE, invoke `body` while loaded, then unload under staged policy. For benchmarks.
    public func withTextEncoderLoaded(
        _ body: @Sendable () async throws -> Void
    ) async throws {
        try await orchestrator.loadTextEncoderExclusive()
        do {
            try await body()
            try await orchestrator.unloadTextEncoderIfStaged()
        } catch {
            try? await orchestrator.unloadTextEncoderIfStaged()
            throw error
        }
    }

    /// Load DiT, invoke `body` while loaded, then unload under staged policy. For benchmarks.
    public func withDiTLoaded(
        _ body: @Sendable () async throws -> Void
    ) async throws {
        try await orchestrator.loadDiTExclusive()
        do {
            try await body()
            try await orchestrator.unloadDiTIfStaged()
        } catch {
            try? await orchestrator.unloadDiTIfStaged()
            throw error
        }
    }

    /// Load VAE, invoke `body` while loaded, then unload under staged policy. For benchmarks.
    public func withVAELoaded(
        _ body: @Sendable () async throws -> Void
    ) async throws {
        try await orchestrator.loadVAEExclusive()
        do {
            try await body()
            try await orchestrator.unloadVAEIfStaged()
        } catch {
            try? await orchestrator.unloadVAEIfStaged()
            throw error
        }
    }

    /// Staged T2I: TE encode → unload → DiT Euler denoise → unload → VAE decode → PNG.
    ///
    /// Pass `trace` to record stage timings and memory samples (benchmarking); no effect on numerics.
    @discardableResult
    public func generate(
        _ request: T2IRequest,
        onProgress: (@Sendable (PipelineProgress) -> Void)? = nil,
        trace: PipelineTrace? = nil
    ) async throws -> URL {
        try beginGeneration()
        defer { generationInFlight = false }

        guard snapshot != nil else {
            throw ImarelloError.weightsNotFound(
                modelID: config.modelID,
                path: ModelPaths.snapshotRoot(
                    modelID: config.modelID,
                    modelsDirectory: config.modelsDirectory
                ).path
            )
        }

        try DimensionValidation.validate(
            width: request.width,
            height: request.height,
            maxSide: config.maxSide,
            tier: config.tier
        )

        onProgress?(PipelineProgress(phase: .preparing))
        trace?.emit(.memorySample(label: "prepare"))

        let (packedH, packedW) = LatentOps.packedSpatial(
            width: request.width, height: request.height)
        let imageSeqLen = packedH * packedW

        var latents = LatentOps.samplePackedNoise(
            width: request.width,
            height: request.height,
            seed: request.seed
        )
        eval(latents)

        let scheduler = Flux2Scheduler(
            numInferenceSteps: request.steps,
            imageSeqLen: imageSeqLen
        )

        // Stages 1–2 nested so prompt embeds + RoPE drop before VAE (lower peak RAM).
        do {
            // --- Stage 1: text encoder (skipped entirely on embed-cache hit) ---
            onProgress?(PipelineProgress(phase: .encodingText))
            var promptEmbeds: MLXArray
            var realTokens = 0
            let embedURL = request.embedCache
                ? PromptEmbedCache.entryURL(prompt: request.prompt, modelID: config.modelID)
                : nil
            if let embedURL, let cached = PromptEmbedCache.load(url: embedURL) {
                promptEmbeds = cached.embeds
                realTokens = cached.realTokens
                eval(promptEmbeds)
                trace?.note("embed_cache=hit", minDensity: .stages)
                trace?.emit(.memorySample(label: "after_embed_cache_hit"))
            } else {
                trace?.emit(.stageBegin("load_te"))
                try await orchestrator.loadTextEncoderExclusive()
                trace?.emit(.stageEnd("load_te"))
                trace?.emit(.memorySample(label: "after_load_te"))
                do {
                    trace?.emit(.stageBegin("encode_te"))
                    // TextEncoderModule.encode already evals embeds.
                    let (embeds, real) = try orchestrator.textEncoder.encode(
                        request.prompt, trace: trace)
                    promptEmbeds = embeds
                    realTokens = real
                    trace?.emit(.stageEnd("encode_te"))
                    trace?.emit(.memorySample(label: "after_encode_te"))
                    if let trace, trace.density != .off {
                        trace.emit(.peakReset(label: "after_te"))
                    }
                    trace?.emit(.stageBegin("unload_te"))
                    try await orchestrator.unloadTextEncoderIfStaged()
                    trace?.emit(.stageEnd("unload_te"))
                    trace?.emit(.memorySample(label: "after_unload_te"))
                } catch {
                    try? await orchestrator.unloadTextEncoderIfStaged()
                    throw error
                }
                if let embedURL {
                    PromptEmbedCache.store(
                        embeds: promptEmbeds, realTokens: realTokens, url: embedURL)
                }
            }

            // Opt-in: trim padded text tokens before the DiT (--text-tokens auto).
            if request.textTokens == .auto {
                promptEmbeds = Self.trimTextTokens(
                    promptEmbeds, realTokens: realTokens, trace: trace)
            }

            let txtIds = LatentOps.textIds(length: promptEmbeds.dim(1))
            let imgIds = LatentOps.imageIds(width: request.width, height: request.height)
            let imgSeq = packedH * packedW
            let joint = promptEmbeds.dim(1) + imgSeq
            trace?.note(
                "canvas=\(request.width)x\(request.height) image_seq=\(imgSeq) joint_seq=\(joint)",
                minDensity: .stages)

            // Distilled klein: no CFG; DiT built without guidance embeds.
            let guidance: MLXArray? = nil

            // --- Stage 2: DiT denoise ---
            trace?.emit(.stageBegin("load_dit"))
            try await orchestrator.loadDiTExclusive()
            trace?.emit(.stageEnd("load_dit"))
            trace?.emit(.memorySample(label: "after_load_dit"))
            do {
                // RoPE + ids are constant across steps for fixed canvas + text length.
                eval(imgIds, txtIds, promptEmbeds)
                let projectedContext = try orchestrator.dit.projectContext(promptEmbeds)
                let rope = try orchestrator.dit.prepareRotaryEmbeddings(
                    imgIds: imgIds, txtIds: txtIds)
                trace?.emit(.memorySample(label: "after_rope"))
                // Host-built training-scale timesteps + Euler dts (eval once before loop).
                let stepTimesteps: [MLXArray] = scheduler.timesteps.map {
                    MLXArray([$0]).asType(.float32)
                }
                let stepDts = LatentOps.eulerDts(sigmas: scheduler.sigmas)
                eval(stepTimesteps)
                // High-res (768²+) on unified memory: drop Metal buffer pool after every
                // step so activation temporaries don't accumulate (8 GB M2 OOMs at 1024²).
                let largeCanvas = Self.isLargeCanvasForCache(
                    maxSide: max(request.width, request.height))
                if largeCanvas {
                    Self.applyPreDenoiseCacheClamp()
                    trace?.emit(.memorySample(label: "after_clear_pre_denoise"))
                }
                if let trace, trace.density != .off {
                    trace.emit(.peakReset(label: "before_denoise"))
                }
                trace?.emit(.stageBegin("denoise"))
                for step in 0 ..< scheduler.numInferenceSteps {
                    onProgress?(
                        PipelineProgress(
                            phase: .denoising, step: step, totalSteps: request.steps))
                    trace?.emit(.denoiseStepBegin(index: step, total: request.steps))
                    trace?.probe(
                        "dit.step\(step).begin", phase: "dit", step: step, minDensity: .denoise)
                    let noisePred = try orchestrator.dit.forward(
                        hiddenStates: latents,
                        encoderHiddenStates: projectedContext,
                        timestep: stepTimesteps[step],
                        imgIds: imgIds,
                        txtIds: txtIds,
                        guidance: guidance,
                        imageRotaryEmb: rope,
                        contextIsProjected: true,
                        trace: trace,
                        stepIndex: step
                    )
                    // Single eval per step: materialize Euler result (pulls noisePred graph).
                    latents = LatentOps.eulerStep(
                        sample: latents,
                        modelOutput: noisePred,
                        dt: stepDts[step]
                    )
                    eval(latents)
                    trace?.probe(
                        "dit.step\(step).after_euler", phase: "dit", step: step, minDensity: .denoise)
                    if largeCanvas, EvalCachePolicy.current.clearCacheAfterDenoiseStep {
                        Memory.clearCache()
                        trace?.probe(
                            "dit.step\(step).after_clear", phase: "dit", step: step,
                            minDensity: .denoise)
                    }
                    trace?.emit(.denoiseStepEnd(index: step, total: request.steps))
                    if step == 0 || step + 1 == scheduler.numInferenceSteps
                        || (trace?.density.instrumentsEveryDenoiseStep == true)
                    {
                        trace?.emit(.memorySample(label: "denoise_step_\(step)"))
                    }
                }
                trace?.emit(.stageEnd("denoise"))
                trace?.emit(.stageBegin("unload_dit"))
                try await orchestrator.unloadDiTIfStaged()
                trace?.emit(.stageEnd("unload_dit"))
                trace?.emit(.memorySample(label: "after_unload_dit"))
            } catch {
                try? await orchestrator.unloadDiTIfStaged()
                throw error
            }
        }
        // Drop TE embeddings / RoPE / ids; keep only latents for VAE.
        Memory.clearCache()

        // --- Stage 3: VAE decode (decode-only weights — ~67 MB less on M2 / Tier L) ---
        onProgress?(PipelineProgress(phase: .decoding))
        trace?.emit(.stageBegin("load_vae"))
        try await orchestrator.loadVAEExclusive(mode: .decodeOnly)
        trace?.emit(.stageEnd("load_vae"))
        trace?.emit(.memorySample(label: "after_load_vae"))
        let rgb: MLXArray
        do {
            trace?.emit(.stageBegin("decode_vae"))
            let spatial = LatentOps.unpackSequence(
                latents, height: packedH, width: packedW)
            let decoded = try orchestrator.vae.decodePacked(spatial)
            eval(decoded)
            rgb = decoded
            trace?.emit(.stageEnd("decode_vae"))
            trace?.emit(.memorySample(label: "after_decode_vae"))
            trace?.emit(.stageBegin("unload_vae"))
            try await orchestrator.unloadVAEIfStaged()
            trace?.emit(.stageEnd("unload_vae"))
            trace?.emit(.memorySample(label: "after_unload_vae"))
        } catch {
            try? await orchestrator.unloadVAEIfStaged()
            throw error
        }

        let outURL = request.outputURL ?? Self.defaultOutputURL(seed: request.seed, prefix: "t2i")
        trace?.emit(.stageBegin("export_png"))
        try ImageExport.writePNG(rgb, to: outURL)
        trace?.emit(.stageEnd("export_png"))
        trace?.emit(.memorySample(label: "after_export"))

        onProgress?(
            PipelineProgress(phase: .finished, step: request.steps, totalSteps: request.steps))
        return outURL
    }

    /// Staged I2I: VAE encode → unload → TE → unload → DiT → unload → VAE decode → PNG.
    ///
    /// Supports classic **strength** init plus Tier-B identity options on
    /// ``I2IRequest/identity`` (reference latents, face-regional σ, clean-latent pull).
    ///
    /// Pass `trace` to record stage timings and memory samples (benchmarking); no effect on numerics.
    @discardableResult
    public func edit(
        _ request: I2IRequest,
        onProgress: (@Sendable (PipelineProgress) -> Void)? = nil,
        trace: PipelineTrace? = nil
    ) async throws -> URL {
        try beginGeneration()
        defer { generationInFlight = false }

        guard snapshot != nil else {
            throw ImarelloError.weightsNotFound(
                modelID: config.modelID,
                path: ModelPaths.snapshotRoot(
                    modelID: config.modelID,
                    modelsDirectory: config.modelsDirectory
                ).path
            )
        }

        if request.strength <= 0 || request.strength > 1 {
            throw ImarelloError.invalidStrength(request.strength)
        }

        let identity = request.identity
        let canvas = try ImageImport.resolveCanvas(
            imageURL: request.imageURL,
            requestWidth: request.width,
            requestHeight: request.height,
            maxSide: config.maxSide
        )
        try DimensionValidation.validate(
            width: canvas.width,
            height: canvas.height,
            maxSide: config.maxSide,
            tier: config.tier
        )

        onProgress?(PipelineProgress(phase: .preparing))
        trace?.emit(.memorySample(label: "prepare"))

        let (packedH, packedW) = LatentOps.packedSpatial(
            width: canvas.width, height: canvas.height)
        let imageSeqLen = packedH * packedW
        let nSteps = request.steps

        // Full N-step schedule from strength noise level → 0 (not a truncated T2I slice).
        // With reference latents, official FLUX.2 I2I starts near pure noise and lets the
        // model attend to the clean ref. Bump effective strength slightly so mid values
        // (e.g. 0.8) still allow wardrobe/scene edits under identity presets.
        let scheduleStrength: Float = {
            if identity.useReferenceLatents {
                return min(1.0, max(request.strength, request.strength * 0.5 + 0.5))
                // s=0.8 → 0.9; s=0.9 → 0.95; s=1.0 → 1.0
            }
            return request.strength
        }()
        let (sigmas, timesteps, startSigma) = Flux2Scheduler.strengthSchedule(
            numInferenceSteps: request.steps,
            strength: scheduleStrength,
            imageSeqLen: imageSeqLen,
            curve: identity.scheduleCurve
        )

        // Optional face mask (CPU/Vision; before Metal-heavy stages).
        var faceMask: MLXArray?
        var faceCount = 0
        if identity.facePreserve || identity.cleanPullAlpha > 0 {
            let built = try FaceIdentityMask.softPackedMask(
                imageURL: request.imageURL,
                width: canvas.width,
                height: canvas.height
            )
            faceMask = built.mask
            faceCount = built.faceCount
            eval(built.mask)
            if faceCount == 0 {
                // No face → disable regional path (avoid accidental full-canvas lock).
                faceMask = nil
            }
        }

        var latents: MLXArray

        // Stages 0–2 nested so pixels / embeds / RoPE drop before final VAE decode.
        do {
            // --- Stage 0: VAE encode reference ---
            onProgress?(PipelineProgress(phase: .encodingImage))
            let imageNCHW = try ImageImport.loadNCHW(
                url: request.imageURL, width: canvas.width, height: canvas.height)
            eval(imageNCHW)

            trace?.emit(.stageBegin("load_vae"))
            // Encode-only weights: decoder (~97 MB) is not needed for the reference encode.
            try await orchestrator.loadVAEExclusive(mode: .encodeOnly)
            trace?.emit(.stageEnd("load_vae"))
            trace?.emit(.memorySample(label: "after_load_vae_enc"))
            let cleanPacked: MLXArray
            do {
                trace?.emit(.stageBegin("encode_vae"))
                let spatial = try orchestrator.vae.encodePackedForDiT(imageNCHW)
                eval(spatial)
                cleanPacked = LatentOps.packSpatial(spatial)
                eval(cleanPacked)
                trace?.emit(.stageEnd("encode_vae"))
                trace?.emit(.stageBegin("unload_vae"))
                try await orchestrator.unloadVAEIfStaged()
                trace?.emit(.stageEnd("unload_vae"))
                trace?.emit(.memorySample(label: "after_unload_vae_enc"))
            } catch {
                try? await orchestrator.unloadVAEIfStaged()
                throw error
            }
            // imageNCHW drops at end of this do; free cache before TE.
            Memory.clearCache()

            // --- Stage 1: text encoder (skipped entirely on embed-cache hit) ---
            onProgress?(PipelineProgress(phase: .encodingText))
            var promptEmbeds: MLXArray
            var realTokens = 0
            let embedURL = request.embedCache
                ? PromptEmbedCache.entryURL(prompt: request.prompt, modelID: config.modelID)
                : nil
            if let embedURL, let cached = PromptEmbedCache.load(url: embedURL) {
                promptEmbeds = cached.embeds
                realTokens = cached.realTokens
                eval(promptEmbeds)
                trace?.note("embed_cache=hit", minDensity: .stages)
            } else {
                trace?.emit(.stageBegin("load_te"))
                try await orchestrator.loadTextEncoderExclusive()
                trace?.emit(.stageEnd("load_te"))
                do {
                    trace?.emit(.stageBegin("encode_te"))
                    let (embeds, real) = try orchestrator.textEncoder.encode(
                        request.prompt, trace: trace)
                    promptEmbeds = embeds
                    realTokens = real
                    trace?.emit(.stageEnd("encode_te"))
                    trace?.emit(.stageBegin("unload_te"))
                    try await orchestrator.unloadTextEncoderIfStaged()
                    trace?.emit(.stageEnd("unload_te"))
                    trace?.emit(.memorySample(label: "after_unload_te"))
                } catch {
                    try? await orchestrator.unloadTextEncoderIfStaged()
                    throw error
                }
                if let embedURL {
                    PromptEmbedCache.store(
                        embeds: promptEmbeds, realTokens: realTokens, url: embedURL)
                }
            }

            // Opt-in: trim padded text tokens before the DiT (--text-tokens auto).
            if request.textTokens == .auto {
                promptEmbeds = Self.trimTextTokens(
                    promptEmbeds, realTokens: realTokens, trace: trace)
            }

            let txtIds = LatentOps.textIds(length: promptEmbeds.dim(1))
            // Denoise target at t=0; optional reference frame at t=10.
            let denoiseImgIds = LatentOps.imageIds(
                width: canvas.width, height: canvas.height, tCoord: 0)
            let refFactor = max(1, identity.refDownsample)
            let refImgIds: MLXArray? = identity.useReferenceLatents
                ? (refFactor > 1
                    ? LatentOps.referenceImageIdsDownsampled(
                        width: canvas.width, height: canvas.height, factor: refFactor, index: 0)
                    : LatentOps.referenceImageIds(
                        width: canvas.width, height: canvas.height, index: 0))
                : nil
            let fullImgIds: MLXArray
            if let refImgIds {
                fullImgIds = concatenated([denoiseImgIds, refImgIds], axis: 1)
            } else {
                fullImgIds = denoiseImgIds
            }

            // Re-noise clean latents to start σ: (1−σ)·x₀ + σ·ε
            // Optional face region uses lower σ (regional strength).
            let noise = LatentOps.sampleNoiseLike(cleanPacked, seed: request.seed)
            let globalNoisy = LatentOps.scaleNoise(
                clean: cleanPacked, noise: noise, sigma: startSigma)
            if identity.facePreserve,
               let mask = faceMask,
               identity.faceStrengthScale < 0.999
            {
                let faceSigma = startSigma * identity.faceStrengthScale
                let faceNoisy = LatentOps.scaleNoise(
                    clean: cleanPacked, noise: noise, sigma: faceSigma)
                latents = LatentOps.regionalBlend(
                    global: globalNoisy, local: faceNoisy, mask: mask)
            } else {
                latents = globalNoisy
            }
            eval(latents)

            // Keep a clean copy for reference-latent conditioning and clean-pull.
            let refClean = cleanPacked
            // Optionally reduced reference tokens for the DiT concat (clean-pull always
            // uses the full-resolution refClean).
            let refTokens = identity.useReferenceLatents && refFactor > 1
                ? LatentOps.downsamplePacked(
                    cleanPacked, height: packedH, width: packedW, factor: refFactor)
                : cleanPacked
            let guidance: MLXArray? = nil
            let useRef = identity.useReferenceLatents
            let pullBase = identity.cleanPullAlpha
            let pullDecay = identity.cleanPullDecay

            // --- Stage 2: DiT denoise (all steps of strength schedule) ---
            trace?.emit(.stageBegin("load_dit"))
            try await orchestrator.loadDiTExclusive()
            trace?.emit(.stageEnd("load_dit"))
            trace?.emit(.memorySample(label: "after_load_dit"))
            do {
                eval(fullImgIds, txtIds, promptEmbeds, refClean, refTokens)
                let projectedContext = try orchestrator.dit.projectContext(promptEmbeds)
                let rope = try orchestrator.dit.prepareRotaryEmbeddings(
                    imgIds: fullImgIds, txtIds: txtIds)
                let stepTimesteps: [MLXArray] = timesteps.map {
                    MLXArray([$0]).asType(.float32)
                }
                let stepDts = LatentOps.eulerDts(sigmas: sigmas)
                eval(stepTimesteps)
                // Ref latents double image seq → treat as large canvas for cache discipline.
                let largeCanvas = Self.isLargeCanvasForCache(
                    maxSide: max(canvas.width, canvas.height), forceLarge: useRef)
                if largeCanvas {
                    Self.applyPreDenoiseCacheClamp()
                }
                trace?.emit(.stageBegin("denoise"))
                for step in 0 ..< nSteps {
                    onProgress?(
                        PipelineProgress(
                            phase: .denoising,
                            step: step,
                            totalSteps: nSteps
                        ))
                    trace?.emit(.denoiseStepBegin(index: step, total: nSteps))

                    let hidden: MLXArray
                    if useRef {
                        hidden = LatentOps.concatImageAndReferences(
                            denoise: latents, references: [refTokens])
                    } else {
                        hidden = latents
                    }

                    let noisePredFull = try orchestrator.dit.forward(
                        hiddenStates: hidden,
                        encoderHiddenStates: projectedContext,
                        timestep: stepTimesteps[step],
                        imgIds: fullImgIds,
                        txtIds: txtIds,
                        guidance: guidance,
                        imageRotaryEmb: rope,
                        contextIsProjected: true,
                        trace: trace,
                        stepIndex: step
                    )
                    let noisePred = useRef
                        ? LatentOps.sliceDenoisePrediction(
                            noisePredFull, denoiseSeqLen: imageSeqLen)
                        : noisePredFull

                    latents = LatentOps.eulerStep(
                        sample: latents,
                        modelOutput: noisePred,
                        dt: stepDts[step]
                    )

                    // Soft pull toward clean identity on face (after Euler).
                    if pullBase > 0, let mask = faceMask {
                        let alpha = LatentOps.cleanPullAlpha(
                            base: pullBase,
                            step: step,
                            totalSteps: nSteps,
                            decay: pullDecay
                        )
                        if alpha > 1e-6 {
                            latents = LatentOps.cleanPull(
                                noisy: latents,
                                clean: refClean,
                                mask: mask,
                                alpha: alpha
                            )
                        }
                    }

                    eval(latents)
                    if largeCanvas, EvalCachePolicy.current.clearCacheAfterDenoiseStep {
                        Memory.clearCache()
                    }
                    trace?.emit(.denoiseStepEnd(index: step, total: nSteps))
                }
                // Face count only for diagnostics in traces (no numeric effect).
                if faceCount > 0 {
                    trace?.emit(.memorySample(label: "i2i_face_count_\(faceCount)"))
                }
                trace?.emit(.stageEnd("denoise"))
                trace?.emit(.stageBegin("unload_dit"))
                try await orchestrator.unloadDiTIfStaged()
                trace?.emit(.stageEnd("unload_dit"))
                trace?.emit(.memorySample(label: "after_unload_dit"))
            } catch {
                try? await orchestrator.unloadDiTIfStaged()
                throw error
            }
        }
        Memory.clearCache()

        // --- Stage 3: VAE decode (decode-only after encode unload) ---
        onProgress?(PipelineProgress(phase: .decoding))
        trace?.emit(.stageBegin("load_vae"))
        try await orchestrator.loadVAEExclusive(mode: .decodeOnly)
        trace?.emit(.stageEnd("load_vae"))
        let rgb: MLXArray
        do {
            trace?.emit(.stageBegin("decode_vae"))
            let spatial = LatentOps.unpackSequence(
                latents, height: packedH, width: packedW)
            let decoded = try orchestrator.vae.decodePacked(spatial)
            eval(decoded)
            rgb = decoded
            trace?.emit(.stageEnd("decode_vae"))
            trace?.emit(.stageBegin("unload_vae"))
            try await orchestrator.unloadVAEIfStaged()
            trace?.emit(.stageEnd("unload_vae"))
            trace?.emit(.memorySample(label: "after_unload_vae_dec"))
        } catch {
            try? await orchestrator.unloadVAEIfStaged()
            throw error
        }

        let outURL = request.outputURL
            ?? Self.defaultOutputURL(seed: request.seed, prefix: "i2i")
        trace?.emit(.stageBegin("export_png"))
        try ImageExport.writePNG(rgb, to: outURL)
        trace?.emit(.stageEnd("export_png"))
        trace?.emit(.memorySample(label: "after_export"))

        onProgress?(PipelineProgress(phase: .finished, step: nSteps, totalSteps: nSteps))
        return outURL
    }

    /// Trim padded text embeddings to real prompt length rounded up to a multiple of 8.
    ///
    /// Padding sits at the end and Qwen attention is causal, so real-token embeddings are
    /// identical trimmed or not — only the DiT joint attention (which has no text mask)
    /// sees fewer tokens. Numerics-changing; gated by `TextTokenMode.auto`.
    /// See `Docs/TEXT_TOKENS.md`.
    private static func trimTextTokens(
        _ embeds: MLXArray, realTokens: Int, trace: PipelineTrace?
    ) -> MLXArray {
        let seqLen = embeds.dim(1)
        guard realTokens > 0 else { return embeds }
        let trimmed = min(seqLen, max(8, (realTokens + 7) / 8 * 8))
        guard trimmed < seqLen else { return embeds }
        trace?.note(
            "text_tokens=" + String(trimmed) + " (auto trim, real=" + String(realTokens)
                + ", padded=" + String(seqLen) + ")",
            minDensity: .stages)
        return embeds[0..., ..<trimmed, 0...]
    }

    private func beginGeneration() throws {
        if generationInFlight {
            throw ImarelloError.concurrentGenerationNotAllowed
        }
        generationInFlight = true
    }

    private static func defaultOutputURL(seed: UInt64?, prefix: String = "t2i") -> URL {
        let dir = AppCache.directory("outputs")
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let seedPart = seed.map { "s\($0)_" } ?? ""
        return dir.appendingPathComponent("\(prefix)_\(seedPart)\(stamp).png")
    }
}
