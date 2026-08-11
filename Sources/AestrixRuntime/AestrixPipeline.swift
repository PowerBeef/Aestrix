import Foundation
import MLX
import AestrixCore
import AestrixWeights
import AestrixText
import AestrixDiT
import AestrixVAE

/// Public entry point. All MLX work is isolated on this actor.
public actor AestrixPipeline {
    public let config: AestrixConfig
    /// Local HF snapshot when present under the Aestrix models cache.
    public let snapshot: ModelSnapshot?
    private let orchestrator: StageOrchestrator
    private var generationInFlight = false

    public init(config: AestrixConfig = .autoDetectingTier()) {
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

    public var hasLocalSnapshot: Bool { snapshot != nil }
    public var snapshotPath: String? { snapshot?.root.path }

    /// Ensure snapshot exists and directory layout is valid.
    public func loadMetadata() async throws {
        _ = try ModelPaths.resolveOrThrow(config: config)
    }

    /// Load DiT quant weights from snapshot, return leaf parameter count, unload if staged.
    public func loadDiT() async throws -> Int {
        try await orchestrator.loadDiTAndCountParameters()
    }

    /// Load VAE weights from snapshot, return leaf parameter count, unload if staged.
    public func loadVAE() async throws -> Int {
        try await orchestrator.loadVAEAndCountParameters()
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

    /// Staged T2I: TE encode → unload → DiT Euler denoise → unload → VAE decode → PNG.
    @discardableResult
    public func generate(
        _ request: T2IRequest,
        onProgress: (@Sendable (PipelineProgress) -> Void)? = nil
    ) async throws -> URL {
        try beginGeneration()
        defer { generationInFlight = false }

        guard snapshot != nil else {
            throw AestrixError.weightsNotFound(
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

        // --- Stage 1: text encoder ---
        onProgress?(PipelineProgress(phase: .encodingText))
        try await orchestrator.loadTextEncoderExclusive()
        let promptEmbeds: MLXArray
        do {
            let (embeds, _) = try orchestrator.textEncoder.encode(request.prompt)
            eval(embeds)
            promptEmbeds = embeds
            try await orchestrator.unloadTextEncoderIfStaged()
        } catch {
            try? await orchestrator.unloadTextEncoderIfStaged()
            throw error
        }

        let txtIds = LatentOps.textIds(length: promptEmbeds.dim(1))
        let imgIds = LatentOps.imageIds(width: request.width, height: request.height)
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

        // Distilled klein: no CFG; DiT built without guidance embeds.
        let guidance: MLXArray? = nil

        // --- Stage 2: DiT denoise ---
        try await orchestrator.loadDiTExclusive()
        do {
            for step in 0 ..< scheduler.numInferenceSteps {
                onProgress?(
                    PipelineProgress(
                        phase: .denoising, step: step, totalSteps: request.steps))
                // Model expects σ-scale in [0,1] or [0,1000]; pass /1000 like diffusers.
                let t = scheduler.timesteps[step] / Float(ModelConstants.numTrainTimesteps)
                let timestep = MLXArray([t]).asType(.float32)
                let noisePred = try orchestrator.dit.forward(
                    hiddenStates: latents,
                    encoderHiddenStates: promptEmbeds,
                    timestep: timestep,
                    imgIds: imgIds,
                    txtIds: txtIds,
                    guidance: guidance
                )
                eval(noisePred)
                latents = LatentOps.eulerStep(
                    sample: latents,
                    modelOutput: noisePred,
                    sigma: scheduler.sigmas[step],
                    sigmaNext: scheduler.sigmas[step + 1]
                )
                eval(latents)
            }
            try await orchestrator.unloadDiTIfStaged()
        } catch {
            try? await orchestrator.unloadDiTIfStaged()
            throw error
        }

        // --- Stage 3: VAE decode ---
        onProgress?(PipelineProgress(phase: .decoding))
        try await orchestrator.loadVAEExclusive()
        let rgb: MLXArray
        do {
            let spatial = LatentOps.unpackSequence(
                latents, height: packedH, width: packedW)
            let decoded = try orchestrator.vae.decodePacked(spatial)
            eval(decoded)
            rgb = decoded
            try await orchestrator.unloadVAEIfStaged()
        } catch {
            try? await orchestrator.unloadVAEIfStaged()
            throw error
        }

        let outURL = request.outputURL ?? Self.defaultOutputURL(seed: request.seed, prefix: "t2i")
        try ImageExport.writePNG(rgb, to: outURL)

        onProgress?(
            PipelineProgress(phase: .finished, step: request.steps, totalSteps: request.steps))
        return outURL
    }

    /// Staged strength I2I: VAE encode → unload → TE → unload → DiT (from strength) → unload → VAE decode → PNG.
    @discardableResult
    public func edit(
        _ request: I2IRequest,
        onProgress: (@Sendable (PipelineProgress) -> Void)? = nil
    ) async throws -> URL {
        try beginGeneration()
        defer { generationInFlight = false }

        guard snapshot != nil else {
            throw AestrixError.weightsNotFound(
                modelID: config.modelID,
                path: ModelPaths.snapshotRoot(
                    modelID: config.modelID,
                    modelsDirectory: config.modelsDirectory
                ).path
            )
        }

        if request.strength <= 0 || request.strength > 1 {
            throw AestrixError.invalidStrength(request.strength)
        }

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

        // --- Stage 0: VAE encode reference ---
        onProgress?(PipelineProgress(phase: .encodingImage))
        let imageNCHW = try ImageImport.loadNCHW(
            url: request.imageURL, width: canvas.width, height: canvas.height)
        eval(imageNCHW)

        try await orchestrator.loadVAEExclusive()
        let cleanPacked: MLXArray
        do {
            let spatial = try orchestrator.vae.encodePackedForDiT(imageNCHW)
            eval(spatial)
            cleanPacked = LatentOps.packSpatial(spatial)
            eval(cleanPacked)
            try await orchestrator.unloadVAEIfStaged()
        } catch {
            try? await orchestrator.unloadVAEIfStaged()
            throw error
        }

        // --- Stage 1: text encoder ---
        onProgress?(PipelineProgress(phase: .encodingText))
        try await orchestrator.loadTextEncoderExclusive()
        let promptEmbeds: MLXArray
        do {
            let (embeds, _) = try orchestrator.textEncoder.encode(request.prompt)
            eval(embeds)
            promptEmbeds = embeds
            try await orchestrator.unloadTextEncoderIfStaged()
        } catch {
            try? await orchestrator.unloadTextEncoderIfStaged()
            throw error
        }

        let (packedH, packedW) = LatentOps.packedSpatial(
            width: canvas.width, height: canvas.height)
        let imageSeqLen = packedH * packedW
        let txtIds = LatentOps.textIds(length: promptEmbeds.dim(1))
        let imgIds = LatentOps.imageIds(width: canvas.width, height: canvas.height)

        // Full N-step schedule from strength noise level → 0 (not a truncated T2I slice).
        let (sigmas, timesteps, startSigma) = Flux2Scheduler.strengthSchedule(
            numInferenceSteps: request.steps,
            strength: request.strength,
            imageSeqLen: imageSeqLen
        )

        // Re-noise clean latents to start σ: (1−σ)·x₀ + σ·ε
        let noise = LatentOps.sampleNoiseLike(cleanPacked, seed: request.seed)
        var latents = LatentOps.scaleNoise(clean: cleanPacked, noise: noise, sigma: startSigma)
        eval(latents)

        let guidance: MLXArray? = nil
        let nSteps = request.steps

        // --- Stage 2: DiT denoise (all steps of strength schedule) ---
        try await orchestrator.loadDiTExclusive()
        do {
            for step in 0 ..< nSteps {
                onProgress?(
                    PipelineProgress(
                        phase: .denoising,
                        step: step,
                        totalSteps: nSteps
                    ))
                let t = timesteps[step] / Float(ModelConstants.numTrainTimesteps)
                let timestep = MLXArray([t]).asType(.float32)
                let noisePred = try orchestrator.dit.forward(
                    hiddenStates: latents,
                    encoderHiddenStates: promptEmbeds,
                    timestep: timestep,
                    imgIds: imgIds,
                    txtIds: txtIds,
                    guidance: guidance
                )
                eval(noisePred)
                latents = LatentOps.eulerStep(
                    sample: latents,
                    modelOutput: noisePred,
                    sigma: sigmas[step],
                    sigmaNext: sigmas[step + 1]
                )
                eval(latents)
            }
            try await orchestrator.unloadDiTIfStaged()
        } catch {
            try? await orchestrator.unloadDiTIfStaged()
            throw error
        }

        // --- Stage 3: VAE decode ---
        onProgress?(PipelineProgress(phase: .decoding))
        try await orchestrator.loadVAEExclusive()
        let rgb: MLXArray
        do {
            let spatial = LatentOps.unpackSequence(
                latents, height: packedH, width: packedW)
            let decoded = try orchestrator.vae.decodePacked(spatial)
            eval(decoded)
            rgb = decoded
            try await orchestrator.unloadVAEIfStaged()
        } catch {
            try? await orchestrator.unloadVAEIfStaged()
            throw error
        }

        let outURL = request.outputURL
            ?? Self.defaultOutputURL(seed: request.seed, prefix: "i2i")
        try ImageExport.writePNG(rgb, to: outURL)

        onProgress?(PipelineProgress(phase: .finished, step: nSteps, totalSteps: nSteps))
        return outURL
    }

    private func beginGeneration() throws {
        if generationInFlight {
            throw AestrixError.concurrentGenerationNotAllowed
        }
        generationInFlight = true
    }

    private static func defaultOutputURL(seed: UInt64?, prefix: String = "t2i") -> URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let dir = caches.appendingPathComponent("Aestrix/outputs", isDirectory: true)
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        let seedPart = seed.map { "s\($0)_" } ?? ""
        return dir.appendingPathComponent("\(prefix)_\(seedPart)\(stamp).png")
    }
}
