import ArgumentParser
import CoreImage
import Foundation
import ImageIO
import UniformTypeIdentifiers
import ImarelloCore
import ImarelloWeights
import ImarelloRuntime
import ImarelloDiT
import ImarelloVAE
import ImarelloEval
import ImarelloBench
import ImarelloDirect
import MLX

@main
struct ImarelloCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "imarello",
        abstract: "Imarello — FLUX.2 klein-4B MLX runtime (low-RAM staged, prequant).",
        subcommands: [
            MemSelfTest.self, Info.self, Schedule.self,
            LoadDiT.self, LoadVAE.self, LoadTE.self,
            EncodePrompt.self, T2I.self, I2I.self, Session.self,
            AnalyzeImage.self, Bench.self, BenchCompare.self,
            DiTCompileSpike.self,
            DirectSpike.self,
            DirectGenerate.self,
        ]
    )
}

func applyAttnF16Threshold(_ value: Int?) {
    guard let value else { return }
    var t = AttentionTuning.current
    t.f16SeqThreshold = value
    AttentionTuning.current = t
}

func applyAttnLinearCompute(_ raw: String?) throws {
    guard let raw else { return }
    var t = AttentionTuning.current
    switch raw {
    case "f16", "float16":
        t.linearF16 = true
    case "f32", "float32":
        t.linearF16 = false
    default:
        throw ValidationError("Unknown --attn-linear-compute '\(raw)'; use f16 | f32")
    }
    AttentionTuning.current = t
}

/// `--steps 0` (or negative) would trap on `Flux2Scheduler`'s precondition.
func validateSteps(_ steps: Int) throws {
    guard steps >= 1 else {
        throw ValidationError("steps must be at least 1, got \(steps)")
    }
}

func applyVAEVariant(_ raw: String, to config: inout ImarelloConfig) throws {
    guard let variant = VAEDecoderVariant(rawValue: raw) else {
        throw ValidationError("Unknown --vae-variant '\(raw)'; use full | small-decoder")
    }
    config.vaeDecoderVariant = variant
}

/// Call at the start of any command that may touch MLX.
func ensureMLXReady() throws {
    MLXBootstrap.ensureMetallibBesideExecutable()
    do {
        try HostPreflight.acquireForCLI()
    } catch let error as ImarelloError {
        fputs("error: \(error.localizedDescription)\n", stderr)
        throw ExitCode.failure
    }
}

// MARK: - Post-generation evaluation (pixel harness + vision brief)

/// Shared options for `t2i` / `i2i` quality workflow (see Docs/EVAL_WORKFLOW.md).
struct GenerationEvalOptions: Sendable {
    var analyze: Bool
    var analyzeJSON: String?
    var visionBrief: Bool
    var failOnPixelGate: Bool
    /// I2I strength for strength-aware SSIM / LPIPS-lite gates.
    var i2iStrength: Float?

    /// True if any eval artifact was requested.
    var isEnabled: Bool { analyze || analyzeJSON != nil || visionBrief }
}

/// Run pixel analysis (+ optional vision brief) after a successful write.
/// - Returns: `true` if pixel hard-fail findings were present (exit 2 when failOnPixelGate).
@discardableResult
func runPostGenerationEval(
    imageURL: URL,
    prompt: String,
    referenceURL: URL?,
    mode: VisionReview.Mode,
    options: GenerationEvalOptions
) throws -> Bool {
    guard options.isEnabled else { return false }

    print("")
    print("=== post-generation eval (Docs/EVAL_WORKFLOW.md) ===")

    let report = try ImageAnalyzer.analyze(
        imageURL: imageURL,
        options: .init(
            prompt: prompt,
            referenceURL: referenceURL,
            maxAnalysisSide: 1024,
            i2iStrength: options.i2iStrength
        )
    )

    let jsonText = try ImageAnalysisReportBuilder.jsonString(report, pretty: true)
    let jsonPath: URL = {
        if let p = options.analyzeJSON {
            return URL(fileURLWithPath: p)
        }
        // Default: sidecar next to PNG
        return imageURL.deletingPathExtension().appendingPathExtension("eval.json")
    }()

    if options.analyze || options.analyzeJSON != nil {
        try FileManager.default.createDirectory(
            at: jsonPath.deletingLastPathComponent(), withIntermediateDirectories: true)
        try jsonText.write(to: jsonPath, atomically: true, encoding: .utf8)
        print("eval_json: \(jsonPath.path)")
    }

    if options.analyze {
        print(ImageAnalysisReportBuilder.textSummary(report))
    }

    if options.visionBrief {
        print("")
        print(VisionReview.agentBrief(report: report, mode: mode))
        let briefPath = imageURL.deletingPathExtension().appendingPathExtension("vision-brief.md")
        try VisionReview.agentBrief(report: report, mode: mode)
            .write(to: briefPath, atomically: true, encoding: .utf8)
        print("vision_brief: \(briefPath.path)")
        print("next: open image with vision tools, fill VisionReview.Assessment, merge (see EVAL_WORKFLOW.md)")
    }

    let fails = report.findings.filter { $0.severity == .fail }
    if !fails.isEmpty {
        fputs(
            "quality_gate: FAIL (\(fails.map(\.code).joined(separator: ","))) — see Docs/EVAL_WORKFLOW.md\n",
            stderr
        )
        if options.failOnPixelGate {
            throw ExitCode(2)
        }
    } else if options.analyze || options.visionBrief {
        print("quality_gate: pixel PASS (still run vision checklist before claiming done)")
    }
    return !fails.isEmpty
}

struct Info: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Show detected tier, default model, snapshot path, and policy."
    )

    func run() async throws {
        let config = ImarelloConfig.autoDetectingTier()
        let pipeline = ImarelloPipeline(config: config)
        let phys = ProcessInfo.processInfo.physicalMemory
        print("Imarello")
        print("  physical_memory: \(MemoryProbe.formatBytes(phys))")
        let caps = ChipCapabilities.detect()
        print("  gpu: \(caps.summary)")
        print("  tier: \(config.tier.rawValue)")
        print("  max_side: \(config.maxSide)")
        print("  memory_policy: \(config.memoryPolicy.rawValue)")
        print("  eval_cache: \(EvalCachePolicy.current.profileName)")
        print("  text_tokens: 512 (default)")
        print("  text_tokens_auto: opt-in trim (faster, weaker conditioning); Docs/TEXT_TOKENS.md")
        let smallReady = ModelPaths.resolveSmallDecoderIfPresent(config: config) != nil
        print("  vae_decoder: \(config.vaeDecoderVariant.rawValue) (default)")
        print("  small_decoder_ready: \(smallReady)")
        if !smallReady {
            print("  small_decoder_hint: \(VAEDecoderVariant.smallDecoder.downloadCommand)")
        }
        print("  weight_preset: \(config.weightPreset.rawValue)")
        print("  model_id: \(config.modelID)")
        print("  model_revision: \(config.revision)")
        print("  peak_budget: \(MemoryProbe.formatBytes(config.tier.peakBudgetBytes))")
        let exe = URL(fileURLWithPath: CommandLine.arguments[0])
        if let metalURL = MetallibVerification.resolveExisting(relativeTo: exe) {
            let metal = MetallibVerification.verify(url: metalURL)
            print("  metallib: \(metal.path)")
            print("  metallib_bytes: \(metal.byteCount)")
            print("  metallib_stub: \(metal.isStub)")
            print("  metallib_steel: \(metal.productReady ? "ok" : "missing")")
            print("  metallib_nax: \(metal.naxPackaged ? "packaged" : "optional-missing")")
            print("  metallib_ready: \(metal.productReady)")
            if !metal.productReady {
                print("  metallib_hint: \(metal.note)")
            }
        } else {
            print("  metallib_ready: false")
            print("  metallib_hint: run Scripts/ensure-metallib.sh")
        }
        if let path = await pipeline.snapshotPath {
            print("  snapshot: \(path)")
            print("  snapshot_ready: true")
            if let localRev = await pipeline.snapshotRevision {
                print("  snapshot_revision: \(localRev)")
                print("  snapshot_revision_match: \(localRev == config.revision)")
            }
        } else {
            print("  snapshot_ready: false")
            print("  snapshot_hint: \(config.downloadCommand)")
        }
    }
}

struct LoadDiT: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "load-dit",
        abstract: "Load quantized DiT weights from the local snapshot (requires Metal)."
    )

    @Flag(name: .long, help: "Run one synthetic DiT step and print activation/weight dtypes (R2 probe).")
    var dumpDtypes: Bool = false

    @Option(name: .long, help: "Canvas width for --dump-dtypes (default 512).")
    var width: Int = 512

    @Option(name: .long, help: "Canvas height for --dump-dtypes (default 512).")
    var height: Int = 512

    func run() async throws {
        try ensureMLXReady()
        let config = ImarelloConfig.autoDetectingTier()
        let pipeline = ImarelloPipeline(config: config)
        guard await pipeline.hasLocalSnapshot else {
            print("error: no local snapshot for \(config.modelID)")
            throw ExitCode.failure
        }
        print("loading DiT from \(await pipeline.snapshotPath ?? "?") …")
        do {
            if dumpDtypes {
                try DimensionValidation.validate(
                    width: width, height: height, maxSide: config.maxSide, tier: config.tier)
                print("compute dtype probe canvas=\(width)x\(height)")
                let report = try await pipeline.probeComputeDtypes(width: width, height: height)
                print(report)
                print("DiT dtype probe OK")
            } else {
                let leaves = try await pipeline.loadDiT()
                print("DiT load OK parameter_leaves=\(leaves)")
            }
        } catch {
            print("error: \(error)")
            throw ExitCode.failure
        }
    }
}

struct LoadVAE: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "load-vae",
        abstract: "Load VAE weights from the local snapshot (requires Metal)."
    )

    @Option(name: .long, help: "Decode graph: small-decoder (default, BFL distilled) | full (klein pack).")
    var vaeVariant: String = "small-decoder"

    func run() async throws {
        try ensureMLXReady()
        var config = ImarelloConfig.autoDetectingTier()
        try applyVAEVariant(vaeVariant, to: &config)
        if config.vaeDecoderVariant == .smallDecoder,
           ModelPaths.resolveSmallDecoderIfPresent(config: config) == nil
        {
            print("error: Small Decoder snapshot missing")
            print("hint: \(VAEDecoderVariant.smallDecoder.downloadCommand)")
            throw ExitCode.failure
        }
        let pipeline = ImarelloPipeline(config: config)
        guard await pipeline.hasLocalSnapshot else {
            print("error: no local snapshot for \(config.modelID)")
            throw ExitCode.failure
        }
        print(
            "loading VAE variant=\(config.vaeDecoderVariant.rawValue) from \(await pipeline.snapshotPath ?? "?") …"
        )
        do {
            let leaves = try await pipeline.loadVAE()
            print("VAE load OK parameter_leaves=\(leaves) variant=\(config.vaeDecoderVariant.rawValue)")
        } catch {
            print("error: \(error)")
            throw ExitCode.failure
        }
    }
}

struct LoadTE: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "load-te",
        abstract: "Load quantized Qwen3 text encoder from the local snapshot (requires Metal)."
    )

    func run() async throws {
        try ensureMLXReady()
        let config = ImarelloConfig.autoDetectingTier()
        let pipeline = ImarelloPipeline(config: config)
        guard await pipeline.hasLocalSnapshot else {
            print("error: no local snapshot for \(config.modelID)")
            throw ExitCode.failure
        }
        print("loading TE from \(await pipeline.snapshotPath ?? "?") …")
        do {
            let leaves = try await pipeline.loadTextEncoder()
            print("TE load OK parameter_leaves=\(leaves)")
        } catch {
            print("error: \(error)")
            throw ExitCode.failure
        }
    }
}

struct EncodePrompt: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "encode-prompt",
        abstract: "Load TE, encode a prompt to [1,512,7680] embeds, unload (staged)."
    )

    @Argument(help: "Prompt text")
    var prompt: String

    func run() async throws {
        try ensureMLXReady()
        let config = ImarelloConfig.autoDetectingTier()
        let pipeline = ImarelloPipeline(config: config)
        guard await pipeline.hasLocalSnapshot else {
            print("error: no local snapshot for \(config.modelID)")
            throw ExitCode.failure
        }
        print("encoding prompt (TE staged) …")
        do {
            let result = try await pipeline.encodePrompt(prompt)
            let shapeStr = result.shape.map(String.init).joined(separator: "×")
            print("encode OK shape=\(shapeStr) real_tokens=\(result.realTokens) pad_to=\(ModelConstants.maxSequenceLength)")
        } catch {
            print("error: \(error)")
            throw ExitCode.failure
        }
    }
}

struct MemSelfTest: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mem-selftest",
        abstract: "Run staged TE→DiT→VAE load/unload without model weights; print memory samples."
    )

    @Option(name: .long, help: "Memory policy: staged | stagedAggressive | resident")
    var memory: String = MemoryPolicy.staged.rawValue

    func run() async throws {
        guard let policy = MemoryPolicy(rawValue: memory) else {
            throw ValidationError("Unknown memory policy: \(memory)")
        }
        var config = ImarelloConfig.autoDetectingTier()
        config.memoryPolicy = policy
        // Avoid loading real weights / Metal during residency smoke test.
        config.modelsDirectory = URL(fileURLWithPath: "/tmp/imarello-no-snapshot", isDirectory: true)

        let pipeline = ImarelloPipeline(config: config)
        print("mem-selftest policy=\(policy.rawValue) tier=\(config.tier.rawValue) dry=true")
        let samples = try await pipeline.memorySelfTest()
        for s in samples {
            let rss = MemoryProbe.formatBytes(s.processResidentBytes)
            let label = s.label.padding(toLength: 22, withPad: " ", startingAt: 0)
            print("  \(label)  rss=\(rss)")
        }
        if let last = samples.last {
            let loadedCheck = samples.filter { $0.label.contains("_loaded") }
            print("samples=\(samples.count) loaded_stages=\(loadedCheck.count)")
            print("peak_rss=\(MemoryProbe.formatBytes(samples.map(\.processResidentBytes).max() ?? 0))")
            print("final_label=\(last.label) OK")
        }
    }
}

struct Schedule: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Print FLUX.2 flow-match Euler sigmas/timesteps for a canvas size."
    )

    @Option var width: Int = 1024
    @Option var height: Int = 1024
    @Option var steps: Int = 4

    func run() async throws {
        try validateSteps(steps)
        try DimensionValidation.validate(
            width: width,
            height: height,
            maxSide: max(width, height),
            tier: .high
        )
        let seq = Flux2Scheduler.imageSeqLen(width: width, height: height)
        let sched = Flux2Scheduler(numInferenceSteps: steps, imageSeqLen: seq)
        let mu = Flux2Scheduler.linearMu(imageSeqLen: seq)
        print("schedule width=\(width) height=\(height) steps=\(steps) image_seq_len=\(seq) mu=\(mu)")
        print("sigmas (\(sched.sigmas.count)):")
        for (i, s) in sched.sigmas.enumerated() {
            print(String(format: "  [%d] %.8f", i, s))
        }
        print("timesteps (\(sched.timesteps.count)):")
        for (i, t) in sched.timesteps.enumerated() {
            print(String(format: "  [%d] %.4f", i, t))
        }
    }
}

struct T2I: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "t2i",
        abstract: "Staged text-to-image: TE → DiT (Euler) → VAE → PNG."
    )

    @Argument(help: "Prompt (BFL klein style: narrative, lighting, subject first).")
    var prompt: String

    @Option var width: Int = 1024
    @Option var height: Int = 1024
    @Option var steps: Int = 4
    @Option var seed: UInt64?
    @Option(name: .long, help: "4bit (the product lock; 6/8-bit pins are non-product references)")
    var weights: String = "4bit"
    @Option(name: .long, help: "Output PNG path (default: ~/Library/Caches/Imarello/outputs/…)")
    var output: String?

    @Flag(name: .long, help: "After write: run pixel eval + print report (writes sidecar .eval.json).")
    var analyze: Bool = false

    @Option(name: .long, help: "After write: write pixel eval JSON to this path (implies analysis).")
    var analyzeJson: String?

    @Flag(name: .long, help: "After write: print vision-review brief + write .vision-brief.md for agents.")
    var visionBrief: Bool = false

    @Flag(name: .long, help: "Exit 2 if pixel quality gate fails (use with --analyze).")
    var failOnPixelGate: Bool = false

    @Option(name: .long, help: "Text tokens to DiT: 512 (default, padded product path) | auto (trim pad; faster, weaker conditioning). Docs/TEXT_TOKENS.md.")
    var textTokens: String = "512"

    @Option(name: .long, help: "EXPERIMENT (ENGINE_RESEARCH §5.1 R4): pad content: prompt (default) | clean (splice empty-prompt pads).")
    var padContent: String = "prompt"

    @Option(name: .long, help: "EXPERIMENT (§5.1 R3): keep only N pads after the real tokens (needs --text-tokens 512).")
    var padKeep: Int?

    @Flag(name: .long, help: "EXPERIMENT (§5.1 R3): ln(removed/kept) bias on kept pads (with --pad-keep).")
    var padBias: Bool = false

    @Option(name: .long, help: "VAE decode graph: small-decoder (default) | full (klein pack).")
    var vaeVariant: String = "small-decoder"

    @Option(name: .long, help: "Seq length above which Q/K/V use f16 (default 512). 2048 restores f32 QKV at 512².")
    var attnF16Threshold: Int?

    @Option(name: .long, help: "4-bit Linear compute: f16 (default, scaled qmm) | f32 (reference GEMM).")
    var attnLinearCompute: String?

    @Flag(name: .long, inversion: .prefixedNo, help: "Cache prompt embeddings on disk; skips TE load+encode on repeat prompts (default on).")
    var embedCache: Bool = true

    @Option(name: .long, help: "TE engine: direct (default since 2026-08-18 — bare-metal encoder, FULL gate passed: 22/22 pixel + vision incl. 1024² anatomy probes) | mlx (previous product path).")
    var teEngine: String = "direct"

    @Option(name: .long, help: "VAE decode engine: direct (default since 2026-08-19 — bare-metal decoder, gate: 15/15 regression + ≤1 LSB output equivalence both canvases + vision) | mlx (previous product path).")
    var vaeEngine: String = "direct"

    @Option(name: .long, help: "DiT denoise engine: direct (default since 2026-08-19 — FULL gate: 15/15 regression + 12 anatomy probes + 1024² trio, e2e −6.7/−9.0%; requires --text-tokens 512, no pad-keep/pad-bias) | mlx (previous product path).")
    var ditEngine: String = "direct"

    @Flag(name: .long, help: "EXPERIMENT (engine plan Q2): compose at 512² (clean anatomy), then refine at the target size via a low-strength I2I pass. Square canvases > 512 only.")
    var twoStage: Bool = false

    @Option(name: .long, help: "Two-stage refine noise level σ in (0, 1); linear schedule, so strength ≡ σ (default 0.15).")
    var refineStrength: Float = 0.15

    @Option(name: .long, help: "Two-stage refine steps (default 2).")
    var refineSteps: Int = 2

    @Option(name: .long, help: "Two-stage upscale filter: bicubic (I2I cover-scale, default) | lanczos (CoreImage, sharper — texture A/B).")
    var refineUpscale: String = "bicubic"

    func run() async throws {
        if teEngine == "direct" {
            guard embedCache else {
                throw ValidationError("--te-engine direct requires the embed cache (drop --no-embed-cache)")
            }
            try ensureMLXReady()
            let config = ImarelloConfig.autoDetectingTier()
            let snap = try ModelPaths.resolveOrThrow(config: config)
            let exe = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
            let metallib = exe.deletingLastPathComponent().appendingPathComponent("mlx.metallib")
            print("te_engine: direct (bare-metal Stage 2) — building engine…")
            let encoder = try DirectTEEncoder(
                teDirectory: snap.textEncoderDirectory,
                tokenizerDirectory: snap.tokenizerDirectory,
                metallibURL: metallib)
            let embeds: MLXArray
            let realTokens: Int
            let ms: Double
            if padContent == "clean" {
                // Splice: real-token causal encode + cached clean-pad bank.
                let (real, r, realMS) = try encoder.encodeRealOnly(prompt)
                let bankURL = PromptEmbedCache.entryURL(prompt: "", modelID: config.modelID)
                var bankMS = 0.0
                let bank: MLXArray
                if let cached = PromptEmbedCache.load(url: bankURL) {
                    bank = cached.embeds
                } else {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let (b, br, _) = try encoder.encode("")
                    PromptEmbedCache.store(embeds: b, realTokens: br, url: bankURL)
                    bank = b
                    bankMS = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                }
                embeds = concatenated([real, bank[0..., r..., 0...]], axis: 1)
                eval(embeds)
                realTokens = r
                ms = realMS
                if bankMS > 0 {
                    print(String(format: "te_engine: clean-pad bank built %.0f ms (one-time, cached)", bankMS))
                }
            } else {
                (embeds, realTokens, ms) = try encoder.encode(prompt)
            }
            let url = PromptEmbedCache.entryURL(
                prompt: prompt, modelID: config.modelID, padContent: padContent)
            PromptEmbedCache.store(embeds: embeds, realTokens: realTokens, url: url)
            print(String(
                format: "te_engine: direct encode %.0f ms (%d real tokens, pad=%@) — cache pre-seeded, TE stage will be skipped",
                ms, realTokens, padContent))
        } else if teEngine != "mlx" {
            throw ValidationError("unknown --te-engine " + teEngine + "; use mlx | direct")
        }
        try ensureMLXReady()
        try validateSteps(steps)
        applyAttnF16Threshold(attnF16Threshold)
        try applyAttnLinearCompute(attnLinearCompute)
        guard let preset = WeightPreset(rawValue: weights) else {
            throw ValidationError("Unknown weights preset: \(weights)")
        }
        guard preset == .bits4 else {
            throw ValidationError(
                "--weights \(weights) is not a product path (4-bit lock, Docs/WEIGHTS.md); "
                    + "the 6/8-bit pins are reference-only and cannot load end-to-end")
        }
        guard let textTokenMode = TextTokenMode(rawValue: textTokens) else {
            throw ValidationError("Unknown --text-tokens '\(textTokens)'; use 512 | auto")
        }
        var config = ImarelloConfig.autoDetectingTier()
        config.apply(preset: preset)
        try applyVAEVariant(vaeVariant, to: &config)
        if config.vaeDecoderVariant == .smallDecoder,
           ModelPaths.resolveSmallDecoderIfPresent(config: config) == nil
        {
            print("error: Small Decoder snapshot missing")
            print("hint: \(VAEDecoderVariant.smallDecoder.downloadCommand)")
            throw ExitCode.failure
        }

        let pipeline = ImarelloPipeline(config: config)
        guard await pipeline.hasLocalSnapshot else {
            print("error: no local snapshot for \(config.modelID) @ \(config.revision)")
            print("hint: \(config.downloadCommand)")
            throw ExitCode.failure
        }

        switch vaeEngine {
        case "direct":
            guard vaeVariant == "small-decoder" else {
                throw ValidationError("--vae-engine direct implements the Small Decoder only; drop --vae-variant full")
            }
            guard let snapV = ModelPaths.resolveIfPresent(config: config),
                let smallDir = ModelPaths.resolveSmallDecoderIfPresent(config: config)
            else {
                print("error: --vae-engine direct needs the klein and Small Decoder snapshots")
                throw ExitCode.failure
            }
            let exeV = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
            await pipeline.setPackedDecoder(DirectVAEPackedDecoder(
                smallDecoderDirectory: smallDir,
                vaeDirectory: snapV.vaeDirectory,
                metallibURL: exeV.deletingLastPathComponent().appendingPathComponent("mlx.metallib")))
        case "mlx":
            break
        default:
            throw ValidationError("unknown --vae-engine " + vaeEngine + "; use mlx | direct")
        }

        switch ditEngine {
        case "direct":
            guard textTokens == "512" else {
                throw ValidationError("--dit-engine direct assumes the 512-token joint sequence; drop --text-tokens auto")
            }
            guard padKeep == nil, !padBias else {
                throw ValidationError("--dit-engine direct does not implement the R3 pad diagnostics (--pad-keep/--pad-bias)")
            }
            guard let snapD = ModelPaths.resolveIfPresent(config: config) else {
                print("error: --dit-engine direct needs the klein snapshot")
                throw ExitCode.failure
            }
            let exeD = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
            await pipeline.setPackedDenoiser(DirectDiTPackedDenoiser(
                transformerDirectory: snapD.root.appendingPathComponent("transformer", isDirectory: true),
                metallibURL: exeD.deletingLastPathComponent().appendingPathComponent("mlx.metallib")))
        case "mlx":
            break
        default:
            throw ValidationError("unknown --dit-engine " + ditEngine + "; use mlx | direct")
        }

        guard let padContentMode = T2IRequest.PadContentMode(rawValue: padContent) else {
            throw ValidationError("Unknown --pad-content '\(padContent)'; use prompt | clean")
        }
        let outURL = output.map { URL(fileURLWithPath: $0) }

        if twoStage {
            try await runTwoStage(
                pipeline: pipeline, textTokenMode: textTokenMode,
                padContentMode: padContentMode, outURL: outURL)
            return
        }

        let request = T2IRequest(
            prompt: prompt,
            width: width,
            height: height,
            steps: steps,
            guidance: 1.0,
            seed: seed,
            outputURL: outURL,
            textTokens: textTokenMode,
            embedCache: embedCache,
            padContent: padContentMode,
            padKeep: padKeep,
            padBias: padBias
        )
        print(
            "t2i width=\(width) height=\(height) steps=\(steps) weights=\(preset.rawValue) vae=\(config.vaeDecoderVariant.rawValue) seed=\(seed.map(String.init) ?? "random") snapshot=\(await pipeline.snapshotPath ?? "?")"
        )
        let physGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0
        if max(width, height) >= 1024, physGB < 10 {
            fputs(
                "note: 1024² on ~\(String(format: "%.0f", physGB)) GB uses checkpointed DiT + tiled VAE (slower than 512²; peak ~3–4 GB MLX).\n",
                stderr
            )
        }
        do {
            let url = try await pipeline.generate(request) { progress in
                switch progress.phase {
                case .encodingText:
                    print("  phase=encoding_text")
                case .denoising:
                    print("  phase=denoising step=\(progress.step + 1)/\(progress.totalSteps)")
                case .decoding:
                    print("  phase=decoding")
                case .finished:
                    print("  phase=finished")
                default:
                    break
                }
            }
            print("wrote \(url.path)")
            try runPostGenerationEval(
                imageURL: url,
                prompt: prompt,
                referenceURL: nil,
                mode: .t2i,
                options: GenerationEvalOptions(
                    analyze: analyze || analyzeJson != nil || visionBrief,
                    analyzeJSON: analyzeJson,
                    visionBrief: visionBrief || analyze,
                    failOnPixelGate: failOnPixelGate
                )
            )
        } catch let code as ExitCode {
            throw code
        } catch let error as ImarelloError {
            print("error: \(error.localizedDescription)")
            throw ExitCode.failure
        } catch {
            print("error: \(error)")
            throw ExitCode.failure
        }
    }

    /// Lanczos upscale via CoreImage on the CPU (software renderer — no Metal
    /// contention, deterministic). Writes a PNG at exactly width×height.
    private static func lanczosUpscale(
        from src: URL, to dst: URL, width: Int, height: Int
    ) throws {
        guard let input = CIImage(contentsOf: src) else {
            throw ImarelloError.imageLoadFailed(path: src.path, reason: "not a decodable image")
        }
        let scale = CGFloat(height) / input.extent.height
        let aspect = (CGFloat(width) / input.extent.width) / scale
        guard let filter = CIFilter(name: "CILanczosScaleTransform") else {
            throw ImarelloError.notImplemented("CILanczosScaleTransform unavailable")
        }
        filter.setValue(input, forKey: kCIInputImageKey)
        filter.setValue(scale, forKey: kCIInputScaleKey)
        filter.setValue(aspect, forKey: kCIInputAspectRatioKey)
        guard let output = filter.outputImage else {
            throw ImarelloError.notImplemented("Lanczos filter produced no output")
        }
        let context = CIContext(options: [.useSoftwareRenderer: true])
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        guard let cg = context.createCGImage(output, from: rect) else {
            throw ImarelloError.notImplemented("Lanczos render failed")
        }
        guard let dest = CGImageDestinationCreateWithURL(
            dst as CFURL, UTType.png.identifier as CFString, 1, nil)
        else {
            throw ImarelloError.unsupportedWeightFormat("CGImageDestination failed for \(dst.path)")
        }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw ImarelloError.unsupportedWeightFormat("failed to write \(dst.path)")
        }
    }

    /// EXPERIMENT (engine plan Q2, MrFlow-style): compose the image at 512²
    /// where Klein 4-step anatomy is reliable, then refine at the target size
    /// with a short low-σ I2I pass. The I2I path cover-scales the 512² PNG to
    /// the target canvas itself (high-quality CGContext interpolation), so no
    /// separate upscaler is involved.
    private func runTwoStage(
        pipeline: ImarelloPipeline,
        textTokenMode: TextTokenMode,
        padContentMode: T2IRequest.PadContentMode,
        outURL: URL?
    ) async throws {
        guard width == height, width > 512 else {
            throw ValidationError(
                "--two-stage needs a square canvas larger than 512 (got \(width)x\(height))")
        }
        guard refineStrength > 0, refineStrength < 1 else {
            throw ValidationError("--refine-strength must be in (0, 1), got \(refineStrength)")
        }
        guard refineUpscale == "bicubic" || refineUpscale == "lanczos" else {
            throw ValidationError("--refine-upscale must be bicubic | lanczos, got '\(refineUpscale)'")
        }
        try validateSteps(refineSteps)
        let finalURL = outURL
            ?? AppCache.directory("outputs")
                .appendingPathComponent("t2i-2stage-\(Int(Date().timeIntervalSince1970)).png")
        let stage1URL = finalURL.deletingPathExtension().appendingPathExtension("stage1.png")
        print(
            "two-stage: compose 512² steps=\(steps) → refine \(width)² σ=\(refineStrength) steps=\(refineSteps)"
        )
        let printProgress: @Sendable (PipelineProgress) -> Void = { progress in
            switch progress.phase {
            case .encodingText: print("  phase=encoding_text")
            case .encodingImage: print("  phase=encoding_image")
            case .denoising:
                print("  phase=denoising step=\(progress.step + 1)/\(progress.totalSteps)")
            case .decoding: print("  phase=decoding")
            default: break
            }
        }
        do {
            _ = try await pipeline.generate(
                T2IRequest(
                    prompt: prompt, width: 512, height: 512, steps: steps,
                    guidance: 1.0, seed: seed, outputURL: stage1URL,
                    textTokens: textTokenMode, embedCache: embedCache,
                    padContent: padContentMode, padKeep: padKeep, padBias: padBias
                ), onProgress: printProgress)
            print("  stage1 wrote \(stage1URL.path)")
            // Optional sharper upscale: pre-scale with CoreImage Lanczos so the
            // refine gets real high-frequency content (the bicubic cover-scale
            // inside I2I softened texture 10-25 pixel-tech points in the Q gate).
            var refineSource = stage1URL
            if refineUpscale == "lanczos" {
                let upURL = finalURL.deletingPathExtension()
                    .appendingPathExtension("stage1up.png")
                try Self.lanczosUpscale(
                    from: stage1URL, to: upURL, width: width, height: height)
                refineSource = upURL
                print("  lanczos upscale wrote \(upURL.path)")
            }
            // Linear curve makes strength ≡ starting σ exactly; every identity
            // mechanism stays off.
            var refineIdentity = IdentityPreserveConfig.disabled
            refineIdentity.scheduleCurve = .linear
            let url = try await pipeline.edit(
                I2IRequest(
                    prompt: prompt, imageURL: refineSource, strength: refineStrength,
                    width: width, height: height, steps: refineSteps,
                    guidance: 1.0, seed: seed, outputURL: finalURL,
                    identity: refineIdentity, textTokens: textTokenMode,
                    embedCache: embedCache
                ), onProgress: printProgress)
            print("wrote \(url.path)")
            try runPostGenerationEval(
                imageURL: url,
                prompt: prompt,
                referenceURL: nil,
                mode: .t2i,
                options: GenerationEvalOptions(
                    analyze: analyze || analyzeJson != nil || visionBrief,
                    analyzeJSON: analyzeJson,
                    visionBrief: visionBrief || analyze,
                    failOnPixelGate: failOnPixelGate
                )
            )
        } catch let code as ExitCode {
            throw code
        } catch let error as ImarelloError {
            print("error: \(error.localizedDescription)")
            throw ExitCode.failure
        } catch {
            print("error: \(error)")
            throw ExitCode.failure
        }
    }
}

struct I2I: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "i2i",
        abstract: "Staged I2I: VAE encode → TE → DiT → VAE decode. Optional Tier-B identity (ref latents, face mask)."
    )

    @Argument(help: "Edit prompt (describe the desired change).")
    var prompt: String

    @Option(name: .long, help: "Path to reference image")
    var image: String

    @Option(help: "Denoise strength in (0, 1]; higher = more change. Color edits often need ≥0.75 (default 0.8).")
    var strength: Float = 0.8

    @Option var width: Int?
    @Option var height: Int?
    @Option var steps: Int = 4
    @Option var seed: UInt64?
    @Option(name: .long, help: "4bit (the product lock; 6/8-bit pins are non-product references)")
    var weights: String = "4bit"
    @Option(name: .long, help: "Output PNG path")
    var output: String?

    @Flag(name: .long, help: "Full Tier-B identity preset: ref latents + face preserve + identity schedule + clean pull.")
    var identity: Bool = false

    @Flag(name: .long, help: "Concatenate clean reference latents into DiT (t=10 RoPE). Implies identity stack piece.")
    var refLatents: Bool = false

    @Flag(name: .long, help: "Vision face mask for regional strength + clean pull.")
    var facePreserve: Bool = false

    @Option(name: .long, help: "Face region σ scale vs global start σ (0…1]. Lower = more face lock (default 0.5 with --identity).")
    var faceStrengthScale: Float?

    @Option(name: .long, help: "Clean-latent pull α on face after each step (0=off; default 0.2 with --identity).")
    var cleanPull: Float?

    @Option(name: .long, help: "Strength curve: color (default) | identity | linear")
    var schedule: String?

    @Option(name: .long, help: "Downsample reference tokens by this factor with --identity/--ref-latents (1 = full ref; 2 = quarter tokens, faster, identity trade-off).")
    var refDownsample: Int = 1

    @Flag(name: .long, help: "After write: run pixel eval + print report (writes sidecar .eval.json).")
    var analyze: Bool = false

    @Option(name: .long, help: "After write: write pixel eval JSON to this path (implies analysis).")
    var analyzeJson: String?

    @Flag(name: .long, help: "After write: print vision-review brief + write .vision-brief.md for agents.")
    var visionBrief: Bool = false

    @Flag(name: .long, help: "Exit 2 if pixel quality gate fails (use with --analyze).")
    var failOnPixelGate: Bool = false

    @Option(name: .long, help: "Text tokens to DiT: 512 (default, padded product path) | auto (trim pad; faster, weaker conditioning). Docs/TEXT_TOKENS.md.")
    var textTokens: String = "512"

    @Option(name: .long, help: "VAE decode graph: small-decoder (default) | full (klein pack). Encoder stays the klein AE.")
    var vaeVariant: String = "small-decoder"

    @Option(name: .long, help: "Seq length above which Q/K/V use f16 (default 512). 2048 restores f32 QKV at 512².")
    var attnF16Threshold: Int?

    @Option(name: .long, help: "4-bit Linear compute: f16 (default, scaled qmm) | f32 (reference GEMM).")
    var attnLinearCompute: String?

    @Flag(name: .long, inversion: .prefixedNo, help: "Cache prompt embeddings on disk; skips TE load+encode on repeat prompts (default on).")
    var embedCache: Bool = true

    func run() async throws {
        try ensureMLXReady()
        try validateSteps(steps)
        applyAttnF16Threshold(attnF16Threshold)
        try applyAttnLinearCompute(attnLinearCompute)
        guard let preset = WeightPreset(rawValue: weights) else {
            throw ValidationError("Unknown weights preset: \(weights)")
        }
        guard preset == .bits4 else {
            throw ValidationError(
                "--weights \(weights) is not a product path (4-bit lock, Docs/WEIGHTS.md); "
                    + "the 6/8-bit pins are reference-only and cannot load end-to-end")
        }
        guard let textTokenMode = TextTokenMode(rawValue: textTokens) else {
            throw ValidationError("Unknown --text-tokens '\(textTokens)'; use 512 | auto")
        }
        if strength <= 0 || strength > 1 {
            throw ValidationError("strength must be in (0, 1], got \(strength)")
        }

        var idCfg = identity ? IdentityPreserveConfig.identityPreset : .disabled
        if refLatents { idCfg.useReferenceLatents = true }
        if facePreserve { idCfg.facePreserve = true }
        if let fss = faceStrengthScale {
            if fss < 0 || fss > 1 {
                throw ValidationError("face-strength-scale must be in [0, 1], got \(fss)")
            }
            idCfg.faceStrengthScale = fss
            idCfg.facePreserve = true
        }
        if let cp = cleanPull {
            if cp < 0 || cp > 1 {
                throw ValidationError("clean-pull must be in [0, 1], got \(cp)")
            }
            idCfg.cleanPullAlpha = cp
            if cp > 0 { idCfg.facePreserve = true }
        }
        if let sched = schedule {
            guard let curve = StrengthScheduleCurve(rawValue: sched) else {
                throw ValidationError("Unknown schedule '\(sched)'; use color|identity|linear")
            }
            idCfg.scheduleCurve = curve
        }
        if refDownsample != 1 {
            guard refDownsample >= 1, refDownsample <= 4 else {
                throw ValidationError("ref-downsample must be 1…4, got \(refDownsample)")
            }
            idCfg.refDownsample = refDownsample
        }

        var config = ImarelloConfig.autoDetectingTier()
        config.apply(preset: preset)
        try applyVAEVariant(vaeVariant, to: &config)
        if config.vaeDecoderVariant == .smallDecoder,
           ModelPaths.resolveSmallDecoderIfPresent(config: config) == nil
        {
            print("error: Small Decoder snapshot missing")
            print("hint: \(VAEDecoderVariant.smallDecoder.downloadCommand)")
            throw ExitCode.failure
        }

        let pipeline = ImarelloPipeline(config: config)
        guard await pipeline.hasLocalSnapshot else {
            print("error: no local snapshot for \(config.modelID) @ \(config.revision)")
            print("hint: \(config.downloadCommand)")
            throw ExitCode.failure
        }

        let refURL = URL(fileURLWithPath: image)
        let request = I2IRequest(
            prompt: prompt,
            imageURL: refURL,
            strength: strength,
            width: width,
            height: height,
            steps: steps,
            guidance: 1.0,
            seed: seed,
            outputURL: output.map { URL(fileURLWithPath: $0) },
            identity: idCfg,
            textTokens: textTokenMode,
            embedCache: embedCache
        )
        let idParts: [String] = [
            idCfg.useReferenceLatents ? "ref-latents" : nil,
            idCfg.facePreserve ? "face" : nil,
            idCfg.cleanPullAlpha > 0 ? String(format: "pull=%.2f", idCfg.cleanPullAlpha) : nil,
            "sched=\(idCfg.scheduleCurve.rawValue)",
        ].compactMap { $0 }
        print(
            "i2i image=\(image) strength=\(strength) steps=\(steps) seed=\(seed.map(String.init) ?? "random") identity=[\(idParts.joined(separator: ","))]"
        )
        do {
            let url = try await pipeline.edit(request) { progress in
                switch progress.phase {
                case .encodingImage:
                    print("  phase=encoding_image")
                case .encodingText:
                    print("  phase=encoding_text")
                case .denoising:
                    print("  phase=denoising step=\(progress.step + 1)/\(progress.totalSteps)")
                case .decoding:
                    print("  phase=decoding")
                case .finished:
                    print("  phase=finished")
                default:
                    break
                }
            }
            print("wrote \(url.path)")
            try runPostGenerationEval(
                imageURL: url,
                prompt: prompt,
                referenceURL: refURL,
                mode: .i2i,
                options: GenerationEvalOptions(
                    analyze: analyze || analyzeJson != nil || visionBrief,
                    analyzeJSON: analyzeJson,
                    visionBrief: visionBrief || analyze,
                    failOnPixelGate: failOnPixelGate,
                    i2iStrength: strength
                )
            )
        } catch let code as ExitCode {
            throw code
        } catch let error as ImarelloError {
            print("error: \(error.localizedDescription)")
            throw ExitCode.failure
        } catch {
            print("error: \(error)")
            throw ExitCode.failure
        }
    }
}

struct AnalyzeImage: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "analyze-image",
        abstract: "Image quality / accuracy harness: technical metrics, optional prompt + reference compare, JSON report."
    )

    @Argument(help: "Path to generated image (PNG/JPEG).")
    var image: String

    @Option(name: .long, help: "Generation prompt (enables color/style alignment heuristics).")
    var prompt: String?

    @Option(name: .long, help: "Reference image path (I2I source or gold) for SSIM/fidelity.")
    var reference: String?

    @Option(name: .long, help: "Write JSON report to this path.")
    var json: String?

    @Flag(name: .long, help: "Print JSON to stdout instead of human text.")
    var jsonStdout: Bool = false

    @Flag(name: .long, help: "Print a vision-review brief (checklist + paths) for multimodal agents.")
    var visionBrief: Bool = false

    @Option(name: .long, help: "Vision mode for checklist: t2i | i2i (default: i2i if --reference set).")
    var visionMode: String?

    @Option(name: .long, help: "Max analysis long-side (default 1024).")
    var maxSide: Int = 1024

    @Option(name: .long, help: "I2I strength used when generating (enables strength-aware SSIM/LPIPS gates).")
    var strength: Float?

    @Flag(name: .long, help: "Skip CLIP / Vision semantic alignment (pixel-only).")
    var skipSemantic: Bool = false

    func run() async throws {
        let imageURL = URL(fileURLWithPath: image)
        let options = ImageAnalyzer.Options(
            prompt: prompt,
            referenceURL: reference.map { URL(fileURLWithPath: $0) },
            maxAnalysisSide: maxSide,
            i2iStrength: strength,
            skipSemantic: skipSemantic
        )
        do {
            let report = try ImageAnalyzer.analyze(imageURL: imageURL, options: options)
            let mode: VisionReview.Mode = {
                if let visionMode {
                    return VisionReview.Mode(rawValue: visionMode) ?? .t2i
                }
                return reference != nil ? .i2i : .t2i
            }()

            let jsonText = try ImageAnalysisReportBuilder.jsonString(report, pretty: true)

            if let jsonPath = json {
                let out = URL(fileURLWithPath: jsonPath)
                try FileManager.default.createDirectory(
                    at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
                try jsonText.write(to: out, atomically: true, encoding: .utf8)
                FileHandle.standardError.write(Data("wrote \(out.path)\n".utf8))
            }

            if visionBrief {
                print(VisionReview.agentBrief(report: report, mode: mode))
            } else if jsonStdout {
                print(jsonText)
            } else {
                print(ImageAnalysisReportBuilder.textSummary(report))
                print("")
                print("(Tip: re-run with --vision-brief for multimodal agent checklist; open the image with vision tools.)")
            }

            let fails = report.findings.filter { $0.severity == .fail }
            if !fails.isEmpty {
                fputs("quality_gate: FAIL (\(fails.map(\.code).joined(separator: ",")))\n", stderr)
                throw ExitCode(2)
            }
        } catch let code as ExitCode {
            throw code
        } catch let error as ImarelloError {
            print("error: \(error.localizedDescription)")
            throw ExitCode.failure
        } catch {
            print("error: \(error)")
            throw ExitCode.failure
        }
    }
}

// MARK: - Performance bench

struct Bench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench",
        abstract: "Performance + pressure harness (timings, MLX memory, DiT block probes). See Docs/PERF.md."
    )

    @Option(name: .long, help: "Mode: t2i | i2i | identity-i2i | pressure-map | dit-one-step | res-ladder | mem-stages | te-only | dit-steps | vae-decode | load-only")
    var mode: String = "t2i"

    @Option(name: .long, help: "Label stored in the report (e.g. baseline, cache-limit-2g).")
    var label: String = "baseline"

    @Option(name: .long, help: "Prompt for T2I / TE modes.")
    var prompt: String = "A red fox in a snowy forest at sunrise, photorealistic."

    @Option(name: .long, help: "Width (multiple of 16). Default 1024.")
    var width: Int = 1024

    @Option(name: .long, help: "Height (multiple of 16). Default 1024.")
    var height: Int = 1024

    @Option(name: .long, help: "Denoise steps.")
    var steps: Int = 4

    @Option(name: .long, help: "Seed.")
    var seed: UInt64 = 42

    @Option(name: .long, help: "Measured trials after warmup.")
    var trials: Int = 3

    @Option(name: .long, help: "Warmup runs (not included in aggregate).")
    var warmup: Int = 1

    @Option(name: .long, help: "Seconds between trials (thermal cooldown).")
    var cooldown: Double = 0

    @Option(name: .long, help: "Write JSON report to this path.")
    var json: String?

    @Option(name: .long, help: "Directory for trial PNGs.")
    var outputDir: String?

    @Option(name: .long, help: "Reference image for i2i / identity-i2i.")
    var image: String?

    @Option(name: .long, help: "I2I strength in (0, 1]. Defaults: 0.8 strength, 0.9 identity.")
    var strength: Float?

    @Flag(name: .long, help: "Enable identity stack when --image is used with a diagnostic mode.")
    var identity: Bool = false

    @Option(name: .long, help: "Optional MLX cacheLimit in bytes (e.g. 2147483648).")
    var cacheLimit: UInt64?

    @Option(name: .long, help: "LEGACY (no effect since 2026-08-18): DiT query-chunked SDPA was deleted; D=128 is always Steel. Kept for provenance.")
    var attnChunkSize: Int?

    @Option(name: .long, help: "LEGACY (no effect since 2026-08-18): see --attn-chunk-size.")
    var attnChunkThreshold: Int?

    @Option(name: .long, help: "Seq length above which Q/K/V use f16 (default 512).")
    var attnF16Threshold: Int?

    @Option(name: .long, help: "4-bit Linear compute: f16 (default, scaled qmm) | f32 (reference GEMM).")
    var attnLinearCompute: String?

    @Option(name: .long, help: "VAE mid-block attention query chunk (default 64). 0 = legacy MLXFast SDPA.")
    var vaeAttnChunk: Int?

    @Option(name: .customLong("pad-content"), help: "T2I pad content: prompt (full-window, default) | clean (TE-splice).")
    var benchPadContent: String?

    @Option(name: .long, help: "Eval/cache profile: product (default) | mid (≥16 GB bench only).")
    var evalCache: String?

    @Flag(name: .long, help: "Allow --eval-cache mid on 8 GB-class hosts (unsafe; research only).")
    var force: Bool = false

    @Option(name: .long, help: "Sequence Linear chunk size (default 512).")
    var attnLinearChunk: Int?

    @Option(name: .long, help: "Seq length above which Linear is chunked (default 1536).")
    var attnLinearThreshold: Int?

    @Option(name: .long, help: "Attention backend: mlx (default) | metal-fa | auto.")
    var attnBackend: String?

    @Option(name: .long, help: "Joint seq above which the DiT clears cache per block (default 1536).")
    var attnBlockClearThreshold: Int?

    @Option(name: .long, help: "Clear cache every N blocks when per-block clears are active (default 2).")
    var attnBlockClearInterval: Int?

    @Option(name: .long, help: "asyncEval per DiT block with a hard eval every N blocks (0 = blocking).")
    var attnAsyncEval: Int?

    @Flag(name: .long, help: "Decide attention store dtype from the joint sequence (f16 double blocks).")
    var attnJointF16: Bool = false

    @Flag(name: .long, help: "S4 EXPERIMENT: full-f16 single-stream epilogue (proj→SwiGLU→concat→to_out input). Pixel-changing; full gate before any promotion.")
    var attnF16FullEpilogue: Bool = false

    @Flag(name: .long, help: "S4 EXPERIMENT: per-tensor dynamic activation scale (amax/64) instead of the flat ÷16.")
    var attnDynamicScale: Bool = false

    @Flag(
        name: .customLong("attn-no-qkv-checkpoint"),
        help: "STAGE-1 EXPERIMENT: skip the per-block QKV checkpoint evals (fewer GPU sync boundaries; raises transient memory).")
    var attnNoQkvCheckpoint: Bool = false

    @Option(name: .long, help: "VAE tile enable threshold in latent px (128 default). WARNING: 136 untiles 1024² decode — measured Metal abort on 8 GB hosts.")
    var vaeTileThreshold: Int?

    @Option(name: .long, help: "Text tokens to DiT for t2i trials: 512 (default, pad) | auto (trim). Docs/TEXT_TOKENS.md.")
    var textTokens: String?

    @Option(name: .long, help: "VAE decode graph: small-decoder (default) | full (klein pack).")
    var vaeVariant: String = "small-decoder"

    @Option(name: .long, help: "VAE decode engine for trials: direct (product default since 2026-08-19) | mlx (previous product path).")
    var vaeEngine: String = "direct"

    @Option(name: .long, help: "DiT denoise engine for trials: direct (product default since 2026-08-19) | mlx (previous product path).")
    var ditEngine: String = "direct"

    @Option(name: .long, help: "Probe density: off | stages | denoise | blocks | max")
    var probeDensity: String = "denoise"

    @Flag(name: .long, help: "Reset MLX peak between TE / DiT / VAE phases.")
    var resetPeakEachPhase: Bool = false

    @Flag(name: .long, help: "Keep writing report when a trial errors (partial timeline).")
    var failSoft: Bool = false

    @Option(name: .long, help: "res-ladder sides as comma list (default 512,640,768,896,1024).")
    var ladder: String = "512,640,768,896,1024"

    @Flag(name: .long, help: "Run pixel quality metrics on each trial PNG.")
    var withQuality: Bool = false

    @Flag(name: .long, help: "Time Steel FA vs FFN vs processQKV glue (GPU-sync; ranking only). See Docs/PERF.md.")
    var opProfile: Bool = false

    func run() async throws {
        try ensureMLXReady()
        try validateSteps(steps)
        guard trials >= 1 else {
            throw ValidationError("trials must be at least 1, got \(trials)")
        }
        guard warmup >= 0 else {
            throw ValidationError("warmup must be at least 0, got \(warmup)")
        }

        guard let benchMode = BenchMode(rawValue: mode) else {
            print("error: unknown mode '\(mode)'. Use: \(BenchMode.allCases.map(\.rawValue).joined(separator: ", "))")
            throw ExitCode.failure
        }
        guard let density = ProbeDensity(rawValue: probeDensity) else {
            print("error: unknown probe-density '\(probeDensity)'")
            throw ExitCode.failure
        }

        let needsImage = benchMode == .i2i || benchMode == .identityI2I || identity
        if needsImage {
            guard let image, FileManager.default.fileExists(atPath: image) else {
                print("error: \(benchMode.rawValue) requires --image PATH to an existing file")
                throw ExitCode.failure
            }
        }
        if let strength, strength <= 0 || strength > 1 {
            print("error: --strength must be in (0, 1]")
            throw ExitCode.failure
        }

        if benchMode == .resLadder {
            try await runResLadder(density: density)
            return
        }

        var effectiveDensity = density
        var effectiveTrials = trials
        var effectiveWarmup = warmup
        var effectiveFailSoft = failSoft
        var effectiveReset = resetPeakEachPhase
        if benchMode == .pressureMap {
            effectiveDensity = density == .off ? .blocks : density
            effectiveTrials = 1
            effectiveWarmup = 0
            effectiveFailSoft = true
            effectiveReset = true
        }
        if benchMode == .ditOneStep {
            // Op-profile ranking wants the product eval path; extra block probes confound it.
            if !opProfile {
                effectiveDensity = density == .off ? .blocks : density
            }
            effectiveTrials = 1
            effectiveWarmup = 0
            effectiveFailSoft = true
        }

        let linearF16: Bool?
        if let raw = attnLinearCompute {
            switch raw {
            case "f16", "float16": linearF16 = true
            case "f32", "float32": linearF16 = false
            default:
                print("error: unknown --attn-linear-compute '\(raw)'. Use: f16 | f32")
                throw ExitCode.failure
            }
        } else {
            linearF16 = nil
        }

        let config = BenchConfig(
            mode: benchMode,
            label: label,
            prompt: prompt,
            width: width,
            height: height,
            steps: benchMode == .ditOneStep ? 1 : steps,
            seed: seed,
            trials: effectiveTrials,
            warmup: effectiveWarmup,
            cooldownSeconds: cooldown,
            withQuality: withQuality,
            outputDirectory: outputDir,
            cacheLimitBytes: cacheLimit,
            probeDensity: effectiveDensity,
            resetPeakEachPhase: effectiveReset,
            failSoft: effectiveFailSoft,
            attentionQueryChunkSize: attnChunkSize,
            attentionQueryChunkThreshold: attnChunkThreshold,
            attentionF16SeqThreshold: attnF16Threshold,
            attentionLinearF16: linearF16 ?? AttentionTuning.current.linearF16,
            attentionLinearChunkSize: attnLinearChunk,
            attentionLinearChunkThreshold: attnLinearThreshold,
            attentionBackend: attnBackend,
            attentionBlockClearSeqThreshold: attnBlockClearThreshold,
            attentionBlockClearInterval: attnBlockClearInterval,
            attentionAsyncEvalInterval: attnAsyncEval,
            attentionJointSeqF16: attnJointF16 ? true : nil,
            attentionF16FullEpilogue: attnF16FullEpilogue ? true : nil,
            attentionDynamicScale: attnDynamicScale ? true : nil,
            attentionQkvCheckpoint: attnNoQkvCheckpoint ? false : nil,
            vaeTileThreshold: vaeTileThreshold,
            // Persist resolved values, not nil-means-default: quality-knob
            // provenance must live in the report, not the filename label.
            textTokens: textTokens ?? TextTokenMode.full512.rawValue,
            padContent: benchPadContent ?? "prompt",
            vaeEngine: vaeEngine,
            ditEngine: ditEngine,
            vaeVariant: vaeVariant,
            evalCache: evalCache ?? EvalCachePolicy.current.profileName,
            vaeAttnChunk: vaeAttnChunk
                ?? (VAEAttentionConfig.current.useMLXFast
                    ? 0 : VAEAttentionConfig.current.queryChunkSize),
            vaeTileSize: VAETileConfig.current.tileSize,
            vaeTileOverlap: VAETileConfig.current.overlap,
            vaeTileBlend: String(describing: VAETileConfig.current.blend),
            opProfile: opProfile,
            imagePath: image,
            strength: strength,
            identity: identity || benchMode == .identityI2I
        )

        if let chunk = vaeAttnChunk {
            if chunk == 0 {
                VAEAttentionConfig.current.useMLXFast = true
            } else {
                VAEAttentionConfig.current.useMLXFast = false
                VAEAttentionConfig.current.queryChunkSize = chunk
            }
        }

        var imarelloConfig = ImarelloConfig.autoDetectingTier()
        try applyVAEVariant(vaeVariant, to: &imarelloConfig)
        if imarelloConfig.vaeDecoderVariant == .smallDecoder,
           ModelPaths.resolveSmallDecoderIfPresent(config: imarelloConfig) == nil
        {
            print("error: Small Decoder snapshot missing")
            print("hint: \(VAEDecoderVariant.smallDecoder.downloadCommand)")
            throw ExitCode.failure
        }
        if let raw = evalCache {
            guard let policy = EvalCachePolicy.named(raw) else {
                print("error: unknown --eval-cache '\(raw)'. Use: product | mid")
                throw ExitCode.failure
            }
            if let reason = policy.refusalReason(tier: imarelloConfig.tier, force: force) {
                print("error: \(reason)")
                throw ExitCode.failure
            }
            EvalCachePolicy.current = policy
        }
        defer {
            VAEAttentionConfig.resetToDefault()
            EvalCachePolicy.resetToDefault()
        }

        let pipeline = ImarelloPipeline(config: imarelloConfig)
        guard await pipeline.hasLocalSnapshot else {
            print("error: no local snapshot for \(imarelloConfig.modelID) @ \(imarelloConfig.revision)")
            print("hint: \(imarelloConfig.downloadCommand)")
            throw ExitCode.failure
        }

        switch vaeEngine {
        case "direct":
            guard vaeVariant == "small-decoder" else {
                print("error: --vae-engine direct implements the Small Decoder only")
                throw ExitCode.failure
            }
            guard let snapV = ModelPaths.resolveIfPresent(config: imarelloConfig),
                let smallDir = ModelPaths.resolveSmallDecoderIfPresent(config: imarelloConfig)
            else {
                print("error: --vae-engine direct needs the klein and Small Decoder snapshots")
                throw ExitCode.failure
            }
            let exeV = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
            await pipeline.setPackedDecoder(DirectVAEPackedDecoder(
                smallDecoderDirectory: smallDir,
                vaeDirectory: snapV.vaeDirectory,
                metallibURL: exeV.deletingLastPathComponent().appendingPathComponent("mlx.metallib")))
        case "mlx":
            break
        default:
            print("error: unknown --vae-engine \(vaeEngine); use mlx | direct")
            throw ExitCode.failure
        }

        switch ditEngine {
        case "direct":
            guard (textTokens ?? "512") == "512" else {
                print("error: --dit-engine direct assumes 512 text tokens")
                throw ExitCode.failure
            }
            guard let snapD = ModelPaths.resolveIfPresent(config: imarelloConfig) else {
                print("error: --dit-engine direct needs the klein snapshot")
                throw ExitCode.failure
            }
            let exeD = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
            await pipeline.setPackedDenoiser(DirectDiTPackedDenoiser(
                transformerDirectory: snapD.root.appendingPathComponent("transformer", isDirectory: true),
                metallibURL: exeD.deletingLastPathComponent().appendingPathComponent("mlx.metallib")))
        case "mlx":
            break
        default:
            print("error: unknown --dit-engine \(ditEngine); use mlx | direct")
            throw ExitCode.failure
        }

        let identityOn = benchMode == .identityI2I || identity
        let (ph, pw) = PressureAnalytics.packedSpatial(width: width, height: height)
        let refSeq = identityOn ? ph * pw : nil
        let analytic = PressureAnalytics.canvasStats(
            width: width, height: height, referenceSeqLen: refSeq)
        print(
            "bench start label=\(label) mode=\(benchMode.rawValue) \(width)x\(height) steps=\(config.steps) density=\(effectiveDensity.rawValue) joint_seq=\(analytic.jointSeqLen)"
        )
        print("  snapshot: \(await pipeline.snapshotPath ?? "?")")
        print("  eval_cache: \(EvalCachePolicy.current.profileName)")
        if opProfile {
            print("  op_profile: on (ranking only; extra eval syncs inflate wall time)")
        }
        print("  linear_compute: \(linearF16 == true || (linearF16 == nil && AttentionTuning.current.linearF16) ? "f16" : "f32")")
        let vaeAttn = VAEAttentionConfig.current
        print(
            "  vae_attn: \(vaeAttn.useMLXFast ? "mlxfast" : "chunk\(vaeAttn.queryChunkSize)")")
        print("  \(analytic.note)")

        do {
            let runner = BenchRunner(pipeline: pipeline, config: config)
            let report = try await runner.run()
            print(BenchReportWriter.textSummary(report))

            let jsonPath = resolveJSONPath()
            try BenchReportWriter.write(report, to: jsonPath)
            print("json: \(jsonPath.path)")

            if report.trials.contains(where: { $0.error != nil }), !effectiveFailSoft {
                throw ExitCode.failure
            }
        } catch let code as ExitCode {
            throw code
        } catch {
            print("error: \(error)")
            throw ExitCode.failure
        }
    }

    private func resolveJSONPath() -> URL {
        if let json {
            return URL(fileURLWithPath: json)
        }
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        return AppCache.directory("bench")
            .appendingPathComponent("\(label)_\(stamp).json")
    }

    /// Spawn one child process per resolution so Metal OOM does not kill the parent.
    private func runResLadder(density: ProbeDensity) async throws {
        let sides = ladder.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard !sides.isEmpty else {
            print("error: empty --ladder")
            throw ExitCode.failure
        }
        let exe = CommandLine.arguments[0]
        var childJSONs: [URL] = []
        print("res-ladder sides=\(sides) density=\(density.rawValue) (subprocess per rung)")

        // The parent only spawns and waits from here on; hand the Metal lock to
        // the serial children and mark this pid ignorable for their preflight.
        HostPreflight.releaseForSubprocesses()
        var childEnv = ProcessInfo.processInfo.environment
        childEnv["IMARELLO_IGNORE_PID"] = String(ProcessInfo.processInfo.processIdentifier)

        for side in sides {
            let out = FileManager.default.temporaryDirectory
                .appendingPathComponent("imarello-ladder-\(side)-\(UUID().uuidString).json")
            var args = [
                "bench",
                "--mode", "pressure-map",
                "--label", "\(label)-\(side)",
                "--width", "\(side)",
                "--height", "\(side)",
                "--steps", "\(steps)",
                "--seed", "\(seed)",
                "--probe-density", density == .off ? "blocks" : density.rawValue,
                "--fail-soft",
                "--json", out.path,
            ]
            if let cacheLimit {
                args += ["--cache-limit", "\(cacheLimit)"]
            }
            if let vaeAttnChunk {
                args += ["--vae-attn-chunk", "\(vaeAttnChunk)"]
            }
            if let evalCache {
                args += ["--eval-cache", evalCache]
            }
            // Forward attention tuning so a ladder can A/B any knob, not just
            // cache/VAE settings (gap noted in the 2026-08-18 engine plan).
            if let attnF16Threshold {
                args += ["--attn-f16-threshold", "\(attnF16Threshold)"]
            }
            if let attnLinearCompute {
                args += ["--attn-linear-compute", attnLinearCompute]
            }
            if let attnLinearChunk {
                args += ["--attn-linear-chunk", "\(attnLinearChunk)"]
            }
            if let attnLinearThreshold {
                args += ["--attn-linear-threshold", "\(attnLinearThreshold)"]
            }
            if let attnBackend {
                args += ["--attn-backend", attnBackend]
            }
            if let attnBlockClearThreshold {
                args += ["--attn-block-clear-threshold", "\(attnBlockClearThreshold)"]
            }
            if let attnBlockClearInterval {
                args += ["--attn-block-clear-interval", "\(attnBlockClearInterval)"]
            }
            if let attnAsyncEval {
                args += ["--attn-async-eval", "\(attnAsyncEval)"]
            }
            if attnJointF16 {
                args += ["--attn-joint-f16"]
            }
            if let vaeTileThreshold {
                args += ["--vae-tile-threshold", "\(vaeTileThreshold)"]
            }
            if force {
                args += ["--force"]
            }
            print("  rung \(side)² …")
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: exe)
            proc.arguments = args
            proc.environment = childEnv
            proc.standardOutput = FileHandle.standardOutput
            proc.standardError = FileHandle.standardError
            do {
                try proc.run()
                proc.waitUntilExit()
            } catch {
                print("  rung \(side) spawn error: \(error)")
                break
            }
            let status = proc.terminationStatus
            if FileManager.default.fileExists(atPath: out.path) {
                childJSONs.append(out)
                print("  rung \(side) exit=\(status) json=\(out.path)")
            } else {
                print("  rung \(side) exit=\(status) (no json — likely Metal abort)")
            }
            // Stop ladder after hard crash without report, or after first error with oom
            if status != 0 {
                if let last = childJSONs.last,
                   let rep = try? BenchReportWriter.loadReport(from: last),
                   rep.trials.contains(where: { $0.error != nil })
                {
                    break
                }
                if status != 0 && !FileManager.default.fileExists(atPath: out.path) {
                    break
                }
            }
        }

        // Merge summary
        var lines = ["imarello res-ladder summary"]
        for url in childJSONs {
            if let r = try? BenchReportWriter.loadReport(from: url) {
                let peak = r.aggregate.peakMlxPeakBytes.map { MemoryProbe.formatBytes(UInt64($0.max)) } ?? "?"
                let e2e = r.aggregate.e2eMs.map { String(format: "%.0fms", $0.mean) } ?? "fail"
                let err = r.trials.first?.error
                let last = r.pressure?.lastProbeBeforeFailure ?? r.trials.first?.lastProbeId ?? "-"
                let joint = r.pressure?.analytic.jointSeqLen ?? 0
                if let err {
                    lines.append(
                        "  \(r.config.width)² joint=\(joint) ERROR last_probe=\(last) — \(err.prefix(80))"
                    )
                } else {
                    lines.append(
                        "  \(r.config.width)² joint=\(joint) e2e=\(e2e) peak_mlx=\(peak) last=\(last)"
                    )
                }
            }
        }
        print(lines.joined(separator: "\n"))
        if let json {
            let summary = lines.joined(separator: "\n")
            try summary.write(to: URL(fileURLWithPath: json), atomically: true, encoding: .utf8)
            print("ladder_summary: \(json)")
        }
    }
}

struct BenchCompare: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench-compare",
        abstract: "Compare two bench JSON reports (percent deltas)."
    )

    @Argument(help: "Baseline report JSON path.")
    var baseline: String

    @Argument(help: "Candidate report JSON path.")
    var candidate: String

    func run() async throws {
        do {
            let a = try BenchReportWriter.loadReport(from: URL(fileURLWithPath: baseline))
            let b = try BenchReportWriter.loadReport(from: URL(fileURLWithPath: candidate))
            print(BenchReportWriter.compareText(baseline: a, candidate: b))
        } catch {
            print("error: \(error)")
            throw ExitCode.failure
        }
    }
}

struct DirectSpike: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "direct-spike",
        abstract: "RESEARCH (Stage 2, research/bare-metal): direct-dispatch spikes against the MLX metallib."
    )

    @Option(name: .long, help: "Text-encoder snapshot directory (default: pinned snapshot).")
    var teDir: String?

    @Option(name: .long, help: "Spike stage: qmm (A) | layer (B) | forward (C).")
    var stage: String = "forward"

    @Option(name: .long, help: "Sequence length for --stage forward (512 = full window, 30 = splice regime).")
    var seq: Int = 512

    func run() async throws {
        try ensureMLXReady()
        let config = ImarelloConfig.autoDetectingTier()
        let dir: URL
        if let teDir {
            dir = URL(fileURLWithPath: teDir)
        } else {
            dir = try ModelPaths.resolveOrThrow(config: config).textEncoderDirectory
        }
        let exe = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        let metallib = exe.deletingLastPathComponent().appendingPathComponent("mlx.metallib")
        switch stage {
        case "qmm":
            print(try DirectQmmSpike.run(teDirectory: dir, metallibURL: metallib))
        case "layer":
            print(try DirectTELayerSpike.run(teDirectory: dir, metallibURL: metallib))
        case "forward":
            print(try DirectTEForward.run(teDirectory: dir, metallibURL: metallib, seqLen: seq))
        case "vae-resnet":
            let snapV = try ModelPaths.resolveOrThrow(config: config)
            _ = snapV
            let sdDir = ModelPaths.smallDecoderSnapshotRoot(modelsDirectory: config.modelsDirectory)
            print(try DirectVAEResnetSpike.run(
                smallDecoderFile: sdDir.appendingPathComponent("small_decoder.safetensors"),
                metallibURL: metallib))
        case "vae-mid":
            let sdDirM = ModelPaths.smallDecoderSnapshotRoot(modelsDirectory: config.modelsDirectory)
            print(try DirectVAEMidSpike.run(
                smallDecoderFile: sdDirM.appendingPathComponent("small_decoder.safetensors"),
                metallibURL: metallib))
        case "vae-decode":
            let snapD = try ModelPaths.resolveOrThrow(config: config)
            let sdDirD = ModelPaths.smallDecoderSnapshotRoot(modelsDirectory: config.modelsDirectory)
            print(try await DirectVAEDecodeSpike.run(
                snapshot: snapD, smallDecoderDirectory: sdDirD,
                metallibURL: metallib, config: config))
        case "vae-tiled":
            let snapT = try ModelPaths.resolveOrThrow(config: config)
            let sdDirT = ModelPaths.smallDecoderSnapshotRoot(modelsDirectory: config.modelsDirectory)
            print(try await DirectVAETiledSpike.run(
                snapshot: snapT, smallDecoderDirectory: sdDirT, metallibURL: metallib))
        case "vae-profile":
            let sdDirP = ModelPaths.smallDecoderSnapshotRoot(modelsDirectory: config.modelsDirectory)
            print(try DirectVAEProfileSpike.run(
                smallDecoderDirectory: sdDirP, metallibURL: metallib))
        case "vae-micro":
            let sdDirU = ModelPaths.smallDecoderSnapshotRoot(modelsDirectory: config.modelsDirectory)
            print(try DirectVAEMicroSpike.run(
                smallDecoderDirectory: sdDirU, metallibURL: metallib))
        case "vae-conv":
            print(try DirectVAESpike.run(metallibURL: metallib))
        case "conditioning":
            let snapC = try ModelPaths.resolveOrThrow(config: config)
            print(try await DirectConditionerSpike.run(snapshot: snapC, metallibURL: metallib))
        case "dit-denoise":
            let snap4 = try ModelPaths.resolveOrThrow(config: config)
            print(try await DirectDiTDenoiseSpike.run(
                snapshot: snap4, metallibURL: metallib, width: seq >= 1024 ? 1024 : 512,
                height: seq >= 1024 ? 1024 : 512))
        case "dit-step":
            let snap3 = try ModelPaths.resolveOrThrow(config: config)
            print(try DirectDiTStepSpike.run(
                transformerDirectory: snap3.root.appendingPathComponent("transformer", isDirectory: true),
                metallibURL: metallib))
        case "dit-double":
            let snap2 = try ModelPaths.resolveOrThrow(config: config)
            print(try DirectDiTDoubleBlockSpike.run(
                transformerDirectory: snap2.root.appendingPathComponent("transformer", isDirectory: true),
                metallibURL: metallib))
        case "dit-block":
            let snap = try ModelPaths.resolveOrThrow(config: config)
            print(try DirectDiTBlockSpike.run(
                transformerDirectory: snap.root.appendingPathComponent("transformer", isDirectory: true),
                metallibURL: metallib))
        case "encode":
            let snap = try ModelPaths.resolveOrThrow(config: config)
            print(try await DirectTEEncodeSpike.run(
                snapshot: snap, metallibURL: metallib,
                prompt: "A red fox sitting in a snowy forest clearing at golden hour, professional wildlife photography"))
        default:
            throw ValidationError("unknown --stage '\(stage)'; use qmm | layer | forward | encode | dit-block | dit-double | dit-step | dit-denoise (--seq 512|1024)")
        }
    }
}

struct DirectGenerate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "direct-generate",
        abstract: "RESEARCH (Stage 2): T2I on the fully bespoke pipeline — direct TE + conditioner + DiT + VAE. Multiple prompts reuse resident engines."
    )

    @Argument(help: "One or more prompts (each becomes an image).")
    var prompts: [String]

    @Option var width: Int = 512
    @Option var height: Int = 512
    @Option var seed: UInt64 = 42
    @Option(name: .long, help: "Output PNG path (index appended for multiple prompts).")
    var output: String

    @Flag(name: .long, help: "Run pixel eval after each write.")
    var analyze: Bool = false

    func run() async throws {
        guard !prompts.isEmpty else { throw ValidationError("at least one prompt required") }
        try ensureMLXReady()
        let config = ImarelloConfig.autoDetectingTier()
        let snap = try ModelPaths.resolveOrThrow(config: config)
        let exe = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        let metallib = exe.deletingLastPathComponent().appendingPathComponent("mlx.metallib")

        let tBuild = CFAbsoluteTimeGetCurrent()
        let pipeline = try await DirectPipeline(snapshot: snap, metallibURL: metallib, config: config)
        print(String(format: "direct-generate: pipeline built in %.0f ms (engines stay resident)",
                     (CFAbsoluteTimeGetCurrent() - tBuild) * 1000))

        let base = URL(fileURLWithPath: output)
        for (idx, prompt) in prompts.enumerated() {
            let outURL: URL
            if prompts.count == 1 {
                outURL = base
            } else {
                outURL = base.deletingPathExtension()
                    .appendingPathExtension("p\(idx).png")
            }
            let tm = try await pipeline.generate(
                prompt: prompt, width: width, height: height, seed: seed, outputURL: outURL)
            print(String(
                format: "direct-generate[%d]: te %.0f ms · dit %.0f ms · vae %.0f ms · total %.0f ms → %@",
                idx, tm.teMS, tm.ditMS, tm.vaeMS, tm.totalMS, outURL.lastPathComponent))
            if analyze {
                _ = try runPostGenerationEval(
                    imageURL: outURL, prompt: prompt, referenceURL: nil, mode: .t2i,
                    options: .init(analyze: true, analyzeJSON: nil, visionBrief: false,
                                   failOnPixelGate: false, i2iStrength: nil))
            }
        }
    }
}
