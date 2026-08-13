import ArgumentParser
import Foundation
import AestrixCore
import AestrixRuntime
import AestrixEval
import AestrixBench

@main
struct AestrixCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "aestrix",
        abstract: "Aestrix — FLUX.2 klein-4B MLX runtime (low-RAM staged, prequant).",
        subcommands: [
            MemSelfTest.self, Info.self, Schedule.self,
            LoadDiT.self, LoadVAE.self, LoadTE.self,
            EncodePrompt.self, T2I.self, I2I.self, Session.self,
            AnalyzeImage.self, Bench.self, BenchCompare.self,
            DiTCompileSpike.self,
        ]
    )
}

/// Call at the start of any command that may touch MLX.
func ensureMLXReady(forceHeadroom: Bool = false) throws {
    MLXBootstrap.ensureMetallibBesideExecutable()
    do {
        try HostPreflight.acquireForCLI(force: forceHeadroom)
    } catch let error as AestrixError {
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
        let config = AestrixConfig.autoDetectingTier()
        let pipeline = AestrixPipeline(config: config)
        let phys = ProcessInfo.processInfo.physicalMemory
        print("Aestrix")
        print("  physical_memory: \(MemoryProbe.formatBytes(phys))")
        print("  tier: \(config.tier.rawValue)")
        print("  max_side: \(config.maxSide)")
        print("  memory_policy: \(config.memoryPolicy.rawValue)")
        print("  weight_preset: \(config.weightPreset.rawValue)")
        print("  model_id: \(config.modelID)")
        print("  peak_budget: \(MemoryProbe.formatBytes(config.tier.peakBudgetBytes))")
        if let path = await pipeline.snapshotPath {
            print("  snapshot: \(path)")
            print("  snapshot_ready: true")
        } else {
            print("  snapshot_ready: false")
            print("  snapshot_hint: download to ~/Library/Caches/Aestrix/models/<org>--<name>/")
        }
    }
}

struct LoadDiT: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "load-dit",
        abstract: "Load quantized DiT weights from the local snapshot (requires Metal)."
    )

    func run() async throws {
        try ensureMLXReady()
        let config = AestrixConfig.autoDetectingTier()
        let pipeline = AestrixPipeline(config: config)
        guard await pipeline.hasLocalSnapshot else {
            print("error: no local snapshot for \(config.modelID)")
            throw ExitCode.failure
        }
        print("loading DiT from \(await pipeline.snapshotPath ?? "?") …")
        do {
            let leaves = try await pipeline.loadDiT()
            print("DiT load OK parameter_leaves=\(leaves)")
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

    func run() async throws {
        try ensureMLXReady()
        let config = AestrixConfig.autoDetectingTier()
        let pipeline = AestrixPipeline(config: config)
        guard await pipeline.hasLocalSnapshot else {
            print("error: no local snapshot for \(config.modelID)")
            throw ExitCode.failure
        }
        print("loading VAE from \(await pipeline.snapshotPath ?? "?") …")
        do {
            let leaves = try await pipeline.loadVAE()
            print("VAE load OK parameter_leaves=\(leaves)")
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
        let config = AestrixConfig.autoDetectingTier()
        let pipeline = AestrixPipeline(config: config)
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
        let config = AestrixConfig.autoDetectingTier()
        let pipeline = AestrixPipeline(config: config)
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
        var config = AestrixConfig.autoDetectingTier()
        config.memoryPolicy = policy
        // Avoid loading real weights / Metal during residency smoke test.
        config.modelsDirectory = URL(fileURLWithPath: "/tmp/aestrix-no-snapshot", isDirectory: true)

        let pipeline = AestrixPipeline(config: config)
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
    @Option(name: .long, help: "3bit|4bit|6bit|8bit (default 4bit — product lock)")
    var weights: String = "4bit"
    @Option(name: .long, help: "Output PNG path (default: ~/Library/Caches/Aestrix/outputs/…)")
    var output: String?

    @Flag(name: .long, help: "After write: run pixel eval + print report (writes sidecar .eval.json).")
    var analyze: Bool = false

    @Option(name: .long, help: "After write: write pixel eval JSON to this path (implies analysis).")
    var analyzeJson: String?

    @Flag(name: .long, help: "After write: print vision-review brief + write .vision-brief.md for agents.")
    var visionBrief: Bool = false

    @Flag(name: .long, help: "Exit 2 if pixel quality gate fails (use with --analyze).")
    var failOnPixelGate: Bool = false

    @Option(name: .long, help: "Text tokens to DiT: 512 (default, padded) | auto (trim to prompt length; experimental).")
    var textTokens: String = "512"

    @Flag(name: .long, inversion: .prefixedNo, help: "Cache prompt embeddings on disk; skips TE load+encode on repeat prompts (default on).")
    var embedCache: Bool = true

    func run() async throws {
        try ensureMLXReady()
        guard let preset = WeightPreset(rawValue: weights) else {
            throw ValidationError("Unknown weights preset: \(weights)")
        }
        guard let textTokenMode = TextTokenMode(rawValue: textTokens) else {
            throw ValidationError("Unknown --text-tokens '\(textTokens)'; use 512 | auto")
        }
        var config = AestrixConfig.autoDetectingTier()
        config.weightPreset = preset
        config.modelID = preset.defaultModelID

        let pipeline = AestrixPipeline(config: config)
        guard await pipeline.hasLocalSnapshot else {
            print("error: no local snapshot for \(config.modelID)")
            print("hint: download mlx-community/FLUX.2-Klein-4B-4bit into Aestrix models cache")
            throw ExitCode.failure
        }

        let outURL = output.map { URL(fileURLWithPath: $0) }
        let request = T2IRequest(
            prompt: prompt,
            width: width,
            height: height,
            steps: steps,
            guidance: 1.0,
            seed: seed,
            outputURL: outURL,
            textTokens: textTokenMode,
            embedCache: embedCache
        )
        print(
            "t2i width=\(width) height=\(height) steps=\(steps) weights=\(preset.rawValue) seed=\(seed.map(String.init) ?? "random") snapshot=\(await pipeline.snapshotPath ?? "?")"
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
        } catch let error as AestrixError {
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
    @Option(name: .long, help: "3bit|4bit|6bit|8bit")
    var weights: String = "4bit"
    @Option(name: .long, help: "Output PNG path")
    var output: String?

    @Flag(name: .long, help: "Full Tier-B identity preset: ref latents + face preserve + identity schedule + clean pull.")
    var identity: Bool = false

    @Flag(name: .long, help: "Concatenate clean reference latents into DiT (t=10 RoPE). Implies identity stack piece.")
    var refLatents: Bool = false

    @Flag(name: .long, help: "Vision face mask for regional strength + clean pull.")
    var facePreserve: Bool = false

    @Option(name: .long, help: "Face region σ scale vs global start σ (0…1]. Lower = more face lock (default 0.45 with --identity).")
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

    @Option(name: .long, help: "Text tokens to DiT: 512 (default, padded) | auto (trim to prompt length; experimental).")
    var textTokens: String = "512"

    @Flag(name: .long, inversion: .prefixedNo, help: "Cache prompt embeddings on disk; skips TE load+encode on repeat prompts (default on).")
    var embedCache: Bool = true

    func run() async throws {
        try ensureMLXReady()
        guard let preset = WeightPreset(rawValue: weights) else {
            throw ValidationError("Unknown weights preset: \(weights)")
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

        var config = AestrixConfig.autoDetectingTier()
        config.weightPreset = preset
        config.modelID = preset.defaultModelID

        let pipeline = AestrixPipeline(config: config)
        guard await pipeline.hasLocalSnapshot else {
            print("error: no local snapshot for \(config.modelID)")
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
        } catch let error as AestrixError {
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
        } catch let error as AestrixError {
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

    @Flag(name: .long, help: "Allow 1024² I2I/identity bench on 8 GB after swap is 0.")
    var forceHeadroom: Bool = false

    @Option(name: .long, help: "Optional MLX cacheLimit in bytes (e.g. 2147483648).")
    var cacheLimit: UInt64?

    @Option(name: .long, help: "DiT SDPA query chunk size (default 256). Sweep: 128|256|512.")
    var attnChunkSize: Int?

    @Option(name: .long, help: "Seq length above which query-chunked SDPA is used (default 1536).")
    var attnChunkThreshold: Int?

    @Option(name: .long, help: "Seq length above which Q/K/V use f16 (default 2048).")
    var attnF16Threshold: Int?

    @Option(name: .long, help: "Sequence Linear chunk size (default 512).")
    var attnLinearChunk: Int?

    @Option(name: .long, help: "Seq length above which Linear is chunked (default 1536).")
    var attnLinearThreshold: Int?

    @Option(name: .long, help: "Attention backend: mlx | metal-fa | auto (default auto).")
    var attnBackend: String?

    @Option(name: .long, help: "Joint seq above which the DiT clears cache per block (default 1536).")
    var attnBlockClearThreshold: Int?

    @Option(name: .long, help: "Clear cache every N blocks when per-block clears are active (default 1).")
    var attnBlockClearInterval: Int?

    @Option(name: .long, help: "Text tokens to DiT for t2i trials: 512 (default) | auto (trim experiment).")
    var textTokens: String?

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

    func run() async throws {
        try ensureMLXReady(forceHeadroom: forceHeadroom)

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
        let tier = DeviceTier.detect()
        if needsImage, tier == .low, max(width, height) >= 1024, !forceHeadroom {
            print("error: 1024² I2I/identity bench on 8 GB-class hosts needs --force-headroom")
            print("  (sysctl vm.swapusage must be 0; prefer --width 512 first)")
            throw ExitCode.failure
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
            effectiveDensity = density == .off ? .blocks : density
            effectiveTrials = 1
            effectiveWarmup = 0
            effectiveFailSoft = true
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
            attentionLinearChunkSize: attnLinearChunk,
            attentionLinearChunkThreshold: attnLinearThreshold,
            attentionBackend: attnBackend,
            attentionBlockClearSeqThreshold: attnBlockClearThreshold,
            attentionBlockClearInterval: attnBlockClearInterval,
            textTokens: textTokens,
            imagePath: image,
            strength: strength,
            identity: identity || benchMode == .identityI2I
        )

        let aestrixConfig = AestrixConfig.autoDetectingTier()
        let pipeline = AestrixPipeline(config: aestrixConfig)
        guard await pipeline.hasLocalSnapshot else {
            print("error: no local snapshot for \(aestrixConfig.modelID)")
            print("  expected under ~/Library/Caches/Aestrix/models/")
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
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        return caches
            .appendingPathComponent("Aestrix/bench", isDirectory: true)
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

        for side in sides {
            let out = FileManager.default.temporaryDirectory
                .appendingPathComponent("aestrix-ladder-\(side)-\(UUID().uuidString).json")
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
            print("  rung \(side)² …")
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: exe)
            proc.arguments = args
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
        var lines = ["aestrix res-ladder summary"]
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
