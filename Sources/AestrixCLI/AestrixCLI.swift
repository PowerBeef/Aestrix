import ArgumentParser
import Foundation
import AestrixCore
import AestrixRuntime
import AestrixEval

@main
struct AestrixCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "aestrix",
        abstract: "Aestrix — FLUX.2 klein-4B MLX runtime (low-RAM staged, prequant).",
        subcommands: [
            MemSelfTest.self, Info.self, Schedule.self,
            LoadDiT.self, LoadVAE.self, LoadTE.self,
            EncodePrompt.self, T2I.self, I2I.self,
            AnalyzeImage.self,
        ]
    )
}

/// Call at the start of any command that may touch MLX.
func ensureMLXReady() {
    MLXBootstrap.ensureMetallibBesideExecutable()
}

// MARK: - Post-generation evaluation (pixel harness + vision brief)

/// Shared options for `t2i` / `i2i` quality workflow (see Docs/EVAL_WORKFLOW.md).
struct GenerationEvalOptions: Sendable {
    var analyze: Bool
    var analyzeJSON: String?
    var visionBrief: Bool
    var failOnPixelGate: Bool

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
            maxAnalysisSide: 1024
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
        ensureMLXReady()
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
        ensureMLXReady()
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
        ensureMLXReady()
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
        ensureMLXReady()
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

    @Option var width: Int = 512
    @Option var height: Int = 512
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

    @Option var width: Int = 512
    @Option var height: Int = 512
    @Option var steps: Int = 4
    @Option var seed: UInt64?
    @Option(name: .long, help: "3bit|4bit|6bit|8bit")
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

    func run() async throws {
        ensureMLXReady()
        guard let preset = WeightPreset(rawValue: weights) else {
            throw ValidationError("Unknown weights preset: \(weights)")
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
            outputURL: outURL
        )
        print(
            "t2i width=\(width) height=\(height) steps=\(steps) seed=\(seed.map(String.init) ?? "random") snapshot=\(await pipeline.snapshotPath ?? "?")"
        )
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
        abstract: "Staged strength I2I: VAE encode → TE → DiT (from strength) → VAE decode → PNG."
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

    @Flag(name: .long, help: "After write: run pixel eval + print report (writes sidecar .eval.json).")
    var analyze: Bool = false

    @Option(name: .long, help: "After write: write pixel eval JSON to this path (implies analysis).")
    var analyzeJson: String?

    @Flag(name: .long, help: "After write: print vision-review brief + write .vision-brief.md for agents.")
    var visionBrief: Bool = false

    @Flag(name: .long, help: "Exit 2 if pixel quality gate fails (use with --analyze).")
    var failOnPixelGate: Bool = false

    func run() async throws {
        ensureMLXReady()
        guard let preset = WeightPreset(rawValue: weights) else {
            throw ValidationError("Unknown weights preset: \(weights)")
        }
        if strength <= 0 || strength > 1 {
            throw ValidationError("strength must be in (0, 1], got \(strength)")
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
            outputURL: output.map { URL(fileURLWithPath: $0) }
        )
        print(
            "i2i image=\(image) strength=\(strength) steps=\(steps) seed=\(seed.map(String.init) ?? "random")"
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

    func run() async throws {
        let imageURL = URL(fileURLWithPath: image)
        let options = ImageAnalyzer.Options(
            prompt: prompt,
            referenceURL: reference.map { URL(fileURLWithPath: $0) },
            maxAnalysisSide: maxSide
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
