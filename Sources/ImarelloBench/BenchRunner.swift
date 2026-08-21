import Foundation
import ImarelloCore
import ImarelloRuntime
import ImarelloDiT
import ImarelloVAE
import ImarelloEval

/// Runs multi-trial generation / micro-benchmarks with full stage + memory metrics.
public actor BenchRunner {
    public let pipeline: ImarelloPipeline
    public let config: BenchConfig

    public init(pipeline: ImarelloPipeline, config: BenchConfig) {
        self.pipeline = pipeline
        self.config = config
    }

    private func applyAttentionTuningFromConfig() {
        var t = AttentionTuning.default
        if let v = config.attentionQueryChunkSize { t.queryChunkSize = v }
        if let v = config.attentionQueryChunkThreshold { t.queryChunkThreshold = v }
        if let v = config.attentionF16SeqThreshold { t.f16SeqThreshold = v }
        if let v = config.attentionLinearF16 { t.linearF16 = v }
        if let v = config.attentionLinearChunkSize { t.linearChunkSize = v }
        if let v = config.attentionLinearChunkThreshold { t.linearChunkThreshold = v }
        if let raw = config.attentionBackend, let backend = AttentionBackend(rawValue: raw) {
            t.backend = backend
        }
        if let v = config.attentionBlockClearSeqThreshold { t.blockCacheClearSeqThreshold = v }
        if let v = config.attentionBlockClearInterval {
            t.blockCacheClearInterval = max(1, v)
        } else {
            t.blockCacheClearInterval = EvalCachePolicy.current.blockCacheClearInterval
        }
        if let v = config.attentionAsyncEvalInterval { t.asyncEvalInterval = max(0, v) }
        if let v = config.attentionJointSeqF16 { t.jointSeqF16 = v }
        if let v = config.attentionF16FullEpilogue { t.linearF16FullEpilogue = v }
        if let v = config.attentionDynamicScale { t.linearDynamicScale = v }
        if let v = config.attentionQkvCheckpoint { t.qkvCheckpoint = v }
        AttentionTuning.current = t
        if let v = config.vaeTileThreshold {
            VAETileConfig.current.enabledThreshold = max(1, v)
        } else {
            VAETileConfig.current = .default
        }
    }

    public func run() async throws -> BenchReport {
        try await MetalWorkLease.withLease {
            try await self.runUnderLease()
        }
    }

    private func runUnderLease() async throws -> BenchReport {
        let previousCacheLimit = MemorySampler.cacheLimit
        let previousAttention = AttentionTuning.current
        let previousVAETile = VAETileConfig.current
        MemorySampler.applyCacheLimit(config.cacheLimitBytes)
        applyAttentionTuningFromConfig()
        defer {
            MemorySampler.applyCacheLimit(previousCacheLimit)
            AttentionTuning.current = previousAttention
            VAETileConfig.current = previousVAETile
        }

        let system = SystemInfo.snapshot(
            mlxCacheLimit: MemorySampler.cacheLimit,
            mlxMemoryLimit: MemorySampler.memoryLimit
        )

        var trials: [BenchTrial] = []
        let effectiveWarmup = config.mode == .pressureMap || config.mode == .ditOneStep
            ? 0 : config.warmup
        let effectiveTrials = config.mode == .pressureMap || config.mode == .ditOneStep
            ? max(1, config.trials) : config.trials
        let runs = effectiveWarmup + effectiveTrials

        for runIndex in 0 ..< runs {
            let isWarmup = runIndex < effectiveWarmup
            let trialIndex = isWarmup ? -(effectiveWarmup - runIndex) : runIndex - effectiveWarmup
            let cold = runIndex == 0

            if config.cooldownSeconds > 0, runIndex > 0 {
                try await Task.sleep(nanoseconds: UInt64(config.cooldownSeconds * 1_000_000_000))
            }

            MemorySampler.clearCache()
            MemorySampler.resetPeak()
            await pipeline.purge()

            let trial = await executeTrial(index: trialIndex, cold: cold)
            if !isWarmup {
                trials.append(trial)
            } else if let err = trial.error, !config.failSoft {
                throw BenchError.warmupFailed(err)
            }
        }

        let aggregate = MetricsAggregator.aggregate(trials: trials)
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime]

        let (ph, pw) = PressureAnalytics.packedSpatial(
            width: config.width, height: config.height)
        let refSeq = identityEnabled ? ph * pw : nil
        let analytic = PressureAnalytics.canvasStats(
            width: config.width, height: config.height, referenceSeqLen: refSeq)
        let failingTrial = trials.first { $0.error != nil }
        let reportTrial = failingTrial ?? trials.last
        let timeline = reportTrial?.memorySamples ?? []
        let (byActive, byDelta) = PressureAnalytics.rank(
            samples: timeline,
            recommendedWS: system.recommendedMaxWorkingSetBytes
        )
        let lastProbe = reportTrial?.lastProbeId
        let failureText = failingTrial?.error?.lowercased()
        let pressure = PressureReport(
            timeline: timeline,
            rankedByActive: byActive,
            rankedByDelta: byDelta,
            analytic: analytic,
            lastProbeBeforeFailure: failingTrial == nil ? nil : lastProbe,
            oom: failureText?.contains("memory") == true
                || failureText?.contains("oom") == true,
            phasePeaks: PressureAnalytics.phasePeaks(samples: timeline),
            probeDensity: config.probeDensity.rawValue
        )

        let pipelineConfig = await pipeline.config
        let pipelineSnapshot = await pipeline.snapshot
        let metallib = MetallibVerification.resolveFromBundles().map {
            MetallibVerification.verify(url: $0)
        }
        let cacheNotes = trials.flatMap(\.memorySamples).compactMap(\.note)
            .filter { $0.hasPrefix("embed_cache=") }
        let cacheHitState: String? = cacheNotes.isEmpty
            ? (config.embedCachePolicy == "disabled" ? "disabled" : "not-observed")
            : Array(Set(cacheNotes)).sorted().joined(separator: ",")
        let successfulSamples = trials.filter { $0.error == nil }.count
        let failedSamples = trials.count - successfulSamples
        let contaminated = trials.contains {
            $0.hostBefore?.contaminated == true || $0.hostAfter?.contaminated == true
        }
        let provenance = BenchProvenance(
            modelRevision: pipelineConfig.revision,
            detectedSnapshotRevision: pipelineSnapshot?.detectedRevision,
            tokenizerFingerprint: pipelineSnapshot.map {
                PromptEmbedCache.tokenizerFingerprint(directory: $0.tokenizerDirectory)
            },
            metallibPath: metallib?.path,
            metallibByteCount: metallib?.byteCount,
            metallibRevision: metallib.flatMap {
                MetallibVerification.recordedRevision(
                    for: URL(fileURLWithPath: $0.path))
            },
            teEngine: config.teEngine,
            cachePolicy: config.embedCachePolicy,
            cacheHitState: cacheHitState,
            buildRevision: system.imarelloGitSha,
            buildDirty: system.imarelloGitDirty,
            successfulSamples: successfulSamples,
            failedSamples: failedSamples,
            contaminated: contaminated)

        return BenchReport(
            schemaVersion: BenchReport.currentSchema,
            label: config.label,
            createdAt: iso.string(from: Date()),
            system: system,
            config: config,
            trials: trials,
            aggregate: aggregate,
            pressure: pressure,
            provenance: provenance
        )
    }

    // MARK: - Trial

    private func executeTrial(index: Int, cold: Bool) async -> BenchTrial {
        let density: ProbeDensity = {
            switch config.mode {
            case .pressureMap:
                return config.probeDensity == .off ? .blocks : config.probeDensity
            case .ditOneStep:
                // Keep --op-profile on the product eval path; block probes confound GPU-sync splits.
                if config.opProfile { return config.probeDensity }
                return config.probeDensity == .off ? .blocks : config.probeDensity
            default:
                return config.probeDensity
            }
        }()
        let collector = TraceCollector(
            density: density,
            resetPeakEachPhase: config.resetPeakEachPhase
        )
        let e2eStart = CFAbsoluteTimeGetCurrent()

        let hostBefore = HostContention.capture()
        if config.opProfile {
            DiTOpProfile.begin()
        } else {
            DiTOpProfile.reset()
        }
        do {
            let outputPath: String?
            switch config.mode {
            case .t2i, .pressureMap:
                outputPath = try await runT2I(collector: collector, steps: config.steps)
            case .i2i:
                outputPath = try await runI2I(
                    collector: collector, steps: config.steps, identity: identityEnabled)
            case .identityI2I:
                outputPath = try await runI2I(
                    collector: collector, steps: config.steps, identity: true)
            case .ditOneStep:
                outputPath = try await runT2I(collector: collector, steps: 1)
            case .memStages:
                try await runMemStages(collector: collector)
                outputPath = nil
            case .teOnly:
                try await runTEOnly(collector: collector)
                outputPath = nil
            case .ditSteps:
                outputPath = try await runT2I(collector: collector, steps: config.steps)
            case .vaeDecode:
                try await runVAEDecode(collector: collector)
                outputPath = nil
            case .loadOnly:
                try await runLoadOnly(collector: collector)
                outputPath = nil
            case .resLadder:
                // Handled by CLI subprocess orchestrator; fall through to single t2i.
                outputPath = try await runT2I(collector: collector, steps: config.steps)
            }

            let e2eMs = (CFAbsoluteTimeGetCurrent() - e2eStart) * 1000
            let hostAfter = HostContention.capture()
            return finishTrial(
                index: index, cold: cold, e2eMs: e2eMs, collector: collector,
                outputPath: outputPath, hostBefore: hostBefore, hostAfter: hostAfter, error: nil)
        } catch {
            let e2eMs = (CFAbsoluteTimeGetCurrent() - e2eStart) * 1000
            let hostAfter = HostContention.capture()
            return finishTrial(
                index: index, cold: cold, e2eMs: e2eMs, collector: collector,
                outputPath: nil, hostBefore: hostBefore, hostAfter: hostAfter,
                error: String(describing: error))
        }
    }

    private func finishTrial(
        index: Int,
        cold: Bool,
        e2eMs: Double,
        collector: TraceCollector,
        outputPath: String?,
        hostBefore: HostContentionSnapshot?,
        hostAfter: HostContentionSnapshot?,
        error: String?
    ) -> BenchTrial {
        let timings = collector.buildTimings(e2eMs: e2eMs)
        let samples = collector.memorySamples
        let peaks = collector.peaks()
        let opProfile: DiTOpProfileReport?
        if config.opProfile {
            let denoiseSec: Double?
            if let total = timings.denoiseTotal {
                denoiseSec = total / 1000
            } else if !timings.denoiseSteps.isEmpty {
                denoiseSec = timings.denoiseSteps.reduce(0, +) / 1000
            } else {
                denoiseSec = nil
            }
            opProfile = DiTOpProfile.snapshot(denoiseSeconds: denoiseSec)
            DiTOpProfile.reset()
        } else {
            opProfile = nil
        }

        var qualityScore: Float?
        var qualityColor: Bool?
        var qualityReferenceSSIM: Float?
        var qualityFidelity: Float?
        var qualityFaceSSIM: Float?
        var qualityFaceFidelity: Float?
        var generatedFaceCount: Int?
        var referenceFaceCount: Int?
        var qualityStatus: QualityEvaluationStatus = config.withQuality
            ? .notApplicable : .notRequested
        var qualityError: String?
        if config.withQuality, let path = outputPath, error == nil {
            do {
                let q = try qualityFromPNG(path: path)
                qualityScore = q.score
                qualityColor = q.colorMatch
                qualityReferenceSSIM = q.referenceSSIM
                qualityFidelity = q.fidelityScore
                qualityFaceSSIM = q.faceReferenceSSIM
                qualityFaceFidelity = q.faceFidelityScore
                generatedFaceCount = q.generatedFaceCount
                referenceFaceCount = q.referenceFaceCount
                qualityStatus = .succeeded
            } catch {
                qualityStatus = .failed
                qualityError = String(describing: error)
            }
        }

        return BenchTrial(
            index: index,
            cold: cold,
            timingsMs: timings,
            memorySamples: samples,
            peakRssBytes: peaks.rss,
            peakMlxActiveBytes: peaks.mlxActive,
            peakMlxPeakBytes: peaks.mlxPeak,
            peakMlxCacheBytes: peaks.mlxCache,
            outputPath: outputPath,
            qualityTechnicalScore: qualityScore,
            qualityColorMatch: qualityColor,
            qualityReferenceSSIM: qualityReferenceSSIM,
            qualityFidelityScore: qualityFidelity,
            qualityFaceReferenceSSIM: qualityFaceSSIM,
            qualityFaceFidelityScore: qualityFaceFidelity,
            qualityStatus: qualityStatus,
            qualityError: qualityError,
            generatedFaceCount: generatedFaceCount,
            referenceFaceCount: referenceFaceCount,
            hostBefore: hostBefore,
            hostAfter: hostAfter,
            error: error,
            lastProbeId: collector.lastProbeId,
            opProfile: opProfile
        )
    }

    // MARK: - Modes

    private func runT2I(collector: TraceCollector, steps: Int) async throws -> String {
        let outDir = resolvedOutputDirectory()
        let outURL = outDir.appendingPathComponent(
            "bench_\(config.label)_s\(config.seed)_t\(Date().timeIntervalSince1970).png"
        )
        let textTokenMode = config.textTokens.flatMap { TextTokenMode(rawValue: $0) } ?? .full512
        let padMode = config.padContent.flatMap { T2IRequest.PadContentMode(rawValue: $0) }
            ?? .prompt
        let request = T2IRequest(
            prompt: config.prompt,
            width: config.width,
            height: config.height,
            steps: steps,
            seed: config.seed,
            outputURL: outURL,
            textTokens: textTokenMode,
            padContent: padMode
        )
        let url = try await pipeline.generate(request, trace: collector.trace)
        return url.path
    }

    private func runI2I(
        collector: TraceCollector,
        steps: Int,
        identity: Bool
    ) async throws -> String {
        guard let imagePath = config.imagePath, !imagePath.isEmpty else {
            throw BenchError.missingImage
        }
        let imageURL = URL(fileURLWithPath: imagePath)
        guard FileManager.default.fileExists(atPath: imageURL.path) else {
            throw BenchError.missingImage
        }

        let outDir = resolvedOutputDirectory()
        let kind = identity ? "identity_i2i" : "i2i"
        let outURL = outDir.appendingPathComponent(
            "bench_\(config.label)_\(kind)_s\(config.seed)_t\(Date().timeIntervalSince1970).png"
        )
        let textTokenMode = config.textTokens.flatMap { TextTokenMode(rawValue: $0) } ?? .full512
        let identityConfig = identity
            ? IdentityPreserveConfig.identityPreset
            : IdentityPreserveConfig.disabled
        let strength = config.strength ?? (identity ? 0.9 : 0.8)
        let request = I2IRequest(
            prompt: config.prompt,
            imageURL: imageURL,
            strength: strength,
            width: config.width,
            height: config.height,
            steps: steps,
            seed: config.seed,
            outputURL: outURL,
            identity: identityConfig,
            textTokens: textTokenMode,
            embedCache: false
        )
        let url = try await pipeline.edit(request, trace: collector.trace)
        return url.path
    }

    private func runMemStages(collector: TraceCollector) async throws {
        collector.sample("start")
        let samples = try await pipeline.memorySelfTest()
        for s in samples {
            collector.sample(s.label)
        }
        collector.sample("end")
        collector.noteStage("mem_stages", ms: 0)
    }

    private func runTEOnly(collector: TraceCollector) async throws {
        collector.timer.begin("load_te")
        collector.sample("before_load_te")
        _ = try await pipeline.loadTextEncoder()
        collector.timer.end("load_te")
        collector.sample("after_load_te")

        collector.timer.begin("encode_te")
        _ = try await pipeline.encodePrompt(config.prompt)
        collector.timer.end("encode_te")
        collector.sample("after_encode_te")

        // Stage-0 probe (bare-metal study §3): two encodes under ONE residency
        // (encodePrompt stages internally, so back-to-back calls never go warm).
        let pair = try await pipeline.encodePromptPairResident(
            config.prompt + " — at dawn",
            config.prompt + " — at night, in the rain")
        collector.timer.record("encode_te_resident_cold", milliseconds: pair.firstMS)
        collector.timer.record("encode_te_resident_warm", milliseconds: pair.secondMS)
        collector.sample("after_encode_te_pair")
        await pipeline.purge()
    }

    private func runVAEDecode(collector: TraceCollector) async throws {
        collector.timer.begin("decode_vae")
        collector.sample("before_decode_vae")
        try await pipeline.decodePackedNoise(
            width: config.width, height: config.height, seed: config.seed)
        collector.timer.end("decode_vae")
        collector.sample("after_decode_vae")
        await pipeline.purge()
    }

    private func runLoadOnly(collector: TraceCollector) async throws {
        collector.sample("start")

        collector.timer.begin("load_te")
        try await pipeline.withTextEncoderLoaded {
            collector.sample("te_loaded")
        }
        collector.timer.end("load_te")
        collector.sample("after_unload_te")
        await pipeline.purge()

        collector.timer.begin("load_dit")
        try await pipeline.withDiTLoaded {
            collector.sample("dit_loaded")
        }
        collector.timer.end("load_dit")
        collector.sample("after_unload_dit")
        await pipeline.purge()

        collector.timer.begin("load_vae")
        try await pipeline.withVAELoaded {
            collector.sample("vae_loaded")
        }
        collector.timer.end("load_vae")
        collector.sample("after_unload_vae")
        await pipeline.purge()
    }

    // MARK: - Helpers

    private func resolvedOutputDirectory() -> URL {
        let path: String
        if let d = config.outputDirectory {
            path = d
        } else {
            path = AppCache.directory("bench").path
        }
        let url = URL(fileURLWithPath: path, isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func qualityFromPNG(
        path: String
    ) throws -> (
        score: Float,
        colorMatch: Bool?,
        referenceSSIM: Float?,
        fidelityScore: Float?,
        faceReferenceSSIM: Float?,
        faceFidelityScore: Float?,
        generatedFaceCount: Int?,
        referenceFaceCount: Int?
    ) {
        let referenceURL = config.imagePath.map { URL(fileURLWithPath: $0) }
        let resolvedStrength = referenceURL == nil
            ? nil
            : (config.strength ?? (identityEnabled ? 0.9 : 0.8))
        let report = try ImageAnalyzer.analyze(
            imageURL: URL(fileURLWithPath: path),
            options: .init(
                prompt: config.prompt,
                referenceURL: referenceURL,
                maxAnalysisSide: 512,
                i2iStrength: resolvedStrength,
                skipSemantic: true,
                vaeTileConfiguration: VAETileEvaluationConfig(
                    enabledThresholdLatentPixels: config.vaeTileThreshold ?? 128,
                    tileSize: config.vaeTileSize ?? 128,
                    overlap: config.vaeTileOverlap ?? 16,
                    blend: config.vaeTileBlend ?? "cosine")
            )
        )
        let face = identityEnabled ? report.faceRegion : nil
        return (
            report.technical.technicalScore,
            report.promptAlignment.colorMatch,
            report.reference?.ssim,
            report.reference?.fidelityScore,
            face?.ssim,
            face?.fidelityScore,
            face?.generatedFaceCount,
            face?.referenceFaceCount
        )
    }

    private var identityEnabled: Bool {
        config.mode == .identityI2I || config.identity == true
    }
}

// MARK: - Errors

public enum BenchError: Error, Sendable, LocalizedError {
    case warmupFailed(String)
    case missingSnapshot
    case missingImage
    case invalidReport(String)

    public var errorDescription: String? {
        switch self {
        case .warmupFailed(let detail): return "Bench warmup failed: \(detail)"
        case .missingSnapshot: return "No local model snapshot for benchmark"
        case .missingImage: return "I2I benchmark requires an existing --image path"
        case .invalidReport(let detail): return "Invalid bench report: \(detail)"
        }
    }
}

// MARK: - Trace collector

final class TraceCollector: @unchecked Sendable {
    let timer = StageTimer()
    private let lock = NSLock()
    private var samples: [MemoryPoint] = []
    private var denoiseSteps: [Double] = []
    private var openStep: (index: Int, start: CFAbsoluteTime)?
    private var extraStages: [String: Double] = [:]
    private var lastActive: UInt64 = 0
    private(set) var lastProbeId: String?
    private let density: ProbeDensity
    private let resetPeakEachPhase: Bool

    var memorySamples: [MemoryPoint] {
        lock.lock()
        defer { lock.unlock() }
        return samples
    }

    lazy var trace: PipelineTrace = {
        PipelineTrace(density: density) { [weak self] event in
            self?.handle(event)
        }
    }()

    init(density: ProbeDensity, resetPeakEachPhase: Bool) {
        self.density = density
        self.resetPeakEachPhase = resetPeakEachPhase
    }

    func sample(_ label: String, note: String? = nil) {
        let point = MemorySampler.sample(
            label: label, previousActive: lastActive == 0 ? nil : lastActive, note: note)
        lock.lock()
        lastActive = point.mlxActiveBytes
        samples.append(point)
        lastProbeId = label
        lock.unlock()
    }

    func noteStage(_ name: String, ms: Double) {
        lock.lock()
        extraStages[name] = ms
        lock.unlock()
    }

    private func handle(_ event: PipelineTraceEvent) {
        switch event {
        case .stageBegin(let name):
            timer.begin(name)
        case .stageEnd(let name):
            _ = timer.end(name)
        case .denoiseStepBegin(let index, _):
            lock.lock()
            openStep = (index, CFAbsoluteTimeGetCurrent())
            lock.unlock()
        case .denoiseStepEnd(let index, _):
            lock.lock()
            if let open = openStep, open.index == index {
                let ms = (CFAbsoluteTimeGetCurrent() - open.start) * 1000
                denoiseSteps.append(ms)
            }
            openStep = nil
            lock.unlock()
        case .memorySample(let label):
            sample(label)
        case .probe(let point):
            if point.sampleMemory {
                sample(point.id)
            } else {
                lock.lock()
                lastProbeId = point.id
                lock.unlock()
            }
        case .note(let text):
            sample("note", note: text)
        case .peakReset(let label):
            if resetPeakEachPhase {
                MemorySampler.resetPeak()
                sample("peak_reset:\(label)")
            }
        }
    }

    func buildTimings(e2eMs: Double) -> StageTimingsMs {
        let completed = timer.allCompletedMs()
        lock.lock()
        let steps = denoiseSteps
        let loadOnly = extraStages
        lock.unlock()

        var loadOnlyMap = completed.filter { key, _ in
            !["load_te", "encode_te", "unload_te", "load_dit", "denoise", "unload_dit",
              "load_vae", "decode_vae", "unload_vae", "export_png", "encode_vae"].contains(key)
        }
        for (k, v) in loadOnly { loadOnlyMap[k] = v }

        return StageTimingsMs(
            e2e: e2eMs,
            loadTe: completed["load_te"],
            encodeTe: completed["encode_te"],
            unloadTe: completed["unload_te"],
            loadDit: completed["load_dit"],
            denoiseTotal: completed["denoise"],
            denoiseSteps: steps,
            unloadDit: completed["unload_dit"],
            loadVae: completed["load_vae"],
            decodeVae: completed["decode_vae"],
            unloadVae: completed["unload_vae"],
            exportPng: completed["export_png"],
            loadOnly: loadOnlyMap
        )
    }

    func peaks() -> (rss: UInt64, mlxActive: UInt64, mlxPeak: UInt64, mlxCache: UInt64) {
        let s = memorySamples
        let live = MemorySampler.sample(label: "final")
        return (
            s.map(\.rssBytes).max() ?? live.rssBytes,
            s.map(\.mlxActiveBytes).max() ?? live.mlxActiveBytes,
            max(s.map(\.mlxPeakBytes).max() ?? 0, live.mlxPeakBytes),
            s.map(\.mlxCacheBytes).max() ?? live.mlxCacheBytes
        )
    }
}
