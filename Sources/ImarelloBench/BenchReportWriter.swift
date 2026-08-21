import Foundation
import ImarelloCore

public enum BenchReportWriter {
    public static func jsonString(_ report: BenchReport, pretty: Bool = true) throws -> String {
        let enc = JSONEncoder()
        if pretty {
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        }
        enc.keyEncodingStrategy = .convertToSnakeCase
        enc.dateEncodingStrategy = .iso8601
        let data = try enc.encode(report)
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    public static func write(_ report: BenchReport, to url: URL) throws {
        let text = try jsonString(report, pretty: true)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    public static func textSummary(_ report: BenchReport) -> String {
        var lines: [String] = []
        let c = report.config
        lines.append(
            "imarello bench label=\(report.label) mode=\(c.mode.rawValue) \(c.width)x\(c.height) steps=\(c.steps) trials=\(report.trials.count)"
        )
        let gpuPart: String
        if let g = report.system.gpuName {
            let na = report.system.hasNeuralAccelerators == true ? " neuralAccel=yes" : " neuralAccel=no"
            let metal = report.system.metalSupport.map { " metal=\($0)" } ?? ""
            gpuPart = " gpu=\(g)\(metal)\(na)"
        } else {
            gpuPart = ""
        }
        lines.append(
            "system: mem=\(MemoryProbe.formatBytes(report.system.physicalMemoryBytes)) thermal=\(report.system.thermalState) host=\(report.system.hostname)\(gpuPart)"
        )
        if let sha = report.system.imarelloGitSha {
            lines.append("git: \(sha)")
        }

        func padName(_ name: String) -> String {
            name.padding(toLength: 16, withPad: " ", startingAt: 0)
        }

        func line(_ name: String, _ s: StatSummary?, bytes: Bool = false) {
            guard let s else { return }
            if bytes {
                lines.append(
                    "  \(padName(name)) mean \(MemoryProbe.formatBytes(UInt64(s.mean)))  p50 \(MemoryProbe.formatBytes(UInt64(s.p50)))  p95 \(MemoryProbe.formatBytes(UInt64(s.p95)))  max \(MemoryProbe.formatBytes(UInt64(s.max)))"
                )
            } else {
                lines.append(
                    String(
                        format: "  %@ mean %7.1f ms  p50 %7.1f  p95 %7.1f  max %7.1f",
                        padName(name), s.mean, s.p50, s.p95, s.max
                    )
                )
            }
        }

        let a = report.aggregate
        lines.append("timings:")
        line("e2e", a.e2eMs)
        line("load_te", a.loadTeMs)
        line("encode_te", a.encodeTeMs)
        line("load_dit", a.loadDitMs)
        line("denoise", a.denoiseTotalMs)
        line("denoise/step", a.denoiseStepMeanMs)
        line("load_vae", a.loadVaeMs)
        line("decode_vae", a.decodeVaeMs)
        line("export_png", a.exportPngMs)
        lines.append("memory peaks:")
        line("peak_rss", a.peakRssBytes, bytes: true)
        line("peak_mlx_active", a.peakMlxActiveBytes, bytes: true)
        line("peak_mlx_peak", a.peakMlxPeakBytes, bytes: true)

        for t in report.trials {
            if let err = t.error {
                let last = t.lastProbeId.map { " last_probe=\($0)" } ?? ""
                lines.append("  trial \(t.index): ERROR \(err)\(last)")
            } else {
                let coldTag = t.cold ? " cold" : ""
                let stepPart: String
                if let steps = t.timingsMs.denoiseStepMeanMs {
                    stepPart = String(format: " denoise/step=%.0fms", steps)
                } else {
                    stepPart = ""
                }
                let contaminated = t.hostBefore?.contaminated == true
                    || t.hostAfter?.contaminated == true
                let hostTag = contaminated ? " CONTAMINATED" : ""
                lines.append(
                    String(
                        format: "  trial %d%@ e2e=%.0fms%@ peak_rss=%@%@",
                        t.index,
                        coldTag,
                        t.timingsMs.e2e,
                        stepPart,
                        MemoryProbe.formatBytes(t.peakRssBytes),
                        hostTag
                    )
                )
                if let status = t.qualityStatus, status != .notRequested {
                    let detail = t.qualityError.map { " — \($0)" } ?? ""
                    lines.append("    quality: \(status.rawValue)\(detail)")
                }
                if contaminated {
                    let notes = (t.hostBefore?.notes ?? []) + (t.hostAfter?.notes ?? [])
                    if !notes.isEmpty {
                        lines.append("    host: \(Array(Set(notes)).sorted().joined(separator: "; "))")
                    }
                }
            }
        }

        if let op = report.trials.compactMap(\.opProfile).last {
            lines.append("op_profile (ranking only):")
            let order = ["qkv_proj", "qkv_rope", "steel_fa", "ffn", "other"]
            for key in order {
                let ms: Double
                if key == "other" {
                    ms = op.otherMs ?? 0
                } else {
                    ms = op.bucketsMs[key] ?? 0
                }
                let share = (op.shares[key] ?? 0) * 100
                let count: String
                if key == "other" {
                    count = ""
                } else if let n = op.counts[key] {
                    count = "  n=\(n)"
                } else {
                    count = ""
                }
                let name = key.padding(toLength: 10, withPad: " ", startingAt: 0)
                lines.append(
                    String(format: "  %@ %7.0f ms  %5.1f%%%@", name, ms, share, count)
                )
            }
            if let denoise = op.denoiseMs {
                let dom = op.dominantBucket ?? "?"
                lines.append(
                    String(
                        format: "  counted=%.0f ms  denoise=%.0f ms  dominant=%@",
                        op.countedMs, denoise, dom
                    )
                )
            }
            for note in op.notes {
                lines.append("  note: \(note)")
            }
        }

        if let p = report.pressure {
            lines.append("pressure:")
            lines.append(
                "  density=\(p.probeDensity) joint_seq=\(p.analytic.jointSeqLen) image_seq=\(p.analytic.imageSeqLen) ref_seq=\(p.analytic.referenceSeqLen ?? 0) packed=\(p.analytic.packedH)x\(p.analytic.packedW)"
            )
            lines.append("  \(p.analytic.note)")
            if !p.phasePeaks.isEmpty {
                let parts = p.phasePeaks.keys.sorted().map { k in
                    "\(k)=\(MemoryProbe.formatBytes(p.phasePeaks[k]!))"
                }
                lines.append("  phase_peaks: \(parts.joined(separator: " "))")
            }
            if let last = p.lastProbeBeforeFailure {
                lines.append("  last_probe: \(last)")
            }
            if p.oom {
                lines.append("  oom: true")
            }
            lines.append("  top_active:")
            for r in p.rankedByActive.prefix(10) {
                let d = r.mlxActiveDeltaBytes
                let dStr = d >= 0 ? "+\(MemoryProbe.formatBytes(UInt64(d)))" : "-\(MemoryProbe.formatBytes(UInt64(-d)))"
                let share = r.shareOfRecommendedWS.map { String(format: " ws=%.0f%%", $0 * 100) } ?? ""
                lines.append(
                    "    \(r.label)  active=\(MemoryProbe.formatBytes(r.mlxActiveBytes))  Δ=\(dStr)\(share)"
                )
            }
            lines.append("  top_delta:")
            for r in p.rankedByDelta.prefix(8) where (r.mlxActiveDeltaBytes) > 0 {
                lines.append(
                    "    \(r.label)  Δ=+\(MemoryProbe.formatBytes(UInt64(r.mlxActiveDeltaBytes)))  active=\(MemoryProbe.formatBytes(r.mlxActiveBytes))"
                )
            }
        }
        return lines.joined(separator: "\n")
    }

    public static func comparisonIssues(
        baseline: BenchReport, candidate: BenchReport
    ) -> [String] {
        var issues: [String] = []
        if baseline.schemaVersion != BenchReport.currentSchema
            || candidate.schemaVersion != BenchReport.currentSchema {
            issues.append("both reports must use schema \(BenchReport.currentSchema)")
        }
        guard let a = baseline.provenance, let b = candidate.provenance else {
            issues.append("schema 1.7 provenance is required on both reports")
            return issues
        }
        func requireEqual<T: Equatable>(_ lhs: T, _ rhs: T, _ label: String) {
            if lhs != rhs { issues.append("\(label) differs") }
        }
        requireEqual(baseline.config.mode, candidate.config.mode, "mode")
        requireEqual(baseline.config.prompt, candidate.config.prompt, "prompt")
        requireEqual(baseline.config.width, candidate.config.width, "width")
        requireEqual(baseline.config.height, candidate.config.height, "height")
        requireEqual(baseline.config.steps, candidate.config.steps, "steps")
        requireEqual(baseline.config.seed, candidate.config.seed, "seed")
        requireEqual(baseline.config.textTokens, candidate.config.textTokens, "text-token mode")
        requireEqual(baseline.config.padContent, candidate.config.padContent, "pad-content mode")
        requireEqual(baseline.config.vaeVariant, candidate.config.vaeVariant, "VAE variant")
        requireEqual(baseline.config.vaeEngine, candidate.config.vaeEngine, "VAE engine")
        requireEqual(baseline.config.ditEngine, candidate.config.ditEngine, "DiT engine")
        requireEqual(baseline.config.cacheLimitBytes, candidate.config.cacheLimitBytes, "MLX cache limit")
        requireEqual(baseline.config.warmup, candidate.config.warmup, "warmup count")
        requireEqual(baseline.config.cooldownSeconds, candidate.config.cooldownSeconds, "cooldown")
        requireEqual(baseline.config.probeDensity, candidate.config.probeDensity, "probe density")
        requireEqual(baseline.config.resetPeakEachPhase, candidate.config.resetPeakEachPhase, "peak-reset mode")
        requireEqual(baseline.config.withQuality, candidate.config.withQuality, "quality-evaluation mode")
        requireEqual(baseline.config.opProfile, candidate.config.opProfile, "operation profiling")
        requireEqual(baseline.config.attentionQueryChunkSize, candidate.config.attentionQueryChunkSize, "attention query chunk")
        requireEqual(baseline.config.attentionQueryChunkThreshold, candidate.config.attentionQueryChunkThreshold, "attention query threshold")
        requireEqual(baseline.config.attentionF16SeqThreshold, candidate.config.attentionF16SeqThreshold, "attention f16 threshold")
        requireEqual(baseline.config.attentionLinearF16, candidate.config.attentionLinearF16, "attention linear compute")
        requireEqual(baseline.config.attentionLinearChunkSize, candidate.config.attentionLinearChunkSize, "attention linear chunk")
        requireEqual(baseline.config.attentionLinearChunkThreshold, candidate.config.attentionLinearChunkThreshold, "attention linear threshold")
        requireEqual(baseline.config.attentionBackend, candidate.config.attentionBackend, "attention backend")
        requireEqual(baseline.config.attentionBlockClearSeqThreshold, candidate.config.attentionBlockClearSeqThreshold, "attention cache-clear threshold")
        requireEqual(baseline.config.attentionBlockClearInterval, candidate.config.attentionBlockClearInterval, "attention cache-clear interval")
        requireEqual(baseline.config.attentionAsyncEvalInterval, candidate.config.attentionAsyncEvalInterval, "attention async-eval interval")
        requireEqual(baseline.config.attentionJointSeqF16, candidate.config.attentionJointSeqF16, "joint attention dtype")
        requireEqual(baseline.config.attentionF16FullEpilogue, candidate.config.attentionF16FullEpilogue, "f16 epilogue")
        requireEqual(baseline.config.attentionDynamicScale, candidate.config.attentionDynamicScale, "dynamic scaling")
        requireEqual(baseline.config.attentionQkvCheckpoint, candidate.config.attentionQkvCheckpoint, "QKV checkpointing")
        requireEqual(baseline.config.vaeTileThreshold, candidate.config.vaeTileThreshold, "VAE tile threshold")
        requireEqual(baseline.config.vaeTileSize, candidate.config.vaeTileSize, "VAE tile size")
        requireEqual(baseline.config.vaeTileOverlap, candidate.config.vaeTileOverlap, "VAE tile overlap")
        requireEqual(baseline.config.vaeTileBlend, candidate.config.vaeTileBlend, "VAE tile blend")
        requireEqual(baseline.config.vaeAttnChunk, candidate.config.vaeAttnChunk, "VAE attention chunk")
        requireEqual(baseline.config.evalCache, candidate.config.evalCache, "evaluation cache profile")
        requireEqual(baseline.config.strength, candidate.config.strength, "I2I strength")
        requireEqual(baseline.config.identity, candidate.config.identity, "identity mode")
        requireEqual(baseline.config.imagePath, candidate.config.imagePath, "input image")
        requireEqual(a.teEngine, b.teEngine, "TE engine")
        requireEqual(a.cachePolicy, b.cachePolicy, "cache policy")
        requireEqual(a.cacheHitState, b.cacheHitState, "cache hit state")
        requireEqual(a.modelRevision, b.modelRevision, "model revision")
        requireEqual(a.detectedSnapshotRevision, b.detectedSnapshotRevision, "snapshot revision")
        requireEqual(a.tokenizerFingerprint, b.tokenizerFingerprint, "tokenizer fingerprint")
        requireEqual(a.metallibRevision, b.metallibRevision, "metallib revision")
        requireEqual(a.metallibByteCount, b.metallibByteCount, "metallib size")
        requireEqual(a.buildDirty, b.buildDirty, "build dirty state")
        requireEqual(baseline.system.hostname, candidate.system.hostname, "host")
        requireEqual(baseline.system.osVersion, candidate.system.osVersion, "OS")
        requireEqual(baseline.system.physicalMemoryBytes, candidate.system.physicalMemoryBytes, "physical memory")
        requireEqual(baseline.system.gpuName, candidate.system.gpuName, "GPU")
        requireEqual(baseline.system.metalSupport, candidate.system.metalSupport, "Metal support")
        requireEqual(baseline.system.appleGpuFamilyRaw, candidate.system.appleGpuFamilyRaw, "GPU family")
        requireEqual(baseline.system.mlxCacheLimitBytes, candidate.system.mlxCacheLimitBytes, "effective MLX cache limit")
        requireEqual(baseline.system.mlxMemoryLimitBytes, candidate.system.mlxMemoryLimitBytes, "MLX memory limit")
        requireEqual(a.successfulSamples, b.successfulSamples, "successful sample count")
        let requiredStrings: [(String, String?)] = [
            ("model revision", a.modelRevision),
            ("snapshot revision", a.detectedSnapshotRevision),
            ("tokenizer fingerprint", a.tokenizerFingerprint),
            ("metallib revision", a.metallibRevision),
            ("TE engine", a.teEngine),
            ("cache policy", a.cachePolicy),
            ("cache hit state", a.cacheHitState),
            ("build revision", a.buildRevision),
            ("candidate model revision", b.modelRevision),
            ("candidate snapshot revision", b.detectedSnapshotRevision),
            ("candidate tokenizer fingerprint", b.tokenizerFingerprint),
            ("candidate metallib revision", b.metallibRevision),
            ("candidate TE engine", b.teEngine),
            ("candidate cache policy", b.cachePolicy),
            ("candidate cache hit state", b.cacheHitState),
            ("candidate build revision", b.buildRevision),
        ]
        for (label, value) in requiredStrings where value?.isEmpty != false {
            issues.append("\(label) is missing")
        }
        if a.metallibByteCount == nil || b.metallibByteCount == nil {
            issues.append("metallib size is missing")
        }
        if a.buildDirty == nil || b.buildDirty == nil {
            issues.append("build dirty state is missing")
        }
        if a.failedSamples > 0 || b.failedSamples > 0 {
            issues.append("one or more samples failed")
        }
        if a.successfulSamples == 0 || b.successfulSamples == 0 {
            issues.append("both reports need successful samples")
        }
        if a.contaminated || b.contaminated {
            issues.append("host contention/thermal contamination was recorded")
        }
        return issues
    }

    public static func compareText(
        baseline: BenchReport, candidate: BenchReport, forceIncomparable: Bool = false
    ) -> String {
        var lines: [String] = []
        lines.append("imarello bench-compare")
        lines.append("  baseline:  \(baseline.label)")
        lines.append("  candidate: \(candidate.label)")
        let issues = comparisonIssues(baseline: baseline, candidate: candidate)
        if !issues.isEmpty {
            lines.append("  comparable: no")
            for issue in issues { lines.append("    - \(issue)") }
            if !forceIncomparable {
                lines.append("  comparison refused; pass --force to print unlabeled raw deltas")
                return lines.joined(separator: "\n")
            }
            lines.append("  forced: raw deltas only; no better/worse conclusion is valid")
        } else {
            lines.append("  comparable: yes")
        }

        func padName(_ name: String) -> String {
            name.padding(toLength: 16, withPad: " ", startingAt: 0)
        }

        func row(_ name: String, _ a: StatSummary?, _ b: StatSummary?, lowerIsBetter: Bool = true) {
            guard let a, let b else { return }
            let pct = MetricsAggregator.percentDelta(baseline: a.mean, candidate: b.mean)
            let arrow: String
            if !issues.isEmpty { arrow = "raw" }
            else if abs(pct) < 0.5 { arrow = "≈" }
            else if (pct < 0) == lowerIsBetter { arrow = "✓ better" }
            else { arrow = "✗ worse" }
            lines.append(
                String(
                    format: "  %@ %+.1f%%  (%.1f → %.1f) %@",
                    padName(name), pct, a.mean, b.mean, arrow
                )
            )
        }

        lines.append("timings (lower better):")
        row("e2e_ms", baseline.aggregate.e2eMs, candidate.aggregate.e2eMs)
        row("denoise_step", baseline.aggregate.denoiseStepMeanMs, candidate.aggregate.denoiseStepMeanMs)
        row("encode_te", baseline.aggregate.encodeTeMs, candidate.aggregate.encodeTeMs)
        row("decode_vae", baseline.aggregate.decodeVaeMs, candidate.aggregate.decodeVaeMs)
        lines.append("memory peaks (lower better):")
        row("peak_rss", baseline.aggregate.peakRssBytes, candidate.aggregate.peakRssBytes)
        row("peak_mlx_active", baseline.aggregate.peakMlxActiveBytes, candidate.aggregate.peakMlxActiveBytes)
        return lines.joined(separator: "\n")
    }

    public static func loadReport(from url: URL) throws -> BenchReport {
        let data = try Data(contentsOf: url)
        let dec = JSONDecoder()
        dec.keyDecodingStrategy = .convertFromSnakeCase
        dec.dateDecodingStrategy = .iso8601
        return try dec.decode(BenchReport.self, from: data)
    }
}
