import Foundation
import AestrixCore

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
            "aestrix bench label=\(report.label) mode=\(c.mode.rawValue) \(c.width)x\(c.height) steps=\(c.steps) trials=\(report.trials.count)"
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
        if let sha = report.system.aestrixGitSha {
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
                lines.append(
                    String(
                        format: "  trial %d%@ e2e=%.0fms%@ peak_rss=%@",
                        t.index,
                        coldTag,
                        t.timingsMs.e2e,
                        stepPart,
                        MemoryProbe.formatBytes(t.peakRssBytes)
                    )
                )
            }
        }

        if let p = report.pressure {
            lines.append("pressure:")
            lines.append(
                "  density=\(p.probeDensity) joint_seq=\(p.analytic.jointSeqLen) image_seq=\(p.analytic.imageSeqLen) packed=\(p.analytic.packedH)x\(p.analytic.packedW)"
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

    public static func compareText(baseline: BenchReport, candidate: BenchReport) -> String {
        var lines: [String] = []
        lines.append("aestrix bench-compare")
        lines.append("  baseline:  \(baseline.label)")
        lines.append("  candidate: \(candidate.label)")

        func padName(_ name: String) -> String {
            name.padding(toLength: 16, withPad: " ", startingAt: 0)
        }

        func row(_ name: String, _ a: StatSummary?, _ b: StatSummary?, lowerIsBetter: Bool = true) {
            guard let a, let b else { return }
            let pct = MetricsAggregator.percentDelta(baseline: a.mean, candidate: b.mean)
            let arrow: String
            if abs(pct) < 0.5 { arrow = "≈" }
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
