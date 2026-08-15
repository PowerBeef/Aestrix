import Testing
import Foundation
@testable import ImarelloBench
import ImarelloDiT

@Suite("ImarelloBench metrics")
struct MetricsAggregatorTests {
    @Test("StatSummary mean/p50/p95")
    func statSummary() {
        let s = StatSummary.from([10, 20, 30, 40, 50])
        #expect(s != nil)
        guard let s else { return }
        #expect(abs(s.mean - 30) < 0.001)
        #expect(s.min == 10)
        #expect(s.max == 50)
        #expect(s.count == 5)
        #expect(s.p50 == 30)
    }

    @Test("StatSummary empty is nil")
    func emptyStat() {
        #expect(StatSummary.from([]) == nil)
    }

    @Test("percentDelta")
    func percentDelta() {
        #expect(abs(MetricsAggregator.percentDelta(baseline: 100, candidate: 90) - (-10)) < 0.001)
        #expect(abs(MetricsAggregator.percentDelta(baseline: 100, candidate: 110) - 10) < 0.001)
        #expect(MetricsAggregator.percentDelta(baseline: 0, candidate: 0) == 0)
    }

    @Test("aggregate ignores failed trials")
    func aggregateSkipsErrors() {
        let ok = BenchTrial(
            index: 0,
            cold: false,
            timingsMs: StageTimingsMs(e2e: 1000, denoiseSteps: [100, 200]),
            memorySamples: [],
            peakRssBytes: 1_000_000_000,
            peakMlxActiveBytes: 500_000_000,
            peakMlxPeakBytes: 600_000_000,
            peakMlxCacheBytes: 100_000_000
        )
        let bad = BenchTrial(
            index: 1,
            cold: false,
            timingsMs: StageTimingsMs(e2e: 99999),
            memorySamples: [],
            peakRssBytes: 0,
            peakMlxActiveBytes: 0,
            peakMlxPeakBytes: 0,
            peakMlxCacheBytes: 0,
            error: "boom"
        )
        let agg = MetricsAggregator.aggregate(trials: [ok, bad])
        #expect(agg.e2eMs?.count == 1)
        #expect(agg.e2eMs?.mean == 1000)
        #expect(agg.denoiseStepMeanMs?.mean == 150)
    }

    @Test("StageTimer measure")
    func stageTimer() {
        let t = StageTimer()
        t.begin("x")
        Thread.sleep(forTimeInterval: 0.01)
        let ms = t.end("x")
        #expect(ms >= 5)
        #expect(t.ms("x") != nil)
    }

    @Test("JSON round-trip report")
    func jsonRoundTrip() throws {
        let report = BenchReport(
            schemaVersion: BenchReport.currentSchema,
            label: "test",
            createdAt: "2026-01-01T00:00:00Z",
            system: SystemSnapshot(
                hostname: "host",
                osVersion: "macOS",
                processId: 1,
                physicalMemoryBytes: 16_000_000_000,
                processorCount: 8,
                recommendedMaxWorkingSetBytes: nil,
                thermalState: "nominal",
                mlxCacheLimitBytes: nil,
                mlxMemoryLimitBytes: nil,
                imarelloGitSha: "abc",
                gpuName: "Apple M2",
                metalSupport: "Metal 4",
                hasNeuralAccelerators: false
            ),
            config: BenchConfig(label: "test", trials: 1, warmup: 0),
            trials: [
                BenchTrial(
                    index: 0,
                    cold: true,
                    timingsMs: StageTimingsMs(e2e: 1234.5, loadTe: 100, denoiseSteps: [50, 60]),
                    memorySamples: [
                        MemoryPoint(
                            label: "x",
                            rssBytes: 1,
                            mlxActiveBytes: 2,
                            mlxCacheBytes: 3,
                            mlxPeakBytes: 4
                        )
                    ],
                    peakRssBytes: 10,
                    peakMlxActiveBytes: 20,
                    peakMlxPeakBytes: 30,
                    peakMlxCacheBytes: 40
                )
            ],
            aggregate: MetricsAggregator.aggregate(trials: [])
        )
        // Re-aggregate properly
        let withAgg = BenchReport(
            schemaVersion: report.schemaVersion,
            label: report.label,
            createdAt: report.createdAt,
            system: report.system,
            config: report.config,
            trials: report.trials,
            aggregate: MetricsAggregator.aggregate(trials: report.trials),
            pressure: nil
        )
        let text = try BenchReportWriter.jsonString(withAgg)
        #expect(text.contains("\"label\" : \"test\"") || text.contains("\"label\": \"test\""))
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("imarello-bench-test-\(UUID().uuidString).json")
        try BenchReportWriter.write(withAgg, to: tmp)
        let loaded = try BenchReportWriter.loadReport(from: tmp)
        #expect(loaded.label == "test")
        #expect(loaded.trials.count == 1)
        #expect(loaded.trials[0].timingsMs.e2e == 1234.5)
        try? FileManager.default.removeItem(at: tmp)
    }

    @Test("text summary prints op profile shares")
    func opProfileSummary() {
        var trial = BenchTrial(
            index: 0,
            cold: true,
            timingsMs: StageTimingsMs(e2e: 1000, denoiseTotal: 800, denoiseSteps: [800]),
            memorySamples: [],
            peakRssBytes: 1,
            peakMlxActiveBytes: 1,
            peakMlxPeakBytes: 1,
            peakMlxCacheBytes: 1
        )
        trial.opProfile = DiTOpProfileReport(
            bucketsMs: ["qkv_proj": 400, "qkv_rope": 50, "steel_fa": 200, "ffn": 150],
            counts: ["qkv_proj": 30, "qkv_rope": 35, "steel_fa": 25, "ffn": 30],
            countedMs: 800,
            denoiseMs: 800,
            otherMs: 0,
            shares: ["qkv_proj": 0.5, "qkv_rope": 0.0625, "steel_fa": 0.25, "ffn": 0.1875, "other": 0],
            notes: ["ranking only"]
        )
        let report = BenchReport(
            schemaVersion: BenchReport.currentSchema,
            label: "op",
            createdAt: "t",
            system: SystemSnapshot(
                hostname: "h", osVersion: "o", processId: 1,
                physicalMemoryBytes: 1, processorCount: 1,
                recommendedMaxWorkingSetBytes: nil, thermalState: "nominal",
                mlxCacheLimitBytes: nil, mlxMemoryLimitBytes: nil, imarelloGitSha: nil,
                gpuName: nil, metalSupport: nil, hasNeuralAccelerators: nil
            ),
            config: BenchConfig(label: "op", opProfile: true),
            trials: [trial],
            aggregate: MetricsAggregator.aggregate(trials: [trial]),
            pressure: nil
        )
        let text = BenchReportWriter.textSummary(report)
        #expect(text.contains("op_profile"))
        #expect(text.contains("qkv_proj"))
        #expect(text.contains("steel_fa"))
        #expect(text.contains("dominant=qkv_proj"))
    }

    @Test("compare text non-empty")
    func compareText() {
        func make(_ label: String, e2e: Double, rss: Double) -> BenchReport {
            let trial = BenchTrial(
                index: 0,
                cold: false,
                timingsMs: StageTimingsMs(e2e: e2e, denoiseSteps: [e2e / 4]),
                memorySamples: [],
                peakRssBytes: UInt64(rss),
                peakMlxActiveBytes: UInt64(rss / 2),
                peakMlxPeakBytes: UInt64(rss / 2),
                peakMlxCacheBytes: 0
            )
            return BenchReport(
                schemaVersion: "1.0",
                label: label,
                createdAt: "t",
                system: SystemSnapshot(
                    hostname: "h", osVersion: "o", processId: 1,
                    physicalMemoryBytes: 1, processorCount: 1,
                    recommendedMaxWorkingSetBytes: nil, thermalState: "nominal",
                    mlxCacheLimitBytes: nil, mlxMemoryLimitBytes: nil, imarelloGitSha: nil,
                    gpuName: nil, metalSupport: nil, hasNeuralAccelerators: nil
                ),
                config: BenchConfig(label: label),
                trials: [trial],
                aggregate: MetricsAggregator.aggregate(trials: [trial]),
                pressure: nil
            )
        }
        let text = BenchReportWriter.compareText(
            baseline: make("a", e2e: 1000, rss: 1e9),
            candidate: make("b", e2e: 900, rss: 0.9e9)
        )
        #expect(text.contains("bench-compare"))
        #expect(text.contains("better") || text.contains("%"))
    }

    @Test("canvas analytics joint seq")
    func canvasAnalytics() {
        let s512 = PressureAnalytics.canvasStats(width: 512, height: 512)
        #expect(s512.imageSeqLen == 1024)
        #expect(s512.jointSeqLen == 512 + 1024)
        let s1024 = PressureAnalytics.canvasStats(width: 1024, height: 1024)
        #expect(s1024.imageSeqLen == 4096)
        #expect(s1024.jointSeqLen == 512 + 4096)

        let identity512 = PressureAnalytics.canvasStats(
            width: 512, height: 512, referenceSeqLen: 1024)
        #expect(identity512.jointSeqLen == 512 + 1024 + 1024)
        #expect(identity512.referenceSeqLen == 1024)
    }

    @Test("host process parser sorts and preserves names")
    func hostProcessParser() {
        let output = """
          101  2.5  1000 /usr/bin/quiet
          202 48.0  2000 /Applications/Cursor Helper
          malformed
          303 17.5  3000 /usr/bin/swift
        """
        let parsed = HostContention.parsePSOutput(output, limit: 2)
        #expect(parsed.count == 2)
        #expect(parsed[0].pid == 202)
        #expect(parsed[0].name == "/Applications/Cursor Helper")
        #expect(parsed[0].rssBytes == 2_048_000)
        #expect(parsed[1].pid == 303)
    }

    @Test("WindowServer and Ghostty are ambient, Cursor Helper is not")
    func ambientDesktopProcesses() {
        #expect(HostContention.isAmbientProcess(
            "/System/Library/PrivateFrameworks/SkyLight.framework/Resources/WindowServer"))
        #expect(HostContention.isAmbientProcess("MTLCompilerService"))
        #expect(HostContention.isAmbientProcess("/Applications/Ghostty.app/Contents/MacOS/ghostty"))
        #expect(HostContention.isAmbientProcess("grok"))
        #expect(!HostContention.isAmbientProcess("/Applications/Cursor Helper"))
        #expect(!HostContention.isAmbientProcess("swift-package"))
    }

    @Test("rank by active and delta")
    func rankPressure() {
        let samples = [
            MemoryPoint(label: "a", rssBytes: 1, mlxActiveBytes: 100, mlxCacheBytes: 0, mlxPeakBytes: 100, mlxActiveDeltaBytes: 100),
            MemoryPoint(label: "b", rssBytes: 1, mlxActiveBytes: 500, mlxCacheBytes: 0, mlxPeakBytes: 500, mlxActiveDeltaBytes: 400),
            MemoryPoint(label: "c", rssBytes: 1, mlxActiveBytes: 200, mlxCacheBytes: 0, mlxPeakBytes: 500, mlxActiveDeltaBytes: -300),
        ]
        let (byA, byD) = PressureAnalytics.rank(samples: samples, recommendedWS: 1000, top: 3)
        #expect(byA.first?.label == "b")
        #expect(byD.first?.label == "b")
        #expect(byA.first?.shareOfRecommendedWS == 0.5)
    }
}
