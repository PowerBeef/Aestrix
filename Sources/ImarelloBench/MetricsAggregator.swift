import Foundation

public enum MetricsAggregator {
    public static func aggregate(trials: [BenchTrial]) -> BenchAggregate {
        let ok = trials.filter { $0.error == nil }
        func mapStat(_ pick: (BenchTrial) -> Double?) -> StatSummary? {
            StatSummary.from(ok.compactMap(pick))
        }
        return BenchAggregate(
            e2eMs: mapStat { $0.timingsMs.e2e },
            denoiseStepMeanMs: mapStat { $0.timingsMs.denoiseStepMeanMs },
            peakRssBytes: mapStat { Double($0.peakRssBytes) },
            peakMlxActiveBytes: mapStat { Double($0.peakMlxActiveBytes) },
            peakMlxPeakBytes: mapStat { Double($0.peakMlxPeakBytes) },
            loadTeMs: mapStat { $0.timingsMs.loadTe },
            encodeTeMs: mapStat { $0.timingsMs.encodeTe },
            loadDitMs: mapStat { $0.timingsMs.loadDit },
            denoiseTotalMs: mapStat { $0.timingsMs.denoiseTotal },
            loadVaeMs: mapStat { $0.timingsMs.loadVae },
            decodeVaeMs: mapStat { $0.timingsMs.decodeVae },
            exportPngMs: mapStat { $0.timingsMs.exportPng }
        )
    }

    /// Percent change of B vs A: (B-A)/A * 100.
    public static func percentDelta(baseline: Double, candidate: Double) -> Double {
        guard baseline != 0 else { return candidate == 0 ? 0 : Double.infinity }
        return (candidate - baseline) / baseline * 100
    }
}
