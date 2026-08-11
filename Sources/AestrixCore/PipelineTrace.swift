import Foundation

/// How densely the pipeline / modules emit memory probes.
///
/// - `off`: no probes (product path; zero instrumentation cost).
/// - `stages`: load/encode/unload/export boundaries only.
/// - `denoise`: stages + every denoise step begin/end (+ euler).
/// - `blocks`: denoise + DiT double blocks + sampled single blocks + TE/VAE sections.
/// - `max`: every single block + every TE layer sample point (slow; diagnosis only).
public enum ProbeDensity: String, Sendable, Codable, CaseIterable, Equatable {
    case off
    case stages
    case denoise
    case blocks
    case max

    public var instrumentsDiTBlocks: Bool {
        self == .blocks || self == .max
    }

    public var instrumentsEveryDenoiseStep: Bool {
        self == .denoise || self == .blocks || self == .max
    }

    public var instrumentsStages: Bool {
        self != .off
    }

    /// Single-stream block indices to sample at `.blocks` density.
    public static let defaultSingleBlockSampleIndices: [Int] = [0, 5, 10, 15, 19]
}

/// Structured probe site for pressure mapping.
public struct ProbePoint: Sendable, Equatable, Codable {
    public var id: String
    public var phase: String
    public var step: Int?
    public var block: Int?
    /// When true, the bench collector should take an MLX/RSS memory snapshot.
    public var sampleMemory: Bool

    public init(
        id: String,
        phase: String,
        step: Int? = nil,
        block: Int? = nil,
        sampleMemory: Bool = true
    ) {
        self.id = id
        self.phase = phase
        self.step = step
        self.block = block
        self.sampleMemory = sampleMemory
    }
}

/// Optional instrumentation hooks for generation (benchmarking / profiling).
///
/// Call sites are best-effort and must not change model numerics when density is `.off`.
public struct PipelineTrace: Sendable {
    public var onEvent: (@Sendable (PipelineTraceEvent) -> Void)?
    public var density: ProbeDensity

    public init(
        density: ProbeDensity = .off,
        onEvent: (@Sendable (PipelineTraceEvent) -> Void)? = nil
    ) {
        self.density = density
        self.onEvent = onEvent
    }

    public func emit(_ event: PipelineTraceEvent) {
        onEvent?(event)
    }

    public func probe(
        _ id: String,
        phase: String,
        step: Int? = nil,
        block: Int? = nil,
        sampleMemory: Bool = true,
        minDensity: ProbeDensity = .stages
    ) {
        guard density.rawValueRank >= minDensity.rawValueRank else { return }
        emit(
            .probe(
                ProbePoint(
                    id: id, phase: phase, step: step, block: block, sampleMemory: sampleMemory)))
    }

    public func memory(_ label: String, minDensity: ProbeDensity = .stages) {
        guard density.rawValueRank >= minDensity.rawValueRank else { return }
        emit(.memorySample(label: label))
    }

    public func note(_ text: String, minDensity: ProbeDensity = .stages) {
        guard density.rawValueRank >= minDensity.rawValueRank else { return }
        emit(.note(text))
    }
}

public enum PipelineTraceEvent: Sendable, Equatable {
    case stageBegin(String)
    case stageEnd(String)
    case denoiseStepBegin(index: Int, total: Int)
    case denoiseStepEnd(index: Int, total: Int)
    case memorySample(label: String)
    case probe(ProbePoint)
    case note(String)
    case peakReset(label: String)
}

extension ProbeDensity {
    /// Ordering for `minDensity` gates (higher = more verbose).
    fileprivate var rawValueRank: Int {
        switch self {
        case .off: return 0
        case .stages: return 1
        case .denoise: return 2
        case .blocks: return 3
        case .max: return 4
        }
    }
}
