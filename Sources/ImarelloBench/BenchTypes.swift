import Foundation
import ImarelloCore
import ImarelloDiT

// MARK: - Config

public enum BenchMode: String, Sendable, Codable, CaseIterable {
    /// Full staged T2I generation.
    case t2i
    /// Full staged strength-only image-to-image.
    case i2i
    /// Full staged identity-preserving I2I (`IdentityPreserveConfig.identityPreset`).
    case identityI2I = "identity-i2i"
    /// Load/unload TE, DiT, VAE only (no forward).
    case memStages = "mem-stages"
    /// Load TE + encode only, then unload.
    case teOnly = "te-only"
    /// Load DiT + timed denoise steps (needs TE encode first).
    case ditSteps = "dit-steps"
    /// Load VAE + decode noise latents only.
    case vaeDecode = "vae-decode"
    /// Load each module and unload (timing only).
    case loadOnly = "load-only"
    /// Single-trial dense DiT block pressure map.
    case pressureMap = "pressure-map"
    /// Subprocess ladder of resolutions until fail.
    case resLadder = "res-ladder"
    /// TE + DiT one forward only (first-step peak).
    case ditOneStep = "dit-one-step"
}

public struct BenchConfig: Sendable, Codable, Equatable {
    public var mode: BenchMode
    public var label: String
    public var prompt: String
    public var width: Int
    public var height: Int
    public var steps: Int
    public var seed: UInt64
    public var trials: Int
    public var warmup: Int
    public var cooldownSeconds: Double
    public var withQuality: Bool
    public var outputDirectory: String?
    public var cacheLimitBytes: UInt64?
    public var probeDensity: ProbeDensity
    public var resetPeakEachPhase: Bool
    public var failSoft: Bool
    /// For res-ladder: sides to try (square). Default 512…1024.
    public var ladderSides: [Int]
    /// Optional DiT attention tuning overrides (nil = product default).
    public var attentionQueryChunkSize: Int?
    public var attentionQueryChunkThreshold: Int?
    public var attentionF16SeqThreshold: Int?
    /// When true, 4-bit Linear GEMMs + SwiGLU run in float16.
    public var attentionLinearF16: Bool?
    public var attentionLinearChunkSize: Int?
    public var attentionLinearChunkThreshold: Int?
    /// `mlx` | `metal-fa` | `auto` (nil = product default).
    public var attentionBackend: String?
    /// Joint seq above which the DiT clears the cache per block (nil = product default).
    public var attentionBlockClearSeqThreshold: Int?
    /// Clear cache every N blocks when per-block clears are active (nil = product default).
    public var attentionBlockClearInterval: Int?
    /// asyncEval per block with a blocking eval every N blocks (nil/0 = blocking every block).
    public var attentionAsyncEvalInterval: Int?
    /// Decide attention store dtype from the joint sequence (nil = per-stream product default).
    public var attentionJointSeqF16: Bool?
    /// S4 experiment: full-f16 single-stream epilogue (nil = product f32 epilogue).
    public var attentionF16FullEpilogue: Bool?
    /// S4 experiment: per-tensor dynamic activation scale (nil = flat ÷16).
    public var attentionDynamicScale: Bool?
    public var attentionQkvCheckpoint: Bool?
    /// Override the VAE tile enable threshold in latent px (nil = product default 128).
    public var vaeTileThreshold: Int?
    /// Text token mode for T2I trials: "512" (pad, product default) | "auto" (trim).
    public var textTokens: String?
    /// Pad content for T2I trials: "prompt" (full-window, default) | "clean" (TE-splice).
    public var padContent: String?
    /// VAE decode engine for T2I trials: "mlx" (product) | "direct" (bare-metal).
    public var vaeEngine: String?
    /// VAE decode variant for provenance: "small-decoder" | "full".
    public var vaeVariant: String?
    /// Resolved eval/cache profile for provenance: "product" | "mid".
    public var evalCache: String?
    /// Resolved VAE mid-block attention chunk for provenance (0 = MLXFast SDPA).
    public var vaeAttnChunk: Int?
    /// Resolved VAE tile geometry for provenance (schema 1.4).
    public var vaeTileSize: Int?
    public var vaeTileOverlap: Int?
    public var vaeTileBlend: String?
    /// GPU-sync Steel FA vs FFN vs processQKV split (ranking only).
    public var opProfile: Bool
    /// Reference image for I2I / identity-I2I modes.
    public var imagePath: String?
    /// I2I denoise strength. Defaults: 0.8 strength / 0.9 identity.
    public var strength: Float?
    /// Enable identity stack when `--image` is used with a diagnostic mode.
    public var identity: Bool?

    public init(
        mode: BenchMode = .t2i,
        label: String = "baseline",
        prompt: String = "A red fox in a snowy forest at sunrise, photorealistic.",
        width: Int = 1024,
        height: Int = 1024,
        steps: Int = 4,
        seed: UInt64 = 42,
        trials: Int = 3,
        warmup: Int = 1,
        cooldownSeconds: Double = 0,
        withQuality: Bool = false,
        outputDirectory: String? = nil,
        cacheLimitBytes: UInt64? = nil,
        probeDensity: ProbeDensity = .denoise,
        resetPeakEachPhase: Bool = false,
        failSoft: Bool = false,
        ladderSides: [Int] = [512, 640, 768, 896, 1024],
        attentionQueryChunkSize: Int? = nil,
        attentionQueryChunkThreshold: Int? = nil,
        attentionF16SeqThreshold: Int? = nil,
        attentionLinearF16: Bool? = nil,
        attentionLinearChunkSize: Int? = nil,
        attentionLinearChunkThreshold: Int? = nil,
        attentionBackend: String? = nil,
        attentionBlockClearSeqThreshold: Int? = nil,
        attentionBlockClearInterval: Int? = nil,
        attentionAsyncEvalInterval: Int? = nil,
        attentionJointSeqF16: Bool? = nil,
        attentionF16FullEpilogue: Bool? = nil,
        attentionDynamicScale: Bool? = nil,
        attentionQkvCheckpoint: Bool? = nil,
        vaeTileThreshold: Int? = nil,
        textTokens: String? = nil,
        padContent: String? = nil,
        vaeEngine: String? = nil,
        vaeVariant: String? = nil,
        evalCache: String? = nil,
        vaeAttnChunk: Int? = nil,
        vaeTileSize: Int? = nil,
        vaeTileOverlap: Int? = nil,
        vaeTileBlend: String? = nil,
        opProfile: Bool = false,
        imagePath: String? = nil,
        strength: Float? = nil,
        identity: Bool? = nil
    ) {
        self.mode = mode
        self.label = label
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.seed = seed
        self.trials = trials
        self.warmup = warmup
        self.cooldownSeconds = cooldownSeconds
        self.withQuality = withQuality
        self.outputDirectory = outputDirectory
        self.cacheLimitBytes = cacheLimitBytes
        self.probeDensity = probeDensity
        self.resetPeakEachPhase = resetPeakEachPhase
        self.failSoft = failSoft
        self.ladderSides = ladderSides
        self.attentionQueryChunkSize = attentionQueryChunkSize
        self.attentionQueryChunkThreshold = attentionQueryChunkThreshold
        self.attentionF16SeqThreshold = attentionF16SeqThreshold
        self.attentionLinearF16 = attentionLinearF16
        self.attentionLinearChunkSize = attentionLinearChunkSize
        self.attentionLinearChunkThreshold = attentionLinearChunkThreshold
        self.attentionBackend = attentionBackend
        self.attentionBlockClearSeqThreshold = attentionBlockClearSeqThreshold
        self.attentionBlockClearInterval = attentionBlockClearInterval
        self.attentionAsyncEvalInterval = attentionAsyncEvalInterval
        self.attentionJointSeqF16 = attentionJointSeqF16
        self.attentionF16FullEpilogue = attentionF16FullEpilogue
        self.attentionDynamicScale = attentionDynamicScale
        self.attentionQkvCheckpoint = attentionQkvCheckpoint
        self.vaeTileThreshold = vaeTileThreshold
        self.textTokens = textTokens
        self.padContent = padContent
        self.vaeEngine = vaeEngine
        self.vaeVariant = vaeVariant
        self.evalCache = evalCache
        self.vaeAttnChunk = vaeAttnChunk
        self.vaeTileSize = vaeTileSize
        self.vaeTileOverlap = vaeTileOverlap
        self.vaeTileBlend = vaeTileBlend
        self.opProfile = opProfile
        self.imagePath = imagePath
        self.strength = strength
        self.identity = identity
    }

    /// Config tuned for pressure-map diagnosis.
    public static func pressureMap(
        width: Int,
        height: Int,
        label: String = "pressure-map",
        seed: UInt64 = 42
    ) -> BenchConfig {
        BenchConfig(
            mode: .pressureMap,
            label: label,
            width: width,
            height: height,
            steps: 4,
            seed: seed,
            trials: 1,
            warmup: 0,
            probeDensity: .blocks,
            resetPeakEachPhase: true,
            failSoft: true
        )
    }
}

// MARK: - Memory

public struct MemoryPoint: Sendable, Codable, Equatable {
    public var label: String
    public var rssBytes: UInt64
    public var mlxActiveBytes: UInt64
    public var mlxCacheBytes: UInt64
    public var mlxPeakBytes: UInt64
    public var timestamp: TimeInterval
    public var mlxActiveDeltaBytes: Int64?
    public var recommendedWorkingSetBytes: UInt64?
    public var headroomBytes: Int64?
    public var note: String?

    public init(
        label: String,
        rssBytes: UInt64,
        mlxActiveBytes: UInt64,
        mlxCacheBytes: UInt64,
        mlxPeakBytes: UInt64,
        timestamp: TimeInterval = Date().timeIntervalSince1970,
        mlxActiveDeltaBytes: Int64? = nil,
        recommendedWorkingSetBytes: UInt64? = nil,
        headroomBytes: Int64? = nil,
        note: String? = nil
    ) {
        self.label = label
        self.rssBytes = rssBytes
        self.mlxActiveBytes = mlxActiveBytes
        self.mlxCacheBytes = mlxCacheBytes
        self.mlxPeakBytes = mlxPeakBytes
        self.timestamp = timestamp
        self.mlxActiveDeltaBytes = mlxActiveDeltaBytes
        self.recommendedWorkingSetBytes = recommendedWorkingSetBytes
        self.headroomBytes = headroomBytes
        self.note = note
    }
}

// MARK: - Pressure

public struct PressureRankedPoint: Sendable, Codable, Equatable {
    public var label: String
    public var mlxActiveBytes: UInt64
    public var mlxActiveDeltaBytes: Int64
    public var shareOfRecommendedWS: Double?
    public var note: String?

    public init(
        label: String,
        mlxActiveBytes: UInt64,
        mlxActiveDeltaBytes: Int64,
        shareOfRecommendedWS: Double? = nil,
        note: String? = nil
    ) {
        self.label = label
        self.mlxActiveBytes = mlxActiveBytes
        self.mlxActiveDeltaBytes = mlxActiveDeltaBytes
        self.shareOfRecommendedWS = shareOfRecommendedWS
        self.note = note
    }
}

public struct CanvasAnalytics: Sendable, Codable, Equatable {
    public var width: Int
    public var height: Int
    public var textSeqLen: Int
    public var imageSeqLen: Int
    public var jointSeqLen: Int
    public var packedH: Int
    public var packedW: Int
    /// Extra packed tokens when identity I2I concatenates a reference (nil for T2I).
    public var referenceSeqLen: Int?
    /// Rough single-stream activation footprint estimate (bytes, f32 activations).
    public var estSingleStreamActBytes: UInt64
    public var note: String

    public init(
        width: Int,
        height: Int,
        textSeqLen: Int,
        imageSeqLen: Int,
        jointSeqLen: Int,
        packedH: Int,
        packedW: Int,
        referenceSeqLen: Int? = nil,
        estSingleStreamActBytes: UInt64,
        note: String
    ) {
        self.width = width
        self.height = height
        self.textSeqLen = textSeqLen
        self.imageSeqLen = imageSeqLen
        self.jointSeqLen = jointSeqLen
        self.packedH = packedH
        self.packedW = packedW
        self.referenceSeqLen = referenceSeqLen
        self.estSingleStreamActBytes = estSingleStreamActBytes
        self.note = note
    }
}

public struct PressureReport: Sendable, Codable, Equatable {
    public var timeline: [MemoryPoint]
    public var rankedByActive: [PressureRankedPoint]
    public var rankedByDelta: [PressureRankedPoint]
    public var analytic: CanvasAnalytics
    public var lastProbeBeforeFailure: String?
    public var oom: Bool
    public var phasePeaks: [String: UInt64]
    public var probeDensity: String

    public init(
        timeline: [MemoryPoint] = [],
        rankedByActive: [PressureRankedPoint] = [],
        rankedByDelta: [PressureRankedPoint] = [],
        analytic: CanvasAnalytics,
        lastProbeBeforeFailure: String? = nil,
        oom: Bool = false,
        phasePeaks: [String: UInt64] = [:],
        probeDensity: String = ProbeDensity.off.rawValue
    ) {
        self.timeline = timeline
        self.rankedByActive = rankedByActive
        self.rankedByDelta = rankedByDelta
        self.analytic = analytic
        self.lastProbeBeforeFailure = lastProbeBeforeFailure
        self.oom = oom
        self.phasePeaks = phasePeaks
        self.probeDensity = probeDensity
    }
}

// MARK: - Timings

public struct StageTimingsMs: Sendable, Codable, Equatable {
    public var e2e: Double
    public var loadTe: Double?
    public var encodeTe: Double?
    public var unloadTe: Double?
    public var loadDit: Double?
    public var denoiseTotal: Double?
    public var denoiseSteps: [Double]
    public var unloadDit: Double?
    public var loadVae: Double?
    public var decodeVae: Double?
    public var unloadVae: Double?
    public var exportPng: Double?
    public var loadOnly: [String: Double]

    public init(
        e2e: Double = 0,
        loadTe: Double? = nil,
        encodeTe: Double? = nil,
        unloadTe: Double? = nil,
        loadDit: Double? = nil,
        denoiseTotal: Double? = nil,
        denoiseSteps: [Double] = [],
        unloadDit: Double? = nil,
        loadVae: Double? = nil,
        decodeVae: Double? = nil,
        unloadVae: Double? = nil,
        exportPng: Double? = nil,
        loadOnly: [String: Double] = [:]
    ) {
        self.e2e = e2e
        self.loadTe = loadTe
        self.encodeTe = encodeTe
        self.unloadTe = unloadTe
        self.loadDit = loadDit
        self.denoiseTotal = denoiseTotal
        self.denoiseSteps = denoiseSteps
        self.unloadDit = unloadDit
        self.loadVae = loadVae
        self.decodeVae = decodeVae
        self.unloadVae = unloadVae
        self.exportPng = exportPng
        self.loadOnly = loadOnly
    }

    public var denoiseStepMeanMs: Double? {
        guard !denoiseSteps.isEmpty else { return nil }
        return denoiseSteps.reduce(0, +) / Double(denoiseSteps.count)
    }
}

// MARK: - Trial / Report

public struct BenchTrial: Sendable, Codable, Equatable {
    public var index: Int
    public var cold: Bool
    public var timingsMs: StageTimingsMs
    public var memorySamples: [MemoryPoint]
    public var peakRssBytes: UInt64
    public var peakMlxActiveBytes: UInt64
    public var peakMlxPeakBytes: UInt64
    public var peakMlxCacheBytes: UInt64
    public var outputPath: String?
    public var qualityTechnicalScore: Float?
    public var qualityColorMatch: Bool?
    public var qualityReferenceSSIM: Float?
    public var qualityFidelityScore: Float?
    public var qualityFaceReferenceSSIM: Float?
    public var qualityFaceFidelityScore: Float?
    public var generatedFaceCount: Int?
    public var referenceFaceCount: Int?
    public var hostBefore: HostContentionSnapshot?
    public var hostAfter: HostContentionSnapshot?
    public var error: String?
    public var lastProbeId: String?
    public var opProfile: DiTOpProfileReport?

    public init(
        index: Int,
        cold: Bool,
        timingsMs: StageTimingsMs,
        memorySamples: [MemoryPoint],
        peakRssBytes: UInt64,
        peakMlxActiveBytes: UInt64,
        peakMlxPeakBytes: UInt64,
        peakMlxCacheBytes: UInt64,
        outputPath: String? = nil,
        qualityTechnicalScore: Float? = nil,
        qualityColorMatch: Bool? = nil,
        qualityReferenceSSIM: Float? = nil,
        qualityFidelityScore: Float? = nil,
        qualityFaceReferenceSSIM: Float? = nil,
        qualityFaceFidelityScore: Float? = nil,
        generatedFaceCount: Int? = nil,
        referenceFaceCount: Int? = nil,
        hostBefore: HostContentionSnapshot? = nil,
        hostAfter: HostContentionSnapshot? = nil,
        error: String? = nil,
        lastProbeId: String? = nil,
        opProfile: DiTOpProfileReport? = nil
    ) {
        self.index = index
        self.cold = cold
        self.timingsMs = timingsMs
        self.memorySamples = memorySamples
        self.peakRssBytes = peakRssBytes
        self.peakMlxActiveBytes = peakMlxActiveBytes
        self.peakMlxPeakBytes = peakMlxPeakBytes
        self.peakMlxCacheBytes = peakMlxCacheBytes
        self.outputPath = outputPath
        self.qualityTechnicalScore = qualityTechnicalScore
        self.qualityColorMatch = qualityColorMatch
        self.qualityReferenceSSIM = qualityReferenceSSIM
        self.qualityFidelityScore = qualityFidelityScore
        self.qualityFaceReferenceSSIM = qualityFaceReferenceSSIM
        self.qualityFaceFidelityScore = qualityFaceFidelityScore
        self.generatedFaceCount = generatedFaceCount
        self.referenceFaceCount = referenceFaceCount
        self.hostBefore = hostBefore
        self.hostAfter = hostAfter
        self.error = error
        self.lastProbeId = lastProbeId
        self.opProfile = opProfile
    }
}

public struct StatSummary: Sendable, Codable, Equatable {
    public var mean: Double
    public var stdev: Double
    public var min: Double
    public var max: Double
    public var p50: Double
    public var p95: Double
    public var count: Int

    public static func from(_ values: [Double]) -> StatSummary? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let n = Double(sorted.count)
        let mean = sorted.reduce(0, +) / n
        let varSum = sorted.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) }
        let stdev = sqrt(varSum / n)
        let p50 = percentile(sorted, 0.50)
        let p95 = percentile(sorted, 0.95)
        return StatSummary(
            mean: mean, stdev: stdev,
            min: sorted.first!, max: sorted.last!,
            p50: p50, p95: p95, count: sorted.count
        )
    }

    private static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let idx = Swift.min(sorted.count - 1, Swift.max(0, Int((Double(sorted.count - 1) * p).rounded())))
        return sorted[idx]
    }
}

public struct BenchAggregate: Sendable, Codable, Equatable {
    public var e2eMs: StatSummary?
    public var denoiseStepMeanMs: StatSummary?
    public var peakRssBytes: StatSummary?
    public var peakMlxActiveBytes: StatSummary?
    public var peakMlxPeakBytes: StatSummary?
    public var loadTeMs: StatSummary?
    public var encodeTeMs: StatSummary?
    public var loadDitMs: StatSummary?
    public var denoiseTotalMs: StatSummary?
    public var loadVaeMs: StatSummary?
    public var decodeVaeMs: StatSummary?
    public var exportPngMs: StatSummary?
}

public struct SystemSnapshot: Sendable, Codable, Equatable {
    public var hostname: String
    public var osVersion: String
    public var processId: Int32
    public var physicalMemoryBytes: UInt64
    public var processorCount: Int
    public var recommendedMaxWorkingSetBytes: UInt64?
    public var thermalState: String
    public var mlxCacheLimitBytes: UInt64?
    public var mlxMemoryLimitBytes: UInt64?
    public var imarelloGitSha: String?
    /// Metal device name (e.g. "Apple M2").
    public var gpuName: String?
    /// e.g. "Metal 4" when known.
    public var metalSupport: String?
    /// True when GPU Neural Accelerators are expected (M5/A19+); false on M1–M4.
    public var hasNeuralAccelerators: Bool?
    /// Highest Apple MTLGPUFamily raw value the device reports (schema 1.4).
    public var appleGpuFamilyRaw: Int? = nil
}

public struct BenchReport: Sendable, Codable, Equatable {
    public var schemaVersion: String
    public var label: String
    public var createdAt: String
    public var system: SystemSnapshot
    public var config: BenchConfig
    public var trials: [BenchTrial]
    public var aggregate: BenchAggregate
    public var pressure: PressureReport?

    /// 1.4 (2026-08-18): + evalCache, vaeAttnChunk, vaeTile{Size,Overlap,Blend} provenance.
    public static let currentSchema = "1.5"
}
