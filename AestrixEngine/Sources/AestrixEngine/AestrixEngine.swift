import Foundation
import MLX

// MARK: - Public configuration & result types

/// Configuration for a generation or edit request.
///
/// Defaults match FLUX.2-klein-4B's distilled recipe: 4 rectified-flow steps and
/// guidance = 1.0 (no classifier-free guidance, so a single forward pass per step).
public struct GenConfig: Sendable {
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidance: Float
    public var seed: UInt64
    public var dtype: MLX.DType

    public init(
        width: Int = 1024,
        height: Int = 1024,
        steps: Int = 4,
        guidance: Float = 1.0,
        seed: UInt64 = 0,
        dtype: MLX.DType = .float16
    ) {
        // FLUX.2 Klein's VAE compresses 16× spatially; both dims must be divisible by 16.
        precondition(width % 16 == 0 && height % 16 == 0, "width/height must be divisible by 16")
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.dtype = dtype
    }
}

/// Platform-neutral generated image stored as 8-bit RGBA. The host app converts this
/// to `UIImage` (kept out of the package so it builds/test on macOS too).
public struct AestrixImage: Sendable {
    public let width: Int
    public let height: Int
    public let rgba: [UInt8]

    public init(width: Int, height: Int, rgba: [UInt8]) {
        precondition(rgba.count == width * height * 4, "rgba buffer must be width*height*4 bytes")
        self.width = width
        self.height = height
        self.rgba = rgba
    }
}

/// Report returned by ``AestrixEngine/load(modelDir:)``.
public struct LoadReport: Sendable {
    public let modelDir: URL
    public let wiredBytes: Int64
    public let modelSizeBytes: Int64

    public init(modelDir: URL, wiredBytes: Int64, modelSizeBytes: Int64) {
        self.modelDir = modelDir
        self.wiredBytes = wiredBytes
        self.modelSizeBytes = modelSizeBytes
    }
}

/// Result of ``AestrixEngine/verifyIntegration()`` — proves MLX executes on this device.
public struct IntegrationReport: Sendable {
    public let ok: Bool
    public let matmulRows: Int
    public let resultSum: Float
    public let message: String
}

// MARK: - Engine façade

/// Highly optimized MLX engine for FLUX.2-klein-4B on iOS.
///
/// `AestrixEngine` is the sole public entry point. It owns the staged model lifecycle
/// (see `ModelContainer`) and exposes text-to-image generation and single-image
/// instruction editing. All MLX state is encapsulated behind this actor so callers
/// never touch the non-`Sendable` `MLXArray` graph directly.
public actor AestrixEngine {

    public init() {}

    /// Load the model from `modelDir` (staged; reports wired bytes). Not yet implemented.
    public func load(modelDir: URL) async throws -> LoadReport {
        throw AestrixError.notImplemented("load(modelDir:) — Milestone 1 (IO layer)")
    }

    /// Generate an image from a text prompt. Not yet implemented.
    public func generate(prompt: String, config: GenConfig) async throws -> AestrixImage {
        throw AestrixError.notImplemented("generate(prompt:config:) — Milestone 5 (t2i pipeline)")
    }

    /// Edit `image` following a natural-language `instruction`. Not yet implemented.
    public func edit(image: AestrixImage, instruction: String, config: GenConfig) async throws -> AestrixImage {
        throw AestrixError.notImplemented("edit(image:instruction:config:) — Milestone 6 (image edit)")
    }

    /// Free all wired memory. Not yet implemented.
    public func unload() async throws {
        throw AestrixError.notImplemented("unload() — Milestone 1 (IO layer)")
    }

    /// Sanity check that MLX is wired up and the GPU backend executes on this device.
    ///
    /// Runs a small fp32 matmul + reduction and forces evaluation. Used by the demo app
    /// (and CI) to confirm the dependency resolved and Metal works on the target.
    public func verifyIntegration() -> IntegrationReport {
        let n = 512
        // fp32 here on purpose: correctness of the sanity sum, not speed.
        let a = MLXRandom.normal([n, n])
        let b = MLXRandom.normal([n, n])
        let c = a.matmul(b)
        let s = c.sum()
        eval(s)
        let value = s.item(Float.self)
        return IntegrationReport(
            ok: true,
            matmulRows: n,
            resultSum: value,
            message: "MLX executed a \(n)×\(n) matmul + sum reduction on-device."
        )
    }
}
