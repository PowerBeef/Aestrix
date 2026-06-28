import Foundation
import MLX
#if canImport(AppKit)
import AppKit
#endif

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

    #if canImport(AppKit)
    /// Save as a PNG file (macOS). Returns the file URL.
    /// Parent directories are created automatically.
    @discardableResult
    public func savePNG(to url: URL) throws -> URL {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let info = CGImageAlphaInfo.premultipliedLast.rawValue
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let cgImage = CGImage(
                width: width, height: height,
                bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4, space: cs,
                bitmapInfo: CGBitmapInfo(rawValue: info),
                provider: provider, decode: nil,
                shouldInterpolate: false, intent: .defaultIntent) else {
            throw AestrixError.invalidInput("failed to create CGImage")
        }
        let bitmap = NSBitmapImageRep(cgImage: cgImage)
        guard let png = bitmap.representation(using: .png, properties: [:]) else {
            throw AestrixError.invalidInput("failed to encode PNG")
        }
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try png.write(to: url)
        return url
    }
    #endif
}

/// Report returned by ``AestrixEngine/load(modelDir:)``.
public struct LoadReport: Sendable {
    public let modelDir: URL
    public let wiredBytes: Int64
    public let modelSizeBytes: Int64
    public let filesPresent: Int
    public let filesTotal: Int
    public let missingFiles: [String]

    /// `true` when every file in the manifest is present and size-verified.
    public var isComplete: Bool { filesPresent == filesTotal }

    public init(modelDir: URL, wiredBytes: Int64, modelSizeBytes: Int64,
                filesPresent: Int, filesTotal: Int, missingFiles: [String]) {
        self.modelDir = modelDir
        self.wiredBytes = wiredBytes
        self.modelSizeBytes = modelSizeBytes
        self.filesPresent = filesPresent
        self.filesTotal = filesTotal
        self.missingFiles = missingFiles
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

    private let manifestFiles: [ModelFile]
    private let manifestBaseURL: URL
    private var modelDir: URL?

    public init(
        files: [ModelFile] = Flux2Klein4B4Bit.files,
        baseURL: URL = Flux2Klein4B4Bit.baseURL
    ) {
        self.manifestFiles = files
        self.manifestBaseURL = baseURL
    }

    /// Default on-device storage: `<Application Support>/Aestrix/models/flux2-klein-4b-4bit`.
    public static func defaultModelDirectory() throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        let dir = appSupport
            .appendingPathComponent("Aestrix", isDirectory: true)
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent("flux2-klein-4b-4bit", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Download every missing model file into `overrideDir` (default: app-support dir).
    /// Idempotent — files already present and size-verified are skipped.
    public func downloadModel(
        into overrideDir: URL? = nil,
        onProgress: @Sendable @escaping (DownloadProgress) -> Void = { _ in }
    ) async throws -> URL {
        let dir = try overrideDir ?? Self.defaultModelDirectory()
        let downloader = ModelDownloader(
            files: manifestFiles, baseURL: manifestBaseURL, destinationRoot: dir)
        try await downloader.ensureDownloaded(onProgress: onProgress)
        return dir
    }

    /// Verify the model at `modelDir` is fully present and mark the engine ready.
    ///
    /// M1 only verifies + records the directory — it does NOT load weights into modules
    /// (that staged wiring begins in M3/M4), so `wiredBytes` is 0 here. Call
    /// ``downloadModel(into:onProgress:)`` first if ``LoadReport/missingFiles`` is non-empty.
    public func load(modelDir: URL) async throws -> LoadReport {
        let total = manifestFiles.count
        let downloader = ModelDownloader(
            files: manifestFiles, baseURL: manifestBaseURL, destinationRoot: modelDir)
        let missing = await downloader.missingFiles().map(\.relativePath)
        let present = total - missing.count
        self.modelDir = modelDir
        return LoadReport(
            modelDir: modelDir,
            wiredBytes: 0,
            modelSizeBytes: manifestFiles.reduce(0) { $0 + $1.size },
            filesPresent: present,
            filesTotal: total,
            missingFiles: missing)
    }

    /// Generate an image from a text prompt.
    ///
    /// Requires the model to be downloaded first (`downloadModel`). Loads all three
    /// submodels, runs the 4-step rectified-flow denoise loop, and decodes via the VAE.
    public func generate(prompt: String, config: GenConfig) async throws -> AestrixImage {
        guard let dir = modelDir else { throw AestrixError.modelNotLoaded }

        let vaeScaleFactor = 8   // VAE spatial downscale
        let latentH = config.height / (vaeScaleFactor * 2)   // H/16 (after patchify)
        let latentW = config.width / (vaeScaleFactor * 2)
        let imageSeqLen = latentH * latentW

        // 1. Tokenize + encode prompt
        AestrixLog.notice("Encoding prompt…")
        let tokenizer = try await AestrixTokenizer(modelDir: dir)
        let inputIds = tokenizer.encode(prompt: prompt)
        let (teWeights, _) = try WeightLoader.loadWeights(submodelDir: dir.appendingPathComponent("text_encoder"))
        let textEncoder = Qwen3TextEncoder()
        textEncoder.loadWeights(teWeights)
        let ctx = textEncoder.promptEmbeds(MLXArray(inputIds, [1, inputIds.count]))
        eval(ctx)

        // 2. Build coordinates
        let txtIds = CoordinateBuilder.textIds(seqLen: 512)
        let imgIds = CoordinateBuilder.imageIds(height: latentH, width: latentW)

        // 3. Noise → patchify → pack
        MLXRandom.seed(config.seed)
        let noise = MLXRandom.normal([1, 128, latentH, latentW])
        var latents = LatentTransforms.pack(noise)   // [1, latentH*latentW, 128]

        // 4. Scheduler
        let scheduler = RectifiedFlowScheduler(numSteps: config.steps, imageSeqLen: imageSeqLen)

        // 5. Load transformer + denoise loop
        AestrixLog.notice("Loading transformer…")
        let (trWeights, _) = try WeightLoader.loadWeights(submodelDir: dir.appendingPathComponent("transformer"))
        let transformer = KleinTransformer()
        transformer.loadWeights(trWeights)

        for t in 0..<config.steps {
            let ts = scheduler.timesteps[t]  // scalar MLXArray (0-dim), passed as-is
            AestrixLog.notice("Denoise step \(t + 1)/\(config.steps)")
            let pred = transformer(latents, ctx, imgIds: imgIds, txtIds: txtIds, timestep: ts)
            latents = scheduler.step(noise: pred, timestepIndex: t, latents: latents)
            eval(latents)
        }

        // 6. Unpack → bn denormalize → unpatchify → VAE decode
        AestrixLog.notice("Decoding…")
        let packed = LatentTransforms.unpack(latents, height: latentH, width: latentW)  // [1, 128, lH, lW]

        // Load VAE weights + bn stats
        let (vaeWeights, vaeQuant) = try WeightLoader.loadWeights(submodelDir: dir.appendingPathComponent("vae"))
        let vae = FluxVAE(precision: .float32)
        vae.loadWeights(vaeWeights)

        // bn denormalize using running_mean/var from the VAE checkpoint
        let bnMean = vaeWeights["bn.running_mean"]!
        let bnVar = vaeWeights["bn.running_var"]!
        let denormed = LatentTransforms.bnDenormalize(packed, runningMean: bnMean, runningVar: bnVar, eps: 1e-4)
        let unpatchified = LatentTransforms.unpatchify(denormed)  // [1, 32, H/8, W/8]
        let decoded = vae.decode(unpatchified)                    // [1, H, W, 3]
        eval(decoded)

        // 7. Convert to AestrixImage
        return ImageConversion.toAestrixImage(decoded)
    }

    /// Edit `image` following a natural-language `instruction`. Not yet implemented.
    public func edit(image: AestrixImage, instruction: String, config: GenConfig) async throws -> AestrixImage {
        throw AestrixError.notImplemented("edit(image:instruction:config:) — Milestone 6 (image edit)")
    }

    /// Free all wired memory. Nothing resident in M1 yet.
    public func unload() async throws {
        modelDir = nil
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
