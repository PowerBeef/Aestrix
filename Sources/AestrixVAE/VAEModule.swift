import Foundation
import MLX
import MLXNN
import AestrixCore
import AestrixWeights

/// Which VAE weights to materialize (M2 / Tier L: prefer decode-only for pure T2I).
public enum VAELoadMode: String, Sendable, Codable, CaseIterable {
    /// Encoder + decoder (~165 MB pack).
    case full
    /// Decoder + BN + post_quant only (~97 MB). T2I default.
    case decodeOnly = "decode-only"
    /// Encoder + BN + quant_conv only (~67 MB). I2I stage-0 encode.
    case encodeOnly = "encode-only"
}

/// Loadable FLUX.2 VAE for staged pipeline residency.
public final class VAEModule: LoadableModule, @unchecked Sendable {
    public let moduleName = "vae"
    public private(set) var isLoaded = false
    public private(set) var loadMode: VAELoadMode = .full

    public private(set) var model: Flux2VAE?
    public private(set) var decodeOnlyModel: Flux2VAEDecoderOnly?
    public private(set) var encodeOnlyModel: Flux2VAEEncoderOnly?

    private let snapshot: ModelSnapshot?

    public init(snapshot: ModelSnapshot? = nil) {
        self.snapshot = snapshot
    }

    public func load() async throws {
        try await load(mode: .full)
    }

    public func load(mode: VAELoadMode) async throws {
        if isLoaded {
            if loadMode == mode { return }
            await unload()
        }
        if let snapshot {
            try snapshot.validateLayout()
            switch mode {
            case .full:
                model = try VAEWeights.load(from: snapshot.vaeDirectory)
                decodeOnlyModel = nil
                encodeOnlyModel = nil
            case .decodeOnly:
                decodeOnlyModel = try VAEWeights.loadDecodeOnly(from: snapshot.vaeDirectory)
                model = nil
                encodeOnlyModel = nil
            case .encodeOnly:
                encodeOnlyModel = try VAEWeights.loadEncodeOnly(from: snapshot.vaeDirectory)
                model = nil
                decodeOnlyModel = nil
            }
        } else {
            model = nil
            decodeOnlyModel = nil
            encodeOnlyModel = nil
        }
        loadMode = mode
        isLoaded = true
    }

    public func unload() async {
        let had = model != nil || decodeOnlyModel != nil || encodeOnlyModel != nil
        model = nil
        decodeOnlyModel = nil
        encodeOnlyModel = nil
        isLoaded = false
        loadMode = .full
        if had {
            Memory.clearCache()
        }
    }

    public var parameterLeafCount: Int {
        if let model { return model.parameters().flattened().count }
        if let decodeOnlyModel { return decodeOnlyModel.parameters().flattened().count }
        if let encodeOnlyModel { return encodeOnlyModel.parameters().flattened().count }
        return 0
    }

    public func encode(_ image: MLXArray) throws -> MLXArray {
        if let model, isLoaded { return model.encode(image) }
        if let encodeOnlyModel, isLoaded { return encodeOnlyModel.encode(image) }
        throw AestrixError.moduleNotLoaded(
            decodeOnlyModel != nil ? "\(moduleName) (decode-only; encode unavailable)" : moduleName)
    }

    /// Encode image to BN-normalized packed latents for DiT I2I (`[B, 128, H/16, W/16]`).
    public func encodePackedForDiT(_ image: MLXArray) throws -> MLXArray {
        if let model, isLoaded { return model.encodePackedForDiT(image) }
        if let encodeOnlyModel, isLoaded { return encodeOnlyModel.encodePackedForDiT(image) }
        throw AestrixError.moduleNotLoaded(
            decodeOnlyModel != nil ? "\(moduleName) (decode-only; encode unavailable)" : moduleName)
    }

    public func decode(_ latents: MLXArray) throws -> MLXArray {
        if let model, isLoaded { return model.decode(latents) }
        if let decodeOnlyModel, isLoaded { return decodeOnlyModel.decode(latents) }
        throw AestrixError.moduleNotLoaded(moduleName)
    }

    public func decodePacked(
        _ packed: MLXArray,
        tileConfig: VAETileConfig = .default
    ) throws -> MLXArray {
        if let model, isLoaded { return model.decodePackedLatents(packed, tileConfig: tileConfig) }
        if let decodeOnlyModel, isLoaded {
            return decodeOnlyModel.decodePackedLatents(packed, tileConfig: tileConfig)
        }
        throw AestrixError.moduleNotLoaded(moduleName)
    }
}
