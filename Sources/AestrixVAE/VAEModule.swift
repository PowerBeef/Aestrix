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
}

/// Loadable FLUX.2 VAE for staged pipeline residency.
public final class VAEModule: LoadableModule, @unchecked Sendable {
    public let moduleName = "vae"
    public private(set) var isLoaded = false
    public private(set) var loadMode: VAELoadMode = .full

    public private(set) var model: Flux2VAE?
    public private(set) var decodeOnlyModel: Flux2VAEDecoderOnly?

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
            case .decodeOnly:
                decodeOnlyModel = try VAEWeights.loadDecodeOnly(from: snapshot.vaeDirectory)
                model = nil
            }
        } else {
            model = nil
            decodeOnlyModel = nil
        }
        loadMode = mode
        isLoaded = true
    }

    public func unload() async {
        let had = model != nil || decodeOnlyModel != nil
        model = nil
        decodeOnlyModel = nil
        isLoaded = false
        loadMode = .full
        if had {
            Memory.clearCache()
        }
    }

    public var parameterLeafCount: Int {
        if let model { return model.parameters().flattened().count }
        if let decodeOnlyModel { return decodeOnlyModel.parameters().flattened().count }
        return 0
    }

    public func encode(_ image: MLXArray) throws -> MLXArray {
        guard let model, isLoaded else {
            throw AestrixError.moduleNotLoaded(
                decodeOnlyModel != nil ? "\(moduleName) (decode-only; encode unavailable)" : moduleName)
        }
        return model.encode(image)
    }

    /// Encode image to BN-normalized packed latents for DiT I2I (`[B, 128, H/16, W/16]`).
    public func encodePackedForDiT(_ image: MLXArray) throws -> MLXArray {
        guard let model, isLoaded else {
            throw AestrixError.moduleNotLoaded(
                decodeOnlyModel != nil ? "\(moduleName) (decode-only; encode unavailable)" : moduleName)
        }
        return model.encodePackedForDiT(image)
    }

    public func decode(_ latents: MLXArray) throws -> MLXArray {
        if let model, isLoaded { return model.decode(latents) }
        if let decodeOnlyModel, isLoaded { return decodeOnlyModel.decode(latents) }
        throw AestrixError.moduleNotLoaded(moduleName)
    }

    public func decodePacked(_ packed: MLXArray) throws -> MLXArray {
        if let model, isLoaded { return model.decodePackedLatents(packed) }
        if let decodeOnlyModel, isLoaded { return decodeOnlyModel.decodePackedLatents(packed) }
        throw AestrixError.moduleNotLoaded(moduleName)
    }
}
