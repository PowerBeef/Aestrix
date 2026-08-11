import Foundation
import MLX
import MLXNN
import AestrixCore
import AestrixWeights

/// Loadable FLUX.2 VAE for staged pipeline residency.
public final class VAEModule: LoadableModule, @unchecked Sendable {
    public let moduleName = "vae"
    public private(set) var isLoaded = false
    public private(set) var model: Flux2VAE?

    private let snapshot: ModelSnapshot?

    public init(snapshot: ModelSnapshot? = nil) {
        self.snapshot = snapshot
    }

    public func load() async throws {
        if isLoaded { throw AestrixError.moduleAlreadyLoaded(moduleName) }
        if let snapshot {
            try snapshot.validateLayout()
            model = try VAEWeights.load(from: snapshot.vaeDirectory)
        } else {
            model = nil
        }
        isLoaded = true
    }

    public func unload() async {
        let had = model != nil
        model = nil
        isLoaded = false
        if had {
            Memory.clearCache()
        }
    }

    public var parameterLeafCount: Int {
        guard let model else { return 0 }
        return model.parameters().flattened().count
    }

    public func encode(_ image: MLXArray) throws -> MLXArray {
        guard let model, isLoaded else { throw AestrixError.moduleNotLoaded(moduleName) }
        return model.encode(image)
    }

    /// Encode image to BN-normalized packed latents for DiT I2I (`[B, 128, H/16, W/16]`).
    public func encodePackedForDiT(_ image: MLXArray) throws -> MLXArray {
        guard let model, isLoaded else { throw AestrixError.moduleNotLoaded(moduleName) }
        return model.encodePackedForDiT(image)
    }

    public func decode(_ latents: MLXArray) throws -> MLXArray {
        guard let model, isLoaded else { throw AestrixError.moduleNotLoaded(moduleName) }
        return model.decode(latents)
    }

    public func decodePacked(_ packed: MLXArray) throws -> MLXArray {
        guard let model, isLoaded else { throw AestrixError.moduleNotLoaded(moduleName) }
        return model.decodePackedLatents(packed)
    }
}
