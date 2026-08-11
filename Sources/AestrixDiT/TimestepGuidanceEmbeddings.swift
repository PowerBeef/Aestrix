import Foundation
import MLX
import MLXNN
import AestrixCore

/// Timestep (+ optional guidance) → MLP embedding. Distilled Klein uses guidance_embeds=false.
public final class Flux2TimestepGuidanceEmbeddings: Module {
    let inChannels: Int
    let embeddingDim: Int
    let guidanceEmbeds: Bool

    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    @ModuleInfo(key: "guidance_linear_1") var guidanceLinear1: Linear?
    @ModuleInfo(key: "guidance_linear_2") var guidanceLinear2: Linear?

    public init(
        inChannels: Int = ModelConstants.timestepEmbedChannels,
        embeddingDim: Int = ModelConstants.innerDim,
        guidanceEmbeds: Bool = false
    ) {
        self.inChannels = inChannels
        self.embeddingDim = embeddingDim
        self.guidanceEmbeds = guidanceEmbeds
        self._linear1.wrappedValue = Linear(inChannels, embeddingDim, bias: false)
        self._linear2.wrappedValue = Linear(embeddingDim, embeddingDim, bias: false)
        if guidanceEmbeds {
            self._guidanceLinear1.wrappedValue = Linear(inChannels, embeddingDim, bias: false)
            self._guidanceLinear2.wrappedValue = Linear(embeddingDim, embeddingDim, bias: false)
        } else {
            self._guidanceLinear1.wrappedValue = nil
            self._guidanceLinear2.wrappedValue = nil
        }
        super.init()
    }

    public func callAsFunction(_ timestep: MLXArray, guidance: MLXArray? = nil) -> MLXArray {
        var t = Self.timestepEmbedding(timestep.asType(.float32), dim: inChannels)
        var emb = linear2(silu(linear1(t)))
        if let guidance, let g1 = guidanceLinear1, let g2 = guidanceLinear2 {
            let g = Self.timestepEmbedding(guidance.asType(.float32), dim: inChannels)
            emb = emb + g2(silu(g1(g)))
        }
        return emb
    }

    static func timestepEmbedding(_ timesteps: MLXArray, dim: Int, flipSinToCos: Bool = true) -> MLXArray {
        let half = dim / 2
        let freqs = exp(-log(Float(10000.0)) * MLXArray(0..<half).asType(.float32) / Float(half))
        // timesteps: [B] → [B, 1]
        let args = timesteps.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        let sins = sin(args)
        let cosines = cos(args)
        var emb: MLXArray
        if flipSinToCos {
            emb = concatenated([cosines, sins], axis: -1)
        } else {
            emb = concatenated([sins, cosines], axis: -1)
        }
        if dim % 2 == 1 {
            emb = concatenated([emb, MLXArray.zeros([emb.dim(0), 1])], axis: -1)
        }
        return emb
    }
}
