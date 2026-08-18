import Foundation
import MLX
import MLXNN
import ImarelloCore
import ImarelloWeights

/// Qwen3-4B 3-layer-tap text encoder (loadable for staged TE→DiT→VAE).
public final class TextEncoderModule: LoadableModule, @unchecked Sendable {
    public let moduleName = "text_encoder"
    public private(set) var isLoaded = false

    public private(set) var model: Qwen3TextEncoder?
    public private(set) var tokenizer: QwenTokenizer?

    private let snapshot: ModelSnapshot?
    private let bits: Int
    private let groupSize: Int

    public init(
        snapshot: ModelSnapshot? = nil,
        bits: Int = TextEncoderWeights.defaultBits,
        groupSize: Int = TextEncoderWeights.defaultGroupSize
    ) {
        self.snapshot = snapshot
        self.bits = bits
        self.groupSize = groupSize
    }

    /// Staged load: with snapshot → quant weights + tokenizer; without → residency flag only.
    public func load() async throws {
        if isLoaded { throw ImarelloError.moduleAlreadyLoaded(moduleName) }
        if let snapshot {
            try snapshot.validateLayout()
            model = try TextEncoderWeights.loadQuantized(
                from: snapshot.textEncoderDirectory,
                bits: bits,
                groupSize: groupSize
            )
            tokenizer = try QwenTokenizer.load(from: snapshot.tokenizerDirectory)
        } else {
            model = nil
            tokenizer = nil
        }
        isLoaded = true
    }

    public func unload() async {
        let had = model != nil
        model = nil
        tokenizer = nil
        isLoaded = false
        if had {
            Memory.clearCache()
        }
    }

    public var parameterLeafCount: Int {
        guard let model else { return 0 }
        return model.parameters().flattened().count
    }

    /// Encode a prompt to Klein DiT text embeddings `[1, 512, 7680]`.
    public func encode(
        _ prompt: String,
        maxLength: Int = ModelConstants.maxSequenceLength,
        trace: PipelineTrace? = nil
    ) throws -> (embeds: MLXArray, realTokens: Int) {
        guard let model, let tokenizer, isLoaded else {
            throw ImarelloError.moduleNotLoaded(moduleName)
        }
        let (ids, mask) = tokenizer.encodePrompt(prompt, maxLength: maxLength)
        let realTokens = mask.reduce(0, +)
        let inputIds = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        let attentionMask = MLXArray(mask.map { Int32($0) }).reshaped([1, mask.count])
        let embeds = model.encode(
            inputIds: inputIds, attentionMask: attentionMask, trace: trace)
        eval(embeds)
        trace?.probe("te.after_eval", phase: "te", minDensity: .blocks)
        return (embeds, realTokens)
    }

    /// TE-splice path: encode ONLY the real (chat-templated) tokens — no pads
    /// in the window at all, so the mask is pure causal (Steel's fast path)
    /// and attention/MLP work scales with the prompt, not with 512.
    ///
    /// Exactness: under causal attention with tail padding, real-token hidden
    /// states never attend pad positions, so rows `< realTokens` are identical
    /// to a full-window encode. Returns `[1, realTokens, jointDim]`.
    public func encodeRealOnly(
        _ prompt: String,
        maxLength: Int = ModelConstants.maxSequenceLength,
        trace: PipelineTrace? = nil
    ) throws -> (embeds: MLXArray, realTokens: Int) {
        guard let model, let tokenizer, isLoaded else {
            throw ImarelloError.moduleNotLoaded(moduleName)
        }
        let ids = tokenizer.encodePromptUnpadded(prompt, maxLength: maxLength)
        let inputIds = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        let embeds = model.encode(inputIds: inputIds, attentionMask: nil, trace: trace)
        eval(embeds)
        trace?.probe("te.after_eval", phase: "te", minDensity: .blocks)
        return (embeds, ids.count)
    }

    /// Encode pre-tokenized ids (for tests / cross-checks).
    public func encode(
        inputIds: MLXArray,
        attentionMask: MLXArray? = nil,
        trace: PipelineTrace? = nil
    ) throws -> MLXArray {
        guard let model, isLoaded else {
            throw ImarelloError.moduleNotLoaded(moduleName)
        }
        let embeds = model.encode(
            inputIds: inputIds, attentionMask: attentionMask, trace: trace)
        eval(embeds)
        return embeds
    }
}
