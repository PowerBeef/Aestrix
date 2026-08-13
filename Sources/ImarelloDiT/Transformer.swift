import Foundation
import MLX
import MLXNN
import ImarelloCore

/// FLUX.2 Klein MMDiT: 5 double + 20 single stream blocks.
public final class Flux2Transformer: Module {
    public let innerDim: Int
    public let outChannels: Int
    public let numLayers: Int
    public let numSingleLayers: Int
    public let jointAttentionDim: Int

    let posEmbed: Flux2PosEmbed
    @ModuleInfo(key: "time_guidance_embed") var timeGuidanceEmbed: Flux2TimestepGuidanceEmbeddings
    @ModuleInfo(key: "double_stream_modulation_img") var doubleStreamModulationImg: Flux2Modulation
    @ModuleInfo(key: "double_stream_modulation_txt") var doubleStreamModulationTxt: Flux2Modulation
    @ModuleInfo(key: "single_stream_modulation") var singleStreamModulation: Flux2Modulation
    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [Flux2TransformerBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks: [Flux2SingleTransformerBlock]
    @ModuleInfo(key: "norm_out") var normOut: AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public init(
        inChannels: Int = ModelConstants.inChannels,
        outChannels: Int? = nil,
        numLayers: Int = ModelConstants.numDoubleBlocks,
        numSingleLayers: Int = ModelConstants.numSingleBlocks,
        attentionHeadDim: Int = ModelConstants.attentionHeadDim,
        numAttentionHeads: Int = ModelConstants.numAttentionHeads,
        jointAttentionDim: Int = ModelConstants.jointAttentionDim,
        timestepGuidanceChannels: Int = ModelConstants.timestepEmbedChannels,
        mlpRatio: Float = ModelConstants.mlpRatio,
        axesDimsRope: [Int] = ModelConstants.ropeAxesDims,
        ropeTheta: Float = ModelConstants.ropeTheta,
        guidanceEmbeds: Bool = false
    ) {
        self.outChannels = outChannels ?? inChannels
        self.innerDim = numAttentionHeads * attentionHeadDim
        self.numLayers = numLayers
        self.numSingleLayers = numSingleLayers
        self.jointAttentionDim = jointAttentionDim

        self.posEmbed = Flux2PosEmbed(theta: ropeTheta, axesDim: axesDimsRope)
        self._timeGuidanceEmbed.wrappedValue = Flux2TimestepGuidanceEmbeddings(
            inChannels: timestepGuidanceChannels,
            embeddingDim: innerDim,
            guidanceEmbeds: guidanceEmbeds
        )
        self._doubleStreamModulationImg.wrappedValue = Flux2Modulation(dim: innerDim, modParamSets: 2)
        self._doubleStreamModulationTxt.wrappedValue = Flux2Modulation(dim: innerDim, modParamSets: 2)
        self._singleStreamModulation.wrappedValue = Flux2Modulation(dim: innerDim, modParamSets: 1)
        self._xEmbedder.wrappedValue = Linear(inChannels, innerDim, bias: false)
        self._contextEmbedder.wrappedValue = Linear(jointAttentionDim, innerDim, bias: false)
        let dim = innerDim
        let nHeads = numAttentionHeads
        let headDim = attentionHeadDim
        let ratio = mlpRatio
        self._transformerBlocks.wrappedValue = (0..<numLayers).map { _ in
            Flux2TransformerBlock(
                dim: dim,
                numAttentionHeads: nHeads,
                attentionHeadDim: headDim,
                mlpRatio: ratio
            )
        }
        self._singleTransformerBlocks.wrappedValue = (0..<numSingleLayers).map { _ in
            Flux2SingleTransformerBlock(
                dim: dim,
                numAttentionHeads: nHeads,
                attentionHeadDim: headDim,
                mlpRatio: ratio
            )
        }
        self._normOut.wrappedValue = AdaLayerNormContinuous(
            embeddingDim: innerDim, conditioningEmbeddingDim: innerDim)
        self._projOut.wrappedValue = Linear(innerDim, outChannels ?? inChannels, bias: false)
        super.init()
    }

    /// Build concatenated (txt∥img) RoPE once per canvas; reuse across denoise steps.
    public func prepareRotaryEmbeddings(
        imgIds: MLXArray,
        txtIds: MLXArray
    ) -> (MLXArray, MLXArray) {
        var iIds = imgIds
        var tIds = txtIds
        if iIds.ndim == 3 { iIds = iIds[0] }
        if tIds.ndim == 3 { tIds = tIds[0] }
        let imgRope = posEmbed(iIds)
        let txtRope = posEmbed(tIds)
        return (
            concatenated([txtRope.0, imgRope.0], axis: 0),
            concatenated([txtRope.1, imgRope.1], axis: 0)
        )
    }

    /// `[B, T, 7680] → [B, T, innerDim]` — run once per generate/edit, not per step.
    public func projectContext(_ encoderHiddenStates: MLXArray) -> MLXArray {
        contextEmbedder(encoderHiddenStates)
    }

    /// Forward pass (fp32 activations recommended for stability).
    ///
    /// - Parameters:
    ///   - timestep: Training-scale timesteps in **[0, 1000]** (host-side). Avoids GPU→CPU
    ///     `item()` syncs that were used when accepting either [0,1] or [0,1000].
    ///   - guidance: Optional guidance in **[0, 1000]** when used (klein distilled: nil).
    ///   - imageRotaryEmb: Optional precomputed (cos, sin) from `prepareRotaryEmbeddings`.
    ///   - contextIsProjected: When true, `encoderHiddenStates` is already
    ///     `contextEmbedder` output `[B, T, innerDim]` (hoisted once per generate).
    ///   - trace / stepIndex / probeDensity: optional pressure probes (no numerics change when off).
    public func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        timestep: MLXArray,
        imgIds: MLXArray,
        txtIds: MLXArray,
        guidance: MLXArray? = nil,
        imageRotaryEmb: (MLXArray, MLXArray)? = nil,
        contextIsProjected: Bool = false,
        trace: PipelineTrace? = nil,
        stepIndex: Int? = nil
    ) -> MLXArray {
        let density = trace?.density ?? .off
        let step = stepIndex
        let prefix = step.map { "dit.step\($0)" } ?? "dit"
        if ComputeDTypeProbe.enabled {
            ComputeDTypeProbe.record("dit.in.hidden", hiddenStates)
            ComputeDTypeProbe.record("dit.in.encoder", encoderHiddenStates)
            ComputeDTypeProbe.record("dit.in.timestep", timestep)
        }

        func sample(_ suffix: String, block: Int? = nil, forceEval: Bool = false) {
            guard let trace, density.instrumentsDiTBlocks else { return }
            if forceEval {
                // Materialize live graph so activeMemory reflects this region (dense diagnosis only).
                eval(hiddenStates)
            }
            trace.probe(
                "\(prefix).\(suffix)",
                phase: "dit",
                step: step,
                block: block,
                sampleMemory: true,
                minDensity: .blocks
            )
        }

        var ts = timestep
        if ts.ndim == 0 {
            ts = MLXArray.full([hiddenStates.dim(0)], values: ts)
        }
        ts = ts.asType(hiddenStates.dtype)

        var g = guidance
        if var gv = g {
            if gv.ndim == 0 {
                gv = MLXArray.full([hiddenStates.dim(0)], values: gv)
            }
            gv = gv.asType(hiddenStates.dtype)
            g = gv
        }

        let temb = timeGuidanceEmbed(ts, guidance: g).asType(.float32)

        var h = xEmbedder(hiddenStates)
        // Prompt→innerDim is invariant across denoise steps; pipeline hoists it.
        var e = contextIsProjected
            ? encoderHiddenStates
            : contextEmbedder(encoderHiddenStates)

        let concatRope = imageRotaryEmb ?? prepareRotaryEmbeddings(imgIds: imgIds, txtIds: txtIds)

        let tembImg = doubleStreamModulationImg(temb)
        let tembTxt = doubleStreamModulationTxt(temb)

        // Per-block cache clears are only needed on large canvases (they made 1024² fit
        // on 8 GB); on small canvases they just thrash the Metal buffer pool.
        let tuning = AttentionTuning.current
        let jointSeq = hiddenStates.dim(1) + encoderHiddenStates.dim(1)
        let clearPerBlock = jointSeq > tuning.blockCacheClearSeqThreshold
        let clearInterval = tuning.blockCacheClearInterval

        // Checkpoint after each major stage: MLX lazy graphs otherwise hold *all* block
        // intermediates until the outer eval — that inflated peak watermark (~8 GB at 768²
        // vs ~2 GB live active) and OOMs 1024² on 8 GB unified memory.
        eval(h, e, temb)
        if ComputeDTypeProbe.enabled {
            ComputeDTypeProbe.record("dit.after_embed.h", h)
            ComputeDTypeProbe.record("dit.after_embed.e", e)
            ComputeDTypeProbe.record("dit.after_embed.temb", temb)
        }
        sample("after_embed")

        for (bi, block) in transformerBlocks.enumerated() {
            let out = block(
                hiddenStates: h,
                encoderHiddenStates: e,
                tembModParamsImg: tembImg,
                tembModParamsTxt: tembTxt,
                imageRotaryEmb: concatRope
            )
            e = out.encoder
            h = out.hidden
            eval(h, e)
            if clearPerBlock, (bi + 1) % clearInterval == 0 {
                Memory.clearCache()
            }
            sample("after_double_\(bi)", block: bi)
        }

        let txtLen = e.dim(1)
        h = concatenated([e, h], axis: 1)
        eval(h)
        if clearPerBlock {
            Memory.clearCache()
        }
        sample("after_concat")

        let tembSingle = singleStreamModulation(temb)[0]
        let singleSample: Set<Int> = density == .max
            ? Set(0 ..< singleTransformerBlocks.count)
            : Set(ProbeDensity.defaultSingleBlockSampleIndices)
        for (bi, block) in singleTransformerBlocks.enumerated() {
            h = block(h, tembModParams: tembSingle, imageRotaryEmb: concatRope)
            // Always checkpoint single-stream (L is largest after concat).
            eval(h)
            if clearPerBlock, (bi + 1) % clearInterval == 0 {
                Memory.clearCache()
            }
            if density.instrumentsDiTBlocks, singleSample.contains(bi) {
                sample("after_single_\(bi)", block: bi)
            }
        }

        h = h[0..., txtLen..., 0...]
        h = normOut(h, textEmbeddings: temb)
        h = projOut(h)
        eval(h)
        sample("after_proj")
        return h
    }
}
