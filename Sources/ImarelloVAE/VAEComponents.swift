import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - ResNet / Attention / Mid / Up-Down

final class Flux2ResnetBlock2D: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    init(inChannels: Int, outChannels: Int, eps: Float = 1e-6, groups: Int = 32) {
        self._norm1.wrappedValue = GroupNorm(
            groupCount: groups, dimensions: inChannels, eps: eps, affine: true, pytorchCompatible: true)
        self._conv1.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1)
        self._norm2.wrappedValue = GroupNorm(
            groupCount: groups, dimensions: outChannels, eps: eps, affine: true, pytorchCompatible: true)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1)
        if inChannels != outChannels {
            self._convShortcut.wrappedValue = Conv2d(
                inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, stride: 1, padding: 0)
        } else {
            self._convShortcut.wrappedValue = nil
        }
        super.init()
    }

    /// Input/output NCHW.
    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var residual = hiddenStates.transposed(0, 2, 3, 1) // NHWC
        var h = residual
        h = norm1(h.asType(.float32)).asType(.float32)
        h = silu(h)
        h = conv1(h)
        h = norm2(h.asType(.float32)).asType(.float32)
        h = silu(h)
        h = conv2(h)
        if let convShortcut {
            residual = convShortcut(residual)
        }
        h = h + residual
        return h.transposed(0, 3, 1, 2)
    }
}

final class Flux2AttentionBlock: Module {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear

    init(channels: Int, groups: Int = 32, eps: Float = 1e-6) {
        self._groupNorm.wrappedValue = GroupNorm(
            groupCount: groups, dimensions: channels, eps: eps, affine: true, pytorchCompatible: true)
        self._toQ.wrappedValue = Linear(channels, channels, bias: true)
        self._toK.wrappedValue = Linear(channels, channels, bias: true)
        self._toV.wrappedValue = Linear(channels, channels, bias: true)
        self._toOut.wrappedValue = Linear(channels, channels, bias: true)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var h = hiddenStates.transposed(0, 2, 3, 1)
        let batch = h.dim(0)
        let height = h.dim(1)
        let width = h.dim(2)
        let channels = h.dim(3)

        let normed = groupNorm(h.asType(.float32)).asType(.float32)
        var q = toQ(normed).reshaped([batch, height * width, 1, channels])
        var k = toK(normed).reshaped([batch, height * width, 1, channels])
        var v = toV(normed).reshaped([batch, height * width, 1, channels])
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        let scale = 1.0 / Foundation.sqrt(Float(channels))
        var attended: MLXArray
        if VAEAttentionConfig.current.useMLXFast {
            attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: nil)
        } else {
            attended = VAEAttention.scaledDotProductAttention(
                query: q, key: k, value: v, scale: scale)
        }
        attended = attended.transposed(0, 2, 1, 3).reshaped([batch, height, width, channels])
        attended = toOut(attended)
        h = h + attended
        return h.transposed(0, 3, 1, 2)
    }
}

final class Flux2UNetMidBlock2D: Module {
    @ModuleInfo(key: "resnets") var resnets: [Flux2ResnetBlock2D]
    @ModuleInfo(key: "attentions") var attentions: [Flux2AttentionBlock]

    init(channels: Int, eps: Float = 1e-6, groups: Int = 32, addAttention: Bool = true) {
        let ch = channels
        self._resnets.wrappedValue = [
            Flux2ResnetBlock2D(inChannels: ch, outChannels: ch, eps: eps, groups: groups),
            Flux2ResnetBlock2D(inChannels: ch, outChannels: ch, eps: eps, groups: groups),
        ]
        self._attentions.wrappedValue = addAttention
            ? [Flux2AttentionBlock(channels: ch, groups: groups, eps: eps)]
            : []
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var h = resnets[0](hiddenStates)
        if !attentions.isEmpty {
            h = attentions[0](h)
        }
        h = resnets[1](h)
        return h
    }
}

final class Flux2Upsample2D: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(channels: Int, outChannels: Int? = nil) {
        let out = outChannels ?? channels
        self._conv.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: out, kernelSize: 3, stride: 1, padding: 1)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var h = repeated(hiddenStates, count: 2, axis: 2)
        h = repeated(h, count: 2, axis: 3)
        h = h.transposed(0, 2, 3, 1)
        h = conv(h)
        return h.transposed(0, 3, 1, 2)
    }
}

final class Flux2Downsample2D: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(channels: Int, padding: Int = 0) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2,
            padding: IntOrPair(padding))
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        // Pad bottom/right by 1 (NCHW)
        var h = padded(
            hiddenStates,
            widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((0, 1)), IntOrPair((0, 1))])
        h = h.transposed(0, 2, 3, 1)
        h = conv(h)
        return h.transposed(0, 3, 1, 2)
    }
}

final class Flux2DownEncoderBlock2D: Module {
    @ModuleInfo(key: "resnets") var resnets: [Flux2ResnetBlock2D]
    @ModuleInfo(key: "downsamplers") var downsamplers: [Flux2Downsample2D]

    init(
        inChannels: Int,
        outChannels: Int,
        numLayers: Int = 2,
        eps: Float = 1e-6,
        groups: Int = 32,
        addDownsample: Bool = true
    ) {
        let chIn = inChannels
        let chOut = outChannels
        self._resnets.wrappedValue = (0..<numLayers).map { i in
            Flux2ResnetBlock2D(
                inChannels: i == 0 ? chIn : chOut,
                outChannels: chOut,
                eps: eps,
                groups: groups
            )
        }
        self._downsamplers.wrappedValue = addDownsample ? [Flux2Downsample2D(channels: chOut, padding: 0)] : []
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var h = hiddenStates
        // Checkpoint like the decoder: without this a 1024² I2I encode builds
        // the whole down-stack lazily (first block holds ~536 MB f32 tensors).
        for r in resnets {
            h = r(h)
            eval(h)
        }
        for d in downsamplers {
            h = d(h)
            eval(h)
        }
        Memory.clearCache()
        return h
    }
}

final class Flux2UpDecoderBlock2D: Module {
    @ModuleInfo(key: "resnets") var resnets: [Flux2ResnetBlock2D]
    @ModuleInfo(key: "upsamplers") var upsamplers: [Flux2Upsample2D]

    init(
        inChannels: Int,
        outChannels: Int,
        numLayers: Int = 3,
        eps: Float = 1e-6,
        groups: Int = 32,
        addUpsample: Bool = true
    ) {
        let chIn = inChannels
        let chOut = outChannels
        self._resnets.wrappedValue = (0..<numLayers).map { i in
            Flux2ResnetBlock2D(
                inChannels: i == 0 ? chIn : chOut,
                outChannels: chOut,
                eps: eps,
                groups: groups
            )
        }
        self._upsamplers.wrappedValue = addUpsample ? [Flux2Upsample2D(channels: chOut)] : []
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var h = hiddenStates
        for r in resnets {
            h = r(h)
            eval(h)
        }
        for u in upsamplers {
            h = u(h)
            eval(h)
        }
        Memory.clearCache()
        return h
    }
}

// MARK: - Encoder / Decoder wrappers

final class Flux2Encoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down_blocks") var downBlocks: [Flux2DownEncoderBlock2D]
    @ModuleInfo(key: "mid_block") var midBlock: Flux2UNetMidBlock2D
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(
        inChannels: Int = 3,
        outChannels: Int = 32,
        blockOutChannels: [Int] = [128, 256, 512, 512],
        layersPerBlock: Int = 2,
        normNumGroups: Int = 32,
        eps: Float = 1e-6
    ) {
        let blocks = blockOutChannels
        self._convIn.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: blocks[0], kernelSize: 3, stride: 1, padding: 1)
        self._downBlocks.wrappedValue = blocks.enumerated().map { i, outputChannel in
            let inputChannel = i > 0 ? blocks[i - 1] : blocks[0]
            let isFinal = i == blocks.count - 1
            return Flux2DownEncoderBlock2D(
                inChannels: inputChannel,
                outChannels: outputChannel,
                numLayers: layersPerBlock,
                eps: eps,
                groups: normNumGroups,
                addDownsample: !isFinal
            )
        }
        self._midBlock.wrappedValue = Flux2UNetMidBlock2D(
            channels: blocks[blocks.count - 1], eps: eps, groups: normNumGroups, addAttention: true)
        self._convNormOut.wrappedValue = GroupNorm(
            groupCount: normNumGroups, dimensions: blocks[blocks.count - 1], eps: eps, affine: true,
            pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(
            inputChannels: blocks[blocks.count - 1], outputChannels: 2 * outChannels, kernelSize: 3,
            stride: 1, padding: 1)
        super.init()
    }

    /// NCHW in/out
    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var h = hiddenStates.transposed(0, 2, 3, 1)
        h = convIn(h)
        h = h.transposed(0, 3, 1, 2)
        for block in downBlocks {
            h = block(h)
        }
        h = midBlock(h)
        h = h.transposed(0, 2, 3, 1)
        h = convNormOut(h.asType(.float32)).asType(.float32)
        h = silu(h)
        h = convOut(h)
        return h.transposed(0, 3, 1, 2)
    }
}

final class Flux2Decoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_block") var midBlock: Flux2UNetMidBlock2D
    @ModuleInfo(key: "up_blocks") var upBlocks: [Flux2UpDecoderBlock2D]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(
        inChannels: Int = 32,
        outChannels: Int = 3,
        blockOutChannels: [Int] = [128, 256, 512, 512],
        layersPerBlock: Int = 2,
        normNumGroups: Int = 32,
        eps: Float = 1e-6
    ) {
        let blocks = blockOutChannels
        let reversedChannels = Array(blocks.reversed())
        self._convIn.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: blocks[blocks.count - 1], kernelSize: 3, stride: 1,
            padding: 1)
        self._midBlock.wrappedValue = Flux2UNetMidBlock2D(
            channels: blocks[blocks.count - 1], eps: eps, groups: normNumGroups, addAttention: true)
        self._upBlocks.wrappedValue = reversedChannels.enumerated().map { i, outputChannel in
            let prev = i == 0 ? outputChannel : reversedChannels[i - 1]
            let isFinal = i == reversedChannels.count - 1
            return Flux2UpDecoderBlock2D(
                inChannels: prev,
                outChannels: outputChannel,
                numLayers: layersPerBlock + 1,
                eps: eps,
                groups: normNumGroups,
                addUpsample: !isFinal
            )
        }
        self._convNormOut.wrappedValue = GroupNorm(
            groupCount: normNumGroups, dimensions: blocks[0], eps: eps, affine: true, pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(
            inputChannels: blocks[0], outputChannels: outChannels, kernelSize: 3, stride: 1, padding: 1)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var h = hiddenStates.transposed(0, 2, 3, 1)
        h = convIn(h)
        h = h.transposed(0, 3, 1, 2)
        eval(h)
        Memory.clearCache()
        h = midBlock(h)
        eval(h)
        Memory.clearCache()
        // Upsample stages grow to full RGB resolution — checkpoint each block so
        // 1024² decode does not hold the entire UNet graph (was the OOM after DiT).
        for block in upBlocks {
            h = block(h)
            eval(h)
            Memory.clearCache()
        }
        h = h.transposed(0, 2, 3, 1)
        h = convNormOut(h.asType(.float32)).asType(.float32)
        h = silu(h)
        h = convOut(h)
        let out = h.transposed(0, 3, 1, 2)
        eval(out)
        return out
    }
}

final class Flux2BatchNormStats: Module {
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray
    let eps: Float

    init(numFeatures: Int, eps: Float = 1e-4) {
        self._runningMean.wrappedValue = MLXArray.zeros([numFeatures], dtype: .float32)
        self._runningVar.wrappedValue = MLXArray.ones([numFeatures], dtype: .float32)
        self.eps = eps
        super.init()
    }
}
