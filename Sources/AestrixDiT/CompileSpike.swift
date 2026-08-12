import Foundation
import MLX
import AestrixCore

/// Block-level `MLX.compile` spike (PERF.md S1 research; resident-DiT scenario only).
///
/// Compares one transformer block forward under three variants:
/// 1. `product`   — uncompiled, internal QKV eval checkpoints on (current tree)
/// 2. `eval-free` — uncompiled, internal evals off (one eval per block)
/// 3. `compiled`  — `compile(inputs: [block])` traced eval-free forward
///
/// The compiled closure only amortizes across steps while the same block instance
/// stays resident; staged unload discards it (see PERF.md S1 verdict).
public enum BlockCompileSpike {
    public struct Timing: Sendable {
        public let label: String
        public let msPerIter: Double
        public init(label: String, msPerIter: Double) {
            self.label = label
            self.msPerIter = msPerIter
        }
    }

    public struct Report: Sendable {
        public let singleBlock: [Timing]
        public let doubleBlock: [Timing]
        public let compileFirstCallMsSingle: Double
        public let compileFirstCallMsDouble: Double
    }

    /// Run the spike on loaded model weights. Restores `AttentionTuning` afterwards.
    public static func run(
        model: Flux2Transformer,
        textSeq: Int = 512,
        imageSide: Int = 32,  // packed 32×32 = 512² canvas
        iters: Int = 6
    ) -> Report {
        let saved = AttentionTuning.current
        defer { AttentionTuning.current = saved }

        let imageSeq = imageSide * imageSide
        let dim = model.innerDim

        MLXRandom.seed(7)
        let hSingle = MLXRandom.normal([1, textSeq + imageSeq, dim], dtype: .float32)
        let hImg = MLXRandom.normal([1, imageSeq, dim], dtype: .float32)
        let hTxt = MLXRandom.normal([1, textSeq, dim], dtype: .float32)
        let temb = MLXRandom.normal([1, dim], dtype: .float32)

        let txtRows = Flux2RoPE.prepareTextIDs(length: textSeq)
        let imgRows = Flux2RoPE.prepareGridIDs(height: imageSide, width: imageSide, tCoord: 0)
        let txtIds = MLXArray(txtRows.flatMap { $0 }).reshaped([textSeq, 4])
        let imgIds = MLXArray(imgRows.flatMap { $0 }).reshaped([imageSeq, 4])
        let rope = model.prepareRotaryEmbeddings(imgIds: imgIds, txtIds: txtIds)

        let modSingle = model.singleStreamModulation(temb)[0]
        let modImg = model.doubleStreamModulationImg(temb)
        let modTxt = model.doubleStreamModulationTxt(temb)
        eval(hSingle, hImg, hTxt, rope.0, rope.1)
        eval(modSingle.0, modSingle.1, modSingle.2)

        let singleBlock = model.singleTransformerBlocks[0]
        let doubleBlock = model.transformerBlocks[0]

        func time(_ label: String, iters: Int, _ body: () -> Void) -> Timing {
            body()  // warmup
            Memory.clearCache()
            let start = DispatchTime.now().uptimeNanoseconds
            for _ in 0 ..< iters { body() }
            let ns = DispatchTime.now().uptimeNanoseconds - start
            return Timing(label: label, msPerIter: Double(ns) / 1e6 / Double(iters))
        }

        // --- Single-stream block ---
        var singleTimings: [Timing] = []
        AttentionTuning.current.qkvCheckpoint = true
        singleTimings.append(time("single/product", iters: iters) {
            let out = singleBlock(hSingle, tembModParams: modSingle, imageRotaryEmb: rope)
            eval(out)
        })
        AttentionTuning.current.qkvCheckpoint = false
        singleTimings.append(time("single/eval-free", iters: iters) {
            let out = singleBlock(hSingle, tembModParams: modSingle, imageRotaryEmb: rope)
            eval(out)
        })
        let compiledSingle = compile(inputs: [singleBlock]) { (args: [MLXArray]) -> [MLXArray] in
            [singleBlock(
                args[0],
                tembModParams: (args[1], args[2], args[3]),
                imageRotaryEmb: (args[4], args[5])
            )]
        }
        let singleArgs = [hSingle, modSingle.0, modSingle.1, modSingle.2, rope.0, rope.1]
        let firstSingleStart = DispatchTime.now().uptimeNanoseconds
        eval(compiledSingle(singleArgs))
        let compileFirstCallMsSingle =
            Double(DispatchTime.now().uptimeNanoseconds - firstSingleStart) / 1e6
        singleTimings.append(time("single/compiled", iters: iters) {
            eval(compiledSingle(singleArgs))
        })

        // --- Double-stream block ---
        var doubleTimings: [Timing] = []
        func runDouble() -> (MLXArray, MLXArray) {
            let out = doubleBlock(
                hiddenStates: hImg,
                encoderHiddenStates: hTxt,
                tembModParamsImg: modImg,
                tembModParamsTxt: modTxt,
                imageRotaryEmb: rope
            )
            return (out.hidden, out.encoder)
        }
        AttentionTuning.current.qkvCheckpoint = true
        doubleTimings.append(time("double/product", iters: iters) {
            let (h, e) = runDouble()
            eval(h, e)
        })
        AttentionTuning.current.qkvCheckpoint = false
        doubleTimings.append(time("double/eval-free", iters: iters) {
            let (h, e) = runDouble()
            eval(h, e)
        })
        let compiledDouble = compile(inputs: [doubleBlock]) { (args: [MLXArray]) -> [MLXArray] in
            let out = doubleBlock(
                hiddenStates: args[0],
                encoderHiddenStates: args[1],
                tembModParamsImg: [(args[2], args[3], args[4]), (args[5], args[6], args[7])],
                tembModParamsTxt: [(args[8], args[9], args[10]), (args[11], args[12], args[13])],
                imageRotaryEmb: (args[14], args[15])
            )
            return [out.hidden, out.encoder]
        }
        let doubleArgs = [
            hImg, hTxt,
            modImg[0].0, modImg[0].1, modImg[0].2, modImg[1].0, modImg[1].1, modImg[1].2,
            modTxt[0].0, modTxt[0].1, modTxt[0].2, modTxt[1].0, modTxt[1].1, modTxt[1].2,
            rope.0, rope.1,
        ]
        let firstDoubleStart = DispatchTime.now().uptimeNanoseconds
        eval(compiledDouble(doubleArgs))
        let compileFirstCallMsDouble =
            Double(DispatchTime.now().uptimeNanoseconds - firstDoubleStart) / 1e6
        doubleTimings.append(time("double/compiled", iters: iters) {
            eval(compiledDouble(doubleArgs))
        })

        return Report(
            singleBlock: singleTimings,
            doubleBlock: doubleTimings,
            compileFirstCallMsSingle: compileFirstCallMsSingle,
            compileFirstCallMsDouble: compileFirstCallMsDouble
        )
    }
}
