import Foundation
import Metal
import MLX
import ImarelloCore
import ImarelloDiT
import ImarelloRuntime

/// `PackedLatentDenoising` adapter: the bespoke DiT behind the product T2I
/// pipeline (opt-in `--dit-engine direct`). Everything — conditioner, RoPE,
/// scheduler, engine, weights — is built inside `denoise` and freed on return,
/// mirroring the product's staged residency. Assumes the product conditioning
/// regime: 512 text tokens, guidance 1.0, no pad-keep/pad-bias diagnostics
/// (the CLI enforces those).
public final class DirectDiTPackedDenoiser: PackedLatentDenoising {
    let transformerDirectory: URL
    let metallibURL: URL
    let useNAX: Bool

    public init(transformerDirectory: URL, metallibURL: URL, useNAX: Bool = false) {
        self.transformerDirectory = transformerDirectory
        self.metallibURL = metallibURL
        self.useNAX = useNAX
    }

    public func denoise(
        noise: MLXArray, embeds: MLXArray,
        width: Int, height: Int, steps: Int,
        onStep: ((Int) throws -> Void)?
    ) throws -> MLXArray {
        try RequestValidation.validate(steps: steps)
        try DimensionValidation.validate(
            width: width, height: height, maxSide: max(width, height), tier: .high)
        let lImg = (width / 16) * (height / 16)
        try DirectTensorValidation.requireShape(
            noise, [1, lImg, 128], name: "direct denoiser noise")
        try DirectTensorValidation.requireShape(
            embeds, [1, 512, 7680], name: "direct denoiser prompt embeddings")

        let conditioner = try DirectConditioner(
            transformerDirectory: transformerDirectory, metallibURL: metallibURL)
        let e0 = try conditioner.projectContext(embeds)

        var arrays: [String: MLXArray] = [:]
        for shard in ["0.safetensors", "1.safetensors"] {
            let url = transformerDirectory.appendingPathComponent(shard)
            if FileManager.default.fileExists(atPath: url.path) {
                for (k, v) in try MLX.loadArrays(url: url) { arrays[k] = v }
            }
        }
        let engine = try DirectDiTStep(lImg: lImg, metallibURL: metallibURL, useNAXQmm: useNAX)
        try engine.loadBlocks(arrays: arrays, nDouble: 5, nSingle: 20)
        let head = try engine.loadHead(arrays: arrays)
        arrays.removeAll()
        Memory.clearCache()

        let imgIds = LatentOps.imageIds(width: width, height: height)
        let txtIds = LatentOps.textIds()
        let rope = conditioner.rope(imgIds: imgIds, txtIds: txtIds)
        let scheduler = Flux2Scheduler(numInferenceSteps: steps, imageSeqLen: lImg)
        let conditioning = try conditioner.stepConditioning(timesteps: scheduler.timesteps)
        let dts: [Float] = (0 ..< scheduler.sigmas.count - 1).map {
            scheduler.sigmas[$0 + 1] - scheduler.sigmas[$0]
        }

        let e0Buf = try engine.upload(e0, "e0")
        eval(noise)
        let noiseData = noise.asType(.float32).asData(noCopy: false)
        noiseData.withUnsafeBytes { raw in
            head.latA.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
        }
        var cur = head.latA, alt = head.latB
        for step in 0 ..< steps {
            try onStep?(step)
            try engine.setStepConditioning(conditioning[step], cos: rope.0, sin: rope.1)
            try engine.encodeDenoiseStep(
                latIn: cur, latOut: alt, e0: e0Buf, head: head, dt: dts[step])
            swap(&cur, &alt)
        }
        var lat = [Float](repeating: 0, count: lImg * 128)
        lat.withUnsafeMutableBufferPointer { p in
            p.baseAddress!.update(
                from: cur.contents().bindMemory(to: Float.self, capacity: lImg * 128),
                count: lImg * 128)
        }
        let latents = MLXArray(lat, [1, lImg, 128])
        eval(latents)
        return latents
    }
}
