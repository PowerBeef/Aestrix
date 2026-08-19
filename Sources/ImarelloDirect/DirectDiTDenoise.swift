import Foundation
import Metal
import MLX
import ImarelloCore
import ImarelloWeights
import ImarelloDiT
import ImarelloRuntime

/// Stage-2 F4: the COMPLETE 4-step denoise stage on the direct engine —
/// x_embedder → 25 blocks → norm_out/proj_out → Euler, one command buffer per
/// step — verified against the full product DiT module end to end.
public enum DirectDiTDenoiseSpike {

    public static func run(
        snapshot: ModelSnapshot, metallibURL: URL, width: Int, height: Int
    ) async throws -> String {
        let lTxt = 512
        let (ph, pw) = { (h: Int, w: Int) -> (Int, Int) in
            (h / 16, w / 16)
        }(height, width)
        let lImg = ph * pw
        let dim = 3072

        // -- Direct engine first: raw dict → buffers, then FREE the dict so it
        // never coexists with the product module (8 GB host).
        var arrays: [String: MLXArray] = [:]
        let tDir = snapshot.root.appendingPathComponent("transformer", isDirectory: true)
        for shard in ["0.safetensors", "1.safetensors"] {
            let url = tDir.appendingPathComponent(shard)
            if FileManager.default.fileExists(atPath: url.path) {
                for (k, v) in try MLX.loadArrays(url: url) { arrays[k] = v }
            }
        }
        let engine = try DirectDiTStep(lImg: lImg, metallibURL: metallibURL)
        let tLoad = CFAbsoluteTimeGetCurrent()
        try engine.loadBlocks(arrays: arrays, nDouble: 5, nSingle: 20)
        let head = try engine.loadHead(arrays: arrays)
        let loadMS = (CFAbsoluteTimeGetCurrent() - tLoad) * 1000
        arrays.removeAll()
        Memory.clearCache()

        // -- Product module: oracle + conditioning source ---------------------
        let dit = DiTModule(snapshot: snapshot)
        try await dit.load()

        let scheduler = Flux2Scheduler(numInferenceSteps: 4, imageSeqLen: lImg)
        let noise = LatentOps.samplePackedNoise(width: width, height: height, seed: 42)
        MLXRandom.seed(99)
        let promptEmbeds = (MLXRandom.normal([1, lTxt, 7680]) * 0.5).asType(.bfloat16)
        let imgIds = LatentOps.imageIds(width: width, height: height)
        let txtIds = LatentOps.textIds()
        eval(noise, promptEmbeds, imgIds, txtIds)

        let e0 = try dit.projectContext(promptEmbeds)
        let rope = try dit.prepareRotaryEmbeddings(imgIds: imgIds, txtIds: txtIds)
        let stepTimesteps: [MLXArray] = scheduler.timesteps.map {
            MLXArray([$0]).asType(.float32)
        }
        let dts = LatentOps.eulerDts(sigmas: scheduler.sigmas)
        let dtFloats: [Float] = (0 ..< scheduler.sigmas.count - 1).map {
            scheduler.sigmas[$0 + 1] - scheduler.sigmas[$0]
        }
        let conditioning = try dit.precomputeStepConditioning(
            timesteps: stepTimesteps, batch: 1, dtype: .float32, guidance: MLXArray(1.0))
        eval(e0, rope.0, rope.1)

        // -- Oracle: product 4-step loop --------------------------------------
        func oracleDenoise() throws -> MLXArray {
            var lat = noise
            for step in 0 ..< 4 {
                let pred = try dit.forward(
                    hiddenStates: lat,
                    encoderHiddenStates: e0,
                    timestep: stepTimesteps[step],
                    imgIds: imgIds,
                    txtIds: txtIds,
                    guidance: MLXArray(1.0),
                    imageRotaryEmb: rope,
                    contextIsProjected: true,
                    stepConditioning: conditioning[step])
                lat = LatentOps.eulerStep(sample: lat, modelOutput: pred, dt: dts[step])
                eval(lat)
                if lImg >= 3000 { Memory.clearCache() }  // product large-canvas discipline
            }
            return lat
        }
        let oLat = try oracleDenoise()
        var oracleMS = 0.0
        do {
            _ = try oracleDenoise()
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try oracleDenoise()
            oracleMS = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        }

        // -- Direct denoise ---------------------------------------------------
        let e0f32 = e0.asType(.float32)
        eval(e0f32)
        let e0Buf = try engine.upload(e0f32, "e0")
        let noiseData = noise.asData(noCopy: false)

        func directDenoise() throws -> MTLBuffer {
            noiseData.withUnsafeBytes { raw in
                head.latA.contents().copyMemory(from: raw.baseAddress!, byteCount: raw.count)
            }
            var cur = head.latA, alt = head.latB
            for step in 0 ..< 4 {
                try engine.setStepConditioning(conditioning[step], cos: rope.0, sin: rope.1)
                try engine.encodeDenoiseStep(
                    latIn: cur, latOut: alt, e0: e0Buf, head: head, dt: dtFloats[step])
                swap(&cur, &alt)
            }
            return cur
        }
        var finalBuf = try directDenoise()
        var directMS = 0.0
        do {
            let t0 = CFAbsoluteTimeGetCurrent()
            finalBuf = try directDenoise()
            directMS = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        }

        // -- Compare ----------------------------------------------------------
        let n = lImg * 128
        let ptr = finalBuf.contents().bindMemory(to: Float.self, capacity: n)
        let ref = oLat.reshaped([lImg, 128]).asType(.float32).asArray(Float.self)
        var dot = 0.0, na = 0.0, nb = 0.0, maxDiff = 0.0
        for i in 0 ..< n {
            let a = Double(ptr[i]), b = Double(ref[i])
            dot += a * b; na += a * a; nb += b * b
            maxDiff = max(maxDiff, abs(a - b))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
        let allocGB = Double(engine.ownedBytes) / 1_073_741_824
        return """
        direct-dit-denoise (F4) — full 4-step stage @\(width)²: embedder + 25 blocks + head + Euler per step
          cosine_final_latents: \(String(format: "%.7f", cosine))
          max_abs_diff:         \(String(format: "%.4f", maxDiff))
          weight_upload:        \(String(format: "%.0f", loadMS)) ms (one-time)
          oracle_4step:         \(String(format: "%.1f", oracleMS)) ms (product DiT, warm)
          direct_4step:         \(String(format: "%.1f", directMS)) ms (one CB per step)
          speedup:              \(String(format: "%.2f", oracleMS / directMS))×
          engine_owned:         \(String(format: "%.2f", allocGB)) GiB (weights + static scratch, deterministic)
          verdict:              \(cosine >= 0.999 ? "PASS" : "investigate")
        """
    }
}
