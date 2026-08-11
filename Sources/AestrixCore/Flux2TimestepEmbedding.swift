import Foundation

/// Sinusoidal timestep / guidance frequency embedding (pre-MLP).
/// Matches mflux `Flux2TimestepGuidanceEmbeddings._timestep_embedding`
/// with `flip_sin_to_cos = true`.
public enum Flux2TimestepEmbedding {
    public static let defaultDim = ModelConstants.timestepEmbedChannels // 256

    /// - Parameters:
    ///   - timesteps: 1D batch of timesteps (already scaled to ~[0,1000] or [0,1])
    ///   - dim: embedding channels (default 256)
    /// - Returns: `[batch][dim]` embeddings
    public static func embed(
        timesteps: [Float],
        dim: Int = defaultDim,
        flipSinToCos: Bool = true
    ) -> [[Float]] {
        let half = dim / 2
        let log10000 = log(Float(10000.0))
        var freqs = [Float](repeating: 0, count: half)
        for i in 0..<half {
            freqs[i] = exp(-log10000 * Float(i) / Float(half))
        }

        var result: [[Float]] = []
        result.reserveCapacity(timesteps.count)
        for t in timesteps {
            var sins = [Float](repeating: 0, count: half)
            var cosines = [Float](repeating: 0, count: half)
            for i in 0..<half {
                let arg = t * freqs[i]
                sins[i] = sin(arg)
                cosines[i] = cos(arg)
            }
            var emb: [Float]
            if flipSinToCos {
                // cos then sin (flip halves)
                emb = cosines + sins
            } else {
                emb = sins + cosines
            }
            if dim % 2 == 1 {
                emb.append(0)
            }
            result.append(emb)
        }
        return result
    }

    /// DiT convention: if all timesteps are ≤ 1, scale by 1000 before embedding.
    public static func scaleTimestepsIfNeeded(_ timesteps: [Float]) -> [Float] {
        guard let maxT = timesteps.max() else { return timesteps }
        if maxT <= 1.0 {
            return timesteps.map { $0 * 1000.0 }
        }
        return timesteps
    }
}
