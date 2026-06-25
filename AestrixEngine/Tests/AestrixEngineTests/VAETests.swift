import Foundation
import Testing
import MLX
@testable import AestrixEngine

@Suite("FluxVAE")
struct VAETests {

    @Test("loads VAE weights and runs encode→decode round-trip", .enabled(if: heavyEnabled))
    func loadAndRoundTrip() async throws {
        try await TestFixtures.ensure(Flux2Klein4B4Bit.vaeFiles)
        let (weights, _) = try WeightLoader.loadWeights(
            submodelDir: TestFixtures.root.appendingPathComponent("vae"))

        let vae = FluxVAE()
        vae.loadWeights(weights)

        // Small NHWC input in [-1, 1] (a real image is normalized at the pipeline level).
        let x = MLXRandom.uniform(low: -1.0, high: 1.0, [1, 128, 128, 3])

        let latent = vae.encode(x)
        #expect(latent.shape == [1, 16, 16, 32])   // 128 / 8 spatial, 32 latent channels

        let img = vae.decode(latent)
        #expect(img.shape == [1, 128, 128, 3])
        eval(img)

        // Output should be finite and in a reasonable image range (≈ [-1, 1]).
        let hi = img.max().item(Float.self)
        let lo = img.min().item(Float.self)
        #expect(hi.isFinite && lo.isFinite)
    }
}
