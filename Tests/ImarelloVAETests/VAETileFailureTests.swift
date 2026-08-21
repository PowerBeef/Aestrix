import MLX
import Testing
@testable import ImarelloVAE

@Suite("Serialized VAE tile failures", .serialized)
struct VAETileFailureTests {
    private enum Expected: Error { case tile }

    @Test("a throwing tile decode propagates instead of trapping")
    func throwingDecodePropagates() {
        let latents = MLXArray.zeros([1, 32, 4, 4], dtype: .float32)
        #expect(throws: Expected.self) {
            try Flux2VAE.decodeLatentsTiled(
                latents,
                decode: { _ in throw Expected.tile },
                grid: 2
            )
        }
    }
}
