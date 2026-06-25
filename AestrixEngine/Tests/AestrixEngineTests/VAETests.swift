import Foundation
import Testing
import MLX
import MLXNN
@testable import AestrixEngine

/// VAE parity vs a Python (mflux/MLX) reference — see `tools/vae_reference.py`.
///
/// Two regimes:
/// - **float32 (definitive):** `vae_reference_fp32.safetensors` + `FluxVAE(precision: .float32)`.
///   float32 MLX reductions are deterministic, so Swift should match mflux to <1e-4 — proving
///   the port has no structural bug.
/// - **bf16 (amplification measurement):** `vae_reference.safetensors` + `.bfloat16`. The encoder
///   matches (latent Δ≈0.03); the decoded image drifts (~0.5) because MLX-Metal bf16 reductions
///   are not bit-deterministic and the decoder amplifies a 1-ULP seed. This is inherent, not a bug
///   (diffusers `AutoencoderKLFlux2.force_upcast=true` decodes in float32 for exactly this reason).
@Suite("FluxVAE")
struct VAETests {

    @Test("loads VAE weights and runs encode→decode round-trip", .enabled(if: heavyEnabled))
    func loadAndRoundTrip() async throws {
        try await TestFixtures.ensure(Flux2Klein4B4Bit.vaeFiles)
        let vae = try await Self.makeVAE(precision: .float32)

        let x = MLXRandom.uniform(low: -1.0, high: 1.0, [1, 128, 128, 3])
        let latent = vae.encode(x)
        #expect(latent.shape == [1, 16, 16, 32])
        let img = vae.decode(latent)
        #expect(img.shape == [1, 128, 128, 3])
        eval(img)
        #expect(img.max().item(Float.self).isFinite)
    }

    @Test("FLOAT32 parity vs Python — proves no structural bug", .enabled(if: heavyEnabled))
    func float32Parity() async throws {
        let vae = try await Self.makeVAE(precision: .float32)
        let ref = try Self.loadReference("vae_reference_fp32.safetensors")

        let latent = vae.encode(ref["input"]!)
        let decoded = vae.decode(latent)
        eval(latent, decoded)

        let latentDiff = Self.maxAbsDiff(latent, ref["latent"]!)
        let decodedDiff = Self.maxAbsDiff(decoded, ref["decoded"]!)
        print("VAE fp32 parity — latent max|Δ|=\(latentDiff)  decoded max|Δ|=\(decodedDiff)")

        #expect(latent.shape == ref["latent"]!.shape)
        #expect(decoded.shape == ref["decoded"]!.shape)
        #expect(latentDiff < 1e-4, "fp32 latent max|Δ|=\(latentDiff)")
        #expect(decodedDiff < 1e-4, "fp32 decoded max|Δ|=\(decodedDiff)")
    }

    @Test("bf16 encode/decode vs Python (amplification measurement)", .enabled(if: heavyEnabled))
    func bf16Amplification() async throws {
        let vae = try await Self.makeVAE(precision: .bfloat16)
        let ref = try Self.loadReference("vae_reference.safetensors")

        let latent = vae.encode(ref["input"]!)
        let decoded = vae.decode(latent)
        eval(latent, decoded)

        let latentDiff = Self.maxAbsDiff(latent, ref["latent"]!)
        let decodedDiff = Self.maxAbsDiff(decoded, ref["decoded"]!)
        print("VAE bf16 — latent max|Δ|=\(latentDiff)  decoded max|Δ|=\(decodedDiff)")

        #expect(latentDiff < 0.05, "bf16 encoder latent parity (target met)")
        // bf16 decoder drift is inherent MLX-Metal non-determinism amplified — see float32Parity
        // for the definitive (structural) correctness check.
        #expect(decodedDiff < 0.75, "bf16 decoded max|Δ|=\(decodedDiff)")
    }

    // MARK: helpers

    private static func loadReference(_ name: String) throws -> [String: MLXArray] {
        let url = TestFixtures.root.appendingPathComponent("parity/\(name)")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AestrixError.invalidInput("missing \(url.path) — run tools/vae_reference.py [--fp32]")
        }
        return try loadArrays(url: url)
    }

    private static func makeVAE(precision: MLX.DType) async throws -> FluxVAE {
        try await TestFixtures.ensure(Flux2Klein4B4Bit.vaeFiles)
        let (weights, _) = try WeightLoader.loadWeights(
            submodelDir: TestFixtures.root.appendingPathComponent("vae"))
        let vae = FluxVAE(precision: precision)
        vae.loadWeights(weights)
        return vae
    }

    private static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        (a - b).abs().max().item(Float.self)
    }
}
