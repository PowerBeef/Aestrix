import Foundation
import Testing
import MLX
@testable import AestrixEngine

@Suite("KleinTransformer")
struct TransformerTests {

    @Test("loads Klein transformer and runs a forward pass", .enabled(if: heavyEnabled))
    func forwardWorks() async throws {
        try await TestFixtures.ensure(Flux2Klein4B4Bit.transformerFiles)
        let (weights, _) = try WeightLoader.loadWeights(
            submodelDir: TestFixtures.root.appendingPathComponent("transformer"))
        let model = KleinTransformer()
        model.loadWeights(weights)

        // Small 256×256 test: latent 16×16 = 256 image tokens + 512 text tokens.
        let latent = MLXRandom.normal([1, 256, 128])
        let ctx = MLXRandom.normal([1, 512, 7680])

        // RoPE coords: img_ids [256,4] = (t=0, h, w, l=0); txt_ids [512,4] = (t=0, h=0, w=0, l=0..511).
        var imgArr: [Int32] = []
        for h in 0..<16 { for w in 0..<16 { imgArr += [0, Int32(h), Int32(w), 0] } }
        var txtArr: [Int32] = []
        for l in 0..<512 { txtArr += [0, 0, 0, Int32(l)] }
        let imgIds = MLXArray(imgArr, [256, 4])
        let txtIds = MLXArray(txtArr, [512, 4])
        let timestep = MLXArray([Float(1.0)])

        let pred = model(latent, ctx, imgIds: imgIds, txtIds: txtIds, timestep: timestep)
        eval(pred)

        #expect(pred.shape == [1, 256, 128], "pred shape \(pred.shape)")
        #expect(pred.max().item(Float.self).isFinite)
        #expect(pred.min().item(Float.self).isFinite)
    }
}
