import Foundation
import Metal
import MLX
import Testing
import ImarelloCore
@testable import ImarelloDirect

@Suite("Direct engine safety")
struct DirectSafetyTests {
    @Test("attention profiles expose pinned Steel and NAX D=128 ABI shapes")
    func attentionProfiles() {
        #expect(DirectAttentionProfile.steelF16.blockQueries == 32)
        #expect(DirectAttentionProfile.naxF16.blockQueries == 64)
        #expect(DirectAttentionProfile.naxF16.blockKeys == 32)
        #expect(DirectAttentionProfile.naxF16.headDimension == 128)
        #expect(DirectAttentionProfile.qualificationJointLengths == [1_536, 2_816, 4_608])
        #expect(DirectKernelABI.naxAttentionSymbols.contains(
            DirectAttentionProfile.naxF16.functionName))
    }
    @Test("odd image and joint token counts cover every row exactly once")
    func oddChunkCoverage() {
        for count in [1, 3, 1_089, 1_601] {
            let ranges = DirectDiTStep.chunkRanges(totalRows: count)
            #expect(ranges.count <= 2)
            #expect(ranges.flatMap(Array.init) == Array(0 ..< count))
        }
    }

    @Test("whole-T2I shell fails closed for unsupported request semantics")
    func wholePipelineCapabilities() {
        let output = URL(fileURLWithPath: "/tmp/imarello-direct-capability.png")
        let supported = T2IRequest(
            prompt: "fox", width: 512, height: 512, steps: 4,
            seed: 42, outputURL: output)
        #expect(DirectT2IBackend.supportsRequest(supported))

        var requests = [T2IRequest]()
        requests.append(T2IRequest(
            prompt: "fox", width: 512, height: 512, steps: 4,
            seed: nil, outputURL: output))
        requests.append(T2IRequest(
            prompt: "fox", width: 512, height: 512, steps: 4,
            seed: 42, outputURL: nil))
        requests.append(T2IRequest(
            prompt: "fox", width: 512, height: 512, steps: 4,
            seed: 42, outputURL: output, textTokens: .auto))
        requests.append(T2IRequest(
            prompt: "fox", width: 512, height: 512, steps: 4,
            seed: 42, outputURL: output, embedCache: true))
        requests.append(T2IRequest(
            prompt: "fox", width: 512, height: 512, steps: 4,
            seed: 42, outputURL: output, padContent: .clean))

        for request in requests {
            #expect(!DirectT2IBackend.supportsRequest(request))
        }
    }
}

@Suite("Serialized direct Metal failures", .serialized)
struct DirectMetalFailureTests {
    @Test("GroupNorm second allocation failure cannot publish partial scratch")
    func groupNormAllocationIsAtomic() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        var calls = 0
        #expect(throws: (any Error).self) {
            _ = try DirectVAE.allocateGroupNormScratch { length in
                calls += 1
                return calls == 1 ? device.makeBuffer(length: length) : nil
            }
        }
        #expect(calls == 2)
    }

    @Test("quantized weight shape mismatches throw before dispatch")
    func packedShapeValidation() {
        let weight = MLXArray.zeros([64, 8], dtype: .uint32)
        let scales = MLXArray.zeros([64, 1], dtype: .float32)
        let biases = MLXArray.zeros([64, 1], dtype: .float32)
        #expect(throws: Never.self) {
            try DirectTensorValidation.requireQuantized(
                weight: weight, scales: scales, biases: biases,
                n: 64, k: 64, name: "valid"
            )
        }
        #expect(throws: (any Error).self) {
            try DirectTensorValidation.requireQuantized(
                weight: weight, scales: scales, biases: biases,
                n: 63, k: 64, name: "invalid"
            )
        }
    }
}
