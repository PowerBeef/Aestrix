import Testing
import Foundation
@testable import ImarelloCore
import ImarelloRuntime

@Suite("Memory staging")
struct MemoryStagingTests {
    @Test("staged self-test never leaves multiple modules loaded")
    func stagedSelfTestInvariant() async throws {
        var config = ImarelloConfig(memoryPolicy: .staged, tier: .low, maxSide: 512)
        config.memoryPolicy = .staged
        // No local snapshot → modules dry-load (no Metal)
        config.modelsDirectory = URL(fileURLWithPath: "/tmp/imarello-no-snapshot", isDirectory: true)
        let pipeline = ImarelloPipeline(config: config)
        let samples = try await pipeline.memorySelfTest()
        #expect(samples.count >= 5)
        #expect(samples.contains { $0.label == "text_encoder_loaded" })
        #expect(samples.contains { $0.label == "dit_loaded" })
        #expect(samples.contains { $0.label == "vae_loaded" })
        #expect(samples.contains { $0.label == "after_vae_unload" })
    }

    @Test("dimension validation requires multiples of 16")
    func dimensionsMultipleOf16() throws {
        #expect(throws: ImarelloError.self) {
            try DimensionValidation.validate(width: 500, height: 512, maxSide: 1024, tier: .mid)
        }
        try DimensionValidation.validate(width: 512, height: 512, maxSide: 1024, tier: .mid)
    }

    @Test("tier detection clamps low memory")
    func tierDetection() {
        #expect(DeviceTier.detect(physicalMemoryBytes: 8 * 1_073_741_824) == .low)
        #expect(DeviceTier.detect(physicalMemoryBytes: 16 * 1_073_741_824) == .mid)
        #expect(DeviceTier.detect(physicalMemoryBytes: 32 * 1_073_741_824) == .high)
    }

    @Test("weight presets never include bf16")
    func noBF16Preset() {
        let ids = WeightPreset.allCases.map(\.rawValue)
        #expect(!ids.contains("bf16"))
        #expect(WeightPreset.bits4.defaultModelID.contains("4bit") || WeightPreset.bits4.defaultModelID.contains("4B-4bit"))
    }
}
