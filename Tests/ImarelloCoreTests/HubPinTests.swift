import Testing
import Foundation
@testable import ImarelloCore
import ImarelloWeights

@Suite("Hub revision pins")
struct HubPinTests {
    @Test("every product preset pins a 40-char commit SHA")
    func pinsAreCommitSHAs() {
        for preset in WeightPreset.allCases {
            #expect(preset.pin.isCommitSHA, "\(preset.rawValue) revision is not a commit SHA")
            #expect(preset.defaultModelID == preset.pin.modelID)
            #expect(preset.pinnedRevision == preset.pin.revision)
        }
        let revisions = Set(WeightPreset.allCases.map(\.pinnedRevision))
        #expect(revisions.count == WeightPreset.allCases.count)
    }

    @Test("default config uses the 4-bit pin")
    func defaultConfigUsesFourBitPin() {
        let config = ImarelloConfig(tier: .low, maxSide: 512)
        #expect(config.weightPreset == .bits4)
        #expect(config.modelID == WeightPreset.bits4.defaultModelID)
        #expect(config.revision == WeightPreset.bits4.pinnedRevision)
        #expect(config.downloadCommand.contains("--revision \(config.revision)"))
        #expect(config.vaeDecoderVariant == .smallDecoder)
    }

    @Test("apply(preset:) updates model id and revision together")
    func applyPresetKeepsPin() {
        var config = ImarelloConfig(weightPreset: .bits4, tier: .low, maxSide: 512)
        config.apply(preset: .bits3)
        #expect(config.weightPreset == .bits3)
        #expect(config.modelID == WeightPreset.bits3.defaultModelID)
        #expect(config.revision == WeightPreset.bits3.pinnedRevision)
    }

    @Test("Docs/hub-pins.json matches WeightPreset")
    func pinsDocumentMatchesPresets() throws {
        let doc = try loadHubPins()
        #expect(doc.schemaVersion == "1.0")
        #expect(Set(doc.packs.keys) == Set(WeightPreset.allCases.map(\.rawValue)))
        for preset in WeightPreset.allCases {
            let pack = try #require(doc.packs[preset.rawValue])
            #expect(pack.modelId == preset.defaultModelID)
            #expect(pack.revision == preset.pinnedRevision)
            #expect(pack.revision.count == 40)
        }
    }

    @Test("Small Decoder pin matches Docs/hub-pins.json")
    func smallDecoderPinMatchesDoc() throws {
        let pin = VAEDecoderVariant.smallDecoderPin
        #expect(pin.isCommitSHA)
        #expect(VAEDecoderVariant.smallDecoder.blockOutChannels == [96, 192, 384, 384])
        #expect(VAEDecoderVariant.full.blockOutChannels == [128, 256, 512, 512])
        let doc = try loadHubPins()
        let pack = try #require(doc.vaeDecoders?["small-decoder"])
        #expect(pack.modelId == pin.modelID)
        #expect(pack.revision == pin.revision)
        #expect(pack.file == VAEDecoderVariant.smallDecoderFileName)
    }

    @Test("WEIGHTS.md and README document the 4-bit pin")
    func docsMentionDefaultPin() throws {
        let root = repoRoot()
        let sha = WeightPreset.bits4.pinnedRevision
        let weights = try String(contentsOf: root.appendingPathComponent("Docs/WEIGHTS.md"), encoding: .utf8)
        let readme = try String(contentsOf: root.appendingPathComponent("README.md"), encoding: .utf8)
        #expect(weights.contains(sha))
        #expect(readme.contains(sha))
        #expect(readme.contains("--revision"))
        #expect(readme.contains(VAEDecoderVariant.smallDecoderPin.revision))
        #expect(weights.contains(VAEDecoderVariant.smallDecoderPin.revision))
    }

    @Test("parses hf download metadata first line as revision")
    func parsesHuggingFaceMetadata() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("imarello_hubpin_\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let meta = tmp.appendingPathComponent(".cache/huggingface/download/vae/0.safetensors.metadata")
        try FileManager.default.createDirectory(at: meta.deletingLastPathComponent(), withIntermediateDirectories: true)
        let sha = "1cebb9b45c21ece14a42615b16bf5fa4de9b56da"
        try "\(sha)\na80f68acb469352cf31abe41a0869c7705ec7b57\n1786385915.786734\n"
            .write(to: meta, atomically: true, encoding: .utf8)
        let snap = ModelSnapshot(modelID: "mlx-community/FLUX.2-Klein-4B-4bit", root: tmp)
        #expect(snap.detectedRevision == sha)
    }

    private struct HubPinsFile: Decodable {
        let schemaVersion: String
        let packs: [String: Pack]
        let vaeDecoders: [String: VAEDecoderPack]?
        struct Pack: Decodable {
            let modelId: String
            let revision: String
        }
        struct VAEDecoderPack: Decodable {
            let modelId: String
            let revision: String
            let file: String?
        }
    }

    private func loadHubPins() throws -> HubPinsFile {
        let url = repoRoot().appendingPathComponent("Docs/hub-pins.json")
        let data = try Data(contentsOf: url)
        let dec = JSONDecoder()
        dec.keyDecodingStrategy = .convertFromSnakeCase
        return try dec.decode(HubPinsFile.self, from: data)
    }

    private func repoRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
