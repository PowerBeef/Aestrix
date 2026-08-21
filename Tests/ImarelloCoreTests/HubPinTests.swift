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
        config.apply(preset: .bits6)
        #expect(config.weightPreset == .bits6)
        #expect(config.modelID == WeightPreset.bits6.defaultModelID)
        #expect(config.revision == WeightPreset.bits6.pinnedRevision)
    }

    @Test("there is no 3-bit preset (product lock)")
    func noThreeBitPreset() {
        #expect(WeightPreset(rawValue: "3bit") == nil)
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

    @Test("snapshot completeness requires tokenizer assets, indexes, and every shard")
    func snapshotCompleteness() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("imarello_snapshot_\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let sha = "1cebb9b45c21ece14a42615b16bf5fa4de9b56da"
        try makeCompleteSnapshot(at: tmp, revision: sha)
        let snapshot = ModelSnapshot(modelID: "m", root: tmp, expectedRevision: sha)
        try snapshot.validateLayout()

        try FileManager.default.removeItem(
            at: tmp.appendingPathComponent("transformer/model-00001-of-00001.safetensors")
        )
        #expect(throws: ImarelloError.self) { try snapshot.validateLayout() }
    }

    @Test("mixed Hugging Face revisions are rejected even without an expected pin")
    func mixedSnapshotRevisions() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("imarello_snapshot_mixed_\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let a = "1cebb9b45c21ece14a42615b16bf5fa4de9b56da"
        let b = "a80f68acb469352cf31abe41a0869c7705ec7b57"
        try makeCompleteSnapshot(at: tmp, revision: a)
        let second = tmp.appendingPathComponent(
            ".cache/huggingface/download/transformer/0.safetensors.metadata"
        )
        try FileManager.default.createDirectory(
            at: second.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        try "\(b)\n".write(to: second, atomically: true, encoding: .utf8)
        #expect(throws: ImarelloError.self) {
            try ModelSnapshot(modelID: "m", root: tmp).validateLayout()
        }
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

    private func makeCompleteSnapshot(at root: URL, revision: String) throws {
        let fm = FileManager.default
        let tokenizer = root.appendingPathComponent("tokenizer", isDirectory: true)
        try fm.createDirectory(at: tokenizer, withIntermediateDirectories: true)
        for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
            try "{}".write(
                to: tokenizer.appendingPathComponent(name), atomically: true, encoding: .utf8
            )
        }
        for component in ["text_encoder", "transformer", "vae"] {
            let directory = root.appendingPathComponent(component, isDirectory: true)
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
            let shard = "model-00001-of-00001.safetensors"
            try Data([1]).write(to: directory.appendingPathComponent(shard))
            let index = #"{"weight_map":{"weight":"model-00001-of-00001.safetensors"}}"#
            try index.write(
                to: directory.appendingPathComponent("model.safetensors.index.json"),
                atomically: true, encoding: .utf8
            )
        }
        let metadata = root.appendingPathComponent(
            ".cache/huggingface/download/tokenizer/tokenizer.json.metadata"
        )
        try fm.createDirectory(at: metadata.deletingLastPathComponent(), withIntermediateDirectories: true)
        try "\(revision)\n".write(to: metadata, atomically: true, encoding: .utf8)
    }

    private func repoRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
