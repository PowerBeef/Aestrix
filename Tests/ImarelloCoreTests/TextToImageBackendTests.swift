import Foundation
import Testing
@testable import ImarelloCore
@testable import ImarelloRuntime

@Suite("Whole text-to-image backend routing", .serialized)
struct TextToImageBackendTests {
    private final class StubBackend: TextToImageGenerationBackend {
        let identifier = "stub-v2"

        func supports(_ request: T2IRequest) -> Bool {
            request.prompt == "supported"
        }

        func generate(
            _ request: T2IRequest,
            onProgress: (@Sendable (PipelineProgress) -> Void)?,
            trace: PipelineTrace?
        ) throws -> URL {
            onProgress?(PipelineProgress(phase: .preparing))
            onProgress?(PipelineProgress(
                phase: .finished, step: request.steps, totalSteps: request.steps))
            return try #require(request.outputURL)
        }
    }

    @Test("supported request routes before any model stage loads")
    func supportedRequestUsesBackend() async throws {
        let fixture = try SnapshotFixture()
        defer { fixture.remove() }

        let pipeline = ImarelloPipeline(config: fixture.config)
        await pipeline.setTextToImageBackend(StubBackend())

        let output = fixture.root.appendingPathComponent("backend-result.png")
        let request = T2IRequest(
            prompt: "supported", width: 512, height: 512, steps: 4,
            seed: 42, outputURL: output)
        let result = try await pipeline.generate(request)
        #expect(result == output)
        #expect(!FileManager.default.fileExists(atPath: output.path))
    }
}

private struct SnapshotFixture {
    let root: URL
    let config: ImarelloConfig

    init() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("imarello-backend-\(UUID().uuidString)", isDirectory: true)
        let config = ImarelloConfig(
            tier: .low, maxSide: 512, modelsDirectory: root)
        let snapshot = root.appendingPathComponent(
            config.modelID.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
        let fm = FileManager.default
        for component in ["text_encoder", "transformer", "vae", "tokenizer"] {
            try fm.createDirectory(
                at: snapshot.appendingPathComponent(component, isDirectory: true),
                withIntermediateDirectories: true)
        }

        for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] {
            try Data("fixture".utf8).write(
                to: snapshot.appendingPathComponent("tokenizer/\(name)"))
        }
        let index = Data(#"{"weight_map":{"fixture":"0.safetensors"}}"#.utf8)
        for component in ["text_encoder", "transformer", "vae"] {
            let directory = snapshot.appendingPathComponent(component, isDirectory: true)
            try index.write(to: directory.appendingPathComponent("model.safetensors.index.json"))
            try Data([0]).write(to: directory.appendingPathComponent("0.safetensors"))
        }

        let metadataDirectory = snapshot.appendingPathComponent(
            ".cache/huggingface/download", isDirectory: true)
        try fm.createDirectory(at: metadataDirectory, withIntermediateDirectories: true)
        try Data("\(config.revision)\n".utf8).write(
            to: metadataDirectory.appendingPathComponent("fixture.metadata"))

        self.root = root
        self.config = config
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }
}
