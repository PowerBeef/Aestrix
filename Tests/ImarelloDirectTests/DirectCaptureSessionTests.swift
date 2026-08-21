import Foundation
import Testing
@testable import ImarelloDirect

@Suite("Direct capture artifacts")
struct DirectCaptureSessionTests {
    @Test("capture is disabled by default and writes atomic manifest when opted in")
    func optInCapture() throws {
        let metadata = metadata()
        #expect(try DirectCaptureSession.fromEnvironment(
            metadata: metadata, environment: [:]) == nil)

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("direct-capture-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let capture = try DirectCaptureSession(directory: directory, metadata: metadata)
        try capture.capture(
            id: "te.tap/9", data: Data([1, 2, 3, 4]),
            dataType: "bfloat16", shape: [2])
        capture.setPlanDigest("plan")
        try capture.finalize()

        let manifestData = try Data(contentsOf: directory.appendingPathComponent(
            "capture-manifest.json"))
        let manifest = try JSONDecoder().decode(DirectCaptureManifest.self, from: manifestData)
        #expect(manifest.run.planDigest == "plan")
        #expect(manifest.entries.map(\.file) == ["te_tap_9.bin"])
    }

    private func metadata() -> DirectCaptureRunMetadata {
        DirectCaptureRunMetadata(
            prompt: "fox", seed: 1, width: 512, height: 512,
            steps: 4, guidance: 1, tokenMode: "512",
            modelRevision: "model", snapshotRevision: "snapshot",
            weightMode: "4bit", backendIdentifier: "direct",
            mlxMetallibPath: "/mlx", directShaderSHA256: "shader",
            directMetallibSHA256: "direct", deviceName: "device",
            operatingSystem: "os", thermalState: "nominal",
            outputPath: "/output.png", planDigest: nil)
    }
}
