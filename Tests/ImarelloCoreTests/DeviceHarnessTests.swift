import Testing
import Foundation
@testable import ImarelloCore

@Suite("Device harness job contract")
struct DeviceHarnessTests {
    @Test("JSON round-trip keeps t2i defaults")
    func jsonRoundTrip() throws {
        let job = DeviceHarnessJob(
            id: "fox-512-s42",
            prompt: DeviceHarnessJob.defaultFoxPrompt,
            width: 512,
            height: 512,
            seed: 42
        )
        let data = try DeviceHarnessPaths.jsonEncoder.encode(job)
        let decoded = try DeviceHarnessPaths.jsonDecoder.decode(DeviceHarnessJob.self, from: data)
        #expect(decoded == job)
        #expect(decoded.textTokens == .full512)
        #expect(decoded.mode == .t2i)
        #expect(decoded.steps == 4)
    }

    @Test("512 and 1024 square jobs validate")
    func allowedSides() throws {
        try DeviceHarnessJob(id: "a", prompt: "fox", width: 512, height: 512).validate(hasLastImage: false)
        try DeviceHarnessJob(id: "b", prompt: "fox", width: 1024, height: 1024).validate(hasLastImage: false)
    }

    @Test("768 and non-square jobs are rejected")
    func rejectBadCanvas() {
        #expect(throws: ImarelloError.self) {
            try DeviceHarnessJob(id: "c", prompt: "fox", width: 768, height: 768).validate(hasLastImage: false)
        }
        #expect(throws: ImarelloError.self) {
            try DeviceHarnessJob(id: "d", prompt: "fox", width: 512, height: 1024).validate(hasLastImage: false)
        }
    }

    @Test("empty prompt is rejected")
    func emptyPrompt() {
        #expect(throws: ImarelloError.self) {
            try DeviceHarnessJob(id: "e", prompt: "   ").validate(hasLastImage: false)
        }
    }

    @Test("i2i without a last image fails closed")
    func i2iNeedsLastImage() throws {
        let job = DeviceHarnessJob(id: "f", mode: .i2i, prompt: "recolor")
        #expect(throws: ImarelloError.self) {
            try job.validate(hasLastImage: false)
        }
        try job.validate(hasLastImage: true)
    }

    @Test("unsafe job ids are sanitized")
    func sanitizeID() {
        #expect(DeviceHarnessJob.sanitizedID("fox-512-s42") == "fox-512-s42")
        #expect(DeviceHarnessJob.sanitizedID("foo/bar") == "foo-bar")
        #expect(DeviceHarnessJob.sanitizedID("") == "job")
    }

    @Test("simulator skip result is skipped not ok")
    func simulatorSkip() {
        let job = DeviceHarnessJob(id: "s", prompt: "fox")
        let result = DeviceHarnessResult.skippedSimulator(job: job)
        #expect(result.status == .skipped)
        #expect(result.error != nil)
    }
}
