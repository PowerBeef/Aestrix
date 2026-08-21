import Foundation
import Testing
import UIKit
@testable import Imarello
import ImarelloCore

@MainActor
@Suite("Generation session")
struct StudioModelTests {
    @Test("Create and Edit have independent repository-locked defaults")
    func defaultDrafts() {
        let fixture = makeFixture()
        defer { fixture.remove() }

        #expect(fixture.session.createDraft.side == 1024)
        #expect(fixture.session.createDraft.seed == 42)
        #expect(fixture.session.editDraft.seed == 42)
        fixture.session.createDraft.prompt = "Create prompt"
        fixture.session.editDraft.prompt = "Edit prompt"
        #expect(fixture.session.createDraft.prompt == "Create prompt")
        #expect(fixture.session.editDraft.prompt == "Edit prompt")
    }

    @Test("generation options validate without mutating the committed seed")
    func transactionalGenerationOptions() {
        let fixture = makeFixture()
        defer { fixture.remove() }
        var draft = GenerationOptionsDraft(seed: fixture.session.createDraft.seed)
        draft.seedText = "18446744073709551616"
        #expect(!draft.isValid)
        #expect(fixture.session.createDraft.seed == 42)

        draft.seedText = "7"
        #expect(draft.isValid)
        fixture.session.setSeed(draft.seed ?? 0, for: .create)
        #expect(fixture.session.createDraft.seed == 7)
        #expect(fixture.session.editDraft.seed == 42)
    }

    @Test("resolution changes only the Create draft")
    func createResolution() {
        let fixture = makeFixture()
        defer { fixture.remove() }
        fixture.session.setCreateResolution(512)
        #expect(fixture.session.createDraft.side == 512)
        #expect(fixture.session.createDraft.seed == 42)
        #expect(fixture.session.editDraft.side == 1024)
    }

    @Test("Edit source locks output size without mutating Create")
    func editSourceSize() {
        let fixture = makeFixture()
        defer { fixture.remove() }
        let record = PrintRecord(
            id: "edit", fileName: "edit.png", prompt: "p", seed: 1,
            side: 512, mode: "i2i", createdAt: .now
        )
        fixture.session.selectEditSource(record)
        fixture.session.setCreateResolution(1024)
        fixture.session.setSeed(8, for: .edit)
        #expect(fixture.session.editDraft.side == 512)
        #expect(fixture.session.createDraft.side == 1024)
        #expect(fixture.session.editSummary.contains("Source 512²"))
        #expect(fixture.session.editStrength == 0.8)
    }

    @Test("zero-based progress uses operation-specific plain language")
    func progressLabel() {
        let fixture = makeFixture()
        defer { fixture.remove() }
        let createOperation = GenerationOperation.create(
            GenerationRequestSnapshot(prompt: "p", side: 1024, seed: 42, source: nil)
        )
        let first = GenerationRun(
            id: UUID(), owner: .user, operation: createOperation,
            startedAt: .now,
            progress: PipelineProgress(phase: .denoising, step: 0, totalSteps: 4)
        )
        fixture.session.installUITestActivity(.running(first))
        #expect(fixture.session.phaseLabel == "Creating 1/4")

        var last = first
        last.progress = PipelineProgress(phase: .denoising, step: 3, totalSteps: 4)
        fixture.session.installUITestActivity(.running(last))
        #expect(fixture.session.phaseLabel == "Creating 4/4")
    }

    @Test("successful Create records an image")
    func completion() async throws {
        let fixture = makeFixture()
        defer { fixture.remove() }
        let output = try makePNG(in: fixture.root, side: 512)
        fixture.engine.behavior = .success(output)
        fixture.session.setCreateResolution(512)

        fixture.session.createImage()
        await waitUntil { fixture.session.completedImageID != nil }

        #expect(fixture.session.currentPrint?.side == 512)
        #expect(fixture.store.prints.count == 1)
        fixture.session.acknowledgeCompletion()
        if case .idle = fixture.session.activity {
            // expected
        } else {
            Issue.record("Completion should acknowledge to idle")
        }
    }

    @Test("failure retries the exact immutable request snapshot")
    func retryProvenance() async throws {
        let fixture = makeFixture()
        defer { fixture.remove() }
        fixture.session.createDraft = GenerationDraft(
            prompt: "Original prompt", side: 512, seed: 7
        )
        fixture.engine.behavior = .failure(FakeGenerationError.expected)

        fixture.session.createImage()
        await waitUntil {
            if case .failed = fixture.session.activity { return true }
            return false
        }
        guard case .failed(let failure) = fixture.session.activity else {
            Issue.record("Expected failure state")
            return
        }
        let original = failure.operation?.snapshot
        #expect(original?.prompt == "Original prompt")
        #expect(original?.side == 512)
        #expect(original?.seed == 7)

        fixture.session.createDraft = GenerationDraft(
            prompt: "Changed after failure", side: 1024, seed: 99
        )
        fixture.engine.behavior = .success(try makePNG(in: fixture.root, side: 512))
        fixture.session.retryFailure()
        await waitUntil { fixture.session.completedImageID != nil }

        #expect(fixture.engine.invocations.last?.prompt == "Original prompt")
        #expect(fixture.engine.invocations.last?.side == 512)
        #expect(fixture.engine.invocations.last?.seed == 7)
    }

    @Test("successful Edit makes the result the next source")
    func iterativeEdit() async throws {
        let fixture = makeFixture()
        defer { fixture.remove() }
        let source = try fixture.store.record(
            outputURL: makePNG(in: fixture.root, name: "source.png", side: 512),
            prompt: "source", seed: 1, side: 512, mode: "t2i"
        )
        fixture.session.selectEditSource(source)
        fixture.session.editDraft.prompt = "Add blue light"
        fixture.engine.behavior = .success(
            try makePNG(in: fixture.root, name: "edited.png", side: 512)
        )

        fixture.session.editImage()
        await waitUntil { fixture.session.completedImageID != nil }

        #expect(fixture.session.editSource?.id == "edited")
        #expect(fixture.session.editSource?.side == 512)
        #expect(fixture.session.editDraft.prompt == "Add blue light")
    }

    @Test("harness activity never overwrites user drafts")
    func harnessPreservesDrafts() {
        let fixture = makeFixture()
        defer { fixture.remove() }
        fixture.session.createDraft = GenerationDraft(prompt: "Create", side: 1024, seed: 42)
        fixture.session.editDraft = GenerationDraft(prompt: "Edit", side: 512, seed: 7)

        fixture.session.beginHarness(
            DeviceHarnessJob(id: "mac", prompt: "Harness", width: 512, seed: 3)
        )

        #expect(fixture.session.createDraft.prompt == "Create")
        #expect(fixture.session.createDraft.side == 1024)
        #expect(fixture.session.editDraft.prompt == "Edit")
        #expect(fixture.session.editDraft.seed == 7)
    }

    @Test("cancel enters stopping and unwinds to idle")
    func cancellation() async {
        let fixture = makeFixture()
        defer { fixture.remove() }
        fixture.engine.behavior = .waitForCancellation

        fixture.session.createImage()
        #expect(fixture.session.isBusy)
        fixture.session.cancel()
        if case .stopping = fixture.session.activity {
            // expected
        } else {
            Issue.record("Cancel should enter stopping immediately")
        }
        await waitUntil {
            if case .idle = fixture.session.activity { return true }
            return false
        }
    }

    @Test("deleting the selected source clears Edit and selects the next image")
    func deletingCurrentImage() throws {
        let fixture = makeFixture()
        defer { fixture.remove() }
        let first = try fixture.store.record(
            outputURL: makePNG(in: fixture.root, name: "first.png", side: 512),
            prompt: "first", seed: 1, side: 512, mode: "t2i"
        )
        let second = try fixture.store.record(
            outputURL: makePNG(in: fixture.root, name: "second.png", side: 512),
            prompt: "second", seed: 2, side: 512, mode: "t2i"
        )
        fixture.session.refreshCurrentPrintFromStore()
        fixture.session.selectEditSource(second)
        try fixture.session.delete(second)
        #expect(fixture.session.currentPrint == first)
        #expect(fixture.session.editSource == nil)
    }

    private func makeFixture() -> StudioFixture {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("StudioSessionTests-\(UUID().uuidString)", isDirectory: true)
        let store = PrintStore(
            durableRoot: root.appendingPathComponent("durable"),
            outputsDirectory: root.appendingPathComponent("outputs"),
            legacyIndexURL: root.appendingPathComponent("legacy.json"),
            fileIO: SystemPrintStoreFileIO()
        )
        let engine = FakeGenerationService(root: root)
        return StudioFixture(
            root: root, engine: engine, store: store,
            session: StudioSession(engine: engine, store: store)
        )
    }

    private func makePNG(
        in root: URL, name: String = "result.png", side: Int
    ) throws -> URL {
        let url = root.appendingPathComponent(name)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(
            size: CGSize(width: side, height: side), format: format
        )
        let image = renderer.image { context in
            UIColor.systemBlue.setFill()
            context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        }
        try #require(image.pngData()).write(to: url)
        return url
    }

    private func waitUntil(_ predicate: @MainActor () -> Bool) async {
        for _ in 0 ..< 1_000 {
            if predicate() { return }
            await Task.yield()
        }
        Issue.record("Timed out waiting for state transition")
    }
}

@MainActor
private struct StudioFixture {
    let root: URL
    let engine: FakeGenerationService
    let store: PrintStore
    let session: StudioSession

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }
}

private enum FakeGenerationError: LocalizedError {
    case expected
    var errorDescription: String? { "Expected generation failure" }
}

@MainActor
private final class FakeGenerationService: GenerationServing {
    struct Invocation: Equatable {
        let prompt: String
        let side: Int
        let seed: UInt64
        let isEdit: Bool
    }

    enum Behavior {
        case success(URL)
        case failure(any Error)
        case waitForCancellation
    }

    var gate: GenerationEngine.RunGate = .ready
    var behavior: Behavior
    var allowsGenerationInCurrentProcess = true
    let expectedModelsDirectory: URL
    private(set) var invocations: [Invocation] = []

    init(root: URL) {
        expectedModelsDirectory = root
        behavior = .failure(FakeGenerationError.expected)
    }

    func refreshGate() {}

    func generate(
        prompt: String, side: Int, seed: UInt64, steps: Int,
        textTokens: TextTokenMode,
        onProgress: @escaping @MainActor (PipelineProgress) -> Void
    ) async throws -> URL {
        invocations.append(Invocation(prompt: prompt, side: side, seed: seed, isEdit: false))
        return try await run(onProgress: onProgress)
    }

    func edit(
        source: URL, prompt: String, side: Int, seed: UInt64,
        strength: Float, steps: Int, textTokens: TextTokenMode,
        onProgress: @escaping @MainActor (PipelineProgress) -> Void
    ) async throws -> URL {
        invocations.append(Invocation(prompt: prompt, side: side, seed: seed, isEdit: true))
        return try await run(onProgress: onProgress)
    }

    private func run(
        onProgress: @escaping @MainActor (PipelineProgress) -> Void
    ) async throws -> URL {
        onProgress(PipelineProgress(phase: .denoising, step: 0, totalSteps: 4))
        switch behavior {
        case .success(let url): return url
        case .failure(let error): throw error
        case .waitForCancellation:
            while true {
                try Task.checkCancellation()
                await Task.yield()
            }
        }
    }
}
