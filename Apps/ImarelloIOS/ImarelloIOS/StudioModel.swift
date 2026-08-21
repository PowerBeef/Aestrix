import Foundation
import Observation
import ImarelloCore

struct GenerationDraft: Equatable {
    var prompt: String
    var side: Int
    var seed: UInt64
}

@MainActor
@Observable
final class StudioSession {
    static let foxPlaceholder =
        "A red fox in a snowy forest at sunrise, photorealistic, golden rim light, shallow depth of field."

    let engine: any GenerationServing
    let store: PrintStore

    var createDraft = GenerationDraft(
        prompt: StudioSession.foxPlaceholder,
        side: 1024,
        seed: 42
    )
    var editDraft = GenerationDraft(prompt: "", side: 1024, seed: 42)
    private(set) var currentPrint: PrintRecord?
    private(set) var editSource: PrintRecord?
    private(set) var activity: GenerationActivity = .idle

    private var runTask: Task<Void, Never>?

    init(engine: any GenerationServing, store: PrintStore) {
        self.engine = engine
        self.store = store
        currentPrint = store.latest
    }

    var createSummary: String { "\(createDraft.side)² · seed \(createDraft.seed)" }
    var editSummary: String {
        guard let source = editSource else { return "Choose a source image" }
        return "Source \(source.side)² · strength 0.8 · seed \(editDraft.seed)"
    }
    var editStrength: Float { 0.8 }
    var isBusy: Bool { activity.isBusy }
    var showsGlobalActivity: Bool { activity.shouldShowGlobalAccessory }
    var completedImageID: String? {
        if case .completed(let image, _, _) = activity { return image.id }
        return nil
    }
    var harnessJobID: String? {
        guard case .harness(let id) = activity.run?.owner else { return nil }
        return id
    }

    var canCreate: Bool { canRun(createDraft) }
    var canEdit: Bool { editSource != nil && canRun(editDraft) }

    func setCreateResolution(_ side: Int) {
        guard DeviceHarnessJob.allowedSides.contains(side), !isBusy else { return }
        createDraft.side = side
    }

    func setSeed(_ seed: UInt64, for workspace: GenerationWorkspace) {
        guard !isBusy else { return }
        switch workspace {
        case .create: createDraft.seed = seed
        case .edit: editDraft.seed = seed
        }
    }

    func selectEditSource(_ record: PrintRecord) {
        guard !isBusy else { return }
        editSource = record
        editDraft.side = record.side
    }

    func clearEditSource() {
        guard !isBusy else { return }
        editSource = nil
    }

    func createImage() {
        let snapshot = GenerationRequestSnapshot(
            prompt: createDraft.prompt,
            side: createDraft.side,
            seed: createDraft.seed,
            source: nil
        )
        guard canCreate, engine.allowsGenerationInCurrentProcess else { return }
        start(.create(snapshot))
    }

    func editImage() {
        guard let source = editSource else { return }
        let snapshot = GenerationRequestSnapshot(
            prompt: editDraft.prompt,
            side: source.side,
            seed: editDraft.seed,
            source: source
        )
        guard canEdit, engine.allowsGenerationInCurrentProcess else { return }
        start(.edit(snapshot))
    }

    func retryFailure() {
        guard case .failed(let failure) = activity,
              failure.owner.isUser,
              let operation = failure.operation,
              engine.allowsGenerationInCurrentProcess
        else { return }
        start(operation)
    }

    func cancel() {
        guard case .running(let run) = activity, run.owner.isUser else { return }
        activity = .stopping(run)
        runTask?.cancel()
    }

    func dismissActivity() {
        guard !activity.isBusy else { return }
        activity = .idle
    }

    func acknowledgeCompletion() {
        if case .completed = activity { activity = .idle }
    }

    func delete(_ record: PrintRecord) throws {
        try store.delete(record)
        if currentPrint?.id == record.id { currentPrint = store.latest }
        if editSource?.id == record.id { editSource = nil }
        if case .failed(let failure) = activity,
           failure.operation?.snapshot.source?.id == record.id {
            activity = .idle
        }
    }

    static func friendlyMessage(for error: Error) -> String {
        if case ImarelloError.weightsNotFound = error {
            return "The model is not on this device yet — sync it from the Mac."
        }
        return error.localizedDescription
    }

    var phaseLabel: String? {
        guard let run = activity.run else { return nil }
        if case .stopping = activity { return "Stopping" }
        let action = run.operation?.actionLabel ?? "Creating"
        switch run.progress.phase {
        case .preparing: return "Preparing"
        case .encodingText: return "Reading the prompt"
        case .encodingImage: return "Reading the source"
        case .denoising:
            if run.progress.totalSteps > 0 {
                let step = min(run.progress.totalSteps, max(1, run.progress.step + 1))
                return "\(action) \(step)/\(run.progress.totalSteps)"
            }
            return action
        case .decoding: return "Finishing the image"
        case .finished: return "Done"
        }
    }

    // MARK: - Harness bridge

    func beginHarness(_ job: DeviceHarnessJob) {
        let run = GenerationRun(
            id: UUID(), owner: .harness(id: job.id), operation: nil,
            startedAt: Date(), progress: PipelineProgress(phase: .preparing)
        )
        activity = .running(run)
    }

    func updateHarnessProgress(_ progress: PipelineProgress) {
        guard case .running(var run) = activity,
              case .harness = run.owner
        else { return }
        run.progress = progress
        activity = .running(run)
    }

    func finishHarness(print: PrintRecord) {
        currentPrint = print
        guard let owner = activity.run?.owner else { return }
        activity = .completed(print: print, owner: owner, operation: nil)
    }

    func failHarness(_ message: String, id: String? = nil) {
        let owner = activity.run?.owner ?? .harness(id: id ?? "monitor")
        activity = .failed(
            GenerationFailure(owner: owner, operation: nil, message: message)
        )
    }

    #if DEBUG
    func installUITestActivity(_ fixture: GenerationActivity) {
        activity = fixture
    }

    func refreshCurrentPrintFromStore() {
        currentPrint = store.latest
    }

    func createOperationSnapshot() -> GenerationOperation {
        .create(
            GenerationRequestSnapshot(
                prompt: createDraft.prompt,
                side: createDraft.side,
                seed: createDraft.seed,
                source: nil
            )
        )
    }
    #endif

    private func canRun(_ draft: GenerationDraft) -> Bool {
        !isBusy
            && !draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && (engine.gate == .ready || engine.gate == .simulator)
    }

    private func start(_ operation: GenerationOperation) {
        let run = GenerationRun(
            id: UUID(), owner: .user, operation: operation,
            startedAt: Date(), progress: PipelineProgress(phase: .preparing)
        )
        activity = .running(run)
        runTask?.cancel()
        let snapshot = operation.snapshot

        runTask = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                let url: URL
                let mode: String
                switch operation {
                case .create:
                    mode = "t2i"
                    url = try await engine.generate(
                        prompt: snapshot.prompt,
                        side: snapshot.side,
                        seed: snapshot.seed
                    ) { [weak self] progress in
                        self?.updateUserProgress(progress, runID: run.id)
                    }
                case .edit:
                    guard let source = snapshot.source else {
                        throw ImarelloError.invalidRequest("An edit source is required")
                    }
                    mode = "i2i"
                    url = try await engine.edit(
                        source: store.url(for: source),
                        prompt: snapshot.prompt,
                        side: snapshot.side,
                        seed: snapshot.seed,
                        strength: editStrength
                    ) { [weak self] progress in
                        self?.updateUserProgress(progress, runID: run.id)
                    }
                }
                try Task.checkCancellation()
                let image = try store.record(
                    outputURL: url,
                    prompt: snapshot.prompt,
                    seed: snapshot.seed,
                    side: snapshot.side,
                    mode: mode
                )
                guard self.matches(run.id) else { return }
                currentPrint = image
                if case .edit = operation {
                    editSource = image
                    editDraft.side = image.side
                }
                activity = .completed(
                    print: image,
                    owner: .user,
                    operation: operation
                )
            } catch is CancellationError {
                if self.matches(run.id) { activity = .idle }
            } catch {
                if self.matches(run.id) {
                    activity = .failed(
                        GenerationFailure(
                            owner: .user,
                            operation: operation,
                            message: Self.friendlyMessage(for: error)
                        )
                    )
                }
            }
            if self.matches(run.id) == false || activity.isBusy == false {
                runTask = nil
            }
        }
    }

    private func updateUserProgress(_ progress: PipelineProgress, runID: UUID) {
        guard case .running(var run) = activity, run.id == runID else { return }
        run.progress = progress
        activity = .running(run)
    }

    private func matches(_ runID: UUID) -> Bool {
        activity.run?.id == runID
    }
}
