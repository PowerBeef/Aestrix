import Foundation
import Observation
import UIKit
import Photos
import ImarelloCore

/// UI state for the studio: the plate (prompt/side/seed), the current print,
/// and the run lifecycle. Talks to `GenerationEngine`; records into `PrintStore`.
@MainActor
@Observable
final class StudioModel {
    enum StudioAction: Equatable {
        case generate
        case edit(PrintRecord)
    }

    static let foxPlaceholder =
        "A red fox in a snowy forest at sunrise, photorealistic, golden rim light, shallow depth of field."

    let engine: GenerationEngine
    let store: PrintStore

    var prompt = StudioModel.foxPlaceholder
    var side = 512
    var seed: UInt64 = 42
    /// Editable digits for the seed field. `TextField(value:format:)` does not
    /// write back while a number pad is focused, so Generate can run the old seed.
    var seedText = "42"

    /// The print on the stage (defaults to the latest from history).
    var currentPrint: PrintRecord?

    /// A print staged for editing: the next Develop runs I2I from it at
    /// strength 0.8. Set from any print in history ("edit-from-any-print").
    var pendingEdit: PrintRecord?

    var isRunning = false
    var phase: PipelineProgress?
    var runStartedAt: Date?
    var errorMessage: String?
    var saveMessage: String?
    private(set) var lastAction: StudioAction?
    /// Set while a Mac harness job owns the pipeline.
    var harnessJobID: String?

    private var runTask: Task<Void, Never>?

    init(engine: GenerationEngine, store: PrintStore) {
        self.engine = engine
        self.store = store
        currentPrint = store.latest
    }

    // MARK: - plate

    @discardableResult
    func commitSeedText() -> UInt64 {
        let digits = seedText.filter(\.isNumber)
        if let value = UInt64(digits), !digits.isEmpty {
            seed = value
        }
        seedText = String(seed)
        return seed
    }

    func randomizeSeed() {
        seed = UInt64.random(in: 0 ... 9_999_999)
        seedText = String(seed)
    }

    var plateSummary: String { "\(side)² · seed \(seedText)" }

    func stageEdit(_ record: PrintRecord) {
        pendingEdit = record
        currentPrint = record
    }

    /// The gold shutter: edit when a print is staged, otherwise generate.
    func develop() {
        if let record = pendingEdit {
            edit(record)
        } else {
            generate()
        }
    }

    // MARK: - run lifecycle

    var canGenerate: Bool {
        !isRunning
            && !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && (engine.gate == .ready || engine.gate == .simulator)
    }

    func generate() {
        commitSeedText()
        guard canGenerate else { return }
        #if targetEnvironment(simulator)
        // Product lock: the Simulator never fakes Klein — Generate is chrome.
        return
        #else
        lastAction = .generate
        let prompt = prompt
        let side = side
        let seed = seed
        startRun { [weak self] in
            guard let self else { return }
            do {
                let url = try await engine.generate(
                    prompt: prompt, side: side, seed: seed
                ) { [weak self] progress in
                    self?.phase = progress
                }
                store.record(outputURL: url, prompt: prompt, seed: seed, side: side, mode: "t2i")
                currentPrint = store.latest
            } catch is CancellationError {
                // cancelled
            } catch {
                errorMessage = error.localizedDescription
            }
        }
        #endif
    }

    /// Edit any in-app print at strength 0.8, at the print's own size.
    func edit(_ record: PrintRecord) {
        commitSeedText()
        guard !isRunning, engine.gate == .ready || engine.gate == .simulator else { return }
        #if targetEnvironment(simulator)
        return
        #else
        lastAction = .edit(record)
        pendingEdit = nil
        let prompt = prompt
        let seed = seed
        let sourceSide = record.side > 0 ? record.side : side
        let source = store.url(for: record)
        startRun { [weak self] in
            guard let self else { return }
            do {
                let url = try await engine.edit(
                    source: source, prompt: prompt, side: sourceSide, seed: seed
                ) { [weak self] progress in
                    self?.phase = progress
                }
                store.record(outputURL: url, prompt: prompt, seed: seed, side: sourceSide, mode: "i2i")
                currentPrint = store.latest
            } catch is CancellationError {
                // cancelled
            } catch {
                errorMessage = error.localizedDescription
            }
        }
        #endif
    }

    func retryLast() {
        switch lastAction {
        case .generate: generate()
        case .edit(let record): edit(record)
        case nil: break
        }
    }

    func cancel() {
        runTask?.cancel()
        runTask = nil
        isRunning = false
        phase = nil
        runStartedAt = nil
    }

    private func startRun(_ work: @escaping @MainActor () async -> Void) {
        errorMessage = nil
        saveMessage = nil
        isRunning = true
        runStartedAt = Date()
        phase = PipelineProgress(phase: .preparing)
        runTask?.cancel()
        runTask = Task { @MainActor in
            await work()
            self.finishRun()
        }
    }

    private func finishRun() {
        isRunning = false
        if phase?.phase != .finished { phase = nil }
        runTask = nil
        runStartedAt = nil
    }

    // MARK: - print actions

    func saveToPhotos(_ record: PrintRecord) async {
        saveMessage = nil
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else {
            errorMessage = "Photos access was denied. Enable it in Settings to save."
            return
        }
        let url = store.url(for: record)
        do {
            try await PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAssetFromImage(atFileURL: url)
            }
            saveMessage = "Saved to Photos"
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func delete(_ record: PrintRecord) {
        store.delete(record)
        if currentPrint?.id == record.id {
            currentPrint = store.latest
        }
        if pendingEdit?.id == record.id {
            pendingEdit = nil
        }
    }

    // MARK: - status copy (one grammar for every state)

    var phaseLabel: String? {
        guard let phase else { return nil }
        switch phase.phase {
        case .preparing: return "Preparing"
        case .encodingText: return "Reading the prompt"
        case .encodingImage: return "Reading the print"
        case .denoising:
            if phase.totalSteps > 0 { return "Developing \(phase.step)/\(phase.totalSteps)" }
            return "Developing"
        case .decoding: return "Fixing the print"
        case .finished: return "Done"
        }
    }
}
