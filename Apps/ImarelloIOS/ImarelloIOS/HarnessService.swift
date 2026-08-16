import Foundation
import ImarelloCore

/// Mac-driven device jobs. The contract is FROZEN — inbox/running/done paths,
/// the job/result schema, and PNG locations are what `Scripts/ios-device-harness.sh`
/// polls. Behavior is a faithful extraction of the original GenerationModel.
@MainActor
enum HarnessService {
    /// Claim at most one inbox job. Safe to call from `.task` / scenePhase.
    static func pollInbox(model: StudioModel) {
        guard !model.isRunning else { return }
        do {
            try DeviceHarnessPaths.ensureDirectories()
            let inbox = try FileManager.default.contentsOfDirectory(
                at: DeviceHarnessPaths.inbox(),
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            )
            .filter { $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            guard let url = inbox.first else { return }
            try claimAndRun(at: url, model: model)
        } catch {
            model.errorMessage = error.localizedDescription
        }
    }

    private static func claimAndRun(at inboxURL: URL, model: StudioModel) throws {
        let data = try Data(contentsOf: inboxURL)
        let job = try DeviceHarnessPaths.jsonDecoder.decode(DeviceHarnessJob.self, from: data)
        let running = DeviceHarnessPaths.runningFile(id: job.id)
        try? FileManager.default.removeItem(at: running)
        try FileManager.default.moveItem(at: inboxURL, to: running)

        #if targetEnvironment(simulator)
        try writeResult(.skippedSimulator(job: job))
        try? FileManager.default.removeItem(at: running)
        #else
        // Reflect the job on the plate so the UI shows what is actually running.
        model.prompt = job.prompt
        model.side = job.width
        model.seed = job.seed
        model.seedText = String(job.seed)
        model.harnessJobID = job.id
        model.isRunning = true
        model.runStartedAt = Date()
        model.phase = PipelineProgress(phase: .preparing)
        Task { @MainActor in
            await run(job, runningURL: running, model: model)
        }
        #endif
    }

    #if !targetEnvironment(simulator)
    private static func run(_ job: DeviceHarnessJob, runningURL: URL, model: StudioModel) async {
        let started = Date()
        let iso = ISO8601DateFormatter().string(from: started)
        defer {
            model.isRunning = false
            model.phase = nil
            model.runStartedAt = nil
            model.harnessJobID = nil
            try? FileManager.default.removeItem(at: runningURL)
        }
        do {
            try job.validate(hasLastImage: model.store.latest != nil)
            try await model.engine.ensureReady()
            let metalNote = MetallibVerification.resolveFromBundles()
                .map { MetallibVerification.verify(url: $0).note }

            let url: URL
            switch job.mode {
            case .t2i:
                url = try await model.engine.generate(
                    prompt: job.prompt, side: job.width, seed: job.seed
                ) { [weak model] progress in
                    model?.phase = progress
                }
                model.store.record(
                    outputURL: url, prompt: job.prompt, seed: job.seed,
                    side: job.width, mode: "t2i")
            case .i2i:
                guard let latest = model.store.latest else {
                    throw ImarelloError.imageLoadFailed(
                        path: "<last-in-app>",
                        reason: "i2i harness job needs a last generated PNG"
                    )
                }
                url = try await model.engine.edit(
                    source: model.store.url(for: latest),
                    prompt: job.prompt, side: job.width, seed: job.seed
                ) { [weak model] progress in
                    model?.phase = progress
                }
                model.store.record(
                    outputURL: url, prompt: job.prompt, seed: job.seed,
                    side: job.width, mode: "i2i")
            }
            model.currentPrint = model.store.latest
            try writeResult(
                DeviceHarnessResult(
                    id: job.id,
                    status: .ok,
                    pngRelativePath: DeviceHarnessPaths.pngContainerPath(
                        filename: url.lastPathComponent),
                    width: job.width,
                    height: job.height,
                    seed: job.seed,
                    elapsedSec: Date().timeIntervalSince(started),
                    startedAt: iso,
                    metallibNote: metalNote
                )
            )
        } catch is CancellationError {
            try? writeResult(
                DeviceHarnessResult(
                    id: job.id, status: .failed, error: "cancelled",
                    width: job.width, height: job.height, seed: job.seed,
                    elapsedSec: Date().timeIntervalSince(started), startedAt: iso
                )
            )
        } catch {
            model.errorMessage = error.localizedDescription
            try? writeResult(
                DeviceHarnessResult(
                    id: job.id, status: .failed, error: error.localizedDescription,
                    width: job.width, height: job.height, seed: job.seed,
                    elapsedSec: Date().timeIntervalSince(started), startedAt: iso
                )
            )
        }
    }
    #endif

    private static func writeResult(_ result: DeviceHarnessResult) throws {
        try DeviceHarnessPaths.ensureDirectories()
        let data = try DeviceHarnessPaths.jsonEncoder.encode(result)
        try data.write(to: DeviceHarnessPaths.doneFile(id: result.id), options: .atomic)
    }
}
