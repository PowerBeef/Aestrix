import Foundation
import ImarelloCore

/// Mac-driven device jobs. The contract is FROZEN — inbox/running/done paths,
/// the job/result schema, and PNG locations are what `Scripts/ios-device-harness.sh`
/// polls. Behavior is a faithful extraction of the original GenerationModel.
@MainActor
enum HarnessService {
    /// The poll loop fires every 2 s; only surface a distinct failure once so a
    /// persistent problem cannot clobber the dismiss button or other errors.
    private static var lastPollErrorMessage: String?
    private static var didSweepStaleJobs = false
    private static let doneFilesKept = 32

    /// Claim at most one inbox job. Safe to call from `.task` / scenePhase.
    static func pollInbox(model: StudioModel) {
        guard !model.isRunning else { return }
        do {
            try DeviceHarnessPaths.ensureDirectories()
            sweepStaleJobsIfNeeded()
            let inbox = try FileManager.default.contentsOfDirectory(
                at: DeviceHarnessPaths.inbox(),
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            )
            .filter { $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            lastPollErrorMessage = nil
            guard let url = inbox.first else { return }
            try claimAndRun(at: url, model: model)
        } catch {
            let message = error.localizedDescription
            if message != lastPollErrorMessage {
                lastPollErrorMessage = message
                model.errorMessage = message
            }
        }
    }

    /// Once per launch: report jobs orphaned by a crash/jetsam mid-run so the
    /// Mac harness gets a result instead of a timeout, and keep `done/` bounded.
    private static func sweepStaleJobsIfNeeded() {
        guard !didSweepStaleJobs else { return }
        didSweepStaleJobs = true
        let fm = FileManager.default
        if let orphans = try? fm.contentsOfDirectory(
            at: DeviceHarnessPaths.running(), includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) {
            for url in orphans where url.pathExtension == "json" {
                let stem = url.deletingPathExtension().lastPathComponent
                var width = 0
                var height = 0
                var seed: UInt64 = 0
                if let data = try? Data(contentsOf: url),
                   let job = try? DeviceHarnessPaths.jsonDecoder.decode(
                    DeviceHarnessJob.self, from: data) {
                    width = job.width
                    height = job.height
                    seed = job.seed
                }
                try? writeResult(
                    DeviceHarnessResult(
                        id: stem, status: .failed,
                        error: "app terminated before the job finished",
                        width: width, height: height, seed: seed,
                        startedAt: ISO8601DateFormatter().string(from: Date())
                    )
                )
                try? fm.removeItem(at: url)
            }
        }
        if let done = try? fm.contentsOfDirectory(
            at: DeviceHarnessPaths.done(),
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) {
            let dated = done
                .filter { $0.pathExtension == "json" }
                .compactMap { url -> (URL, Date)? in
                    let date = (try? url.resourceValues(forKeys: [.contentModificationDateKey]))?
                        .contentModificationDate
                    return (url, date ?? .distantPast)
                }
                .sorted { $0.1 > $1.1 }
            for (url, _) in dated.dropFirst(doneFilesKept) {
                try? fm.removeItem(at: url)
            }
        }
    }

    private static func claimAndRun(at inboxURL: URL, model: StudioModel) throws {
        let data = try Data(contentsOf: inboxURL)
        let job: DeviceHarnessJob
        do {
            job = try DeviceHarnessPaths.jsonDecoder.decode(DeviceHarnessJob.self, from: data)
        } catch {
            // An undecodable job must not stay in the inbox: `pollInbox` always
            // retries the lexicographic first file, so leaving it would wedge the
            // queue forever. Quarantine it and report failure under the filename
            // stem (the harness script names inbox files by job id).
            quarantine(inboxURL, decodeError: error, model: model)
            return
        }
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
                    prompt: job.prompt, side: job.width, seed: job.seed,
                    steps: job.steps, textTokens: job.textTokens
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
                    prompt: job.prompt, side: job.width, seed: job.seed,
                    strength: job.strength, steps: job.steps, textTokens: job.textTokens
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
            model.errorMessage = StudioModel.friendlyMessage(for: error)
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

    /// Move an undecodable inbox job aside and report it, so the queue keeps
    /// moving and the Mac harness sees a `failed` result instead of a timeout.
    private static func quarantine(_ inboxURL: URL, decodeError: Error, model: StudioModel) {
        let fm = FileManager.default
        let stem = inboxURL.deletingPathExtension().lastPathComponent
        let failedDir = DeviceHarnessPaths.root()
            .appendingPathComponent("failed", isDirectory: true)
        try? fm.createDirectory(at: failedDir, withIntermediateDirectories: true)
        let destination = failedDir.appendingPathComponent(inboxURL.lastPathComponent)
        try? fm.removeItem(at: destination)
        do {
            try fm.moveItem(at: inboxURL, to: destination)
        } catch {
            try? fm.removeItem(at: inboxURL)
        }
        try? writeResult(
            DeviceHarnessResult(
                id: stem, status: .failed,
                error: "undecodable job JSON: \(decodeError.localizedDescription)",
                width: 0, height: 0, seed: 0,
                startedAt: ISO8601DateFormatter().string(from: Date())
            )
        )
        model.errorMessage = "Harness job \(stem) could not be read; moved aside."
    }

    private static func writeResult(_ result: DeviceHarnessResult) throws {
        try DeviceHarnessPaths.ensureDirectories()
        let data = try DeviceHarnessPaths.jsonEncoder.encode(result)
        try data.write(to: DeviceHarnessPaths.doneFile(id: result.id), options: .atomic)
    }
}
