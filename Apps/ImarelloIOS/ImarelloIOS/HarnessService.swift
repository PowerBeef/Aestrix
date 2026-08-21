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
    private static let persistence = HarnessPersistence(
        root: DeviceHarnessPaths.root(),
        fileIO: SystemPrintStoreFileIO()
    )

    /// Claim at most one inbox job. Safe to call from `.task` / scenePhase.
    static func pollInbox(session: StudioSession) {
        guard !session.isBusy else { return }
        do {
            try DeviceHarnessPaths.ensureDirectories()
            try sweepStaleJobsIfNeeded()
            let inbox = try FileManager.default.contentsOfDirectory(
                at: DeviceHarnessPaths.inbox(),
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            )
            .filter { $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            lastPollErrorMessage = nil
            guard let url = inbox.first else { return }
            try claimAndRun(at: url, session: session)
        } catch {
            let message = error.localizedDescription
            if message != lastPollErrorMessage {
                lastPollErrorMessage = message
                session.failHarness(message)
            }
        }
    }

    /// Once per launch: report jobs orphaned by a crash/jetsam mid-run so the
    /// Mac harness gets a result instead of a timeout, and keep `done/` bounded.
    private static func sweepStaleJobsIfNeeded() throws {
        guard !didSweepStaleJobs else { return }
        let fm = FileManager.default
        if let orphans = try? fm.contentsOfDirectory(
            at: DeviceHarnessPaths.running(), includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) {
            for url in orphans where url.pathExtension == "json" {
                let stem = url.deletingPathExtension().lastPathComponent
                if persistence.hasDurableResult(id: stem) {
                    // A prior finalize committed the result but was interrupted
                    // while removing the recovery marker. The durable result wins.
                    try fm.removeItem(at: url)
                    continue
                }
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
                try finalize(
                    DeviceHarnessResult(
                        id: stem, status: .failed,
                        error: "app terminated before the job finished",
                        width: width, height: height, seed: seed,
                        startedAt: ISO8601DateFormatter().string(from: Date())
                    ),
                    runningURL: url
                )
            }
        }
        didSweepStaleJobs = true
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

    private static func claimAndRun(at inboxURL: URL, session: StudioSession) throws {
        let data = try Data(contentsOf: inboxURL)
        let job: DeviceHarnessJob
        do {
            job = try DeviceHarnessPaths.jsonDecoder.decode(DeviceHarnessJob.self, from: data)
        } catch {
            // An undecodable job must not stay in the inbox: `pollInbox` always
            // retries the lexicographic first file, so leaving it would wedge the
            // queue forever. Quarantine it and report failure under the filename
            // stem (the harness script names inbox files by job id).
            try quarantine(inboxURL, decodeError: error, session: session)
            return
        }
        do {
            guard inboxURL.deletingPathExtension().lastPathComponent == job.id else {
                throw ImarelloError.invalidRequest(
                    "harness filename must match the embedded job id"
                )
            }
            try job.validate(hasLastImage: session.store.latest != nil)
            let running = DeviceHarnessPaths.runningFile(id: job.id)
            guard !FileManager.default.fileExists(atPath: running.path) else {
                throw ImarelloError.invalidRequest("harness job id is already running")
            }
            let done = DeviceHarnessPaths.doneFile(id: job.id)
            guard !FileManager.default.fileExists(atPath: done.path) else {
                throw ImarelloError.invalidRequest("harness job id has already been used")
            }
        } catch {
            try reject(inboxURL, job: job, error: error, session: session)
            return
        }
        let running = DeviceHarnessPaths.runningFile(id: job.id)
        try FileManager.default.moveItem(at: inboxURL, to: running)

        #if targetEnvironment(simulator)
        try finalize(.skippedSimulator(job: job), runningURL: running)
        #else
        // Reflect the job on the plate so the UI shows what is actually running.
        session.beginHarness(job)
        Task { @MainActor in
            await run(job, runningURL: running, session: session)
        }
        #endif
    }

    #if !targetEnvironment(simulator)
    private static func run(
        _ job: DeviceHarnessJob, runningURL: URL, session: StudioSession
    ) async {
        let started = Date()
        let iso = ISO8601DateFormatter().string(from: started)
        let result: DeviceHarnessResult
        var finishedPrint: PrintRecord?
        var failureMessage: String?
        do {
            let metalNote = MetallibVerification.resolveFromBundles()
                .map { MetallibVerification.verify(url: $0).note }

            let url: URL
            switch job.mode {
            case .t2i:
                url = try await session.engine.generate(
                    prompt: job.prompt, side: job.width, seed: job.seed,
                    steps: job.steps, textTokens: job.textTokens
                ) { progress in
                    session.updateHarnessProgress(progress)
                }
                finishedPrint = try session.store.record(
                    outputURL: url, prompt: job.prompt, seed: job.seed,
                    side: job.width, mode: "t2i")
            case .i2i:
                guard let latest = session.store.latest else {
                    throw ImarelloError.imageLoadFailed(
                        path: "<last-in-app>",
                        reason: "i2i harness job needs a last generated PNG"
                    )
                }
                url = try await session.engine.edit(
                    source: session.store.url(for: latest),
                    prompt: job.prompt, side: job.width, seed: job.seed,
                    strength: job.strength, steps: job.steps, textTokens: job.textTokens
                ) { progress in
                    session.updateHarnessProgress(progress)
                }
                finishedPrint = try session.store.record(
                    outputURL: url, prompt: job.prompt, seed: job.seed,
                    side: job.width, mode: "i2i")
            }
            result = DeviceHarnessResult(
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
        } catch is CancellationError {
            failureMessage = "cancelled"
            result = DeviceHarnessResult(
                id: job.id, status: .failed, error: "cancelled",
                width: job.width, height: job.height, seed: job.seed,
                elapsedSec: Date().timeIntervalSince(started), startedAt: iso
            )
        } catch {
            failureMessage = StudioSession.friendlyMessage(for: error)
            result = DeviceHarnessResult(
                id: job.id, status: .failed, error: error.localizedDescription,
                width: job.width, height: job.height, seed: job.seed,
                elapsedSec: Date().timeIntervalSince(started), startedAt: iso
            )
        }
        do {
            try finalize(result, runningURL: runningURL)
            if let finishedPrint {
                session.finishHarness(print: finishedPrint)
            } else {
                session.failHarness(failureMessage ?? "The Mac run failed.", id: job.id)
            }
        } catch {
            // The running marker is deliberately retained. On the next poll or
            // launch the stale sweep can durably report the interrupted job.
            session.failHarness(
                "Harness result could not be saved; recovery state was retained. \(error.localizedDescription)",
                id: job.id
            )
        }
    }
    #endif

    /// Move an undecodable inbox job aside and report it, so the queue keeps
    /// moving and the Mac harness sees a `failed` result instead of a timeout.
    private static func quarantine(
        _ inboxURL: URL, decodeError: Error, session: StudioSession
    ) throws {
        let stem = inboxURL.deletingPathExtension().lastPathComponent
        try writeResult(
            DeviceHarnessResult(
                id: stem, status: .failed,
                error: "undecodable job JSON: \(decodeError.localizedDescription)",
                width: 0, height: 0, seed: 0,
                startedAt: ISO8601DateFormatter().string(from: Date())
            )
        )
        try moveToFailed(inboxURL)
        session.failHarness("Harness job \(stem) could not be read; moved aside.", id: stem)
    }

    private static func reject(
        _ inboxURL: URL, job: DeviceHarnessJob, error: Error, session: StudioSession
    ) throws {
        let running = DeviceHarnessPaths.runningFile(id: job.id)
        let collidesWithActiveOrCompletedJob = FileManager.default.fileExists(
            atPath: running.path
        ) || persistence.hasDurableResult(id: job.id)
        // A duplicate must never overwrite the result or recovery marker that
        // belongs to the original run. Fresh invalid jobs still get a durable
        // failure result under the frozen wire contract.
        if !collidesWithActiveOrCompletedJob {
            try writeResult(
                DeviceHarnessResult(
                    id: job.id, status: .failed,
                    error: "invalid harness job: \(error.localizedDescription)",
                    width: job.width, height: job.height, seed: job.seed,
                    startedAt: ISO8601DateFormatter().string(from: Date())
                )
            )
        }
        try moveToFailed(inboxURL)
        session.failHarness(
            "Harness job \(job.id) was rejected: \(error.localizedDescription)", id: job.id
        )
    }

    private static func moveToFailed(_ source: URL) throws {
        let fm = FileManager.default
        let failedDir = DeviceHarnessPaths.root()
            .appendingPathComponent("failed", isDirectory: true)
        try fm.createDirectory(at: failedDir, withIntermediateDirectories: true)
        let destination = failedDir.appendingPathComponent(
            "\(UUID().uuidString)-\(source.lastPathComponent)"
        )
        try fm.moveItem(at: source, to: destination)
    }

    private static func finalize(_ result: DeviceHarnessResult, runningURL: URL) throws {
        try persistence.finalize(result, runningURL: runningURL)
    }

    private static func writeResult(_ result: DeviceHarnessResult) throws {
        try persistence.writeResult(result)
    }
}
