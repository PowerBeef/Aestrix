import Foundation
import ImarelloCore

#if DEBUG
enum UITestScenario: String {
    case empty
    case library
    case singleSource = "single-source"
    case sourceLoadFailure = "source-load-failure"
    case manySources = "many-sources"
    case pendingEdit = "pending-edit"
    case running
    case failed

    static var current: UITestScenario? {
        let arguments = ProcessInfo.processInfo.arguments
        if let flag = arguments.firstIndex(of: "--ui-test-scenario"),
           arguments.indices.contains(flag + 1) {
            return UITestScenario(rawValue: arguments[flag + 1])
        }
        return ProcessInfo.processInfo.environment["IMARELLO_UI_TEST_SCENARIO"]
            .flatMap(UITestScenario.init(rawValue:))
    }

    @MainActor
    func makeStore() -> PrintStore {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ImarelloUITest-\(UUID().uuidString)", isDirectory: true)
        return PrintStore(
            durableRoot: root.appendingPathComponent("durable", isDirectory: true),
            outputsDirectory: root.appendingPathComponent("outputs", isDirectory: true),
            legacyIndexURL: root.appendingPathComponent("legacy.json"),
            fileIO: SystemPrintStoreFileIO()
        )
    }

    @MainActor
    func install(into session: StudioSession) {
        guard self != .empty else { return }

        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("ImarelloUITestFixtures-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        for fixture in fixtures {
            guard let bundled = Bundle.main.url(
                forResource: fixture.asset, withExtension: "png"
            ) else { continue }
            let output = scratch.appendingPathComponent("\(fixture.id).png")
            guard (try? FileManager.default.copyItem(at: bundled, to: output)) != nil else {
                continue
            }
            _ = try? session.store.record(
                outputURL: output, prompt: fixture.prompt,
                seed: fixture.seed, side: 1024, mode: fixture.mode
            )
        }
        session.refreshCurrentPrintFromStore()

        if self == .sourceLoadFailure, let latest = session.store.prints.first {
            try? FileManager.default.removeItem(at: session.store.url(for: latest))
        }

        switch self {
        case .empty, .library, .singleSource, .sourceLoadFailure, .manySources:
            break
        case .pendingEdit:
            if let source = session.store.prints.last {
                session.selectEditSource(source)
            }
        case .running:
            let run = GenerationRun(
                id: UUID(), owner: .user,
                operation: session.createOperationSnapshot(),
                startedAt: Date().addingTimeInterval(-7),
                progress: PipelineProgress(phase: .denoising, step: 1, totalSteps: 4)
            )
            session.installUITestActivity(.running(run))
        case .failed:
            session.installUITestActivity(
                .failed(
                    GenerationFailure(
                        owner: .user,
                        operation: session.createOperationSnapshot(),
                        message: "The image could not be prepared."
                    )
                )
            )
        }
    }

    private var fixtures: [Fixture] {
        let text = Fixture(
            id: "woman_t2i", asset: "woman_t2i",
            prompt: "Editorial portrait in warm window light.",
            seed: 42, mode: "t2i"
        )
        let edit = Fixture(
            id: "woman_i2i", asset: "woman_i2i",
            prompt: "Add a painterly blue jacket while preserving the portrait.",
            seed: 43, mode: "i2i"
        )
        switch self {
        case .empty:
            return []
        case .singleSource, .sourceLoadFailure:
            return [edit]
        case .manySources:
            return [
                text,
                edit,
                Fixture(id: "portrait_3", asset: "woman_t2i", prompt: text.prompt, seed: 44, mode: "t2i"),
                Fixture(id: "portrait_4", asset: "woman_i2i", prompt: edit.prompt, seed: 45, mode: "i2i"),
                Fixture(id: "portrait_5", asset: "woman_t2i", prompt: text.prompt, seed: 46, mode: "t2i"),
                Fixture(id: "portrait_6", asset: "woman_i2i", prompt: edit.prompt, seed: 47, mode: "i2i"),
            ]
        case .library, .pendingEdit, .running, .failed:
            return [text, edit]
        }
    }
}

private struct Fixture {
    let id: String
    let asset: String
    let prompt: String
    let seed: UInt64
    let mode: String
}
#endif
