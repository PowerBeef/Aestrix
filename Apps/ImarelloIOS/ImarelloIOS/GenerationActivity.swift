import Foundation
import ImarelloCore

enum GenerationOwner: Equatable {
    case user
    case harness(id: String)

    var isUser: Bool {
        if case .user = self { return true }
        return false
    }

    var label: String { self == .user ? "On this device" : "Mac run" }
}

struct GenerationRequestSnapshot: Equatable {
    let prompt: String
    let side: Int
    let seed: UInt64
    let source: PrintRecord?
}

enum GenerationOperation: Equatable {
    case create(GenerationRequestSnapshot)
    case edit(GenerationRequestSnapshot)

    var snapshot: GenerationRequestSnapshot {
        switch self {
        case .create(let snapshot), .edit(let snapshot): return snapshot
        }
    }

    var destination: AppTab {
        switch self {
        case .create: return .create
        case .edit: return .edit
        }
    }

    var actionLabel: String {
        switch self {
        case .create: return "Creating"
        case .edit: return "Editing"
        }
    }
}

struct GenerationRun {
    let id: UUID
    let owner: GenerationOwner
    let operation: GenerationOperation?
    let startedAt: Date
    var progress: PipelineProgress
}

struct GenerationFailure {
    let owner: GenerationOwner
    let operation: GenerationOperation?
    let message: String
}

enum GenerationActivity {
    case idle
    case running(GenerationRun)
    case stopping(GenerationRun)
    case completed(
        print: PrintRecord,
        owner: GenerationOwner,
        operation: GenerationOperation?
    )
    case failed(GenerationFailure)

    var isBusy: Bool {
        switch self {
        case .running, .stopping: return true
        case .idle, .completed, .failed: return false
        }
    }

    var run: GenerationRun? {
        switch self {
        case .running(let run), .stopping(let run): return run
        case .idle, .completed, .failed: return nil
        }
    }

    var shouldShowGlobalAccessory: Bool {
        if case .idle = self { return false }
        return true
    }

    var operation: GenerationOperation? {
        switch self {
        case .running(let run), .stopping(let run): return run.operation
        case .completed(_, _, let operation): return operation
        case .failed(let failure): return failure.operation
        case .idle: return nil
        }
    }

    var destination: AppTab { operation?.destination ?? .create }
}
