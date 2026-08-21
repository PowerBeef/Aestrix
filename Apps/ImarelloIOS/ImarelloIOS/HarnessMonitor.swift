import Foundation
import Observation

@MainActor
@Observable
final class HarnessMonitor {
    typealias Poll = @MainActor (StudioSession) -> Void

    private let session: StudioSession
    private let poll: Poll
    private var pollTask: Task<Void, Never>?

    private(set) var isMonitoring = false

    init(
        session: StudioSession,
        poll: @escaping Poll = { HarnessService.pollInbox(session: $0) }
    ) {
        self.session = session
        self.poll = poll
    }

    func setActive(_ active: Bool) {
        if active { start() } else { stop() }
    }

    private func start() {
        guard pollTask == nil else { return }
        isMonitoring = true
        session.engine.refreshGate()
        poll(session)
        pollTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: .seconds(2))
                } catch {
                    break
                }
                guard let self, !Task.isCancelled else { break }
                self.poll(self.session)
            }
        }
    }

    private func stop() {
        pollTask?.cancel()
        pollTask = nil
        isMonitoring = false
    }
}
