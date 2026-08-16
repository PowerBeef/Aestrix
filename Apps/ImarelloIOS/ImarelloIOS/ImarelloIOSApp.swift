import SwiftUI

@main
struct ImarelloIOSApp: App {
    @State private var engine: GenerationEngine
    @State private var store: PrintStore
    @State private var model: StudioModel
    @Environment(\.scenePhase) private var scenePhase

    init() {
        let engine = GenerationEngine()
        let store = PrintStore()
        _engine = State(initialValue: engine)
        _store = State(initialValue: store)
        _model = State(initialValue: StudioModel(engine: engine, store: store))
    }

    var body: some Scene {
        WindowGroup {
            StudioRootView()
                .environment(model)
                .environment(store)
                .environment(engine)
                .tint(ImarelloTheme.copper)
                .preferredColorScheme(.dark)
                .task {
                    engine.refreshGate()
                    HarnessService.pollInbox(model: model)
                    while !Task.isCancelled {
                        try? await Task.sleep(for: .seconds(2))
                        HarnessService.pollInbox(model: model)
                    }
                }
                .onChange(of: scenePhase) { _, phase in
                    if phase == .active {
                        engine.refreshGate()
                        HarnessService.pollInbox(model: model)
                    }
                }
        }
    }
}
