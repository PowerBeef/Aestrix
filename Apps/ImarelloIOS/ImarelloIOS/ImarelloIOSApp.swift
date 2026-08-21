import SwiftUI

@main
struct ImarelloIOSApp: App {
    @State private var engine: GenerationEngine
    @State private var store: PrintStore
    @State private var session: StudioSession
    @State private var navigation: AppNavigation
    @State private var photoExporter: PhotoExportService
    @State private var harnessMonitor: HarnessMonitor
    @Environment(\.scenePhase) private var scenePhase

    private let imageLoader = PrintImageLoader()
    private let usesUITestFixture: Bool

    init() {
        let engine = GenerationEngine()
        #if DEBUG
        let scenario = UITestScenario.current
        let store = scenario?.makeStore() ?? PrintStore()
        #else
        let store = PrintStore()
        #endif
        let session = StudioSession(engine: engine, store: store)
        #if DEBUG
        scenario?.install(into: session)
        usesUITestFixture = scenario != nil
        #else
        usesUITestFixture = false
        #endif

        _engine = State(initialValue: engine)
        _store = State(initialValue: store)
        _session = State(initialValue: session)
        _navigation = State(initialValue: AppNavigation())
        _photoExporter = State(initialValue: PhotoExportService())
        _harnessMonitor = State(initialValue: HarnessMonitor(session: session))
    }

    var body: some Scene {
        WindowGroup {
            StudioRootView()
                .environment(session)
                .environment(store)
                .environment(engine)
                .environment(navigation)
                .environment(photoExporter)
                .environment(\.printImageLoader, imageLoader)
                .tint(ImarelloTheme.copper)
                .preferredColorScheme(.dark)
                .task {
                    engine.refreshGate()
                    if !usesUITestFixture {
                        harnessMonitor.setActive(scenePhase == .active)
                    }
                }
                .onChange(of: scenePhase) { _, phase in
                    guard !usesUITestFixture else { return }
                    if phase == .active { engine.refreshGate() }
                    harnessMonitor.setActive(phase == .active)
                }
        }
    }
}
