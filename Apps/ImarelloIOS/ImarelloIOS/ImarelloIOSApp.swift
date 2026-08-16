import SwiftUI

@main
struct ImarelloIOSApp: App {
    @State private var model = GenerationModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(model)
                .tint(ImarelloTheme.copper)
                .preferredColorScheme(.dark)
                .task {
                    model.refreshGate()
                    model.pollHarnessInbox()
                    while !Task.isCancelled {
                        try? await Task.sleep(for: .seconds(2))
                        model.pollHarnessInbox()
                    }
                }
                .onChange(of: scenePhase) { _, phase in
                    if phase == .active {
                        model.refreshGate()
                        model.pollHarnessInbox()
                    }
                }
        }
    }
}
