import SwiftUI

@main
struct ImarelloIOSApp: App {
    @State private var model = GenerationModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(model)
                .tint(ImarelloTheme.copper)
                .preferredColorScheme(.dark)
                .task {
                    model.refreshGate()
                }
        }
    }
}
