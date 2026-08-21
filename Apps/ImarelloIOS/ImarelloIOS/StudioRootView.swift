import SwiftUI
import UIKit

struct StudioRootView: View {
    @Environment(StudioSession.self) private var session
    @Environment(AppNavigation.self) private var navigation
    @Environment(\.printImageLoader) private var imageLoader

    var body: some View {
        @Bindable var navigation = navigation
        TabView(selection: $navigation.selectedTab) {
            NavigationStack(path: $navigation.createPath) {
                CreatePage()
                    .withImageDestinations()
            }
            .tabItem {
                Label("Create", systemImage: "sparkles")
            }
            .tag(AppTab.create)
            .accessibilityIdentifier("tab.create")

            NavigationStack(path: $navigation.editPath) {
                EditPage()
                    .withImageDestinations()
            }
            .tabItem {
                Label("Edit", systemImage: "wand.and.sparkles")
            }
            .tag(AppTab.edit)
            .accessibilityIdentifier("tab.edit")

            NavigationStack(path: $navigation.galleryPath) {
                GalleryView()
                    .withImageDestinations()
            }
            .tabItem {
                Label("Gallery", systemImage: "square.grid.2x2")
            }
            .tag(AppTab.gallery)
            .accessibilityIdentifier("tab.gallery")

            NavigationStack(path: $navigation.settingsPath) {
                SettingsPage()
                    .withImageDestinations()
            }
            .tabItem {
                Label("Settings", systemImage: "gearshape")
            }
            .tag(AppTab.settings)
            .accessibilityIdentifier("tab.settings")
        }
        .tabBarMinimizeBehavior(.never)
        .tabViewBottomAccessory(isEnabled: session.showsGlobalActivity) {
            GenerationAccessory()
        }
        .sheet(item: $navigation.presentedSheet) { sheet in
            switch sheet {
            case .generationOptions(let workspace):
                GenerationOptionsSheet(session: session, workspace: workspace)
            }
        }
        .onChange(of: session.completedImageID) { _, completedID in
            guard completedID != nil,
                  navigation.selectedTab == session.activity.destination,
                  navigation.isAtRoot(navigation.selectedTab)
            else { return }
            session.acknowledgeCompletion()
        }
        .onReceive(
            NotificationCenter.default.publisher(
                for: UIApplication.didReceiveMemoryWarningNotification
            )
        ) { _ in
            Task { await imageLoader.removeAll() }
        }
        .background(ImarelloTheme.canvas.ignoresSafeArea())
    }
}

private extension View {
    func withImageDestinations() -> some View {
        navigationDestination(for: AppRoute.self) { route in
            switch route {
            case .image(let id):
                PrintDetailView(initialID: id)
            }
        }
    }
}

#Preview {
    let engine = GenerationEngine()
    let store = PrintStore()
    let session = StudioSession(engine: engine, store: store)
    let navigation = AppNavigation()
    return StudioRootView()
        .environment(session)
        .environment(store)
        .environment(engine)
        .environment(navigation)
        .environment(PhotoExportService())
        .tint(ImarelloTheme.copper)
        .preferredColorScheme(.dark)
}
