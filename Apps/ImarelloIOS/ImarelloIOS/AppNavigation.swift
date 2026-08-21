import Observation

enum AppTab: String, CaseIterable, Hashable, Identifiable {
    case create
    case edit
    case gallery
    case settings

    var id: Self { self }
}

enum AppRoute: Hashable {
    case image(id: String)
}

enum GenerationWorkspace: String, Hashable {
    case create
    case edit
}

enum AppSheet: Hashable, Identifiable {
    case generationOptions(GenerationWorkspace)

    var id: String {
        switch self {
        case .generationOptions(let workspace): return "generation-options-\(workspace.rawValue)"
        }
    }
}

@MainActor
@Observable
final class AppNavigation {
    var selectedTab: AppTab = .create
    var createPath: [AppRoute] = []
    var editPath: [AppRoute] = []
    var galleryPath: [AppRoute] = []
    var settingsPath: [AppRoute] = []
    var presentedSheet: AppSheet?

    func openImage(id: String, from tab: AppTab? = nil) {
        let source = tab ?? selectedTab
        selectedTab = source
        switch source {
        case .create:
            createPath.append(.image(id: id))
        case .edit:
            editPath.append(.image(id: id))
        case .gallery:
            galleryPath.append(.image(id: id))
        case .settings:
            settingsPath.append(.image(id: id))
        }
    }

    func showCreateRoot() {
        selectedTab = .create
        createPath.removeAll()
    }

    func showEditRoot() {
        selectedTab = .edit
        editPath.removeAll()
    }

    func showGalleryRoot() {
        selectedTab = .gallery
        galleryPath.removeAll()
    }

    func showRoot(for tab: AppTab) {
        switch tab {
        case .create: showCreateRoot()
        case .edit: showEditRoot()
        case .gallery:
            selectedTab = .gallery
            galleryPath.removeAll()
        case .settings:
            selectedTab = .settings
            settingsPath.removeAll()
        }
    }

    func isAtRoot(_ tab: AppTab) -> Bool {
        switch tab {
        case .create: return createPath.isEmpty
        case .edit: return editPath.isEmpty
        case .gallery: return galleryPath.isEmpty
        case .settings: return settingsPath.isEmpty
        }
    }
}
