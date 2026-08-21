import Photos
import SwiftUI
import UIKit

struct SettingsPage: View {
    @Environment(GenerationEngine.self) private var engine
    @Environment(PrintStore.self) private var store
    @Environment(AppNavigation.self) private var navigation

    @State private var storageBytes: Int64?
    @State private var photoStatus = PHAuthorizationStatus.notDetermined

    private let storageService = GalleryStorageService()

    var body: some View {
        Form {
            Section("Device and Model") {
                LabeledContent("Status") {
                    Label(modelStatus.title, systemImage: modelStatus.symbol)
                        .foregroundStyle(modelStatus.color)
                }
                LabeledContent("Runtime", value: runtimeLabel)
            }

            Section("Gallery") {
                LabeledContent("Images", value: String(store.prints.count))
                LabeledContent("Storage", value: storageLabel)
                Button {
                    navigation.showGalleryRoot()
                } label: {
                    Label("Open Gallery", systemImage: "square.grid.2x2")
                }
            }

            Section("Photos") {
                LabeledContent("Access", value: photoPresentation.title)
                if photoPresentation.canOpenSettings {
                    Button {
                        guard let url = URL(string: UIApplication.openSettingsURLString) else {
                            return
                        }
                        UIApplication.shared.open(url)
                    } label: {
                        Label("Open System Settings", systemImage: "arrow.up.forward.app")
                    }
                }
            }

            Section("About") {
                LabeledContent("App", value: "Imarello")
                LabeledContent("Version", value: versionLabel)
                LabeledContent("Engine", value: "FLUX.2 Klein 4B")
            }
        }
        .scrollContentBackground(.hidden)
        .background(ImarelloTheme.canvas.ignoresSafeArea())
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.large)
        .task(id: store.prints.map(\.id)) {
            let urls = store.prints.map { store.url(for: $0) }
            storageBytes = await storageService.bytes(for: urls)
        }
        .onAppear {
            photoStatus = PHPhotoLibrary.authorizationStatus(for: .addOnly)
        }
        .accessibilityIdentifier("settings.root")
    }

    private var modelStatus: ModelStatusPresentation {
        switch engine.gate {
        case .ready:
            return ModelStatusPresentation(
                title: "Ready",
                symbol: "checkmark.circle.fill",
                color: .green
            )
        case .missingWeights:
            return ModelStatusPresentation(
                title: "Model unavailable",
                symbol: "exclamationmark.triangle.fill",
                color: .orange
            )
        case .simulator:
            return ModelStatusPresentation(
                title: "Interface preview",
                symbol: "iphone.gen3",
                color: .secondary
            )
        }
    }

    private var runtimeLabel: String {
        #if targetEnvironment(simulator)
        "Simulator · UI only"
        #else
        "On-device"
        #endif
    }

    private var storageLabel: String {
        guard let storageBytes else { return "Calculating…" }
        return ByteCountFormatter.string(fromByteCount: storageBytes, countStyle: .file)
    }

    private var photoPresentation: PhotoAccessPresentation {
        PhotoAccessPresentation(status: photoStatus)
    }

    private var versionLabel: String {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString")
            as? String ?? "—"
        let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion")
            as? String ?? "—"
        return "\(version) (\(build))"
    }
}

struct PhotoAccessPresentation: Equatable {
    let title: String
    let canOpenSettings: Bool

    init(status: PHAuthorizationStatus) {
        switch status {
        case .authorized: self = Self(title: "Allowed", canOpenSettings: false)
        case .limited: self = Self(title: "Limited", canOpenSettings: false)
        case .denied: self = Self(title: "Denied", canOpenSettings: true)
        case .restricted: self = Self(title: "Restricted", canOpenSettings: true)
        case .notDetermined: self = Self(title: "Not requested", canOpenSettings: false)
        @unknown default: self = Self(title: "Unknown", canOpenSettings: false)
        }
    }

    private init(title: String, canOpenSettings: Bool) {
        self.title = title
        self.canOpenSettings = canOpenSettings
    }
}

private struct ModelStatusPresentation {
    let title: String
    let symbol: String
    let color: Color
}
