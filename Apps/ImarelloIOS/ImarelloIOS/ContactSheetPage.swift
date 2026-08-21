import SwiftUI

struct GalleryView: View {
    @Environment(PrintStore.self) private var store
    @Environment(AppNavigation.self) private var navigation
    @Environment(\.printImageLoader) private var imageLoader

    @State private var backdropImage: UIImage?

    private let columns = [GridItem(.adaptive(minimum: 110), spacing: 2)]

    var body: some View {
        ZStack {
            GalleryAtmosphere(image: backdropImage)

            Group {
                if store.prints.isEmpty {
                    emptyLibrary
                } else {
                    ScrollView {
                        LazyVGrid(columns: columns, spacing: 2) {
                            ForEach(Array(store.prints.enumerated()), id: \.element.id) { index, record in
                                GalleryThumbnailCell(
                                    record: record,
                                    position: index + 1,
                                    count: store.prints.count
                                ) {
                                    navigation.openImage(id: record.id, from: .gallery)
                                }
                            }
                        }
                        .padding(.bottom, ImarelloTheme.Space.xl)
                    }
                    .accessibilityIdentifier("gallery.grid")
                }
            }
        }
        .navigationTitle("Gallery")
        .navigationBarTitleDisplayMode(.inline)
        .navigationSubtitle(countLabel)
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                ImarelloToolbarMark()
            }
        }
        .task(id: store.prints.first?.id) {
            await loadBackdrop()
        }
    }

    private var countLabel: String {
        "\(store.prints.count) image\(store.prints.count == 1 ? "" : "s")"
    }

    private var emptyLibrary: some View {
        ContentUnavailableView {
            Label("No Images Yet", systemImage: "photo.stack")
        } description: {
            Text("Create an image and it will appear here.")
        } actions: {
            Button("Go to Create", action: navigation.showCreateRoot)
                .buttonStyle(.glass)
        }
        .accessibilityIdentifier("gallery.empty")
    }

    private func loadBackdrop() async {
        backdropImage = nil
        guard let latest = store.prints.first else { return }
        do {
            let loaded = try await imageLoader.image(
                at: store.url(for: latest),
                recordID: latest.id,
                maxPixel: 320
            )
            guard !Task.isCancelled else { return }
            backdropImage = loaded
        } catch {
            backdropImage = nil
        }
    }
}

private struct GalleryAtmosphere: View {
    let image: UIImage?

    var body: some View {
        ZStack {
            ImarelloTheme.stage
            if let image {
                AtmosphericImageBackdrop(image: image, dimmingOpacity: 0.72)
                TonalStageScrims()
            }
        }
        .ignoresSafeArea()
        .accessibilityHidden(true)
    }
}

private struct GalleryThumbnailCell: View {
    @Environment(PrintStore.self) private var store
    @Environment(\.printImageLoader) private var imageLoader

    let record: PrintRecord
    let position: Int
    let count: Int
    let action: () -> Void

    @State private var image: UIImage?

    var body: some View {
        Button(action: action) {
            ZStack(alignment: .bottomLeading) {
                Rectangle()
                    .fill(Color(.secondarySystemBackground))
                    .aspectRatio(1, contentMode: .fit)
                    .overlay {
                        if let image {
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFill()
                        } else {
                            ProgressView()
                                .controlSize(.small)
                        }
                    }
                    .clipped()
                if record.mode == "i2i" {
                    Label("Edit", systemImage: "wand.and.sparkles")
                        .labelStyle(.titleOnly)
                        .font(.caption2.weight(.semibold))
                        .textCase(.uppercase)
                        .kerning(0.8)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(.black.opacity(0.68), in: Capsule())
                        .foregroundStyle(ImarelloTheme.cream)
                        .padding(4)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityHint("Opens the image detail")
        .accessibilityIdentifier("gallery.cell.\(record.id)")
        .task(id: record.id) {
            await load()
        }
        .onDisappear { image = nil }
    }

    private var accessibilityLabel: String {
        let seed = record.seed.map(String.init) ?? "unknown"
        let provenance = record.mode == "i2i" ? ", edited from another image" : ""
        return "Image \(position) of \(count), \(record.side) by \(record.side), seed \(seed)\(provenance)"
    }

    private func load() async {
        let url = store.url(for: record)
        do {
            let loaded = try await imageLoader.image(
                at: url, recordID: record.id, maxPixel: 240
            )
            guard !Task.isCancelled else { return }
            image = loaded
        } catch {
            image = nil
        }
    }
}
