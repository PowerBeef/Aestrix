import SwiftUI

struct EditPage: View {
    @Environment(StudioSession.self) private var session
    @Environment(PrintStore.self) private var store
    @Environment(AppNavigation.self) private var navigation
    @Environment(\.printImageLoader) private var imageLoader
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var sourceImage: UIImage?

    var body: some View {
        Group {
            if session.editSource == nil {
                EditSourcePicker()
            } else {
                editor
            }
        }
        .navigationTitle("Edit")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                ImarelloToolbarMark()
            }
            if let source = session.editSource {
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button {
                        navigation.presentedSheet = .generationOptions(.edit)
                    } label: {
                        Image(systemName: "slider.horizontal.3")
                            .frame(width: 44, height: 44)
                    }
                    .disabled(session.isBusy)
                    .accessibilityLabel("Generation options")
                    .accessibilityHint("Change the seed")
                    .accessibilityIdentifier("edit.options")

                    Button {
                        session.clearEditSource()
                    } label: {
                        Image(systemName: "photo.on.rectangle")
                            .frame(width: 44, height: 44)
                    }
                    .disabled(session.isBusy)
                    .accessibilityLabel("Choose another source")
                    .accessibilityValue("Current source is \(source.caption)")
                    .accessibilityIdentifier("edit.change-source")
                }
            }
        }
        .task(id: session.editSource?.id) {
            await loadSource()
        }
        .animation(reduceMotion ? nil : .snappy, value: session.showsGlobalActivity)
    }

    private var editor: some View {
        ZStack {
            ImarelloTheme.stage.ignoresSafeArea()
            sourceStage
            TonalStageScrims()
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            if !session.showsGlobalActivity {
                GenerationComposer(workspace: .edit)
                    .padding(.horizontal, ImarelloTheme.Space.md)
                    .padding(.bottom, ImarelloTheme.Space.xs)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
    }

    @ViewBuilder
    private var sourceStage: some View {
        if let source = session.editSource, let sourceImage {
            AtmosphericImageStage(image: sourceImage)
            .ignoresSafeArea()
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Source image, \(source.caption)")
        } else {
            ProgressView("Loading source")
                .tint(.secondary)
        }
    }

    private func loadSource() async {
        guard let source = session.editSource else {
            sourceImage = nil
            return
        }
        do {
            let loaded = try await imageLoader.image(
                at: store.url(for: source),
                recordID: source.id,
                maxPixel: 1024
            )
            guard !Task.isCancelled else { return }
            sourceImage = loaded
        } catch is CancellationError {
            return
        } catch {
            sourceImage = nil
        }
    }
}

struct EditSourceCollection {
    let featured: PrintRecord?
    let remaining: [PrintRecord]

    init(records: [PrintRecord]) {
        featured = records.first
        remaining = Array(records.dropFirst())
    }
}

private struct EditSourcePicker: View {
    @Environment(StudioSession.self) private var session
    @Environment(PrintStore.self) private var store
    @Environment(AppNavigation.self) private var navigation
    @Environment(\.printImageLoader) private var imageLoader

    @State private var featuredImage: UIImage?
    @State private var featuredLoadFailed = false

    private var sources: EditSourceCollection {
        EditSourceCollection(records: store.prints)
    }

    var body: some View {
        ZStack {
            ImarelloTheme.stage.ignoresSafeArea()
            sourceContent
            if sources.featured != nil {
                TonalStageScrims()
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            if let featured = sources.featured {
                EditSourceBar(
                    featured: featured,
                    remaining: sources.remaining,
                    totalCount: store.prints.count,
                    featuredLoadFailed: featuredLoadFailed,
                    select: session.selectEditSource
                )
            }
        }
        .task(id: sources.featured?.id) {
            await loadFeaturedSource()
        }
    }

    @ViewBuilder
    private var sourceContent: some View {
        if let featured = sources.featured {
            if let featuredImage {
                Button {
                    session.selectEditSource(featured)
                } label: {
                    AtmosphericImageStage(image: featuredImage)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .ignoresSafeArea()
                .accessibilityLabel(
                    "Featured source, \(featured.side) by \(featured.side), \(featured.caption)"
                )
                .accessibilityHint("Selects this image as the edit source")
                .accessibilityIdentifier("edit.featured-source")
            } else if featuredLoadFailed {
                EditSourceUnavailableState()
            } else {
                ProgressView("Loading source")
                    .tint(.secondary)
                    .accessibilityIdentifier("edit.featured-loading")
            }
        } else {
            EditSourceEmptyState(action: navigation.showCreateRoot)
        }
    }

    private func loadFeaturedSource() async {
        featuredImage = nil
        featuredLoadFailed = false
        guard let featured = sources.featured else { return }
        do {
            guard let loaded = try await imageLoader.image(
                at: store.url(for: featured),
                recordID: featured.id,
                maxPixel: 1024
            ) else {
                featuredLoadFailed = true
                return
            }
            guard !Task.isCancelled else { return }
            featuredImage = loaded
        } catch is CancellationError {
            return
        } catch {
            featuredLoadFailed = true
        }
    }
}

private struct EditSourceBar: View {
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    let featured: PrintRecord
    let remaining: [PrintRecord]
    let totalCount: Int
    let featuredLoadFailed: Bool
    let select: (PrintRecord) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: ImarelloTheme.Space.xs) {
            Group {
                if dynamicTypeSize.isAccessibilitySize {
                    VStack(alignment: .leading, spacing: ImarelloTheme.Space.xxs) {
                        sourceHeading
                        sourceReading
                    }
                } else {
                    HStack(spacing: ImarelloTheme.Space.sm) {
                        sourceHeading
                        Spacer(minLength: ImarelloTheme.Space.sm)
                        sourceReading
                    }
                }
            }

            if remaining.isEmpty {
                Text(featuredLoadFailed ? "No other images are available." : "Tap the image to begin editing.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            } else {
                ScrollView(.horizontal) {
                    LazyHStack(spacing: ImarelloTheme.Space.xs) {
                        ForEach(Array(remaining.enumerated()), id: \.element.id) { index, record in
                            EditSourceThumbnail(
                                record: record,
                                position: index + 2,
                                count: totalCount
                            ) {
                                select(record)
                            }
                        }
                    }
                }
                .frame(height: 76)
                .scrollIndicators(.hidden)
                .accessibilityIdentifier("edit.source-strip")
            }
        }
        .padding(.horizontal, ImarelloTheme.Space.md)
        .padding(.top, ImarelloTheme.Space.sm)
        .padding(
            .bottom,
            dynamicTypeSize.isAccessibilitySize ? ImarelloTheme.Space.xl + 8 : ImarelloTheme.Space.xs
        )
        .background {
            LinearGradient(
                colors: [.black.opacity(0.9), ImarelloTheme.stage.opacity(0.96)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea(edges: .bottom)
        }
        .accessibilityIdentifier("edit.source-bar")
    }

    private var sourceHeading: some View {
        Label("Choose a source", systemImage: "photo.on.rectangle.angled")
            .font(.headline)
            .foregroundStyle(ImarelloTheme.cream)
            .lineLimit(2)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var sourceReading: some View {
        Text("\(featured.caption) · \(totalCount) image\(totalCount == 1 ? "" : "s")")
            .font(.caption)
            .monospacedDigit()
            .foregroundStyle(.secondary)
            .lineLimit(2)
            .fixedSize(horizontal: false, vertical: true)
    }
}

private struct EditSourceThumbnail: View {
    @Environment(PrintStore.self) private var store
    @Environment(\.printImageLoader) private var imageLoader

    let record: PrintRecord
    let position: Int
    let count: Int
    let action: () -> Void

    @State private var image: UIImage?

    var body: some View {
        Button(action: action) {
            Rectangle()
                .fill(Color(.secondarySystemBackground))
                .frame(width: 76, height: 76)
                .overlay {
                    if let image {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFill()
                    } else {
                        ProgressView().controlSize(.small)
                    }
                }
                .clipped()
        }
        .buttonStyle(.plain)
        .accessibilityLabel(
            "Source \(position) of \(count), \(record.side) by \(record.side), \(record.caption)"
        )
        .accessibilityHint("Selects this image as the edit source")
        .accessibilityIdentifier("edit.source.\(record.id)")
        .task(id: record.id) { await load() }
        .onDisappear { image = nil }
    }

    private func load() async {
        do {
            let loaded = try await imageLoader.image(
                at: store.url(for: record),
                recordID: record.id,
                maxPixel: 240
            )
            guard !Task.isCancelled else { return }
            image = loaded
        } catch {
            image = nil
        }
    }
}

private struct EditSourceEmptyState: View {
    let action: () -> Void

    var body: some View {
        VStack(spacing: ImarelloTheme.Space.sm) {
            Image("Mark")
                .resizable()
                .scaledToFit()
                .frame(width: 72, height: 72)
                .opacity(0.9)
                .accessibilityHidden(true)
            Text("No images to edit")
                .font(.title2.weight(.semibold))
                .foregroundStyle(ImarelloTheme.cream)
            Text("Create an image first, then return here to edit it.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button("Go to Create", action: action)
                .buttonStyle(.glass)
                .padding(.top, ImarelloTheme.Space.xxs)
        }
        .padding(ImarelloTheme.Space.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityIdentifier("edit.empty")
    }
}

private struct EditSourceUnavailableState: View {
    var body: some View {
        VStack(spacing: ImarelloTheme.Space.sm) {
            Image(systemName: "photo.badge.exclamationmark")
                .font(.largeTitle)
                .foregroundStyle(ImarelloTheme.cream)
            Text("Source unavailable")
                .font(.title2.weight(.semibold))
                .foregroundStyle(ImarelloTheme.cream)
            Text("Choose another image below.")
                .font(.body)
                .foregroundStyle(.secondary)
        }
        .padding(ImarelloTheme.Space.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("edit.source-unavailable")
    }
}
