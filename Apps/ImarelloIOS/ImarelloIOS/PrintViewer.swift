import SwiftUI

struct PrintDetailView: View {
    @Environment(StudioSession.self) private var session
    @Environment(PrintStore.self) private var store
    @Environment(PhotoExportService.self) private var photoExporter
    @Environment(AppNavigation.self) private var navigation
    @Environment(\.dismiss) private var dismiss
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    let initialID: String

    @State private var selection: String
    @State private var confirmDelete = false
    @State private var message: String?
    @State private var errorMessage: String?
    @State private var retry: DetailRetry?

    init(initialID: String) {
        self.initialID = initialID
        _selection = State(initialValue: initialID)
    }

    private var current: PrintRecord? {
        store.prints.first { $0.id == selection } ?? store.prints.first
    }

    var body: some View {
        ZStack {
            ImarelloTheme.enlarger.ignoresSafeArea()
            pages
        }
        .safeAreaInset(edge: .bottom, spacing: ImarelloTheme.Space.xs) {
            VStack(spacing: ImarelloTheme.Space.xs) {
                if session.showsGlobalActivity {
                    CompactGenerationActivity(onOpen: returnToActivity)
                        .frame(maxWidth: .infinity)
                        .padding(.horizontal, ImarelloTheme.Space.xs)
                        .glassEffect(.regular, in: .capsule)
                }
                if let record = current {
                    PrintCaption(record: record, message: message, error: errorMessage)
                    if retry != nil {
                        Button("Try Again", action: retryDetailOperation)
                            .frame(minHeight: 44)
                            .buttonStyle(.glass)
                            .accessibilityIdentifier("print.retry")
                    }
                    PrintActionDock(
                        record: record,
                        isBusy: session.isBusy,
                        isSaving: photoExporter.isSaving,
                        accessibilityLayout: dynamicTypeSize.isAccessibilitySize,
                        edit: { edit(record) },
                        save: { save(record) },
                        delete: { confirmDelete = true }
                    )
                }
            }
            .padding(.horizontal, ImarelloTheme.Space.md)
            .padding(.bottom, ImarelloTheme.Space.xs)
        }
        .navigationTitle(current?.caption ?? "Image")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbarVisibility(.hidden, for: .tabBar)
        .preferredColorScheme(.dark)
        .confirmationDialog(
            "Delete this image?", isPresented: $confirmDelete,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) { deleteCurrent() }
        } message: {
            Text("The image is removed from Gallery and any active edit source.")
        }
        .onChange(of: store.prints.map(\.id)) { _, ids in
            guard !ids.contains(selection) else { return }
            if let first = ids.first { selection = first } else { dismiss() }
        }
    }

    private var pages: some View {
        TabView(selection: $selection) {
            ForEach(Array(store.prints.enumerated()), id: \.element.id) { index, record in
                PrintPage(
                    record: record,
                    shouldLoad: isNearSelection(index: index),
                    isSelected: record.id == selection,
                    position: index + 1,
                    count: store.prints.count
                )
                .tag(record.id)
            }
        }
        .tabViewStyle(.page(indexDisplayMode: .never))
        .ignoresSafeArea()
    }

    private func isNearSelection(index: Int) -> Bool {
        guard let selectedIndex = store.prints.firstIndex(where: { $0.id == selection }) else {
            return false
        }
        return abs(selectedIndex - index) <= 1
    }

    private func edit(_ record: PrintRecord) {
        session.selectEditSource(record)
        navigation.showEditRoot()
    }

    private func returnToActivity() {
        navigation.showRoot(for: session.activity.destination)
        session.acknowledgeCompletion()
    }

    private func save(_ record: PrintRecord) {
        message = nil
        errorMessage = nil
        Task { @MainActor in
            do {
                try await photoExporter.save(store.url(for: record))
                message = "Saved to Photos"
                retry = nil
            } catch {
                errorMessage = error.localizedDescription
                retry = .save(record)
            }
        }
    }

    private func deleteCurrent() {
        guard let record = current else { return }
        delete(record)
    }

    private func delete(_ record: PrintRecord) {
        guard store.prints.contains(record) else {
            retry = nil
            errorMessage = "That image is no longer in the Gallery."
            return
        }
        let next = neighbor(of: record)
        do {
            try session.delete(record)
            retry = nil
            errorMessage = nil
            if let next { selection = next.id } else { dismiss() }
        } catch {
            errorMessage = StudioSession.friendlyMessage(for: error)
            retry = .delete(record)
        }
    }

    private func retryDetailOperation() {
        guard let retry else { return }
        switch retry {
        case .save(let record): save(record)
        case .delete(let record): delete(record)
        }
    }

    private func neighbor(of record: PrintRecord) -> PrintRecord? {
        guard let index = store.prints.firstIndex(of: record) else { return nil }
        let remaining = store.prints.enumerated()
            .filter { $0.offset != index }
            .map(\.element)
        guard !remaining.isEmpty else { return nil }
        return remaining[min(index, remaining.count - 1)]
    }
}

private enum DetailRetry {
    case save(PrintRecord)
    case delete(PrintRecord)
}

private struct PrintCaption: View {
    let record: PrintRecord
    let message: String?
    let error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: ImarelloTheme.Space.xxs) {
            HStack {
                Text(record.caption)
                    .instrumentReading()
                if record.mode == "i2i" {
                    Label("Edit", systemImage: "wand.and.sparkles")
                        .instrumentLabel()
                }
                Spacer()
                if let message {
                    Label(message, systemImage: "checkmark.circle")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            if !record.prompt.isEmpty {
                Text(record.prompt)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            if let error {
                Label(error, systemImage: "exclamationmark.triangle")
                    .font(.footnote)
                    .foregroundStyle(.orange)
                    .lineLimit(3)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(ImarelloTheme.Space.md)
        .background(.black.opacity(0.72), in: .rect(cornerRadius: ImarelloTheme.Radius.control))
        .accessibilityElement(children: .combine)
    }
}

private struct PrintActionDock: View {
    @Environment(PrintStore.self) private var store

    let record: PrintRecord
    let isBusy: Bool
    let isSaving: Bool
    let accessibilityLayout: Bool
    let edit: () -> Void
    let save: () -> Void
    let delete: () -> Void

    var body: some View {
        GlassEffectContainer(spacing: ImarelloTheme.Space.xs) {
            if accessibilityLayout {
                Grid(horizontalSpacing: ImarelloTheme.Space.xs, verticalSpacing: ImarelloTheme.Space.xs) {
                    GridRow {
                        editButton
                        shareButton
                    }
                    GridRow {
                        saveButton
                        deleteButton
                    }
                }
            } else {
                HStack(spacing: ImarelloTheme.Space.xs) {
                    editButton
                    shareButton
                    saveButton
                    deleteButton
                }
            }
        }
    }

    private var editButton: some View {
        Button(action: edit) {
            StackedActionLabel(title: "Edit", systemImage: "wand.and.sparkles")
                .frame(maxWidth: .infinity, minHeight: 52)
        }
        .buttonStyle(.glassProminent)
        .foregroundStyle(ImarelloTheme.stage)
        .disabled(isBusy)
        .accessibilityHint("Uses this image as the edit source at strength 0.8")
        .accessibilityIdentifier("print.edit")
    }

    private var shareButton: some View {
        ShareLink(item: store.url(for: record)) {
            StackedActionLabel(title: "Share", systemImage: "square.and.arrow.up")
                .frame(maxWidth: .infinity, minHeight: 52)
        }
        .buttonStyle(.glass)
        .accessibilityIdentifier("print.share")
    }

    private var saveButton: some View {
        Button(action: save) {
            StackedActionLabel(
                title: isSaving ? "Saving" : "Save",
                systemImage: isSaving ? "hourglass" : "square.and.arrow.down"
            )
            .frame(maxWidth: .infinity, minHeight: 52)
        }
        .buttonStyle(.glass)
        .disabled(isSaving)
        .accessibilityHint("Saves this image to Photos")
        .accessibilityIdentifier("print.save")
    }

    private var deleteButton: some View {
        Button(role: .destructive, action: delete) {
            StackedActionLabel(title: "Delete", systemImage: "trash")
                .frame(maxWidth: .infinity, minHeight: 52)
        }
        .buttonStyle(.glass)
        .disabled(isBusy)
        .accessibilityIdentifier("print.delete")
    }
}

private struct StackedActionLabel: View {
    let title: String
    let systemImage: String

    var body: some View {
        VStack(spacing: ImarelloTheme.Space.xxs) {
            Image(systemName: systemImage)
                .font(.body.weight(.medium))
            Text(title)
                .font(.caption.weight(.medium))
        }
        .padding(.horizontal, ImarelloTheme.Space.xs)
    }
}

private struct PrintPage: View {
    @Environment(PrintStore.self) private var store
    @Environment(\.printImageLoader) private var imageLoader

    let record: PrintRecord
    let shouldLoad: Bool
    let isSelected: Bool
    let position: Int
    let count: Int

    @State private var image: UIImage?

    var body: some View {
        ZoomablePrint(image: image, record: record, position: position, count: count)
            .id("\(record.id)#\(isSelected)")
            .allowsHitTesting(isSelected)
            .task(id: "\(record.id)#\(shouldLoad)") {
                guard shouldLoad else {
                    image = nil
                    return
                }
                await load()
            }
            .onDisappear { image = nil }
    }

    private func load() async {
        let url = store.url(for: record)
        do {
            let loaded = try await imageLoader.image(
                at: url, recordID: record.id
            )
            guard !Task.isCancelled else { return }
            image = loaded
        } catch {
            image = nil
        }
    }
}

private struct ZoomablePrint: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    let image: UIImage?
    let record: PrintRecord
    let position: Int
    let count: Int

    @State private var scale: CGFloat = 1
    @State private var steadyScale: CGFloat = 1
    @State private var offset: CGSize = .zero
    @State private var steadyOffset: CGSize = .zero

    var body: some View {
        GeometryReader { proxy in
            Group {
                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                } else {
                    ProgressView()
                        .tint(.secondary)
                }
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
            .scaleEffect(scale)
            .offset(offset)
            .gesture(zoomGesture(in: proxy.size))
            .highPriorityGesture(scale > 1.01 ? panGesture(in: proxy.size) : nil)
            .onTapGesture(count: 2) { toggleZoom(in: proxy.size) }
        }
        .accessibilityElement()
        .accessibilityLabel("Image \(position) of \(count), \(record.caption)")
        .accessibilityValue("Zoom \(Int(scale * 100)) percent")
        .accessibilityZoomAction { action in
            let target = action.direction == .zoomIn ? scale * 1.5 : scale / 1.5
            setScale(target)
        }
    }

    private func zoomGesture(in size: CGSize) -> some Gesture {
        MagnifyGesture()
            .onChanged { value in
                scale = min(4, max(1, steadyScale * value.magnification))
                offset = clamped(offset, scale: scale, in: size)
            }
            .onEnded { _ in
                steadyScale = scale
                offset = clamped(offset, scale: scale, in: size)
                steadyOffset = offset
            }
    }

    private func panGesture(in size: CGSize) -> some Gesture {
        DragGesture()
            .onChanged { value in
                let proposed = CGSize(
                    width: steadyOffset.width + value.translation.width,
                    height: steadyOffset.height + value.translation.height
                )
                offset = clamped(proposed, scale: scale, in: size)
            }
            .onEnded { _ in
                offset = clamped(offset, scale: scale, in: size)
                steadyOffset = offset
            }
    }

    private func toggleZoom(in size: CGSize) {
        withAnimation(reduceMotion ? nil : .snappy) {
            if scale > 1.01 {
                reset()
            } else {
                scale = 2.5
                steadyScale = 2.5
                offset = clamped(offset, scale: scale, in: size)
            }
        }
    }

    private func setScale(_ target: CGFloat) {
        withAnimation(reduceMotion ? nil : .snappy) {
            scale = min(4, max(1, target))
            steadyScale = scale
            if scale <= 1.01 { reset() }
        }
    }

    private func reset() {
        scale = 1
        steadyScale = 1
        offset = .zero
        steadyOffset = .zero
    }

    private func clamped(_ proposed: CGSize, scale: CGFloat, in size: CGSize) -> CGSize {
        let maxX = max(0, size.width * (scale - 1) / 2)
        let maxY = max(0, size.height * (scale - 1) / 2)
        return CGSize(
            width: min(maxX, max(-maxX, proposed.width)),
            height: min(maxY, max(-maxY, proposed.height))
        )
    }
}
