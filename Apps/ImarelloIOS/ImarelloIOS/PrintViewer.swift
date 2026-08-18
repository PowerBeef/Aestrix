import SwiftUI

/// The enlarger: one print at full size. Swipe between prints, pinch to zoom,
/// act on the print (edit from it, share, save, delete).
struct PrintViewer: View {
    @Environment(StudioModel.self) private var model
    @Environment(PrintStore.self) private var store

    let initial: PrintRecord
    let onEdit: (PrintRecord) -> Void
    let onClose: () -> Void

    @State private var selection: String
    @State private var confirmDelete = false

    init(initial: PrintRecord, onEdit: @escaping (PrintRecord) -> Void, onClose: @escaping () -> Void) {
        self.initial = initial
        self.onEdit = onEdit
        self.onClose = onClose
        _selection = State(initialValue: initial.id)
    }

    private var current: PrintRecord? {
        store.prints.first { $0.id == selection } ?? store.prints.first
    }

    var body: some View {
        ZStack {
            ImarelloTheme.enlarger.ignoresSafeArea()

            TabView(selection: $selection) {
                ForEach(store.prints) { record in
                    PrintPage(record: record)
                        .tag(record.id)
                }
            }
            .tabViewStyle(.page(indexDisplayMode: .never))
            .ignoresSafeArea()

            VStack {
                HStack {
                    Button {
                        onClose()
                    } label: {
                        Image(systemName: "xmark")
                            .font(.body.weight(.semibold))
                            .frame(width: 40, height: 40)
                    }
                    .buttonStyle(.glass)
                    .accessibilityLabel("Close the viewer")
                    Spacer()
                }
                Spacer()
                if let record = current {
                    caption(record)
                    actions(record)
                }
            }
            .padding(.horizontal, ImarelloTheme.Space.md)
            .padding(.vertical, ImarelloTheme.Space.xs)
        }
        .preferredColorScheme(.dark)
        .confirmationDialog(
            "Delete this print?", isPresented: $confirmDelete, titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                guard let record = current else { return }
                let next = neighbor(of: record)
                model.delete(record)
                if let next {
                    selection = next.id
                } else {
                    onClose()
                }
            }
        } message: {
            Text("The print is removed from the sheet and the studio.")
        }
    }

    private func neighbor(of record: PrintRecord) -> PrintRecord? {
        guard let index = store.prints.firstIndex(of: record) else { return nil }
        let remaining = store.prints.enumerated().filter { $0.offset != index }.map(\.element)
        guard !remaining.isEmpty else { return nil }
        return remaining[min(index, remaining.count - 1)]
    }

    private func caption(_ record: PrintRecord) -> some View {
        VStack(alignment: .leading, spacing: ImarelloTheme.Space.xxs) {
            HStack(spacing: ImarelloTheme.Space.xs) {
                Text(record.caption)
                    .instrumentReading()
                if record.mode == "i2i" {
                    Text("Edit")
                        .instrumentLabel()
                }
                Spacer()
                if let error = model.errorMessage {
                    // Errors must be visible here too: a denied Photos save
                    // would otherwise give no feedback at all in the viewer.
                    Text(error)
                        .font(.footnote)
                        .foregroundStyle(ImarelloTheme.cream)
                        .lineLimit(2)
                        .onTapGesture { model.errorMessage = nil }
                        .accessibilityHint("Tap to dismiss")
                } else if model.saveMessage != nil {
                    Text(model.saveMessage ?? "")
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
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(ImarelloTheme.Space.md)
        .glassEffect(.regular, in: RoundedRectangle(cornerRadius: ImarelloTheme.Radius.control, style: .continuous))
        .accessibilityElement(children: .combine)
    }

    private func actions(_ record: PrintRecord) -> some View {
        HStack(spacing: ImarelloTheme.Space.sm) {
            Button {
                onEdit(record)
            } label: {
                Label("Edit", systemImage: "wand.and.sparkles")
                    .frame(maxWidth: .infinity)
            }
            .accessibilityHint("Stages this print for an edit at strength 0.8")

            ShareLink(item: store.url(for: record)) {
                Label("Share", systemImage: "square.and.arrow.up")
                    .frame(maxWidth: .infinity)
            }

            Button {
                Task { await model.saveToPhotos(record) }
            } label: {
                Label("Save", systemImage: "square.and.arrow.down")
                    .frame(maxWidth: .infinity)
            }
            .accessibilityHint("Saves the print to Photos")

            Button(role: .destructive) {
                confirmDelete = true
            } label: {
                Label("Delete", systemImage: "trash")
                    .frame(maxWidth: .infinity)
            }
        }
        .labelStyle(StackedActionLabel())
        .padding(.vertical, ImarelloTheme.Space.sm)
        .padding(.horizontal, ImarelloTheme.Space.xs)
        .glassEffect(.regular, in: RoundedRectangle(cornerRadius: ImarelloTheme.Radius.control, style: .continuous))
    }
}

/// One page of the enlarger. The paged TabView is not lazy — its ForEach body
/// runs for every print — so the full-resolution decode happens here, only
/// while the page is (near) on screen, and is dropped when it leaves.
private struct PrintPage: View {
    @Environment(PrintStore.self) private var store
    let record: PrintRecord

    @State private var image: UIImage?

    var body: some View {
        ZoomablePrint(image: image)
            .task(id: record.id) {
                let path = store.url(for: record).path
                image = await Task.detached(priority: .userInitiated) {
                    UIImage(contentsOfFile: path)
                }.value
            }
            .onDisappear { image = nil }
    }
}

/// Viewer actions read as icon over word — no symbol has to carry a verb alone.
private struct StackedActionLabel: LabelStyle {
    func makeBody(configuration: Configuration) -> some View {
        VStack(spacing: ImarelloTheme.Space.xxs) {
            configuration.icon
                .font(.body.weight(.medium))
            configuration.title
                .font(.caption2.weight(.medium))
        }
    }
}

/// Pinch-to-zoom + pan for one print; double-tap toggles 1× / 2.5×.
private struct ZoomablePrint: View {
    let image: UIImage?

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
                    Image(systemName: "photo")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
            .scaleEffect(scale)
            .offset(offset)
            .gesture(zoomGesture)
            .simultaneousGesture(scale > 1.01 ? panGesture : nil)
            .onTapGesture(count: 2) {
                withAnimation(.snappy) {
                    if scale > 1.01 {
                        scale = 1
                        steadyScale = 1
                        offset = .zero
                        steadyOffset = .zero
                    } else {
                        scale = 2.5
                        steadyScale = 2.5
                    }
                }
            }
            .accessibilityLabel("Print")
            .accessibilityZoomAction { action in
                withAnimation(.snappy) {
                    switch action.direction {
                    case .zoomIn: scale = min(4, scale * 1.5)
                    case .zoomOut: scale = max(1, scale / 1.5)
                    }
                    steadyScale = scale
                    if scale <= 1.01 {
                        offset = .zero
                        steadyOffset = .zero
                    }
                }
            }
        }
    }

    private var zoomGesture: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                scale = min(4, max(1, steadyScale * value.magnification))
            }
            .onEnded { _ in
                steadyScale = scale
                if scale <= 1.01 {
                    withAnimation(.snappy) {
                        offset = .zero
                        steadyOffset = .zero
                    }
                }
            }
    }

    private var panGesture: some Gesture {
        DragGesture()
            .onChanged { value in
                offset = CGSize(
                    width: steadyOffset.width + value.translation.width,
                    height: steadyOffset.height + value.translation.height
                )
            }
            .onEnded { _ in
                steadyOffset = offset
            }
    }
}
