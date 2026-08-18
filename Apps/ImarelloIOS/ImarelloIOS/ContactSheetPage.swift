import SwiftUI

/// Page two of the spread: the Contact Sheet. A tight film grid of every
/// print; tap to enlarge in the viewer.
struct ContactSheetPage: View {
    @Environment(StudioModel.self) private var model
    @Environment(PrintStore.self) private var store

    let openViewer: (PrintRecord) -> Void
    let backToStage: () -> Void

    private let columns = [GridItem(.adaptive(minimum: 110), spacing: 2)]

    var body: some View {
        ZStack {
            ImarelloTheme.canvas.ignoresSafeArea()

            if store.prints.isEmpty {
                emptySheet
                    .safeAreaInset(edge: .top) { header }
            } else {
                ScrollView {
                    LazyVGrid(columns: columns, spacing: 2) {
                        ForEach(store.prints) { record in
                            cell(record)
                        }
                    }
                    .padding(.bottom, ImarelloTheme.Space.xl)
                }
                .safeAreaInset(edge: .top) { header }
            }
        }
        // The same single instrument voice as the stage — run state, gate, and
        // errors must not go invisible while the sheet is the front page.
        .safeAreaInset(edge: .bottom) {
            StatusRow()
                .padding(.horizontal, ImarelloTheme.Space.md)
                .padding(.bottom, ImarelloTheme.Space.xs)
        }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: ImarelloTheme.Space.sm) {
            Button(action: backToStage) {
                Image(systemName: "chevron.left")
                    .font(.body.weight(.semibold))
                    .frame(width: 40, height: 40)
            }
            .buttonStyle(.glass)
            .accessibilityLabel("Back to the stage")
            Text("Contact Sheet")
                .font(.title2.weight(.semibold))
                .foregroundStyle(ImarelloTheme.cream)
            Spacer()
            Text("\(store.prints.count) print\(store.prints.count == 1 ? "" : "s")")
                .instrumentLabel()
        }
        .padding(.horizontal, ImarelloTheme.Space.md)
        .padding(.top, ImarelloTheme.Space.xxs)
        .padding(.bottom, ImarelloTheme.Space.sm)
        .background(ImarelloTheme.canvas)
    }

    private func cell(_ record: PrintRecord) -> some View {
        Button {
            openViewer(record)
        } label: {
            ZStack(alignment: .bottomLeading) {
                Color(.systemGray6)
                    .aspectRatio(1, contentMode: .fit)
                    .overlay {
                        if let thumb = store.thumbnail(for: record) {
                            Image(uiImage: thumb)
                                .resizable()
                                .scaledToFill()
                        }
                    }
                    .clipped()
                if record.mode == "i2i" {
                    Text("Edit")
                        .font(.system(size: 10, weight: .semibold))
                        .textCase(.uppercase)
                        .kerning(0.8)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(.black.opacity(0.55), in: Capsule())
                        .foregroundStyle(ImarelloTheme.cream)
                        .padding(4)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Print, \(record.caption)\(record.mode == "i2i" ? ", edit" : "")")
        .accessibilityHint("Opens the print viewer")
    }

    private var emptySheet: some View {
        VStack(spacing: ImarelloTheme.Space.sm) {
            Image(systemName: "photo.stack")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text("Nothing on the sheet yet")
                .font(.title3.weight(.semibold))
                .foregroundStyle(ImarelloTheme.cream)
            Text("Prints land here as you develop them.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Button("To the stage", action: backToStage)
                .buttonStyle(.glass)
                .padding(.top, ImarelloTheme.Space.xs)
        }
        .padding(ImarelloTheme.Space.xl)
    }

}
