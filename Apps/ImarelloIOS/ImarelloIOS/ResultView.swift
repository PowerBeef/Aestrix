import SwiftUI

struct ResultView: View {
    @Environment(GenerationModel.self) private var model

    var body: some View {
        VStack(spacing: ImarelloTheme.Space.xs) {
            Group {
                if let image = model.lastImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: ImarelloTheme.Radius.stage, style: .continuous))
                        .accessibilityLabel("Generated image")
                } else {
                    RoundedRectangle(cornerRadius: ImarelloTheme.Radius.stage, style: .continuous)
                        .fill(ImarelloTheme.stage)
                        .overlay {
                            RoundedRectangle(cornerRadius: ImarelloTheme.Radius.stage, style: .continuous)
                                .strokeBorder(ImarelloTheme.copper.opacity(0.28), lineWidth: 1)
                        }
                        .accessibilityLabel("Empty preview")
                }
            }
            .aspectRatio(1, contentMode: .fit)
            .frame(maxWidth: .infinity)
            .shadow(color: ImarelloTheme.stage.opacity(0.55), radius: ImarelloTheme.Space.md, x: 0, y: ImarelloTheme.Space.xs)

            if let url = model.lastImageURL {
                GlassEffectContainer(spacing: ImarelloTheme.Space.sm) {
                    HStack(spacing: ImarelloTheme.Space.sm) {
                        ShareLink(item: url) {
                            Label("Share", systemImage: "square.and.arrow.up")
                                .lineLimit(1)
                                .minimumScaleFactor(0.8)
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.glass)
                        .buttonSizing(.flexible)
                        .controlSize(.large)

                        Button {
                            Task { await model.saveToPhotos() }
                        } label: {
                            Label("Save", systemImage: "square.and.arrow.down")
                                .lineLimit(1)
                                .minimumScaleFactor(0.8)
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.glass)
                        .buttonSizing(.flexible)
                        .controlSize(.large)
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }
}
