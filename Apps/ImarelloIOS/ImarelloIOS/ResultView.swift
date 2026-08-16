import SwiftUI

struct ResultView: View {
    @Environment(GenerationModel.self) private var model
    var onPreviewTap: (() -> Void)?

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
                            StageEmptyState()
                        }
                        .overlay {
                            RoundedRectangle(cornerRadius: ImarelloTheme.Radius.stage, style: .continuous)
                                .strokeBorder(ImarelloTheme.copper.opacity(0.28), lineWidth: 1)
                        }
                        .accessibilityElement(children: .combine)
                }
            }
            .aspectRatio(1, contentMode: .fit)
            .frame(maxWidth: .infinity)
            .shadow(color: ImarelloTheme.stage.opacity(0.55), radius: ImarelloTheme.Space.md, x: 0, y: ImarelloTheme.Space.xs)
            .contentShape(Rectangle())
            .onTapGesture { onPreviewTap?() }

            if let seed = model.lastSeed, model.lastImage != nil {
                Text("\(model.lastSide) · seed \(seed)")
                    .font(.caption.monospacedDigit().weight(.medium))
                    .foregroundStyle(ImarelloTheme.cream.opacity(0.72))
                    .accessibilityLabel("Generated at \(model.lastSide) with seed \(seed)")
            }

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

/// The gate story lives on the stage: ready invites, missing-weights explains,
/// Simulator says what this build can and cannot do.
private struct StageEmptyState: View {
    @Environment(GenerationModel.self) private var model

    var body: some View {
        VStack(spacing: ImarelloTheme.Space.sm) {
            Image(systemName: model.emptyStateIcon)
                .font(.title)
                .foregroundStyle(ImarelloTheme.copper)
            Text(model.emptyStateTitle)
                .font(.headline)
                .foregroundStyle(ImarelloTheme.cream)
            Text(model.emptyStateDetail)
                .font(.subheadline)
                .foregroundStyle(ImarelloTheme.cream.opacity(0.72))
            if let caption = model.emptyStateCaption {
                Text(caption)
                    .font(.caption.monospaced())
                    .foregroundStyle(ImarelloTheme.cream.opacity(0.55))
                    .lineLimit(2)
            }
        }
        .multilineTextAlignment(.center)
        .padding(ImarelloTheme.Space.lg)
    }
}
