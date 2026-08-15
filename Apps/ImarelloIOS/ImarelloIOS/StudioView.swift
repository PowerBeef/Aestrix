import SwiftUI

struct StudioView: View {
    @Environment(GenerationModel.self) private var model
    @FocusState private var promptFocused: Bool

    var body: some View {
        @Bindable var model = model
        VStack(alignment: .leading, spacing: ImarelloTheme.Space.lg) {
            if let banner = model.bannerText {
                WeightStatusView(message: banner, modelsPath: model.expectedModelsDirectory.path)
            }

            StudioBrief(model: model, promptFocused: $promptFocused)

            ResultView()
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            if showsStatus {
                statusStrip
            }
        }
        .padding(.horizontal, ImarelloTheme.Space.md)
        .padding(.top, ImarelloTheme.Space.xs)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(ImarelloTheme.canvas.ignoresSafeArea())
        .safeAreaInset(edge: .bottom, spacing: 0) {
            StudioDock(model: model, promptFocused: $promptFocused)
        }
        .scrollDismissesKeyboard(.interactively)
    }

    private var showsStatus: Bool {
        model.isRunning || model.errorMessage != nil || model.saveMessage != nil
    }

    @ViewBuilder
    private var statusStrip: some View {
        VStack(alignment: .leading, spacing: ImarelloTheme.Space.xs) {
            if model.isRunning {
                HStack(spacing: ImarelloTheme.Space.sm) {
                    ProgressView()
                        .tint(ImarelloTheme.copper)
                    Text(model.phaseLabel ?? "Working…")
                        .font(.subheadline)
                        .foregroundStyle(ImarelloTheme.cream.opacity(0.72))
                    Spacer(minLength: ImarelloTheme.Space.xs)
                    Button("Stop", role: .cancel) {
                        model.cancel()
                    }
                    .buttonStyle(.glass)
                    .controlSize(.large)
                }
            }

            if let error = model.errorMessage {
                Text(error)
                    .font(.callout)
                    .foregroundStyle(ImarelloTheme.copper)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityLabel("Error: \(error)")
            }

            if let saved = model.saveMessage {
                Text(saved)
                    .font(.callout)
                    .foregroundStyle(ImarelloTheme.cream.opacity(0.72))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

private struct StudioBrief: View {
    @Bindable var model: GenerationModel
    @FocusState.Binding var promptFocused: Bool
    @ScaledMetric(relativeTo: .body) private var seedWidth = ImarelloTheme.Size.seedDigits
    @State private var instrumentHeight: CGFloat = 0

    var body: some View {
        VStack(alignment: .leading, spacing: ImarelloTheme.Space.xs) {
            GlassField {
                TextField("Describe the image", text: $model.prompt, axis: .vertical)
                    .font(.body)
                    .lineLimit(2...3)
                    .textFieldStyle(.plain)
                    .foregroundStyle(ImarelloTheme.cream)
                    .tint(ImarelloTheme.copper)
                    .focused($promptFocused)
                    .accessibilityLabel("Prompt")
            }

            GlassEffectContainer(spacing: ImarelloTheme.Space.sm) {
                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .center, spacing: ImarelloTheme.Space.sm) {
                        sizePicker
                            .onGeometryChange(for: CGFloat.self) { proxy in
                                proxy.size.height
                            } action: { instrumentHeight = $0 }
                        seedCluster
                            .frame(height: instrumentHeight > 0 ? instrumentHeight : nil)
                    }
                    VStack(alignment: .leading, spacing: ImarelloTheme.Space.sm) {
                        sizePicker
                        seedCluster
                    }
                }
                .controlSize(.large)
            }
        }
    }

    private var sizePicker: some View {
        Picker("Size", selection: $model.side) {
            Text("512").tag(512)
            Text("1024").tag(1024)
        }
        .pickerStyle(.segmented)
        .controlSize(.large)
        .tint(ImarelloTheme.copper)
        .accessibilityLabel("Canvas size")
        .frame(maxWidth: .infinity)
    }

    private var seedCluster: some View {
        HStack(spacing: ImarelloTheme.Space.xs) {
            TextField("Seed", value: $model.seed, format: .number.grouping(.never))
                .font(.body.monospacedDigit())
                .controlSize(.large)
                .keyboardType(.numberPad)
                .multilineTextAlignment(.trailing)
                .textFieldStyle(.plain)
                .foregroundStyle(ImarelloTheme.cream)
                .tint(ImarelloTheme.copper)
                .accessibilityLabel("Seed")
                .frame(minWidth: seedWidth)

            Button("Random") {
                model.seed = UInt64.random(in: 0...9_999_999)
            }
            .font(.body.weight(.semibold))
            .controlSize(.large)
            .foregroundStyle(ImarelloTheme.copper)
            .buttonStyle(.plain)
            .accessibilityLabel("Random seed")
        }
        .padding(.horizontal, ImarelloTheme.Space.md)
        .controlSize(.large)
        .frame(maxHeight: .infinity)
        .glassEffect(.regular, in: Capsule())
        .fixedSize(horizontal: true, vertical: false)
    }
}

private struct StudioDock: View {
    @Bindable var model: GenerationModel
    @FocusState.Binding var promptFocused: Bool

    var body: some View {
        GlassEffectContainer(spacing: ImarelloTheme.Space.sm) {
            ViewThatFits(in: .horizontal) {
                dockRow(editTitle: "Edit last")
                dockRow(editTitle: "Edit")
            }
            .controlSize(.large)
        }
        .padding(.horizontal, ImarelloTheme.Space.md)
        .padding(.top, ImarelloTheme.Space.sm)
        .padding(.bottom, ImarelloTheme.Space.xs)
    }

    private func dockRow(editTitle: String) -> some View {
        HStack(spacing: ImarelloTheme.Space.sm) {
            dockButton(
                title: "Generate",
                systemImage: "sparkles",
                tinted: true,
                disabled: !model.canGenerate
            ) {
                promptFocused = false
                model.generate()
            }
            .accessibilityHint("Create a new image from the prompt")

            dockButton(
                title: editTitle,
                systemImage: "slider.horizontal.3",
                tinted: false,
                disabled: !model.canEdit
            ) {
                promptFocused = false
                model.editLast()
            }
            .accessibilityHint("Edit the last generated image. No photo import.")
            .accessibilityLabel("Edit last")
        }
    }

    private func dockButton(
        title: String,
        systemImage: String,
        tinted: Bool,
        disabled: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .font(.body.weight(.semibold))
                .labelStyle(.titleAndIcon)
                .frame(maxWidth: .infinity, minHeight: ImarelloTheme.Size.dock, maxHeight: ImarelloTheme.Size.dock)
                .foregroundStyle(tinted ? Color.white : ImarelloTheme.cream.opacity(disabled ? 0.42 : 0.92))
                .background(tinted ? ImarelloTheme.copper : ImarelloTheme.stage, in: Capsule())
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .frame(maxWidth: .infinity, minHeight: ImarelloTheme.Size.dock, maxHeight: ImarelloTheme.Size.dock)
    }
}
