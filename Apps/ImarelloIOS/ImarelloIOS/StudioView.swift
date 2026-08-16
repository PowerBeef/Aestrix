import SwiftUI

private enum StudioFocus: Hashable {
    case prompt
    case seed
}

struct StudioView: View {
    @Environment(GenerationModel.self) private var model
    @FocusState private var focus: StudioFocus?

    var body: some View {
        @Bindable var model = model
        VStack(alignment: .leading, spacing: ImarelloTheme.Space.lg) {
            StudioBrief(model: model, focus: $focus)

            ResultView(onPreviewTap: dismissKeyboard)
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
            StudioDock(model: model, dismissKeyboard: dismissKeyboard)
        }
        .toolbar {
            ToolbarItemGroup(placement: .keyboard) {
                Spacer()
                Button("Done") { dismissKeyboard() }
                    .fontWeight(.semibold)
                    .foregroundStyle(ImarelloTheme.copper)
                    .accessibilityLabel("Dismiss keyboard")
            }
        }
        .onChange(of: focus) { _, newFocus in
            if newFocus != .seed {
                model.commitSeedText()
            }
        }
    }

    private func dismissKeyboard() {
        model.commitSeedText()
        focus = nil
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
                    if let started = model.runStartedAt {
                        TimelineView(.periodic(from: started, by: 1)) { context in
                            Text(elapsedLabel(from: started, to: context.date))
                                .font(.subheadline.monospacedDigit())
                                .foregroundStyle(ImarelloTheme.cream.opacity(0.55))
                                .accessibilityLabel("Elapsed \(elapsedLabel(from: started, to: context.date))")
                        }
                    }
                    Spacer(minLength: ImarelloTheme.Space.xs)
                    Button("Stop", role: .cancel) {
                        model.cancel()
                    }
                    .buttonStyle(.glass)
                    .controlSize(.large)
                }
            }

            if let error = model.errorMessage {
                HStack(alignment: .firstTextBaseline, spacing: ImarelloTheme.Space.sm) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(ImarelloTheme.copper)
                        .accessibilityHidden(true)
                    Text(error)
                        .font(.callout)
                        .foregroundStyle(ImarelloTheme.cream)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityLabel("Error: \(error)")
                    Spacer(minLength: ImarelloTheme.Space.xs)
                    if model.lastAction != nil, !model.isRunning {
                        Button("Try Again") {
                            model.retryLast()
                        }
                        .buttonStyle(.glass)
                        .accessibilityHint("Run the last generate or edit again")
                    }
                }
            }

            if let saved = model.saveMessage {
                Text(saved)
                    .font(.callout)
                    .foregroundStyle(ImarelloTheme.cream.opacity(0.72))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func elapsedLabel(from start: Date, to now: Date) -> String {
        let seconds = max(0, Int(now.timeIntervalSince(start)))
        if seconds < 60 { return "· \(seconds) s" }
        return "· \(seconds / 60) m \(seconds % 60) s"
    }
}

private struct StudioBrief: View {
    @Bindable var model: GenerationModel
    @FocusState.Binding var focus: StudioFocus?
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
                    .focused($focus, equals: .prompt)
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

            if model.side == 1024 {
                Text("1024² prints take several minutes on iPhone")
                    .font(.caption)
                    .foregroundStyle(ImarelloTheme.cream.opacity(0.6))
                    .padding(.leading, ImarelloTheme.Space.xxs)
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
        .onChange(of: model.side) { _, _ in
            model.commitSeedText()
            focus = nil
        }
    }

    private var seedCluster: some View {
        HStack(spacing: ImarelloTheme.Space.xs) {
            TextField("Seed", text: $model.seedText)
                .font(.body.monospacedDigit())
                .controlSize(.large)
                .keyboardType(.numberPad)
                .textContentType(.none)
                .multilineTextAlignment(.trailing)
                .textFieldStyle(.plain)
                .foregroundStyle(ImarelloTheme.cream)
                .tint(ImarelloTheme.copper)
                .focused($focus, equals: .seed)
                .accessibilityLabel("Seed")
                .accessibilityValue(model.seedText)
                .frame(minWidth: seedWidth)

            Button("Random") {
                model.randomizeSeed()
                focus = nil
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
    var dismissKeyboard: () -> Void

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
                dismissKeyboard()
                model.generate()
            }
            .accessibilityHint("Create a new image from the prompt")

            dockButton(
                title: editTitle,
                systemImage: "slider.horizontal.3",
                tinted: false,
                disabled: !model.canEdit
            ) {
                dismissKeyboard()
                model.editLast()
            }
            .accessibilityHint("Re-runs the last print at \(model.lastSide) square, strength 0.8. No photo import.")
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
