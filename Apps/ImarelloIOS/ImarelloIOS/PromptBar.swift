import SwiftUI

/// The floating prompt bar: the field and the gold shutter. The shutter is the
/// only prominent glass in the app.
struct PromptBar: View {
    @Environment(StudioModel.self) private var model
    @FocusState private var promptFocused: Bool

    private var shutterTitle: String {
        if model.isRunning { return "Stop" }
        return model.pendingEdit == nil ? "Develop" : "Edit"
    }

    var body: some View {
        @Bindable var model = model
        HStack(spacing: ImarelloTheme.Space.xs) {
            TextField("Describe the print…", text: $model.prompt, axis: .vertical)
                .lineLimit(1 ... 3)
                .focused($promptFocused)
                .submitLabel(.done)
                .onChange(of: model.prompt) { old, new in
                    // Vertical-axis fields keep Return as a newline; treat it as Done.
                    if new.contains("\n") {
                        model.prompt = new.replacingOccurrences(of: "\n", with: " ")
                        promptFocused = false
                    }
                }
                .padding(.horizontal, ImarelloTheme.Space.md)
                .padding(.vertical, ImarelloTheme.Space.sm)
                .glassEffect(.regular, in: RoundedRectangle(cornerRadius: ImarelloTheme.Radius.control, style: .continuous))
                .accessibilityLabel("Prompt")

            Button {
                promptFocused = false
                if model.isRunning {
                    model.cancel()
                } else {
                    model.develop()
                }
            } label: {
                Text(shutterTitle)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
                    .frame(minHeight: ImarelloTheme.Size.shutter)
                    .padding(.horizontal, ImarelloTheme.Space.md)
                    .contentTransition(.identity)
            }
            .buttonStyle(.glassProminent)
            // A Mac harness job owns the pipeline; the shutter can't stop it.
            .disabled(model.harnessJobID != nil || (!model.isRunning && !model.canGenerate))
            .accessibilityLabel(
                model.isRunning
                    ? "Stop the run"
                    : model.pendingEdit == nil ? "Develop a print" : "Develop the edit"
            )
        }
        // The shutter answers physically: a thud when the run starts, a
        // success tap when a new print lands on the sheet.
        .sensoryFeedback(.impact(weight: .medium), trigger: model.isRunning) { old, new in
            !old && new
        }
        .sensoryFeedback(.success, trigger: model.store.prints.count) { old, new in
            new > old
        }
    }
}
