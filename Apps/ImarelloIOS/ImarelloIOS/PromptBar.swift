import SwiftUI

struct GenerationComposer: View {
    @Environment(StudioSession.self) private var session
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @FocusState private var promptFocused: Bool

    let workspace: GenerationWorkspace

    private var actionTitle: String { workspace == .create ? "Create" : "Edit" }
    private var promptPlaceholder: String {
        workspace == .create ? "Describe an image…" : "Describe the change…"
    }

    var body: some View {
        GlassEffectContainer(spacing: ImarelloTheme.Space.xs) {
            Group {
                if dynamicTypeSize.isAccessibilitySize {
                    VStack(alignment: .leading, spacing: ImarelloTheme.Space.xs) {
                        provenance
                        promptField(text: promptBinding)
                        actionButton
                    }
                } else {
                    VStack(alignment: .leading, spacing: ImarelloTheme.Space.xs) {
                        provenance
                        HStack(alignment: .bottom, spacing: ImarelloTheme.Space.xs) {
                            promptField(text: promptBinding)
                            actionButton
                        }
                    }
                }
            }
        }
        .sensoryFeedback(.impact(weight: .medium), trigger: session.isBusy) { old, new in
            !old && new
        }
    }

    private var provenance: some View {
        Label(summary, systemImage: workspace == .create ? "viewfinder" : "photo.badge.checkmark")
            .font(.caption)
            .monospacedDigit()
            .foregroundStyle(.secondary)
            .lineLimit(dynamicTypeSize.isAccessibilitySize ? 3 : 1)
            .accessibilityElement(children: .combine)
    }

    private var summary: String {
        workspace == .create ? session.createSummary : session.editSummary
    }

    private var promptBinding: Binding<String> {
        Binding(
            get: {
                workspace == .create
                    ? session.createDraft.prompt
                    : session.editDraft.prompt
            },
            set: { value in
                if workspace == .create {
                    session.createDraft.prompt = value
                } else {
                    session.editDraft.prompt = value
                }
            }
        )
    }

    private func promptField(text: Binding<String>) -> some View {
        TextField(promptPlaceholder, text: text, axis: .vertical)
            .lineLimit(1 ... 4)
            .focused($promptFocused)
            .submitLabel(.done)
            .onChange(of: text.wrappedValue) { _, newValue in
                guard newValue.contains("\n") else { return }
                text.wrappedValue = newValue.replacingOccurrences(of: "\n", with: " ")
                promptFocused = false
            }
            .padding(.horizontal, ImarelloTheme.Space.md)
            .padding(.vertical, ImarelloTheme.Space.sm)
            .frame(minHeight: 50)
            .glassEffect(
                .regular.interactive(),
                in: .rect(cornerRadius: ImarelloTheme.Radius.control)
            )
            .accessibilityLabel(workspace == .create ? "Image prompt" : "Edit prompt")
            .accessibilityIdentifier("\(workspace.rawValue).prompt")
    }

    private var actionButton: some View {
        Button(actionTitle, action: performAction)
            .font(.headline)
            .lineLimit(1)
            .frame(
                maxWidth: dynamicTypeSize.isAccessibilitySize ? .infinity : nil,
                minHeight: ImarelloTheme.Size.shutter
            )
            .padding(.horizontal, ImarelloTheme.Space.md)
            .buttonStyle(.glassProminent)
            .foregroundStyle(ImarelloTheme.stage)
            .frame(minHeight: 50)
            .padding(.vertical, 4)
            .contentShape(Rectangle())
            .disabled(workspace == .create ? !session.canCreate : !session.canEdit)
            .accessibilityLabel(workspace == .create ? "Create image" : "Apply edit")
            .accessibilityIdentifier("\(workspace.rawValue).action")
    }

    private func performAction() {
        promptFocused = false
        switch workspace {
        case .create: session.createImage()
        case .edit: session.editImage()
        }
    }
}
