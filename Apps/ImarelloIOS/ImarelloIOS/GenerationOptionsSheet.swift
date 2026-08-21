import SwiftUI

struct GenerationOptionsDraft: Equatable {
    var seedText: String

    init(seed: UInt64) {
        seedText = String(seed)
    }

    var seed: UInt64? {
        guard !seedText.isEmpty, seedText.allSatisfy(\.isNumber) else { return nil }
        return UInt64(seedText)
    }

    var isValid: Bool { seed != nil }

    mutating func randomize() {
        seedText = String(UInt64.random(in: 0 ... 9_999_999))
    }
}

struct GenerationOptionsSheet: View {
    @Environment(\.dismiss) private var dismiss
    @FocusState private var seedFocused: Bool

    let session: StudioSession
    let workspace: GenerationWorkspace
    @State private var draft: GenerationOptionsDraft

    init(session: StudioSession, workspace: GenerationWorkspace) {
        self.session = session
        self.workspace = workspace
        let seed = workspace == .create
            ? session.createDraft.seed
            : session.editDraft.seed
        _draft = State(initialValue: GenerationOptionsDraft(seed: seed))
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Seed", text: $draft.seedText)
                        .keyboardType(.numberPad)
                        .focused($seedFocused)
                        .monospacedDigit()
                        .accessibilityIdentifier("generation-options.seed")

                    Button {
                        draft.randomize()
                    } label: {
                        Label("Randomize Seed", systemImage: "dice")
                    }
                    .frame(minHeight: 44)

                    if !draft.seedText.isEmpty, draft.seed == nil {
                        Label(
                            "Enter a whole number that fits in the seed field.",
                            systemImage: "exclamationmark.triangle"
                        )
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .accessibilityIdentifier("generation-options.seed-error")
                    }
                } header: {
                    Text("Seed")
                } footer: {
                    Text("Use the same seed to reproduce an image with the same settings.")
                }
            }
            .scrollContentBackground(.hidden)
            .background(ImarelloTheme.canvas)
            .navigationTitle("Generation Options")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel", action: dismiss.callAsFunction)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done", action: commit)
                        .disabled(!draft.isValid)
                }
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { seedFocused = false }
                }
            }
        }
        .presentationDetents([.medium])
        .presentationDragIndicator(.visible)
        .presentationBackground(ImarelloTheme.canvas)
    }

    private func commit() {
        guard let seed = draft.seed else { return }
        session.setSeed(seed, for: workspace)
        dismiss()
    }
}
