import SwiftUI

/// The plate: size, seed, and the fixed edit strength. No hidden physics —
/// everything the pipeline will use is on this card.
struct PlateSheet: View {
    @Environment(StudioModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @FocusState private var seedFocused: Bool

    var body: some View {
        @Bindable var model = model
        NavigationStack {
            Form {
                Section {
                    Picker("Canvas", selection: $model.side) {
                        Text("512").tag(512)
                        Text("1024").tag(1024)
                    }
                    .pickerStyle(.segmented)
                } header: {
                    Text("Canvas")
                } footer: {
                    Text(model.side == 512 ? "About 12 seconds on device." : "A few minutes on device.")
                }

                Section("Seed") {
                    HStack {
                        TextField("Seed", text: $model.seedText)
                            .keyboardType(.numberPad)
                            .focused($seedFocused)
                            .monospacedDigit()
                        Button {
                            model.randomizeSeed()
                        } label: {
                            Image(systemName: "dice")
                        }
                        .accessibilityLabel("Random seed")
                    }
                }

                Section {
                    LabeledContent("Strength", value: "0.8")
                } header: {
                    Text("Edit")
                } footer: {
                    Text("Edits develop from a print at a fixed strength of 0.8.")
                }
            }
            .scrollContentBackground(.hidden)
            .background(ImarelloTheme.canvas)
            .navigationTitle("The Plate")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        model.commitSeedText()
                        dismiss()
                    }
                }
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") {
                        model.commitSeedText()
                        seedFocused = false
                    }
                }
            }
        }
        .onDisappear {
            model.commitSeedText()
        }
    }
}
