import SwiftUI

struct CreatePage: View {
    @Environment(StudioSession.self) private var session
    @Environment(GenerationEngine.self) private var engine
    @Environment(AppNavigation.self) private var navigation
    @Environment(\.printImageLoader) private var imageLoader
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var image: UIImage?

    var body: some View {
        ZStack {
            ImarelloTheme.stage.ignoresSafeArea()
            stage
            TonalStageScrims()
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            if !session.showsGlobalActivity {
                GenerationComposer(workspace: .create)
                    .padding(.horizontal, ImarelloTheme.Space.md)
                    .padding(.bottom, ImarelloTheme.Space.xs)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
        .navigationTitle("Create")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                ImarelloToolbarMark()
            }
            ToolbarItemGroup(placement: .topBarTrailing) {
                Menu {
                    resolutionButton(512)
                    resolutionButton(1024)
                } label: {
                    HStack(spacing: ImarelloTheme.Space.xxs) {
                        Image(systemName: "viewfinder")
                        Text(verbatim: "\(session.createDraft.side)²")
                    }
                    .font(.footnote.weight(.medium))
                    .monospacedDigit()
                    .frame(minWidth: 44, minHeight: 44)
                }
                .disabled(session.isBusy)
                .accessibilityLabel("Image resolution")
                .accessibilityValue(
                    "\(session.createDraft.side) by \(session.createDraft.side) pixels"
                )
                .accessibilityHint(resolutionHint)
                .accessibilityIdentifier("create.resolution")
                .accessibilityShowsLargeContentViewer {
                    Label {
                        Text(verbatim: "\(session.createDraft.side)²")
                    } icon: {
                        Image(systemName: "viewfinder")
                    }
                }

                Button {
                    navigation.presentedSheet = .generationOptions(.create)
                } label: {
                    Image(systemName: "slider.horizontal.3")
                        .frame(width: 44, height: 44)
                }
                .disabled(session.isBusy)
                .accessibilityLabel("Generation options")
                .accessibilityHint("Change the seed")
                .accessibilityIdentifier("create.options")
            }
        }
        .task(id: session.currentPrint?.id) {
            await loadCurrentPrint()
        }
        .animation(reduceMotion ? nil : .snappy, value: session.showsGlobalActivity)
    }

    private var resolutionHint: String {
        if session.isBusy { return "Unavailable while an image is being created" }
        return "Choose 512 or 1024 pixels"
    }

    private func resolutionButton(_ side: Int) -> some View {
        Button {
            session.setCreateResolution(side)
        } label: {
            if session.createDraft.side == side {
                Label {
                    Text(verbatim: "\(side) × \(side)")
                } icon: {
                    Image(systemName: "checkmark")
                }
            } else {
                Text(verbatim: "\(side) × \(side)")
            }
        }
    }

    @ViewBuilder
    private var stage: some View {
        if let record = session.currentPrint, let image {
            Button {
                navigation.openImage(id: record.id, from: .create)
            } label: {
                AtmosphericImageStage(image: image)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .ignoresSafeArea()
            .accessibilityLabel("Current image, \(record.caption)")
            .accessibilityHint("Opens the image detail")
        } else {
            EmptyCreateState(gate: engine.gate)
        }
    }

    private func loadCurrentPrint() async {
        guard let record = session.currentPrint else {
            image = nil
            return
        }
        let url = session.store.url(for: record)
        do {
            let loaded = try await imageLoader.image(
                at: url, recordID: record.id, maxPixel: 1024
            )
            guard !Task.isCancelled else { return }
            image = loaded
        } catch is CancellationError {
            return
        } catch {
            image = nil
        }
    }
}

private struct EmptyCreateState: View {
    let gate: GenerationEngine.RunGate

    var body: some View {
        VStack(spacing: ImarelloTheme.Space.sm) {
            Image("Mark")
                .resizable()
                .scaledToFit()
                .frame(width: 72, height: 72)
                .opacity(0.9)
                .accessibilityHidden(true)
            Text(title)
                .font(.title2.weight(.semibold))
                .foregroundStyle(ImarelloTheme.cream)
            Text(message)
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(ImarelloTheme.Space.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("create.empty")
    }

    private var title: String {
        switch gate {
        case .ready: return "No images yet"
        case .missingWeights: return "Model unavailable"
        case .simulator: return "Simulator preview"
        }
    }

    private var message: String {
        switch gate {
        case .ready: return "Describe an image below, then tap Create."
        case .missingWeights: return "Sync the Klein model from the Mac, then return here."
        case .simulator: return "The interface is available here; image generation runs only on device."
        }
    }
}
