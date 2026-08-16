import SwiftUI

/// Page one of the spread: the Stage. The print owns the screen; every
/// instrument floats on glass above it.
struct StudioPage: View {
    @Environment(StudioModel.self) private var model
    @Environment(PrintStore.self) private var store
    @Environment(GenerationEngine.self) private var engine
    @Environment(\.dynamicTypeSize) private var typeSize

    let openSheet: () -> Void
    let openViewer: (PrintRecord) -> Void

    @State private var showPlate = false

    var body: some View {
        ZStack {
            ImarelloTheme.stage.ignoresSafeArea()

            stageContent

            VStack(spacing: ImarelloTheme.Space.sm) {
                header
                Spacer()
                StatusRow()
                PromptBar()
            }
            .padding(.horizontal, ImarelloTheme.Space.md)
            .padding(.vertical, ImarelloTheme.Space.xs)
            // The print is the content; this is the instrument panel around it.
            // Past accessibility2 the controls stop fitting the glass at all.
            .dynamicTypeSize(...DynamicTypeSize.accessibility2)
        }
        .sheet(isPresented: $showPlate) {
            PlateSheet()
                .presentationDetents([.medium])
                .presentationDragIndicator(.visible)
        }
    }

    // MARK: - stage

    @ViewBuilder
    private var stageContent: some View {
        if let record = model.currentPrint, let image = store.thumbnail(for: record, maxPixel: 512) {
            GeometryReader { proxy in
                ZStack {
                    // Overscan of the print itself so the Stage owns the whole
                    // screen while the print stays uncropped above it.
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(width: proxy.size.width, height: proxy.size.height)
                        .clipped()
                        .blur(radius: 48)
                        .overlay(ImarelloTheme.stage.opacity(0.55))
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(width: proxy.size.width, height: proxy.size.height)
                }
                .overlay {
                    if model.isRunning {
                        Color.black.opacity(0.35)
                    }
                }
                .animation(.easeInOut(duration: 0.3), value: model.isRunning)
            }
            .ignoresSafeArea()
            .onTapGesture {
                if !model.isRunning { openViewer(record) }
            }
            .accessibilityElement()
            .accessibilityLabel("Current print, \(record.caption)")
            .accessibilityHint("Opens the print viewer")
            .accessibilityAddTraits(.isButton)
        } else {
            emptyStage
        }
    }

    private var emptyStage: some View {
        VStack(spacing: ImarelloTheme.Space.sm) {
            Image("Mark")
                .resizable()
                .scaledToFit()
                .frame(width: 72, height: 72)
                .opacity(0.9)
            switch engine.gate {
            case .ready:
                Text("The stage is dark")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(ImarelloTheme.cream)
                Text("Set a plate and develop your first print.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            case .missingWeights:
                Text("No plates in the studio")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(ImarelloTheme.cream)
                Text("Sync the Klein weights from the Mac, then return here.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            case .simulator:
                Text("Simulator preview")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(ImarelloTheme.cream)
                Text("The chrome only — Klein develops on device.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(ImarelloTheme.Space.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - chrome

    private var header: some View {
        HStack {
            HStack(spacing: ImarelloTheme.Space.xs) {
                Image("Mark")
                    .resizable()
                    .scaledToFit()
                    .frame(width: ImarelloTheme.Size.mark, height: ImarelloTheme.Size.mark)
                // At accessibility sizes the wordmark would only ever show a
                // fragment; the Mark carries the brand on its own.
                if !typeSize.isAccessibilitySize {
                    Text("Imarello")
                        .font(.headline)
                        .foregroundStyle(ImarelloTheme.cream)
                        .lineLimit(1)
                }
            }
            .accessibilityHidden(true)

            Spacer()

            Button {
                showPlate = true
            } label: {
                Text(model.plateSummary)
                    .font(.footnote.weight(.medium))
                    .monospacedDigit()
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                    .padding(.horizontal, ImarelloTheme.Space.sm)
                    .frame(height: ImarelloTheme.Size.headerControl)
            }
            .buttonStyle(.glass)
            // The chip is functional; the wordmark beside it is decoration and
            // gives up its width first at large text sizes.
            .layoutPriority(1)
            .accessibilityLabel("Plate: \(model.side) by \(model.side), seed \(model.seed)")
            .accessibilityHint("Opens size and seed controls")

            Button(action: openSheet) {
                Image(systemName: "square.grid.2x2")
                    .font(.footnote.weight(.semibold))
                    .frame(width: ImarelloTheme.Size.headerControl,
                           height: ImarelloTheme.Size.headerControl)
            }
            .buttonStyle(.glass)
            .accessibilityLabel("Contact sheet")
            .accessibilityHint("Shows all prints")
        }
    }
}
