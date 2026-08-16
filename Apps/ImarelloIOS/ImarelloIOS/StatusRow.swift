import SwiftUI

/// The one instrument grammar for every state: gate, caption, progress, edit
/// staging, and errors all speak through this glass row — cells flip, nothing
/// improvises its own chrome.
struct StatusRow: View {
    @Environment(StudioModel.self) private var model
    @Environment(GenerationEngine.self) private var engine

    private enum Cell: Equatable {
        case running(label: String)
        case error(String)
        case saved(String)
        case editStaged(PrintRecord)
        case gate(GenerationEngine.RunGate)
        case caption(PrintRecord)
        case idle
    }

    private var cell: Cell {
        if model.isRunning { return .running(label: model.phaseLabel ?? "Developing") }
        if let message = model.errorMessage { return .error(message) }
        if let message = model.saveMessage { return .saved(message) }
        if let staged = model.pendingEdit { return .editStaged(staged) }
        if engine.gate != .ready { return .gate(engine.gate) }
        if let record = model.currentPrint { return .caption(record) }
        return .idle
    }

    private var stateTag: String {
        switch cell {
        case .running: return model.harnessJobID == nil ? "Developing" : "Mac run"
        case .error: return "Stopped"
        case .saved: return "Saved"
        case .editStaged: return "Edit · 0.8"
        case .gate(.missingWeights): return "No plates"
        case .gate(.simulator): return "Preview"
        case .gate(.ready): return "Ready"
        case .caption: return "Print"
        case .idle: return "Ready"
        }
    }

    var body: some View {
        HStack(spacing: ImarelloTheme.Space.sm) {
            Text(stateTag)
                .instrumentLabel()
                .lineLimit(1)
                .fixedSize()

            reading
                .lineLimit(2)
                .frame(maxWidth: .infinity, alignment: .leading)

            trailingAction
        }
        .padding(.horizontal, ImarelloTheme.Space.md)
        .padding(.vertical, ImarelloTheme.Space.sm)
        .glassEffect(.regular, in: RoundedRectangle(cornerRadius: ImarelloTheme.Radius.control, style: .continuous))
        .animation(.snappy, value: cell)
        .task(id: model.saveMessage) {
            guard model.saveMessage != nil else { return }
            try? await Task.sleep(for: .seconds(2.5))
            model.saveMessage = nil
        }
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder
    private var reading: some View {
        switch cell {
        case .running:
            HStack(spacing: ImarelloTheme.Space.xs) {
                Text(model.phaseLabel ?? "Developing")
                    .instrumentReading()
                    .contentTransition(.numericText())
                if let started = model.runStartedAt {
                    ElapsedReading(since: started)
                }
            }
            .transition(.push(from: .bottom))
        case .error(let message):
            Text(message)
                .font(.footnote)
                .foregroundStyle(ImarelloTheme.cream)
                .lineLimit(2)
                .transition(.push(from: .bottom))
        case .saved(let message):
            Text(message)
                .instrumentReading()
                .transition(.push(from: .bottom))
        case .editStaged(let record):
            Text("Develops from \(record.caption)")
                .instrumentReading()
                .lineLimit(1)
                .transition(.push(from: .bottom))
        case .gate(.missingWeights):
            Text("Sync weights from the Mac")
                .instrumentReading()
                .transition(.push(from: .bottom))
        case .gate(.simulator):
            Text("Klein develops on device")
                .instrumentReading()
                .transition(.push(from: .bottom))
        case .gate(.ready), .idle:
            Text("Set a plate and develop")
                .instrumentReading()
                .transition(.push(from: .bottom))
        case .caption(let record):
            Text(record.caption + (record.mode == "i2i" ? " · edit" : ""))
                .instrumentReading()
                .lineLimit(1)
                .transition(.push(from: .bottom))
        }
    }

    @ViewBuilder
    private var trailingAction: some View {
        switch cell {
        case .running:
            if model.harnessJobID == nil {
                Button {
                    model.cancel()
                } label: {
                    Image(systemName: "xmark")
                        .font(.footnote.weight(.semibold))
                }
                .accessibilityLabel("Cancel the run")
            } else {
                ProgressView()
                    .controlSize(.small)
            }
        case .error:
            HStack(spacing: ImarelloTheme.Space.sm) {
                if model.lastAction != nil {
                    Button("Try Again") {
                        model.errorMessage = nil
                        model.retryLast()
                    }
                    .font(.footnote.weight(.semibold))
                }
                Button {
                    model.errorMessage = nil
                } label: {
                    Image(systemName: "xmark")
                        .font(.footnote.weight(.semibold))
                }
                .accessibilityLabel("Dismiss error")
            }
        case .editStaged:
            Button {
                model.pendingEdit = nil
            } label: {
                Image(systemName: "xmark")
                    .font(.footnote.weight(.semibold))
            }
            .accessibilityLabel("Cancel edit")
        case .saved, .gate, .caption, .idle:
            EmptyView()
        }
    }
}

/// Ticking seconds since the run started, tabular so the row never jitters.
private struct ElapsedReading: View {
    let since: Date

    var body: some View {
        TimelineView(.periodic(from: since, by: 1)) { context in
            let seconds = max(0, Int(context.date.timeIntervalSince(since)))
            Text("\(seconds) s")
                .font(.subheadline)
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .contentTransition(.numericText())
        }
    }
}
