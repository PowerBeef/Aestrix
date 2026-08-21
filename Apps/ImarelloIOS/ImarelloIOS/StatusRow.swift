import SwiftUI

struct GenerationAccessory: View {
    @Environment(StudioSession.self) private var session
    @Environment(AppNavigation.self) private var navigation
    @Environment(\.tabViewBottomAccessoryPlacement) private var placement

    var body: some View {
        Group {
            if placement == .inline {
                CompactGenerationActivity(onOpen: openActivity)
            } else {
                expanded
            }
        }
    }

    private var expanded: some View {
        HStack(spacing: ImarelloTheme.Space.sm) {
            Button(action: openActivity) {
                HStack(spacing: ImarelloTheme.Space.sm) {
                    activitySymbol
                    VStack(alignment: .leading, spacing: ImarelloTheme.Space.xxs) {
                        Text(title)
                            .font(.caption.weight(.semibold))
                            .textCase(.uppercase)
                            .kerning(0.8)
                            .foregroundStyle(.secondary)
                        Text(detail)
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(ImarelloTheme.cream)
                            .monospacedDigit()
                            .contentTransition(.numericText())
                            .lineLimit(2)
                    }
                    Spacer(minLength: 0)
                    if let run = session.activity.run {
                        ElapsedReading(since: run.startedAt)
                    }
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            trailingActions
        }
        .padding(.horizontal, ImarelloTheme.Space.md)
        .padding(.vertical, ImarelloTheme.Space.xs)
    }

    private var activitySymbol: some View {
        Group {
            switch session.activity {
            case .running, .stopping:
                ProgressView()
            case .completed:
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(ImarelloTheme.copper)
            case .failed:
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
            case .idle:
                EmptyView()
            }
        }
        .frame(width: 28, height: 28)
    }

    @ViewBuilder
    private var trailingActions: some View {
        switch session.activity {
        case .running(let run):
            if run.owner.isUser {
                Button("Cancel", action: session.cancel)
                    .frame(minHeight: 44)
                    .accessibilityIdentifier("generation.cancel")
            }
        case .stopping:
            ProgressView()
                .controlSize(.small)
                .accessibilityLabel("Stopping")
        case .completed:
            Button(action: session.dismissActivity) {
                Image(systemName: "xmark")
                    .frame(width: 44, height: 44)
            }
            .accessibilityLabel("Dismiss completion")
        case .failed(let failure):
            if failure.owner.isUser, failure.operation != nil {
                Button("Retry", action: session.retryFailure)
                    .frame(minHeight: 44)
                    .accessibilityIdentifier("generation.retry")
            }
            Button(action: session.dismissActivity) {
                Image(systemName: "xmark")
                    .frame(width: 44, height: 44)
            }
            .accessibilityLabel("Dismiss error")
        case .idle:
            EmptyView()
        }
    }

    private var title: String {
        switch session.activity {
        case .running(let run), .stopping(let run):
            return run.owner.isUser
                ? run.operation?.actionLabel ?? "Creating"
                : run.owner.label
        case .completed: return "Image ready"
        case .failed: return "Stopped"
        case .idle: return "Ready"
        }
    }

    private var detail: String {
        switch session.activity {
        case .running, .stopping:
            return session.phaseLabel ?? "Creating"
        case .completed(let image, _, _):
            return image.caption
        case .failed(let failure):
            return failure.message
        case .idle:
            return "Ready"
        }
    }

    private func openActivity() {
        navigation.showRoot(for: session.activity.destination)
        session.acknowledgeCompletion()
    }
}

struct CompactGenerationActivity: View {
    @Environment(StudioSession.self) private var session
    let onOpen: () -> Void

    var body: some View {
        HStack(spacing: ImarelloTheme.Space.xs) {
            Button(action: onOpen) {
                HStack(spacing: ImarelloTheme.Space.xs) {
                    compactProgress
                    Text(compactLabel)
                        .font(.caption.weight(.semibold))
                        .monospacedDigit()
                        .contentTransition(.numericText())
                        .lineLimit(1)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if case .running(let run) = session.activity, run.owner.isUser {
                Button(action: session.cancel) {
                    Image(systemName: "xmark")
                        .frame(width: 44, height: 44)
                }
                .accessibilityLabel("Cancel the run")
                .accessibilityIdentifier("generation.cancel")
            } else if case .failed(let failure) = session.activity {
                if failure.owner.isUser, failure.operation != nil {
                    Button("Retry", action: session.retryFailure)
                        .frame(minHeight: 44)
                        .accessibilityIdentifier("generation.retry")
                }
                Button(action: session.dismissActivity) {
                    Image(systemName: "xmark")
                        .frame(width: 44, height: 44)
                }
                .accessibilityLabel("Dismiss error")
            } else if case .completed = session.activity {
                Button(action: session.dismissActivity) {
                    Image(systemName: "xmark")
                        .frame(width: 44, height: 44)
                }
                .accessibilityLabel("Dismiss completion")
            }
        }
        .padding(.leading, ImarelloTheme.Space.md)
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder
    private var compactProgress: some View {
        switch session.activity {
        case .running(let run), .stopping(let run):
            if run.progress.phase == .denoising, run.progress.totalSteps > 0 {
                ProgressView(
                    value: Double(min(run.progress.totalSteps, max(1, run.progress.step + 1))),
                    total: Double(run.progress.totalSteps)
                )
                .frame(width: 44)
            } else {
                ProgressView()
                    .controlSize(.small)
            }
        case .completed:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(ImarelloTheme.copper)
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
        case .idle:
            EmptyView()
        }
    }

    private var compactLabel: String {
        switch session.activity {
        case .running, .stopping: return session.phaseLabel ?? "Creating"
        case .completed: return "Image ready"
        case .failed: return "Stopped"
        case .idle: return "Ready"
        }
    }
}

private struct ElapsedReading: View {
    let since: Date

    var body: some View {
        TimelineView(.periodic(from: since, by: 1)) { context in
            Text("\(max(0, Int(context.date.timeIntervalSince(since)))) s")
                .font(.footnote)
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .contentTransition(.numericText())
        }
    }
}
