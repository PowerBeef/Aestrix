import SwiftUI

struct WeightStatusView: View {
    let message: String
    let modelsPath: String

    var body: some View {
        VStack(alignment: .leading, spacing: ImarelloTheme.Space.xxs) {
            Text(message)
                .font(.subheadline)
                .foregroundStyle(ImarelloTheme.cream)
                .fixedSize(horizontal: false, vertical: true)
            Text(displayPath)
                .font(.caption.monospaced())
                .foregroundStyle(ImarelloTheme.cream.opacity(0.62))
                .textSelection(.enabled)
                .lineLimit(2)
        }
        .padding(ImarelloTheme.Space.sm)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(.regular, in: RoundedRectangle(cornerRadius: ImarelloTheme.Radius.control, style: .continuous))
        .accessibilityElement(children: .combine)
    }

    private var displayPath: String {
        let url = URL(fileURLWithPath: modelsPath)
        let parts = url.pathComponents.suffix(3)
        return parts.joined(separator: "/")
    }
}
