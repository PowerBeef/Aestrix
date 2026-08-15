import SwiftUI

struct RootView: View {
    @Environment(GenerationModel.self) private var model
    @ScaledMetric(relativeTo: .title2) private var markSide = ImarelloTheme.Size.mark

    var body: some View {
        NavigationStack {
            StudioView()
                .navigationBarTitleDisplayMode(.inline)
                .toolbarBackground(.hidden, for: .navigationBar)
                .toolbarColorScheme(.dark, for: .navigationBar)
                .toolbar {
                    ToolbarItem(placement: .topBarLeading) {
                        Image("Mark")
                            .resizable()
                            .interpolation(.high)
                            .scaledToFit()
                            .frame(width: markSide, height: markSide)
                            .accessibilityHidden(true)
                    }
                    .sharedBackgroundVisibility(.hidden)

                    ToolbarItem(placement: .principal) {
                        Text("Imarello")
                            .font(.title2.weight(.semibold))
                            .foregroundStyle(ImarelloTheme.cream)
                            .lineLimit(1)
                            .minimumScaleFactor(0.75)
                            .accessibilityAddTraits(.isHeader)
                    }
                    .sharedBackgroundVisibility(.hidden)
                }
        }
    }
}

#Preview {
    RootView()
        .environment(GenerationModel())
        .preferredColorScheme(.dark)
}
