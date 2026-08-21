import SwiftUI

struct AtmosphericImageStage: View {
    let image: UIImage

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                AtmosphericImageBackdrop(image: image)

                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(
                        width: min(proxy.size.width, 640),
                        height: proxy.size.height
                    )
            }
        }
    }
}

struct AtmosphericImageBackdrop: View {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    let image: UIImage
    var dimmingOpacity = 0.55

    var body: some View {
        GeometryReader { proxy in
            Image(uiImage: image)
                .resizable()
                .scaledToFill()
                .frame(width: proxy.size.width, height: proxy.size.height)
                .clipped()
                .blur(radius: 48)
                .overlay(
                    ImarelloTheme.stage.opacity(
                        reduceTransparency ? max(dimmingOpacity, 0.82) : dimmingOpacity
                    )
                )
        }
        .accessibilityHidden(true)
    }
}

struct TonalStageScrims: View {
    var body: some View {
        VStack(spacing: 0) {
            LinearGradient(
                colors: [.black.opacity(0.7), .clear],
                startPoint: .top,
                endPoint: .bottom
            )
            .frame(height: 150)

            Spacer()

            LinearGradient(
                colors: [.clear, .black.opacity(0.72)],
                startPoint: .top,
                endPoint: .bottom
            )
            .frame(height: 240)
        }
        .ignoresSafeArea()
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

struct ImarelloToolbarMark: View {
    var body: some View {
        Image("Mark")
            .resizable()
            .scaledToFit()
            .frame(width: 28, height: 28)
            .accessibilityHidden(true)
    }
}
