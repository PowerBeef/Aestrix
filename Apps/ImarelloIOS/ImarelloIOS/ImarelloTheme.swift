import SwiftUI

enum ImarelloTheme {
    static let canvas = Color("StudioBackground")
    static let stage = Color("StageGround")
    static let cream = Color("Cream")
    static let copper = Color.accentColor

    /// 4-pt scale. Tight inside a group, generous between groups.
    enum Space {
        static let xxs: CGFloat = 4
        static let xs: CGFloat = 8
        static let sm: CGFloat = 12
        static let md: CGFloat = 16
        static let lg: CGFloat = 20
        static let xl: CGFloat = 24
    }

    enum Radius {
        static let control: CGFloat = 16
        static let stage: CGFloat = 20
    }

    enum Size {
        static let mark: CGFloat = 32
        static let seedDigits: CGFloat = 64
        static let dock: CGFloat = 52
    }
}

struct GlassField<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(.horizontal, ImarelloTheme.Space.sm)
            .padding(.vertical, ImarelloTheme.Space.sm)
            .glassEffect(.regular, in: RoundedRectangle(cornerRadius: ImarelloTheme.Radius.control, style: .continuous))
    }
}
