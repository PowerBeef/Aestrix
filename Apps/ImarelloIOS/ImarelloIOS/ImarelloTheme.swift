import SwiftUI

enum ImarelloTheme {
    static let canvas = Color("StudioBackground")
    static let stage = Color("StageGround")
    static let cream = Color("Cream")
    static let copper = Color.accentColor
    /// The enlarger room: the darkroom taken to full black for the viewer.
    static let enlarger = Color(red: 0.024, green: 0.027, blue: 0.031)

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
    }

    enum Size {
        static let mark: CGFloat = 32
        static let shutter: CGFloat = 50
        static let headerControl: CGFloat = 38
    }
}

extension View {
    /// Uppercase instrument label — the darkroom's engraved lettering.
    func instrumentLabel() -> some View {
        font(.caption2.weight(.semibold))
            .textCase(.uppercase)
            .kerning(1.1)
            .foregroundStyle(.secondary)
    }

    /// One reading on the status row — cream ink, tabular digits.
    func instrumentReading() -> some View {
        font(.subheadline.weight(.medium))
            .monospacedDigit()
            .foregroundStyle(ImarelloTheme.cream)
    }
}
