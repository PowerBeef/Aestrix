// DIRECTION CONTRACT (impeccable; seed 72f2b522, surface scope, mode operate)
// THESIS: Two full-bleed pages — the Stage and the Contact Sheet — swiped like
//   camera modes; each page owns the whole screen for its one job. Refuses the
//   category default (prompt-form above a result card with a history tab).
// OWN-WORLD: espresso ground, iris-gold accent, cream ink; Liquid Glass Regular
//   only; darkroom register (prints, stage, plate); recognizable with all
//   content removed by the gold shutter on espresso and the flip-cell status row.
// STORY: the owner opens onto their print, tweaks the plate, taps the gold
//   shutter, watches the row develop, swipes left to live among their prints.
// FIRST VIEWPORT: the print full-bleed to the safe area; plate chip top-trailing;
//   one glass status row (single flip-cell grammar for caption, gate, progress,
//   and errors) above a floating prompt bar with the circular gold Generate;
//   a slim glass Sheet handle on the trailing edge invites the swipe.
// FORM: Spread Deck — index 5 of 7 on the ordered structure list (the roll).
// FINISH: unreviewed and undocumented is unfinished; this build ends with the
//   finish review, the verdict, and DESIGN.md.

import SwiftUI

/// Root: the two-page spread. Page one is the Stage, page two the Contact Sheet.
struct StudioRootView: View {
    @Environment(StudioModel.self) private var model
    @State private var page = 0
    @State private var viewerSelection: PrintRecord?

    var body: some View {
        TabView(selection: $page) {
            StudioPage(openSheet: { withAnimation(.snappy) { page = 1 } },
                       openViewer: { viewerSelection = $0 })
                .tag(0)
            ContactSheetPage(openViewer: { viewerSelection = $0 },
                             backToStage: { withAnimation(.snappy) { page = 0 } })
                .tag(1)
        }
        .tabViewStyle(.page(indexDisplayMode: .never))
        .ignoresSafeArea()
        .background(ImarelloTheme.canvas.ignoresSafeArea())
        .fullScreenCover(item: $viewerSelection) { record in
            PrintViewer(
                initial: record,
                onEdit: { staged in
                    viewerSelection = nil
                    model.stageEdit(staged)
                    withAnimation(.snappy) { page = 0 }
                },
                onClose: { viewerSelection = nil }
            )
        }
    }
}

#Preview {
    let engine = GenerationEngine()
    let store = PrintStore()
    return StudioRootView()
        .environment(StudioModel(engine: engine, store: store))
        .environment(store)
        .environment(engine)
        .tint(ImarelloTheme.copper)
        .preferredColorScheme(.dark)
}
