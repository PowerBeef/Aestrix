---
name: Imarello iOS
description: A native image creator where the current image owns the screen and system chrome stays quiet.
colors:
  stage-ground: "#0D0C0B"
  studio-canvas: "#191716"
  enlarger-black: "#060708"
  iris-gold: "#F0B03A"
  cream-ink: "#F4E8D6"
rounded:
  control: "16pt continuous"
  capsule: "system capsule"
  image: "0pt"
spacing:
  xxs: "4pt"
  xs: "8pt"
  sm: "12pt"
  md: "16pt"
  xl: "24pt"
  film-gutter: "2pt"
---

# Imarello iOS design authority

## North star: image first, system native

Imarello is an image creator, not an AI dashboard. The image is the only lit, saturated object; controls recede around it. The app uses four persistent native tabs—Create, Edit, Gallery, and Settings—with independent `NavigationStack` histories, seed-only Generation Options, a system tab accessory for global generation, and pushed immersive detail.

The brand is restrained: official iris mark, near-black rooms, cream type, one iris-gold action. There are no shadows, ornamental gradients, rounded image cards, or decorative glass panels.

## Screen graph

```text
Create tab ── tap image ─┐
                         ├─> Image detail ── Edit ──> Edit root
Gallery tab ── tap cell ─┘        │
                                  └─ swipe adjacent images

Edit tab ── choose source ──> edit workspace ──> iterative result
Create/Edit toolbar ── Generation Options (seed draft → Cancel/Done)
Any tab ── generation accessory ── originating Create/Edit root
Settings tab ── readiness · storage · Photos access · app information
```

Navigation paths are session-only. Relaunch starts on Create with the newest durable image. Image detail hides the tab bar but retains the native back button and edge-swipe gesture. The system owns tab-bar width, Liquid Glass, spacing, safe areas, and selection motion; `.tabBarMinimizeBehavior(.never)` keeps all four destinations available.

## Create

- The current image is centered and uncropped with `scaledToFit`.
- A downsampled duplicate fills the stage, blurred at 48 points under a dark overlay. Top and bottom scrims protect system chrome and the composer.
- Tapping the image pushes detail; the image is inert during active generation.
- The system navigation bar owns the Create title, mark, resolution-only picker, and separate Generation Options button.
- The bottom composer shows resolution and seed truth, the 1–4-line prompt field, and the only prominent action: Create.
- While activity is global, the composer withdraws. It returns after completion acknowledgement or failure dismissal.

## Edit

- With no source, feature the newest Gallery image in the same immersive stage as Create. Tapping it selects it immediately.
- A compact bottom source area occupies the compositional role of Create's composer: it names the source task and presents the remaining newest-first images in a horizontal strip without duplicating the featured image.
- The empty state uses the official mark, direct guidance, and one native glass action back to Create; do not offer Photos or file import.
- Once selected, the source owns the same immersive stage treatment as Create. The composer has its own persistent prompt and seed draft.
- Source dimensions lock output resolution, and strength 0.8 is always visible as non-editable provenance.
- Change Source returns to the picker. A successful result becomes the next source without overwriting the Create draft.

## Gallery

- A native inline navigation root uses the same leading mark and quiet toolbar treatment as Create, with the image count as its subtitle.
- A newest-first adaptive `LazyVGrid` remains the primary content and extends beneath the native navigation and tab chrome.
- A low-resolution copy of the newest image supplies the heavily dimmed atmospheric background. The grid stays sharp and glass-free; a missing or failed backdrop falls back to the solid stage ground.
- Cells are 1:1, hard-clipped, and separated by exactly 2 points—no radius, border, card, or shadow.
- Thumbnails decode asynchronously and clear off-screen.
- Edit provenance is a compact text badge with a non-color symbol/label equivalent for accessibility.
- Gallery scrolling never minimizes the four-item native tab bar.

## Image detail

- Near-black immersive background, pushed navigation, system back gesture.
- Horizontal paging among durable images; only the visible and adjacent images decode.
- Pinch, double-tap, pan, and VoiceOver zoom clamp to 1×…4× and reset per page.
- Caption uses a subdued opaque black reading surface, not glass.
- Edit, Share, Save, and Delete are labeled icon-over-word controls. Edit is the one prominent action in detail.
- During generation, Edit and Delete are disabled; Share and Save remain available. Compact global activity stays reachable while the tab bar is hidden.
- Save/delete errors and retry remain local to the detail that initiated them.

## Global generation accessory

Use `tabViewBottomAccessory(isEnabled:content:)`.

- Expanded: owner/state, phase or human-readable step, elapsed time, and actions.
- Inline: compact progress/state plus a 44-point Cancel, Retry, or Dismiss control where applicable.
- User work can be cancelled. Harness work reads “Mac run” and has no in-app cancellation.
- Tapping the reading selects the originating Create or Edit tab and pops its stack.
- Completion while browsing reads “Image ready” until opened or dismissed.
- Failure retains an immutable request snapshot without clearing either prompt draft or the selected source.

## Generation Options

A system `Form` in an item-driven sheet owns a seed-only draft. Cancel and interactive dismissal discard. Done validates a `UInt64` seed before committing it to the originating workspace. Resolution never appears in this sheet: Create owns a separate 512²/1024² picker and Edit inherits the source size.

## Settings

- Use a native `Form` with Device and Model, Gallery, Photos, and About sections.
- Readiness is the existing fail-closed gate and never initializes MLX for presentation.
- Gallery count and asynchronous disk usage are informational. Deletion remains per image.
- Photos access reports current add-only authorization and offers Open System Settings only when denied or restricted.

## Materials, color, and shape

- Permanent dark appearance.
- Regular Liquid Glass only. Native bars and sheets adopt it automatically.
- Custom glass is allowed only for interactive control/navigation surfaces: composer field, generation control, and action buttons.
- Group nearby custom glass with `GlassEffectContainer`; never nest glass and never put glass on an image or Gallery cell.
- Gold means action, never status. Destructive actions use the system destructive role.
- Rectangular control radius is 16 points continuous; control pills use capsules; images always have square corners.

## Typography and motion

Use system semantic text styles. Instrument readings and all changing numbers use monospaced digits and numeric content transitions. Uppercase/kerned caption labels are reserved for short machine state, never headings or prose.

No Dynamic Type cap. At accessibility sizes the composer stacks, detail actions become a 2×2 grid, labels wrap, and decoration yields before controls shrink. Motion uses short system transitions and disappears under Reduce Motion.

## Accessibility completion

- VoiceOver announces Gallery/source-picker position, dimensions, seed, and edit provenance.
- Images have useful labels; atmospheric duplicates and the mark are decorative.
- Every custom action has a 44-point hit region and a text label; toolbar/tab controls rely on native system hit handling.
- State is never color-only: symbols and words accompany success, failure, edit, and progress.
- Keyboard presentation must leave the prompt and Create/Edit action reachable.
- Respect Reduce Transparency with stronger solid scrims, Increased Contrast through semantic foregrounds, Button Shapes through native buttons, and Larger Content Viewer on compact custom readouts.

## Do not reintroduce

- The former page-style swipe deck.
- A modal/full-screen-cover viewer with a custom close button.
- A global untyped error string or a view-owned infinite harness poll loop.
- Synchronous image decoding from a SwiftUI `body`.
- A Dynamic Type cap, gesture-only action, or symbol-only destructive/editor action.
- Brown/espresso grounds, rounded images, shadows, Clear glass, nested glass, or more than one prominent action per context.
