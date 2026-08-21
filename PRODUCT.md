# Product

<!-- product-schema 1 -->

## Platform

iOS 26.2, iPhone-first with a functional adaptive iPad layout. No Mac Catalyst.

## Users and purpose

Imarello is Patrice's private on-device image creator and live showpiece for the native Swift + MLX FLUX.2 Klein 4B engine. The primary loop is prompt → image → edit → image again, with no server round-trip and no imported photos.

The Simulator is UI-only. Physical-device generation remains blocked before MLX until the main app embeds and selects a complete `iphoneos` no-JIT metallib. Historical pre-no-JIT timings are context, not current functionality.

## Product locks

- FLUX.2 Klein 4B, prequantized 4-bit weights, staged residency.
- 1024 × 1024 default; 512 × 512 is an explicit resolution choice.
- Four steps, guidance 1.0, seeded output, no negative prompt.
- Text-to-image and edit-from-any-Gallery-image at fixed strength 0.8.
- No photo or file import, presets, favorites, folders, or cloud inference.
- Image metadata records what ran: `{side} · seed {n}`.

Weights live under `Caches/Imarello/models/` and are never bundled. Canonical images and the versioned `PrintStore` index live in Application Support; harness pull copies remain under `Caches/Imarello/outputs/`.

## Information architecture

The app has four first-class tabs, each with independent session-only navigation:

1. **Create** — the current image, resolution picker, independent prompt draft, seed options, and Create action.
2. **Edit** — a canvas-first source picker features the newest Gallery image with a compact strip of remaining sources, followed by an independent edit workspace. Output size is locked to the source and strength is fixed at 0.8.
3. **Gallery** — the persistent newest-first image grid over the same restrained atmospheric ground as Create.
4. **Settings** — model/device readiness, Gallery storage, Photos access, and app/build information.

Create and Gallery push the same immersive image detail. Detail retains the system back gesture and provides swipe, zoom, Edit, Share, Save, and Delete. Edit selects that image as the source, switches to the Edit root, locks the source size, and exposes strength 0.8. A successful edit becomes the next source for iterative editing.

Generation is app-global. A `tabViewBottomAccessory` keeps phase, step, elapsed time, completion, failure, and user cancellation visible while browsing. User work can be cancelled; Mac harness work is identified and cannot be cancelled in-app. Activity returns to the originating Create or Edit workspace, while harness activity defaults to Create.

Generation Options is transactional: Cancel or interactive dismissal discards the draft; Done validates and commits only the seed. Create resolution remains a separate 512²/1024² control; Edit resolution is locked to the selected source.

## Operating contracts

- The frozen Mac harness polls every two seconds only while the scene is active. Its inbox/running/done paths, JSON schema, PNG paths, validation, and durable completion semantics do not change with the UI.
- The app always fails closed on missing weights or an incomplete no-JIT metallib. Simulator Create/Edit never invokes or imitates the model.
- Debug UI automation may use `--ui-test-scenario empty|library|pending-edit|running|failed`. These are static UI states backed by real repository sample images and never initialize MLX.
- `project.yml` is authoritative for the Xcode project, capabilities, entitlements, sources, and test targets.

## Brand and interaction principles

- The image is the product. Chrome is furniture around it.
- Permanent darkroom black, cream ink, iris-gold action tint, official iris mark.
- Generated imagery is the only saturated surface. Images are square-cornered with 2-point Gallery gutters.
- Create, Edit, and Gallery share one atmospheric image language: a sharp image layer over a heavily dimmed, low-resolution blurred backdrop. Gallery applies that atmosphere behind its grid, never to the thumbnails themselves.
- Use native tabs, navigation bars, sheets, toolbars, and Regular Liquid Glass controls. Do not nest glass or apply it to image/grid content.
- Exactly one prominent action in context: Create, Edit, or the Edit action in image detail.
- Use direct, plain-language navigation and action labels: Create, Edit, Gallery, Settings, Image, Source, Resolution, and Generation Options.
- Never fake progress, runtime readiness, or generation.

## Accessibility contract

No Dynamic Type cap. Composers, activity, actions, and Generation Options reflow at accessibility sizes. Interactive elements have system or explicit 44-point hit regions, meaningful VoiceOver labels and order, non-color state cues, keyboard-safe placement, and support for Button Shapes, Larger Content Viewer, Reduce Motion, Reduce Transparency, and Increased Contrast.
