---
name: Imarello iOS Studio
description: A darkroom on the phone — the print owns the screen, every instrument floats on glass around it.
colors:
  stage-ground: "#0D0C0B"
  studio-canvas: "#191716"
  enlarger-black: "#060708"
  iris-gold: "#F0B03A"
  cream-ink: "#F4E8D6"
typography:
  display:
    fontFamily: "SF Pro (system), Text style .title2"
    fontSize: "22pt"
    fontWeight: 600
    letterSpacing: "normal"
  headline:
    fontFamily: "SF Pro (system), Text style .title3"
    fontSize: "20pt"
    fontWeight: 600
  title:
    fontFamily: "SF Pro (system), Text style .headline"
    fontSize: "17pt"
    fontWeight: 600
  reading:
    fontFamily: "SF Pro (system), Text style .subheadline"
    fontSize: "15pt"
    fontWeight: 500
    fontFeature: "monospacedDigit"
  body:
    fontFamily: "SF Pro (system), Text style .subheadline"
    fontSize: "15pt"
    fontWeight: 400
  caption:
    fontFamily: "SF Pro (system), Text style .footnote"
    fontSize: "13pt"
    fontWeight: 400
  label:
    fontFamily: "SF Pro (system), Text style .caption2"
    fontSize: "11pt"
    fontWeight: 600
    letterSpacing: "1.1pt"
rounded:
  control: "16px"
  capsule: "9999px"
  print: "0px"
spacing:
  xxs: "4px"
  xs: "8px"
  sm: "12px"
  md: "16px"
  xl: "24px"
  film-gutter: "2px"
components:
  button-develop:
    backgroundColor: "{colors.iris-gold}"
    textColor: "{colors.stage-ground}"
    typography: "{typography.reading}"
    rounded: "{rounded.capsule}"
    padding: "0 16px"
    height: "50px"
  button-glass:
    textColor: "{colors.iris-gold}"
    typography: "{typography.caption}"
    rounded: "{rounded.capsule}"
    height: "38px"
  input-prompt:
    textColor: "{colors.cream-ink}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "12px 16px"
  card-status:
    textColor: "{colors.cream-ink}"
    typography: "{typography.reading}"
    rounded: "{rounded.control}"
    padding: "12px 16px"
  chip-edit-badge:
    backgroundColor: "rgba(0,0,0,0.55)"
    textColor: "{colors.cream-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.capsule}"
    padding: "2px 5px"
---

# Design System: Imarello iOS Studio

## Overview

**Creative North Star: "The Darkroom Bench"**

This is a darkroom, not a form. The room is dark so that the thing being made is the only lit object in it; every control is an instrument on the bench beside the print, never a panel the print has to make room for. The two pages of the spread — the Stage and the Contact Sheet — are both full-bleed and both single-purpose, swiped like camera modes rather than navigated like a tab bar.

The character is quiet, dark, and instrumental. Chrome is near-monochrome: near-black grounds, cream ink, one gold accent, and Liquid Glass for anything that floats over an image. Warmth is reserved. The generated print supplies all of the color the screen has, and the chrome deliberately supplies none — the ground was espresso brown until 2026-08-16 and was rejected exactly because a brown room competed with the prints for warmth. The register is consistent and functional, not cute: generations are *prints*, the canvas is the *stage*, capture settings are the *plate*, running is *developing*.

Density is low on the Stage (three floating instruments over an uncropped print) and maximal on the Contact Sheet (a 2 pt film grid edge to edge). That contrast is the system: the Stage is one print treated as precious, the Sheet is every print treated as a strip of film.

**Key Characteristics:**
- Dark-first, always; the app pins `.preferredColorScheme(.dark)` and has no light mode.
- The print is the only warm, saturated thing on screen.
- Chrome floats on Liquid Glass Regular; only one element is prominent glass.
- One radius (16 pt) for every rectangular surface; capsules for every control.
- No shadows anywhere; depth comes from glass and a blurred overscan of the print itself.
- Instrument labels are uppercase, kerned, and secondary; readings are cream and tabular.

## Colors

A three-tone darkroom: two near-black grounds, one cream ink, one gold accent — and the print, which is the only chromatic surface in the app.

### Primary
- **Iris Gold** (`{colors.iris-gold}`): the app tint and the single accent. It carries the Develop pill (the one filled surface in the app), the glyph and label in every glass control, and the interactive words in the viewer's action bar. It is the color of "you can act here."

### Neutral
- **Stage Ground** (`{colors.stage-ground}`): the Stage page's ground, the darkest of the two working blacks. It sits behind the print and takes a 55% overlay on top of the print's blurred overscan so the letterbox reads as room, not as crop.
- **Studio Canvas** (`{colors.studio-canvas}`): the Contact Sheet ground and the Plate sheet's form background — one step lighter than the Stage, which is how the second page announces itself as a different room.
- **Enlarger Black** (`{colors.enlarger-black}`): the print viewer only. Near-pure black so a single enlarged print has nothing at all to compete with.
- **Cream Ink** (`{colors.cream-ink}`): all primary text — the wordmark, page titles, status readings, empty-state headings, badge text. Never pure white.

Secondary text (`.secondary`) is the system's own dim of cream ink and carries instrument labels, elapsed time, prompt echoes, and empty-state subtitles. Destructive actions use the system red role, unmodified.

### Named Rules
**The Only Warm Thing Rule.** The print supplies the warmth; the room does not. Chrome stays on the near-black / cream / gold set — no brown, no tinted surface, no colored card. If a new surface needs warmth, it needs a print behind it instead.

**The One Gold Rule.** Gold means "act." It appears on the Develop pill, on glass control labels, and on nothing that is merely information. A gold status readout, a gold divider, or a gold heading is out of system.

**The Room Ramp Rule.** Three grounds, three rooms: Stage `#0D0C0B`, Contact Sheet `#191716`, viewer `#060708`. Depth of focus goes with darkness — the more singular the content, the blacker the room.

## Typography

**Font:** SF Pro throughout, via Dynamic Type text styles. No custom or display face; the mark carries the brand, the type carries the readings.

**Character:** Instrument-panel typography. Structure comes from three moves — uppercase kerned labels, tabular readings, and a semibold page title — rather than from size contrast. The ramp is shallow (22 → 11 pt) because nothing on screen should out-shout the print.

### Hierarchy
- **Display** (semibold, `.title2` 22pt): the page title on the Contact Sheet. The only type at this size in the app.
- **Headline** (semibold, `.title3` 20pt): empty-state headlines ("The stage is dark", "Nothing on the sheet yet").
- **Title** (semibold, `.headline` 17pt): the "Imarello" wordmark beside the mark.
- **Reading** (medium, `.subheadline` 15pt, monospaced digits, cream ink): every instrument value — the status row's current reading, print captions, elapsed seconds. Tabular so a ticking row never jitters.
- **Body** (regular, `.subheadline` 15pt): empty-state subtitles and the prompt field's text.
- **Caption** (regular/medium, `.footnote` 13pt): the plate chip (`512² · seed 42`, monospaced digits), prompt echoes in the viewer caption card, error messages, inline control labels.
- **Label** (semibold, `.caption2` 11pt, uppercase, 1.1pt kerning, secondary): state tags on the status row (`PREVIEW`, `DEVELOPING`, `EDIT · 0.8`), the print count on the Contact Sheet header, the Edit marker on the viewer caption.

### Named Rules
**The Engraved Label Rule.** Uppercase + kerning + secondary is reserved for machine state and counts — the engraving on an instrument. It is never applied to a phrase, a heading, or a piece of marketing copy above a title.

**The Tabular Reading Rule.** Any number that changes while you watch it (step counts, elapsed seconds, seeds, canvas sizes) is set in monospaced digits with `.contentTransition(.numericText())`. Numbers change value, not width.

## Layout

Two full-bleed pages in a paged `TabView`, both ignoring the safe area; the content owns the screen and the chrome is inset over it. There is no tab bar, no navigation bar, and no persistent nav chrome anywhere except the modal Plate sheet.

The Stage is a single vertical stack pinned to the screen edges: header at the top, a `Spacer`, then the status row and the prompt bar at the bottom, all at 16 pt horizontal / 8 pt vertical inset with 12 pt between the instruments. The print behind it is uncropped (`scaledToFit`), so the letterbox is real space and the instruments sit in it.

The Contact Sheet is an adaptive `LazyVGrid` with a 110 pt minimum cell and a **2 pt gutter**, running edge to edge with no page margin, under a solid-ground header pinned as a top safe-area inset. Cells are square (1:1) and fill (`scaledToFill`), which is what makes it read as a strip of film rather than a photo gallery.

Spacing is a 4 pt scale: 4 (inside a label pair), 8 (between adjacent controls), 12 (between instruments and inside glass), 16 (page inset and glass horizontal padding), 24 (empty-state padding). The film gutter (2 pt) is deliberately outside the scale and belongs to the grid alone.

**Responsive / Dynamic Type.** Chrome is capped at `DynamicTypeSize.accessibility2`; past that the controls stop fitting the glass. The print is never capped. At accessibility sizes the wordmark is removed entirely and the plate chip takes layout priority — the functional control keeps its width, the decoration gives up its own.

### Named Rules
**The Content-Owns-The-Screen Rule.** Every page is full-bleed and single-purpose. Instruments float over the content; content never gets pushed into a card to make room for a control.

**The Decoration Yields First Rule.** When space runs out, decoration disappears before function shrinks. The wordmark goes at accessibility sizes; the plate chip does not.

## Elevation & Depth

No shadows. There is not a single `shadow` in the app, and there is no elevation ramp. Depth is built two ways: **Liquid Glass** (`.glassEffect(.regular, ...)` and `.buttonStyle(.glass)`) for anything that floats over an image, and **atmospheric blur** on the Stage — the current print is drawn twice, once as a `scaledToFill` overscan blurred at 48 pt under a 55% ground overlay, once uncropped on top. The room behind the print is literally made of the print.

During a run the print dims under a 35% black wash, cross-faded over 0.3 s. That wash is the only opacity-based state signal in the app.

### Named Rules
**The Glass-Only Rule.** Anything that floats over the print is Liquid Glass Regular, never a tinted or opaque card. If a surface can't be glass, it belongs on the ground instead.

**The Single Prominent Rule.** Exactly one element in the app is `.glassProminent`: the Develop pill. Prominence is how the primary action is identified — a second prominent control would erase it.

## Shapes

One radius: **16 pt continuous** on every rectangular floating surface (prompt field, status row, viewer caption card, viewer action bar). Every control that isn't a rectangle is a **capsule** — the Develop pill, the plate chip, the 38 pt glass header buttons, the 40 pt glass chevron and close buttons, the Edit badge. Print thumbnails and the print itself have **no radius at all**: square-cornered, hard-clipped, laid on a 2 pt gutter like a contact sheet. The mark is the only circular form.

Borders do not exist as a device. Separation comes from glass against ground, or from the 2 pt gutter.

### Named Rules
**The Square Print Rule.** Prints are never rounded, never inset in a card, never given a border. The chrome is soft; the image is not.

## Components

### Buttons
- **Develop pill (primary):** a capsule of prominent glass filled iris gold with dark text, minimum 50 pt tall, 16 pt horizontal padding, subheadline semibold. Its label is a **word** — "Develop" / "Edit" / "Stop" — sized to fit (`fixedSize`) and swapped with `.contentTransition(.identity)` so it never animates through a wrong word. It is disabled while a Mac harness job owns the pipeline. It answers physically: a medium impact when a run starts, a success tap when a print lands.
- **Glass controls (secondary):** `.buttonStyle(.glass)`, capsule, gold content, no fill. Header controls are 38×38; page-level chevron and close buttons are 40×40. Used for the plate chip, the contact-sheet button, the back chevron, the viewer close, and the empty-state "To the stage".
- **Viewer actions:** four equal-width buttons in one glass bar, each an **icon over its word** (`StackedActionLabel`: body-medium symbol above a caption2-medium title, 4 pt apart). Edit / Share / Save in gold, Delete in the destructive role.

### Chips
- **Plate chip:** a glass capsule reading the plate state (`512² · seed 42`) in footnote-medium with monospaced digits, single line, scaling to 70% before truncating. Tapping it opens the Plate sheet. It is a readout and a control at once.
- **Edit badge:** on Contact Sheet cells only. Uppercase 10pt semibold cream on a 55% black capsule, inset 4 pt from the cell's bottom-leading corner. It marks provenance (i2i), never decoration.

### Cards / Containers
- **Status row:** 16 pt continuous glass, 16/12 padding, one horizontal grammar for **every** state — a fixed uppercase state tag, one flexible cream reading, and an optional trailing action. Gate, caption, progress, edit staging, saved confirmations, and errors all flip through this one row with `.push(from: .bottom)` transitions on `.snappy`. Nothing improvises its own chrome.
- **Viewer caption card:** same glass and radius, holding the print's caption reading, an Edit marker, and up to two lines of the prompt in secondary footnote.

### Inputs / Fields
- **Prompt field:** glass at 16 pt radius, 16/12 padding, vertical-axis text field growing 1–3 lines. Return is coerced into Done (the newline is stripped and focus drops) rather than inserting a break. No visible stroke, no focus glow — the glass is the field.
- **Plate sheet:** a standard `Form` at a `.medium` detent with the system background hidden and the studio canvas behind it. Segmented canvas picker, monospaced numeric seed field with a dice randomizer, and strength shown as read-only `LabeledContent` at 0.8 — the fixed value is displayed, not editable.

### Navigation
There is no navigation chrome. The spread is a paged `TabView` with the index dots hidden; movement is a swipe, or the grid button (Stage → Sheet) and back chevron (Sheet → Stage) in the two headers. The affordance lives **in the header**, never as a floating edge handle over the print. The viewer is a `fullScreenCover` with its own close button and its own paged `TabView` for swiping between prints.

### Signature: the Stage
The current print, full-bleed and uncropped, with a blurred overscan of *itself* filling the letterbox. Tapping it opens the viewer; it is inert while a run is in flight. This is the app's recognizable silhouette with all content removed: a dark room, a gold pill, and one glass row.

## Do's and Don'ts

### Do:
- **Do** let the print be the only saturated, warm thing on screen; keep new chrome on near-black, cream, and gold.
- **Do** float every over-image surface on Liquid Glass Regular at 16 pt continuous radius.
- **Do** route every new state — progress, error, confirmation, staging — through the existing status row's flip-cell grammar instead of adding a banner, toast, or alert.
- **Do** label the primary action with a verb in words ("Develop", "Edit", "Stop"), sized to fit.
- **Do** set every changing number in monospaced digits with a numeric-text transition.
- **Do** cap new chrome at `DynamicTypeSize.accessibility2` and let decoration drop out before function shrinks.
- **Do** keep the darkroom register (print, stage, plate, develop) in labels and VoiceOver, and keep it functional.

### Don't:
- **Don't** add a shadow. The system has none; depth is glass and the blurred overscan of the print.
- **Don't** make a second control `.glassProminent`. There is exactly one prominent element and it is the Develop pill.
- **Don't** round, inset, or border a print. Thumbnails are square and hard-clipped on a 2 pt gutter.
- **Don't** use gold for information — no gold headings, readouts, dividers, or backgrounds behind text.
- **Don't** reintroduce a warm ground (the espresso brown was tried and rejected on 2026-08-16).
- **Don't** put a navigation affordance on top of the print; page-switching lives in the header.
- **Don't** ship a light appearance or an unpinned color scheme; the app is dark-first by construction.
- **Don't** let a symbol carry a verb alone in the viewer — actions are icon over word.
