# Product

<!-- impeccable:product-schema 1 -->

## Platform

ios

## Users

The developer-owner (Patrice) as the primary user, plus people they demo to in person. The owner knows what seeds, canvas sizes, and Klein are; the design may assume that context. No anonymous installs are planned — this is a personal studio and a live showpiece for the Imarello engine.

## Product Purpose

`Apps/ImarelloIOS` (product name **Imarello**) runs the full FLUX.2 klein 4B pipeline on-device (iPhone 17 Pro class) and lets the owner generate and iterate on images anywhere. Success is the **fast iterate loop**: prompt → print → tweak → print again with minimum friction — the measured ~11.6 s 512² generate *is* the product; every surface serves iteration speed.

## Positioning

A from-scratch native Swift + MLX runtime (not a wrapper or a server round-trip): the whole diffusion stack — Qwen3 text encoder, MMDiT, VAE — runs staged on the phone's own silicon, offline, in ~12 s at 512². Neighboring apps cannot truthfully claim the same mechanism.

## Operating Context

- Generation runs on a physical iPhone; the **Simulator is UI-only** (MLX has no Simulator Metal — Generate is a chrome no-op, never faked).
- Weights (~5 GB) live in the app container (`Caches/Imarello/models/`), synced from a Mac; never bundled. A fresh install may need a resync; the app's gate states must tell that story honestly.
- A Mac-driven **harness** drops job JSONs into `Caches/Imarello/jobs/inbox/`; the app polls every 2 s while active and runs them through the same pipeline. This contract (paths, result schema, PNG locations under `Caches/Imarello/outputs/`) is frozen — repo scripts depend on it.
- Generated PNGs persist in `Caches/Imarello/outputs/`.
- Measured on-device: 512² ≈ 11.6 s; 1024² takes minutes and its anatomy is vision-gated (Klein 4-step limitation).

## Capabilities and Constraints

- v1 capabilities: text-to-image (512² / 1024², 4 steps, guidance 1.0, seeded) and in-app image-to-image at **fixed strength 0.8** (value must be visible in the UI — no hidden physics). Editing may start from any in-app print (confirmed): same pipeline, **no photo import** of outside images.
- Persistent generation **history** with edit-from-any-print is in scope (confirmed).
- iPhone-first; iPad (`TARGETED_DEVICE_FAMILY "1,2"`) must not break but gets no bespoke layout. **No Mac Catalyst.**
- Seed entry is a numeric string with a number-pad Done bar; the seed commits before generate; the caption for a finished print is exactly `{side} · seed {n}` for what actually ran.
- Kernel entitlements (`increased-memory-limit`, `extended-virtual-addressing`) are pinned via `project.yml` `SystemCapabilities`; the Xcode project regenerates via `Scripts/generate-ios-project.sh` — file-set changes require a regen.
- Engine truth lives in the repo (`CLAUDE.md`, `Docs/IOS.md`); the app links `ImarelloRuntime` from the local package and adds no model code.

## Brand Commitments

- Name **Imarello**; the official mark is the 3D cream/gold iris (`Mark` asset, `Docs/assets/readme/imarello-mark.*`).
- Palette commitment: espresso ground, iris-gold/copper accent, cream ink (existing colorsets `StudioBackground`, `StageGround`, `AccentColor`, `Cream`). Dark-first.
- **Liquid Glass Regular** (never Clear); prominent/tinted glass on the primary action only.
- Voice (confirmed): the **darkroom/atelier register** — generations are **prints**, the canvas is the **stage**, capture settings are the **plate**. Used consistently in labels and VoiceOver, without becoming cute or obscuring function.

## Evidence on Hand

- Real on-device output PNGs in the app container and repo (`Docs/assets/readme/*.jpg` samples; eval artifacts). No fabricated screenshots or claims.
- Measured performance numbers in `Docs/PERF.md` (device: 512² 11.6 s, 2026-08-16).
- Prior UI critique history: `.impeccable/critique/2026-08-15…studioview….md` (15/40 against the original form-first build).

## Product Principles

1. **The print is the product** — the current image (or its honest absence) owns the screen; controls are furniture.
2. **Iteration speed over ceremony** — the loop prompt → print → tweak must never gain a step that isn't earning its keep.
3. **Honest gates** — never fake generation (Simulator), never hide why a control is disabled, never invent progress.
4. **No hidden physics** — strength 0.8, sizes, seeds, and which print actually ran are always visible truths.
5. **One brand, precisely** — espresso/iris-gold/cream and the darkroom register, executed in details rather than decoration.
