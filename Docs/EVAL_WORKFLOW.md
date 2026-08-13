# Generation evaluation workflow

This is the **mandatory quality procedure** for Aestrix image generations (agents and humans).

It has two layers:

| Layer | Tool | What it catches |
|-------|------|-----------------|
| **A. Pixel harness** | `AestrixEval` / `aestrix analyze-image` | Sharpness, clip, hue, SSIM, color-word heuristics, structured findings |
| **B. Vision review** | Multimodal agent (open PNG) | Subject match, real color, text/logo, hands, artifacts, “edit applied”, aesthetics |

Neither layer alone is enough. Pixel metrics are CI-friendly; vision covers semantics.

| Doc | Role |
|-----|------|
| **This file** | End-to-end procedure (when / how / gates) |
| [IMAGE_ANALYSIS.md](IMAGE_ANALYSIS.md) | Metrics schema & library API |
| [eval-prompts.md](eval-prompts.md) | BFL-style regression prompts |
| [AGENTS.md](../AGENTS.md) | Agent contract (must follow) |

---

## When this applies

Run the full procedure after **any** of:

- `aestrix t2i`
- `aestrix i2i`
- Manual generation used as a quality gate / regression sample
- Before claiming “generation works” or “color edit works” in a session

Skip only for pure load/math tests (`load-te`, `mem-selftest`, unit tests without PNGs).

---

## Quick path (recommended)

### T2I — generate and eval in one command

```bash
swift build && ./Scripts/ensure-metallib.sh

.build/debug/aestrix t2i "YOUR PROMPT" \
  --width 512 --height 512 --steps 4 --seed 42 \
  --output /tmp/aestrix_out.png \
  --analyze --vision-brief
```

### I2I

```bash
.build/debug/aestrix i2i "YOUR EDIT PROMPT" \
  --image /path/to/source.png \
  --strength 0.8 --steps 4 --seed 7 \
  --output /tmp/aestrix_edit.png \
  --analyze --vision-brief

# People / character consistency (Tier B identity stack)
.build/debug/aestrix i2i \
  "Same person, identical face and pose; golden hour outdoor balcony, emerald silk top" \
  --image /path/to/portrait.png \
  --strength 0.9 --identity --seed 7 \
  --output /tmp/aestrix_edit_id.png \
  --analyze --vision-brief
```

**Identity flags** (`aestrix i2i`):

| Flag | Effect |
|------|--------|
| `--identity` | Full preset: ref latents + face mask + clean-pull 0.2 + `identity` schedule |
| `--ref-latents` | DiT attends to clean reference tokens (`t=10` RoPE) |
| `--face-preserve` | Vision soft face mask for regional σ + pull |
| `--face-strength-scale F` | Face start-σ = global × F (default 0.5) |
| `--clean-pull A` | Blend clean latent into face after each step |
| `--schedule color\|identity\|linear` | Strength → start-noise curve |

**Strength bands**

| Intent | Strength | Prompt style |
|--------|----------|--------------|
| Mild grade / lighting | 0.35–0.55 | Say what stays |
| Color / wardrobe on objects | **≥ 0.8** | Hex + color words |
| People identity + scene/wardrobe | **0.85–0.9** + `--identity` | Face/pose lock **and** what changes |
| Recolor same garment | 0.85–0.9 + `--identity` | “exact same cut; only fabric color …” |
| New outfit | 0.85–0.9 + `--identity` | “replace the blouse with a different …” |

Match canvas to the reference (omit `--width`/`--height` or set native size) so 1024 sources are not downscaled to 512.

**512² pixel loop:** `Scripts/eval-regression.sh` (T2I eval-prompts 1–5 × seeds 42/0/7).

### Sidecar files (next to the PNG)

| File | Contents |
|------|----------|
| `out.eval.json` | Pixel report (Codable JSON) |
| `out.vision-brief.md` | Paths + pixel summary + vision checklist |

| Flag | Effect |
|------|--------|
| `--analyze` | Print pixel report + write sidecars + vision brief |
| `--analyze-json PATH` | Write pixel JSON to PATH |
| `--vision-brief` | Vision brief only (also implied by `--analyze`) |
| `--fail-on-pixel-gate` | Exit **2** if pixel findings include `fail` |

### Eval an existing PNG

```bash
./Scripts/eval-generation.sh /tmp/aestrix_out.png \
  --prompt "YOUR PROMPT" \
  --mode t2i

./Scripts/eval-generation.sh /tmp/aestrix_edit.png \
  --prompt "EDIT PROMPT" \
  --reference /path/to/source.png \
  --mode i2i
```

---

## Step-by-step procedure

```text
┌─────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Generate   │────▶│  A. Pixel eval   │────▶│  B. Vision review  │
│  t2i / i2i  │     │  analyze-image   │     │  open PNG + check  │
└─────────────┘     └────────┬─────────┘     └─────────┬──────────┘
                             │                         │
                             │    .eval.json           │ Assessment
                             ▼                         ▼
                    ┌────────────────────────────────────────┐
                    │  Merge (optional): mergingVision(...)  │
                    │  Final gate: no unjustified fails      │
                    └────────────────────────────────────────┘
```

### Phase A — Pixel harness (automated)

1. Ensure output PNG exists.
2. Run analysis (`--analyze` on gen, `analyze-image`, or `Scripts/eval-generation.sh`).
3. Read scores + `findings[]` (`severity`, `code`, `message`).
4. **Pixel gate:** any `fail` → failed unless vision later proves a documented false positive.

| Code | Meaning | Typical fix |
|------|---------|-------------|
| `color_mismatch` | Prompt color not found (fail if single color; **warn** if multi-color) | Raise I2I `--strength`, clearer color words; multi-color → vision |
| `possible_tile_seam` | High midline discontinuity on ≥768 canvas | Inspect flat regions; VAE tile blend |
| `vae_tile_expected` | Info: large canvas likely tiled decode | Check `tile_seam_score` |
| `soft_focus` | Very low Laplacian variance | Confirm blur; steps/weights |
| `highlight_clip` | Blown whites | Softer lighting in prompt |
| `high_structure_fidelity` | I2I ≈ reference | Strength too low for the edit |
| `low_structure_fidelity` | Large change vs ref | Expected at high strength |
| `strength_too_low_for_edit` | strength≥0.75 + color edit but SSIM still high | Raise strength / stronger prompt |
| `expected_structure_change` | strength≥0.75 + low SSIM | OK for strong edits |
| `low_semantic_alignment` | CLIP/proxy score low | Vision review subject |
| `semantic_score` | Info: CLIP or Vision proxy score | Prefer Core ML CLIP when installed |

**Backend note:** Steel fused FA / decode-only VAE do not change the PNG eval path. Cosine **tiled VAE** (≥768) is covered by seam metrics. Schema **1.3**: semantic (CLIP/proxy) + LPIPS-lite + strength-aware I2I.

```bash
# Strength-aware I2I eval (also automatic after i2i --analyze)
aestrix analyze-image edit.png --reference src.png --prompt "make it blue" --strength 0.85

# CI golden floors (no weights)
./Scripts/ci-eval-floors.sh
```

### Phase B — Vision review (multimodal agent)

1. Open **generated** image with a vision tool (`read_file` on the PNG).
2. For I2I, also open **reference**.
3. Open `*.vision-brief.md` (or re-run with `--vision-brief`).
4. Answer every checklist item.
5. Fill `VisionReview.Assessment` (see template).
6. Merge:

```swift
let final = ImageAnalysisReportBuilder.mergingVision(pixelReport, assessment)
// overall ≈ 0.45 * pixel + 0.55 * vision
```

7. **Final gate:** no critical vision fails; pixel fails fixed or waived with vision evidence.

### Checklist

**T2I:** `subject`*, `color`*, `lighting`, `artifacts`*, `composition`, `text`*, `anatomy`*, `photoreal`  
**I2I adds:** `identity`, `edit_applied`*  

\* = critical when applicable.

Template: [templates/vision-assessment.example.json](../templates/vision-assessment.example.json).

---

## Agent session contract

Agents **must not** claim generation quality without:

1. Pixel report on the artifact path, and  
2. Vision inspection of the same PNG (and reference for I2I).

**Definition of done:**

- [ ] PNG written  
- [ ] Pixel eval run (`--analyze` or `eval-generation.sh`)  
- [ ] Vision checklist completed against the image  
- [ ] Failures fixed **or** explicitly waived with reason  
- [ ] Paths + scores recorded in the session summary  

### False positives / negatives

| Case | Action |
|------|--------|
| Pixel says fail, vision sees success (e.g. blue mug on wood → hue “orange”) | Waive with vision note; check `top_chromatic_hues` |
| Pixel says pass, vision sees wrong color/subject | **Fail** — trust vision; re-gen |
| High SSIM but edit requested | Strength too low or prompt weak |

---

## CI / regression recipe

```bash
# Always (no GPU weights)
swift test --filter AestrixEvalTests
swift test --filter AestrixCoreTests

# Optional smoke gen + hard pixel gate (needs snapshot + metallib)
./Scripts/ensure-metallib.sh
swift build
.build/debug/aestrix t2i "A red fox in a snowy forest at sunrise, photorealistic." \
  --width 512 --height 512 --steps 4 --seed 42 \
  --output /tmp/aestrix_ci_t2i.png \
  --analyze --fail-on-pixel-gate
```

Vision review is **agent/human** (not plain CI) unless a VLM job is added later.

Regression prompts: [eval-prompts.md](eval-prompts.md).

---

## Library API

```swift
import AestrixEval

let pixel = try ImageAnalyzer.analyze(
    imageURL: outURL,
    options: .init(prompt: prompt, referenceURL: refURL)
)
print(VisionReview.agentBrief(report: pixel, mode: .i2i))

// After viewing image(s):
let assessment = VisionReview.Assessment(
    mode: .i2i,
    imagePath: outURL.path,
    referencePath: refURL?.path,
    prompt: prompt,
    caption: "…",
    answers: ["subject": "…", "color": "…", "edit_applied": "…"],
    findings: [.init(severity: .info, code: "ok", message: "…")],
    visionScore: 88,
    verdict: "pass — …"
)
let final = ImageAnalysisReportBuilder.mergingVision(pixel, assessment)
```

---

## Related commands

| Command | Role |
|---------|------|
| `aestrix t2i … --analyze --vision-brief` | Generate + full eval kickoff |
| `aestrix i2i … --analyze --vision-brief` | Edit + full eval kickoff |
| `aestrix analyze-image` | Pixel / brief only |
| `Scripts/eval-generation.sh` | Sidecars for an existing PNG |
| `Scripts/ensure-metallib.sh` | Required before generate on clean builds |
