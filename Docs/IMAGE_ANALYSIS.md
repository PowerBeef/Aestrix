# Image quality & accuracy harness

`AestrixEval` provides **no-Metal** pixel analysis so agents and CI can score generations without loading weights.

**End-to-end procedure (generate → pixel → vision → gate):**  
**[`Docs/EVAL_WORKFLOW.md`](EVAL_WORKFLOW.md)** — mandatory for agents after `t2i` / `i2i`.

**Multimodal agents** should **supplement** pixel metrics: open the PNG, answer the vision checklist, merge into the report.

Schema version: **1.3** (semantic/CLIP + LPIPS-lite + strength-aware I2I).

---

## CLI

```bash
# Technical quality only
aestrix analyze-image out.png

# Prompt alignment (color / style heuristics)
aestrix analyze-image out.png --prompt "a cobalt blue ceramic mug on wood"

# I2I fidelity vs source
aestrix analyze-image edited.png --reference source.png --prompt "make it blue" --json report.json

# Vision brief for agents (paths + checklist)
aestrix analyze-image edited.png --reference source.png --prompt "make it blue" --vision-brief

# Machine-readable stdout
aestrix analyze-image out.png --prompt "…" --json-stdout
```

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | OK (no `fail` findings) |
| 2 | Hard failure (e.g. single-color `color_mismatch`) |
| 1 | Load/analysis error |

---

## Metrics

### Technical (no reference)

| Field | Meaning |
|-------|---------|
| `sharpness_laplacian_var` | Laplacian variance of luminance (blur detector) |
| `mean_gradient` | Edge energy |
| `std_luminance` / `contrast_rms` | Global contrast |
| `clip_black_fraction` / `clip_white_fraction` | Crushed shadows / blown highlights |
| `mean_saturation` | HSV saturation average |
| `dominant_hue` + fraction | Coarse color bucket |
| `hue_weights` / `top_chromatic_hues` | Center-weighted chromatic mass |
| `noise_proxy` | High-frequency residual after blur |
| `luminance_entropy` | Histogram richness |
| `expects_vae_tiling` | `true` when max(side) ≥ 768 (Aestrix auto-tiles decode) |
| `tile_seam_score` / `_vertical` / `_horizontal` | Midline discontinuity / mean gradient (clean often &lt;2.5) |
| `technical_score` | 0–100 composite (includes mild seam penalty when tiling expected) |

### Reference (I2I / gold)

| Field | Meaning |
|-------|---------|
| `ssim` | Structural similarity (luminance windows) |
| `psnr` / `mse` / `mean_abs_error` | Pixel error |
| `mean_color_distance` | L2 of mean RGB |
| `histogram_correlation` | Color distribution similarity |
| `mean_delta_e` | Cheap perceptual color distance |
| `fidelity_score` | 0–100 “looks like reference” |

### Prompt alignment (heuristic, not a VLM)

- Extracts **color words** (blue/cobalt/red/…) and compares to dominant / chromatic hues  
- Prompt length style vs BFL klein narrative guidance  
- Flags unverifiable terms (`hands`, `text`, `logo`, …) for human/VLM review  
- **Single-color miss → `fail`**; **multi-color miss → `warn`** (false positives common)  
- **Does not** claim full prompt adherence — only cheap signals agents can act on  

### Semantic alignment (P1 — CLIP / Vision proxy)

| Field | Meaning |
|-------|---------|
| `semantic.score` | 0…100 prompt–image agreement |
| `semantic.backend` | `coreml_clip` \| `vision_proxy` \| `unavailable` |
| `semantic.cosine` | CLIP cosine when Core ML models present |
| `semantic.top_labels` | Vision classify labels (proxy backend) |

**Core ML CLIP (optional):** place compiled models at  
`~/Library/Caches/Aestrix/models/clip-coreml/image_encoder.mlmodelc` and `text_encoder.mlmodelc`  
(string text input + embedding multiarray output). See `Scripts/fetch-clip-coreml.sh`.

**Default without models:** `VNClassifyImageRequest` label–token overlap (`vision_proxy`). Always available; weaker than real CLIP.

CLI: `--skip-semantic` for pure pixel path.

### Perceptual distance (P2 — LPIPS-lite)

Always on for `--reference` compares (no neural weights):

| Field | Meaning |
|-------|---------|
| `perceptual_distance` | Lower = more similar (~0 identical) |
| `perceptual_score` | 0…100 (higher = more similar) |
| `ms_ssim` | Multi-scale SSIM component |

Fidelity score blends SSIM + LPIPS-lite + color + histogram.

### Strength-aware I2I gates (P2)

Pass `--strength` to `analyze-image` (or automatic after `i2i --analyze`):

| Condition | Finding |
|-----------|---------|
| strength ≥ 0.75, color edit, SSIM > 0.85 | `strength_too_low_for_edit` (warn) |
| strength ≥ 0.75, SSIM < 0.35 | `expected_structure_change` (info) |
| strength < 0.4, SSIM < 0.5 | `unexpected_identity_drift` (warn) |
| strength ≥ 0.75, color edit, LPIPS-lite ≪ 0.08 | `low_perceptual_change` (warn) |

### Findings codes (selected)

| Code | Severity | Notes |
|------|----------|--------|
| `color_mismatch` | fail or warn | Fail if one color intent; warn if multi-color |
| `possible_tile_seam` | warn | seam_score &gt; 3 on ≥768 canvas |
| `vae_tile_expected` | info | ≥768 canvas; report seam_score |
| `soft_focus` / `highlight_clip` / `low_contrast` | warn | Technical gates |
| `high_structure_fidelity` / `low_structure_fidelity` | info | I2I SSIM vs reference |
| `needs_visual_review` | info | Prompt mentions text/hands/… |

---

## Backend alignment (runtime vs eval)

| Runtime (Aestrix generate) | Eval harness impact |
|----------------------------|---------------------|
| **1024² default** | `maxAnalysisSide` default **1024** — full-res analysis for default canvas |
| **Steel fused FA** (DiT) | Does **not** change PNG path; no metric change required |
| **VAE cosine-tiled decode** (spatial ≥ 96 ≈ **768 px**) | **`expects_vae_tiling` + `tile_seam_*`** detect hard seams |
| **Decode-only VAE** | N/A to PNG metrics |
| **f16 QKV long seq** | Quality via vision / technical; no direct metric |
| Bench `--with-quality` | Still uses `technical_score` + `color_match` only |

Pixel eval is **intentionally independent of MLX** so CI/agents work without weights. It scores **output PNGs**, not intermediate latents.

---

## Research notes: improving the harness (2026-08)

### What industry uses (T2I)

| Family | Examples | Role | Fit for Aestrix |
|--------|----------|------|-----------------|
| Distribution | FID, KID, Precision/Recall | Dataset vs real | Needs many gens + ref set; heavy for local CLI |
| Prompt alignment | **CLIPScore**, BLIP | Image–text cosine | Best automated next step (Core ML / MLX CLIP optional) |
| Human preference | **ImageReward**, HPSv2/v3, VisionReward, DreamSim | RLHF-style rankers | Strong correlation with humans; Python/weights heavy |
| Full-ref I2I | **LPIPS**, SSIM, PSNR | Identity / edit | We have SSIM/PSNR; **LPIPS** would beat SSIM for perception |
| No-ref quality | Aesthetic predictors, BRISQUE-class | Beauty / artifacts | Optional; biases toward “vivid” aesthetics |
| Composition / identity | DINOv2, IRF | Subject consistency | Edit / multi-ref later |

Sources: [PrunaAI objective metrics](https://huggingface.co/blog/PrunaAI/objective-metrics-for-image-generation-assessment), [Awesome Evaluation of Visual Generation](https://github.com/ziqihuangg/Awesome-Evaluation-of-Visual-Generation), ImageReward / HPSv2 literature, SHINE/ComplexCompo (DINOv2 + DreamSim + ImageReward).

### Gaps in current harness

1. **No CLIPScore** — color words only; cannot score “fisherman at helm” vs wrong scene.  
2. **No LPIPS** — I2I fidelity over-relies on SSIM (identity, not “good edit”).  
3. **No human-preference model** — `technical_score` ≠ aesthetics or prompt adherence.  
4. **Vision layer is agent-manual** — not automated; CI stays pixel-only (by design).  
5. **Color gate false positives** — multi-object multi-color scenes; mitigated (warn) but not solved.  
6. **No regression gold set** — roadmap: golden metric floors in CI.  
7. **Tile seams** — now instrumented; was a blind spot after cosine VAE tiles.  

### Recommended roadmap (priority)

| P | Item | Status |
|---|------|--------|
| P0 | Tile-seam metric + VAE-tile findings | **Done** (1.2) |
| P1 | CLIPScore optional + Vision proxy | **Done** (1.3) |
| P1 | Golden metric floors in CI | **Done** (`GoldenMetricFloorsTests`, `Scripts/ci-eval-floors.sh`) |
| P2 | LPIPS-lite for I2I | **Done** (MS-SSIM multi-scale) |
| P2 | Strength-aware I2I gates | **Done** |
| P3 | ImageReward / HPS offline batch | Parked |
| P3 | Automated VLM / true AlexNet LPIPS weights | Parked |

```bash
./Scripts/ci-eval-floors.sh
./Scripts/fetch-clip-coreml.sh   # optional real CLIP Core ML
```

---

## Agent workflow

1. Generate: `aestrix t2i` / `aestrix i2i` → PNG  
2. Analyze: `aestrix analyze-image … --json /tmp/report.json`  
3. Read findings: severity `fail` / `warn` / `info` + `code`  
4. Act: e.g. `color_mismatch` → increase I2I `--strength`, clarify color in prompt, re-run  
5. Vision: open PNG with checklist from `VisionReview`  

Example gate:

```bash
aestrix analyze-image out.png --prompt "$PROMPT" --json /tmp/r.json || echo "quality gate failed"
```

## Library API

```swift
import AestrixEval

let report = try ImageAnalyzer.analyze(
    imageURL: outURL,
    options: .init(prompt: prompt, referenceURL: refURL)
)
print(report.summary)
print(report.findings)
let json = try ImageAnalysisReportBuilder.jsonString(report)
```

## Vision layer (agent multimodal)

```swift
let pixel = try ImageAnalyzer.analyze(imageURL: out, options: .init(prompt: p, referenceURL: ref))
print(VisionReview.agentBrief(report: pixel, mode: .i2i))
// Agent: open images with vision, fill Assessment, merge
let combined = ImageAnalysisReportBuilder.mergingVision(pixel, assessment)
// combined.overallScore blends 45% pixel + 55% vision
```

Checklist IDs: `subject`, `color`, `lighting`, `artifacts`, `composition`, `text`, `anatomy`, `photoreal`, plus I2I `identity`, `edit_applied`.

## Limits

- Pixel layer is not a substitute for vision on text/hands/aesthetics  
- SSIM/fidelity measure **identity**, not “good edit” — high strength I2I can correctly lower SSIM  
- Color heuristics use chromatic + center weighting; multi-color prompts demoted to warn  
- Tile-seam score is a **heuristic** (midline jumps); confirm with vision on flat regions  
- Always open the image when claiming generation quality  
