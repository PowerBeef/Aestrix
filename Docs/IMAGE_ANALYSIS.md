# Image quality & accuracy harness

`AestrixEval` provides **no-Metal** pixel analysis so agents and CI can score generations without loading weights.

**End-to-end procedure (generate → pixel → vision → gate):**  
**[`Docs/EVAL_WORKFLOW.md`](EVAL_WORKFLOW.md)** — mandatory for agents after `t2i` / `i2i`.

**Multimodal agents** should **supplement** pixel metrics: open the PNG, answer the vision checklist, merge into the report.

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
| 2 | Hard failure (e.g. `color_mismatch`) |
| 1 | Load/analysis error |

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
| `noise_proxy` | High-frequency residual after blur |
| `luminance_entropy` | Histogram richness |
| `technical_score` | 0–100 composite |

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

- Extracts **color words** (blue/cobalt/red/…) and compares to dominant hue  
- Prompt length style vs BFL klein narrative guidance  
- Flags unverifiable terms (`hands`, `text`, `logo`, …) for human/VLM review  
- **Does not** claim full prompt adherence — only cheap signals agents can act on  

## Agent workflow

1. Generate: `aestrix t2i` / `aestrix i2i` → PNG  
2. Analyze: `aestrix analyze-image … --json /tmp/report.json`  
3. Read findings: severity `fail` / `warn` / `info` + `code`  
4. Act: e.g. `color_mismatch` → increase I2I `--strength`, clarify color in prompt, re-run  

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
// Agent: read_file(out) + read_file(ref) with vision, fill Assessment
let assessment = VisionReview.Assessment(
    mode: .i2i,
    imagePath: out.path,
    referencePath: ref?.path,
    prompt: p,
    caption: "…",
    answers: ["subject": "yes", "color": "blue mug", "edit_applied": "yes", …],
    findings: [.init(severity: .info, code: "vision_ok", message: "…")],
    visionScore: 88,
    verdict: "pass — blue glaze edit clear, layout preserved"
)
let combined = ImageAnalysisReportBuilder.mergingVision(pixel, assessment)
// combined.overallScore blends 45% pixel + 55% vision
```

Checklist IDs: `subject`, `color`, `lighting`, `artifacts`, `composition`, `text`, `anatomy`, `photoreal`, plus I2I `identity`, `edit_applied`.

## Limits

- Pixel layer is not a substitute for vision on text/hands/aesthetics  
- SSIM/fidelity measure **identity**, not “good edit” — high strength I2I can correctly lower SSIM  
- Color heuristics use chromatic + center weighting; still verify with vision  
- Always open the image when claiming generation quality 
