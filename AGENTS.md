# Aestrix — Agent Instructions

Aestrix is a **from-scratch** native **Swift + MLX** runtime for **Black Forest Labs FLUX.2 [klein] 4B** (Apache-2.0) on Apple Silicon. It is **not** a fork of existing Swift ports; public MLX ports are oracles only.

## Product locks

| Rule | Detail |
|------|--------|
| Model | **Klein 4B only** — do not ship Klein 9B (Non-Commercial) or FLUX.2 Dev (32B) |
| Weights | **Pre-quantized only** (default **4-bit**). **No user-facing bf16** download or quantize-from-bf16 at runtime |
| Memory | **Staged pipeline is the default**: never co-reside text encoder + DiT + VAE |
| Platforms | macOS library + CLI first; iOS 26 uses the same staged core |
| v1 features | Text-to-image + single-image I2I (strength-based first) |
| Guidance | Distilled defaults: **4 steps**, **guidance = 1.0**, **no negative prompts**, no prompt-upsampling by default |

## Skills to load

### Product / prompting (project-scoped)

- **`flux-best-practices`** at `.grok/skills/flux-best-practices/` (vendored from [black-forest-labs/skills](https://github.com/black-forest-labs/skills), pinned SHA in `.upstream-sha`)
  - Especially: `rules/flux2-models.md` (klein), `t2i-prompting.md`, `i2i-prompting.md`, `negative-prompt-alternatives.md`, `core-principles.md`
  - Use for CLI examples, eval prompts, I2I UX, quality gates
  - **Do not** install flux-3 video skills into this repo

### Engineering

| Work | Skill / tool |
|------|----------------|
| MLX arrays, NN, quant, wired memory, eval | `mlx-swift` |
| Qwen3 text-encoder port patterns | `mlx-swift-lm` (+ model-porting) |
| Library API truth | Context7 MCP |
| Hugging Face download / inspect | `hf-cli` / `hf` |
| Build / sim / device | XcodeBuildMCP + `axiom-xcode-mcp` / `axiom-build` |
| Concurrency / actors | `axiom-concurrency` |
| Memory / perf audits | `axiom-performance`, memory auditor |
| Tests | `axiom-testing` |
| Image quality / gen feedback | **`Docs/EVAL_WORKFLOW.md`** + `aestrix analyze-image` + vision tools |
| “Done” claims | `verification-before-completion` + **EVAL_WORKFLOW** on sample PNGs |

BFL skills cover **prompting/product behavior**, not DiT/VAE math. MLX skills cover **implementation**.

## Architecture reminders

1. **Serial residency**: TE encode → unload → DiT denoise → unload → VAE decode → unload; `Memory.clearCache()` after unload.
2. **Qwen3 TE**: chat template required; layers **9/18/27** concat → **7680**; full **512** padded tokens to DiT.
3. **DiT**: MMDiT **5 double + 20 single** blocks; 4-axis RoPE θ=2000; inner dim 3072.
4. **Scheduler**: linear-μ exponential time-shift Euler; timestep ×1000 when max(t) ≤ 1 (model does this if caller passes σ∈[0,1]).
5. **Default resolution**: Tier L **512²**; Tier M+ may use 1024².
6. **Canonical weights**: `mlx-community/FLUX.2-Klein-4B-4bit` (module-split TE/DiT/VAE). See `Docs/WEIGHTS.md`.
7. **Text RoPE ids** (FLUX.2): `[t,h,w,l] = [0,0,0,token_i]` — not `[token_i,0,0,0]`.
8. **Latents**: noise as packed `[B, H/16·W/16, 128]` (32 ch × 2×2 patch); VAE decode uses BN denorm + unpatchify.
9. **I2I strength**: full N-step strength schedule + mid-range curve; color edits often need **≥ 0.8**.

## Phase status

| Phase | Status | Notes |
|-------|--------|--------|
| 0 Scaffold / snapshot | **Done** | SPM targets, cache layout, `aestrix info` |
| 1 Pure math | **Done** | RoPE, timestep emb, modulation, scheduler |
| 2 DiT load | **Done** | `Flux2Transformer` + 4-bit load; `load-dit` |
| 3 VAE load | **Done** | `Flux2VAE` + load; `load-vae` |
| 4 Qwen3 TE | **Done** | 3-layer tap, chat template, BPE tokenizer |
| 5 Staged T2I | **Done** | `aestrix t2i` |
| 6 I2I strength | **Done** | `aestrix i2i` |
| Eval workflow | **Done** | Pixel + vision procedure (`Docs/EVAL_WORKFLOW.md`) |
| 7 iOS host | Pending | Same staged core |

## Process rule (blocking)

**Always address issues that surface before starting the next phase.**  
Do not stack untested phase work on a broken foundation (e.g. metallib, load failures, parity fails, failed eval gates).

## Metal / metallib (SPM CLI)

- `swift build` alone often leaves a **stub** `default.metallib` (~3KB). Forward kernels require a **full** metallib (~130MB).
- After `swift build` / package resolve: run **`Scripts/ensure-metallib.sh`**.
- Large metallib is **gitignored**; regenerate after clean checkouts.
- Smoke load: `aestrix load-te` · `load-dit` · `load-vae` (needs snapshot + metallib).

---

## Generation evaluation workflow (mandatory)

**Canonical doc:** [`Docs/EVAL_WORKFLOW.md`](Docs/EVAL_WORKFLOW.md)

After every `t2i` / `i2i` used to judge quality, agents **must** run pixel eval **and** open the PNG with vision.

### One-shot generate + eval

```bash
swift build && ./Scripts/ensure-metallib.sh

# T2I
.build/debug/aestrix t2i "$PROMPT" --width 512 --height 512 --steps 4 --seed 42 \
  --output /tmp/out.png --analyze --vision-brief

# I2I
.build/debug/aestrix i2i "$PROMPT" --image "$REF" --strength 0.8 \
  --output /tmp/edit.png --analyze --vision-brief
```

Writes sidecars: `out.eval.json`, `out.vision-brief.md`.

### Existing PNG

```bash
./Scripts/eval-generation.sh /tmp/out.png --prompt "$PROMPT" --mode t2i
./Scripts/eval-generation.sh /tmp/edit.png --prompt "$PROMPT" --reference "$REF" --mode i2i
```

### Vision pass (you)

1. `read_file` the generated PNG (and reference for I2I).  
2. Answer the checklist in the vision brief.  
3. Fill `VisionReview.Assessment` (see `templates/vision-assessment.example.json`).  
4. Merge: `ImageAnalysisReportBuilder.mergingVision(pixel, assessment)`.  

### Definition of done (generation claim)

- [ ] PNG written  
- [ ] Pixel eval run  
- [ ] Vision checklist completed  
- [ ] Fails fixed or waived with reason  
- [ ] Paths + scores in session summary  

| Layer | Covers |
|-------|--------|
| Pixel | Sharpness, clip, hue, SSIM, color-word heuristics, exit 2 on hard fail |
| Vision | Subject, real color, text, hands, artifacts, edit applied, aesthetics |

Do **not** claim “blue mug works” from metrics alone without opening the image.

Metrics reference: [`Docs/IMAGE_ANALYSIS.md`](Docs/IMAGE_ANALYSIS.md).

---

## Coding conventions

- Swift 6, `actor` isolation for all MLX state (`MLXArray` is not `Sendable`).
- Pin `mlx-swift` deliberately after validation (`0.31.6`).
- Prefer `MLXFast.scaledDotProductAttention`.
- Public API lives in `AestrixRuntime`; modules are loadable independently.
- Do not add BFL cloud API as a runtime dependency.

## CLI map

| Command | Role |
|---------|------|
| `aestrix info` | Tier, policy, snapshot path |
| `aestrix mem-selftest` | Dry staged residency (no weights) |
| `aestrix schedule` | Print sigmas/timesteps |
| `aestrix load-te` / `load-dit` / `load-vae` | Staged weight load smoke |
| `aestrix encode-prompt` | TE only → shape/token count |
| `aestrix t2i … --analyze --vision-brief` | Generate + eval kickoff |
| `aestrix i2i … --analyze --vision-brief` | Edit + eval kickoff |
| `aestrix analyze-image` | Pixel / brief only |
| `Scripts/eval-generation.sh` | Eval existing PNG |
| `Scripts/ensure-metallib.sh` | Build/install full Metal library |

## Out of scope (v1)

- Klein 9B / FLUX.2 Dev shipping  
- Multi-reference edit (>1 image) and KV-cache  
- LoRA training  
- Base CFG 28-step path  
- bf16 product path  
