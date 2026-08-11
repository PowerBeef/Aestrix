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
| Default canvas | **1024²** (4-bit staged). Lower sizes via `--width` / `--height`. |

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
| Memory / perf audits | `axiom-performance`, **`Docs/PERF.md`**, `aestrix bench` |
| Tests | `axiom-testing` |
| Image quality / gen feedback | **`Docs/EVAL_WORKFLOW.md`** + `aestrix analyze-image` + vision tools |
| “Done” claims | `verification-before-completion` + **EVAL_WORKFLOW** on sample PNGs |

BFL skills cover **prompting/product behavior**, not DiT/VAE math. MLX skills cover **implementation**.

## Architecture reminders

1. **Serial residency**: TE encode → unload → DiT denoise → unload → VAE decode → unload; `Memory.clearCache()` after unload and between large-canvas stages.
2. **Qwen3 TE**: chat template required; layers **9/18/27** concat → **7680**; full **512** padded tokens to DiT by default.
3. **DiT**: MMDiT **5 double + 20 single** blocks; 4-axis RoPE θ=2000; inner dim 3072. Long sequences: **block checkpointing**, **chunked SDPA/Linears**, **f16 Q/K/V** when seq > 2048.
4. **Scheduler**: match mflux/diffusers (time-shift / sigma); training-scale timesteps **[0, 1000]** passed from pipeline (no host `item()` sync).
5. **Default resolution**: **1024²** (4-bit). On ~8 GB unified: use release build + full metallib; measured ~**97 s** e2e / peak MLX watermark ~**3.3 GB** on M2 after low-RAM path.
6. **VAE**: T2I / final I2I decode use **decode-only** weights (~97 MB). Large canvases use **tiled 2×2 latent decode**.
7. **Canonical weights**: `mlx-community/FLUX.2-Klein-4B-4bit` (module-split TE/DiT/VAE). See `Docs/WEIGHTS.md`.
8. **Text RoPE ids** (FLUX.2): `[t,h,w,l] = [0,0,0,token_i]`.
9. **Latents**: packed `[B, H/16·W/16, 128]`; VAE decode uses BN denorm + unpatchify.
10. **I2I strength**: full N-step strength schedule + mid-range curve; color edits often need **≥ 0.8**.

## Phase status

**Authoritative backlog / parked work:** [`Docs/ROADMAP.md`](Docs/ROADMAP.md)

| Phase | Status | Notes |
|-------|--------|--------|
| 0–6 + Eval | **Done** | macOS library + CLI (T2I, I2I, eval workflow) |
| **P9 Performance harness** | **Done / partial** | `AestrixBench`, pressure probes; 1024² low-RAM path measured |
| **P7 iOS host** | **Parked** | Resume via `Docs/ROADMAP.md` § P7 |
| **P8 macOS polish** | **Parked** | Regression suite, release pins |
| Out of v1 | Tracked only | Multi-ref, CFG, LoRA, bf16 — see roadmap |

## Process rule (blocking)

**Always address issues that surface before starting the next phase.**  
Do not stack untested phase work on a broken foundation (e.g. metallib, load failures, parity fails, failed eval gates, unexplained OOM).

## Metal / metallib (SPM CLI)

- `swift build` alone often leaves a **stub** `default.metallib` (~3KB). Forward kernels require a **full** metallib (~130MB).
- After `swift build` / package resolve: run **`Scripts/ensure-metallib.sh`**.
- Large metallib is **gitignored**; regenerate after clean checkouts.
- Prefer **`swift build -c release`** for generation / benches.
- Smoke: `aestrix load-te` · `load-dit` · `load-vae` · `t2i` / `bench` (needs snapshot + metallib).

---

## Generation evaluation workflow (mandatory)

**Canonical doc:** [`Docs/EVAL_WORKFLOW.md`](Docs/EVAL_WORKFLOW.md)

After every `t2i` / `i2i` used to judge quality, agents **must** run pixel eval **and** open the PNG with vision.

### One-shot generate + eval

```bash
swift build -c release && ./Scripts/ensure-metallib.sh

# T2I (default 1024²; use --width 512 for faster smoke)
.build/release/aestrix t2i "$PROMPT" --width 1024 --height 1024 --steps 4 --seed 42 \
  --output /tmp/out.png --analyze --vision-brief

# I2I
.build/release/aestrix i2i "$PROMPT" --image "$REF" --strength 0.8 \
  --output /tmp/edit.png --analyze --vision-brief
```

### Performance / pressure (mandatory before “faster / leaner” claims)

**Canonical doc:** [`Docs/PERF.md`](Docs/PERF.md)

```bash
# Multi-trial timings + memory
.build/release/aestrix bench --width 512 --height 512 --warmup 1 --trials 3 \
  --json /tmp/bench.json

# DiT block-level pressure map
.build/release/aestrix bench --mode pressure-map --width 768 --height 768 \
  --probe-density blocks --json /tmp/pressure.json

# Resolution ladder (subprocess per side)
.build/release/aestrix bench --mode res-ladder --ladder 512,768,1024
```

Do **not** claim Tier L/M readiness or 1024² support without measured peak RSS / MLX watermark.

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

---

## Coding conventions

- Swift 6, `actor` isolation for all MLX state (`MLXArray` is not `Sendable`).
- Pin `mlx-swift` deliberately after validation (`0.31.6`).
- Prefer `MLXFast.scaledDotProductAttention`.
- Public API lives in `AestrixRuntime`; modules are loadable independently.
- Instrumentation via `PipelineTrace` / `ProbeDensity` must not change numerics when density is `.off`.
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
| `aestrix bench` | Timings, memory, pressure probes (`pressure-map`, `dit-one-step`, `res-ladder`) |
| `aestrix bench-compare A B` | Percent deltas between JSON reports |
| `Scripts/eval-generation.sh` | Eval existing PNG |
| `Scripts/ensure-metallib.sh` | Build/install full Metal library |

## Out of scope (v1)

- Klein 9B / FLUX.2 Dev shipping  
- Multi-reference edit (>1 image) and KV-cache  
- LoRA training  
- Base CFG 28-step path  
- bf16 product path  
- User-facing bit-depth other than prequant Hub packs (3/4/6/8) without product decision  
