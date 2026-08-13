# Aestrix — Agent Instructions

Aestrix is a **from-scratch** native **Swift + MLX** runtime for **Black Forest Labs FLUX.2 [klein] 4B** (Apache-2.0) on Apple Silicon. It is **not** a fork of existing Swift ports; public MLX ports are oracles only.

## Product locks

| Rule | Detail |
|------|--------|
| Model | **Klein 4B only** — do not ship Klein 9B (Non-Commercial) or FLUX.2 Dev (32B) |
| Weights | **Pre-quantized only** (default **4-bit**). **No user-facing bf16** download or quantize-from-bf16 at runtime |
| Memory | **Staged pipeline is the default**: never co-reside text encoder + DiT + VAE |
| Platforms | macOS library + CLI first; iOS 26 uses the same staged core |
| v1 features | Text-to-image + single-image I2I (strength + optional **identity** stack) |
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
3. **DiT**: MMDiT **5 double + 20 single** blocks; 4-axis RoPE θ=2000; inner dim 3072. Long sequences: **block checkpointing**, **MLX Steel fused FA** (simdgroup MMA, full Q, D=128); **f16 Q/K/V** when seq > 512 (512² image tokens included). Prompt context is **`projectContext` once per generate** (`7680→3072`), not every denoise step.
4. **Scheduler**: match mflux/diffusers (time-shift / sigma); training-scale timesteps **[0, 1000]** passed from pipeline (no host `item()` sync).
5. **Default resolution**: **1024²** (4-bit). On ~8 GB unified: release + full metallib; 2026-08-13 `hoist-*`: 512² e2e ~**27.5 s**, 1024² ~**87.7 s**; peak active ~**2.04–2.05 GiB**, watermark ~**2.99 / 3.76 GiB**. See `Docs/PERF.md`.
6. **VAE**: T2I / final I2I decode use **decode-only** weights (~97 MB). Large canvases use **tiled decode** (`VAETileConfig`: default overlap + cosine blend).
7. **Canonical weights**: `mlx-community/FLUX.2-Klein-4B-4bit` @ `1cebb9b45c21ece14a42615b16bf5fa4de9b56da` (module-split TE/DiT/VAE). Pins: `WeightPreset.pin`, `Docs/hub-pins.json`, `Docs/WEIGHTS.md`.
8. **Text RoPE ids** (FLUX.2): `[t,h,w,l] = [0,0,0,token_i]`.
9. **Latents**: packed `[B, H/16·W/16, 128]`; VAE decode uses BN denorm + unpatchify.
10. **I2I strength**: full N-step strength schedule + mid-range curve; color edits often need **≥ 0.8**.
11. **I2I identity (Tier B)**: `--identity` enables reference latents (`t=10`), Vision face mask (regional σ + clean-pull), and milder `identity` schedule. Strength bands: color/object **≥ 0.8**; people + scene **0.85–0.9** + `--identity`. Say **recolor same cut** vs **replace outfit**. Default I2I remains strength-only.
12. **Text-stage shortcuts**: prompt-embed disk cache is **default on** (`~/Library/Caches/Aestrix/embeds`, keyed by prompt+model+bits+len; `--no-embed-cache` opts out; hit skips the whole TE stage, byte-identical). `--text-tokens auto` trims padding tokens (numerics differ from the full-512 reference — **opt-in / experimental**; big win at 512²). `--identity --ref-downsample N` pools reference tokens for cheaper identity I2I.

## Phase status

**Authoritative backlog / parked work:** [`Docs/ROADMAP.md`](Docs/ROADMAP.md)

| Phase | Status | Notes |
|-------|--------|--------|
| 0–6 + Eval | **Done** | macOS library + CLI (T2I, I2I, eval workflow) |
| **P6c Identity I2I** | **Done** | Ref latents (`t=10`), Vision face mask, clean-pull, schedule curves; `aestrix i2i --identity` |
| **P9 Performance harness** | **Done** | `AestrixBench` (t2i / i2i / identity-i2i + host contention); Steel FA + tiled VAE + **context hoist**; 2026-08-13 `hoist-512` / `hoist-1024` in `Docs/PERF.md` |
| **P7 iOS host** | **Parked** | Resume via `Docs/ROADMAP.md` § P7 |
| **P8 macOS polish** | **Partial** | Hub pin + GitHub eval-floors CI done; I2I strength-curve doc still open |
| Out of v1 | Tracked only | Multi-ref (>1 image), CFG, LoRA, bf16 — see roadmap |

## Process rule (blocking)

**Always address issues that surface before starting the next phase.**  
Do not stack untested phase work on a broken foundation (e.g. metallib, load failures, parity fails, failed eval gates, unexplained OOM).

## Metal / metallib (SPM CLI)

- `swift build` alone often leaves a **stub** `default.metallib` (~3KB). Forward kernels require a **full** metallib (~130MB).
- After `swift build` / package resolve: run **`Scripts/ensure-metallib.sh`**.
- Large metallib is **gitignored**; regenerate after clean checkouts.
- Prefer **`swift build -c release`** for generation / benches.
- Smoke: `aestrix load-te` · `load-dit` · `load-vae` · `t2i` / `bench` (needs snapshot + metallib).

## Host safety (8 GB Mac mini — blocking)

This machine is **8 GB unified** (`Mac14,3`). Cursor agents + `swift build`/`swift test` + `MTLCompilerService` + optional DiT (~2 GiB) starved **WindowServer** for 127 s and triggered a **watchdog kernel panic** (2026-08-13). Separate `aestrix` Metal command-buffer aborts (VAE decode) are process crashes that can worsen compositor stalls.

**Rules for every agent on this host:**

1. **One Metal owner.** Do not run `aestrix` generate/bench/compile-spike while Xcode, another `aestrix`, or a second IDE is compiling Metal.
2. **Default to filtered unit tests and 512² smokes.** 1024² T2I bench is OK when asked (measured ~88 s, watermark ~3.76 GiB). Do **not** start a 4-trial `identity-i2i` at 1024 (joint ~8704) unless the user explicitly wants that.
3. **Never** `MLX.compile` the full DiT; **never** `dit-compile-spike` on this host without `--force` on an idle machine.
4. **Never** relax cache-clear / `EvalCachePolicy.high` (that type lives only on `cursor-opt-quarantine`).
5. After any reboot, hang, or new `aestrix-*.ips`, **stop and inspect** DiagnosticReports before retrying.
6. Experimental Cursor work is on **`cursor-opt-quarantine`** — do not merge VAE D=512 `evalEachChunk` or `.high` cache policy.
7. **Ambient ≠ contaminated:** `WindowServer`, Ghostty, `grok`, `MTLCompilerService` do not mark bench trials dirty. Cursor / `swift-package` still do.

CLI: `HostPreflight` takes `~/Library/Caches/Aestrix/aestrix.lock` and refuses a second instance. **Swap is not a run gate.** Details: [`Docs/HOST_SAFETY.md`](Docs/HOST_SAFETY.md).

**Tests:** do not assume unfiltered `swift test` is safe (Metal FA tests have hung after GPU aborts). Prefer:

```bash
swift test --filter 'HostPreflight|GoldenMetric|Flux2Math|IdentityPreserve|AestrixBench|HubPin'
```

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
# 512² eval-prompts × seeds 42/0/7 (pixel gate; PNGs under /tmp)
AESTRIX=.build/release/aestrix ./Scripts/eval-regression.sh

# I2I (strength-only color/style)
.build/release/aestrix i2i "$PROMPT" --image "$REF" --strength 0.8 \
  --output /tmp/edit.png --analyze --vision-brief

# I2I identity (people / character consistency — Tier B)
.build/release/aestrix i2i "$PROMPT" --image "$REF" --strength 0.9 --identity \
  --output /tmp/edit-id.png --analyze --vision-brief
```

### Performance / pressure (mandatory before “faster / leaner” claims)

**Canonical doc:** [`Docs/PERF.md`](Docs/PERF.md)

```bash
# Multi-trial timings + memory (512 smoke; 1024 is product default)
.build/release/aestrix bench --width 512 --height 512 --warmup 1 --trials 3 \
  --json /tmp/bench.json

# Identity I2I bench (512² first)
.build/release/aestrix bench --mode identity-i2i --image "$REF" \
  --width 512 --height 512 --strength 0.9 --with-quality \
  --json /tmp/id-i2i.json

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
| `aestrix info` | Tier, policy, snapshot path, pinned Hub revision |
| `aestrix mem-selftest` | Dry staged residency (no weights) |
| `aestrix schedule` | Print sigmas/timesteps |
| `aestrix load-te` / `load-dit` / `load-vae` | Staged weight load smoke; `load-dit --dump-dtypes` = compute-dtype probe |
| `aestrix encode-prompt` | TE only → shape/token count |
| `aestrix t2i … --analyze --vision-brief` | Generate + eval kickoff |
| `aestrix i2i … --analyze --vision-brief` | Edit + eval kickoff |
| `aestrix analyze-image` | Pixel / brief only |
| `aestrix bench` | Timings, memory, pressure; modes `t2i` \| `i2i` \| `identity-i2i` \| `pressure-map` \| `dit-one-step` \| `res-ladder` |
| `aestrix bench-compare A B` | Percent deltas between JSON reports |
| `aestrix session` | Warm multi-prompt loop; modules resident (≥16 GB gate, `--force-resident`) |
| `aestrix dit-compile-spike` | Research: block-level `MLX.compile` (NO-GO; refused on 8 GB without `--force`) |
| `Scripts/eval-generation.sh` | Eval existing PNG |
| `Scripts/eval-regression.sh` | 512² T2I eval-prompts × seeds 42/0/7 + pixel gate |
| `Scripts/ci-eval-floors.sh` | Hub pins + synthetic golden floors (no weights; GitHub Actions) |
| `Scripts/ensure-metallib.sh` | Build/install full Metal library |

## Out of scope (v1)

- Klein 9B / FLUX.2 Dev shipping  
- Multi-reference edit (>1 image) and KV-cache  
- LoRA training  
- Base CFG 28-step path  
- bf16 product path  
- User-facing bit-depth other than prequant Hub packs (3/4/6/8) without product decision  
