# Imarello performance harness & optimization map

**Measurement first.** Ship and trust the bench harness before large optimizations. Every speed/RAM change must show deltas via `imarello bench` + `imarello bench-compare`.

Related: `Docs/MEMORY.md` (staged policy), `Docs/ARCHITECTURE.md`, `Docs/EVAL_WORKFLOW.md` (quality, not speed).

---

## Quick start

```bash
swift build && ./Scripts/ensure-metallib.sh

# Full staged T2I (default canvas 1024² — may OOM on 8 GB)
.build/release/imarello bench --label baseline --width 512 --height 512 --json /tmp/imarello-baseline.json

# Pressure map: DiT block-level MLX active samples (use for 1024 diagnosis)
.build/release/imarello bench --mode pressure-map --width 512 --height 512 \
  --probe-density blocks --json /tmp/p512.json

# One denoise step only (first-step peak)
.build/release/imarello bench --mode dit-one-step --width 768 --height 768 \
  --probe-density blocks --json /tmp/p768-step0.json

# Resolution ladder (subprocess per side; survives Metal abort)
.build/release/imarello bench --mode res-ladder --ladder 512,640,768,896,1024 \
  --probe-density blocks

# Compare two reports
.build/release/imarello bench-compare /tmp/imarello-baseline.json /tmp/imarello-candidate.json

# I2I / identity
.build/release/imarello bench --mode identity-i2i --image /path/to/ref.png \
  --width 512 --height 512 --strength 0.9 --with-quality \
  --json /tmp/id-i2i.json
```

Snapshot path (default):

`~/Library/Caches/Imarello/models/mlx-community--FLUX.2-Klein-4B-4bit`

---

## CLI reference

### `imarello bench`

| Flag | Default | Meaning |
|------|---------|---------|
| `--mode` | `t2i` | `t2i` \| `i2i` \| `identity-i2i` \| `pressure-map` \| `dit-one-step` \| `res-ladder` \| `mem-stages` \| `te-only` \| `dit-steps` \| `vae-decode` \| `load-only` |
| `--image` | — | Required for `i2i` / `identity-i2i` |
| `--strength` | 0.8 / 0.9 | I2I denoise strength (identity default 0.9) |
| `--probe-density` | `denoise` | `off` \| `stages` \| `denoise` \| `blocks` \| `max` |
| `--fail-soft` | off | Keep JSON when a trial errors |
| `--reset-peak-each-phase` | off | Reset MLX peak between TE/DiT/VAE |
| `--ladder` | 512…1024 | Sides for `res-ladder` |
| `--label` | `baseline` | Stored in report for compare |
| `--prompt` | fox sample | T2I / TE prompt |
| `--width` / `--height` | `512` | Canvas (multiples of 16) |
| `--steps` | `4` | Distilled default |
| `--seed` | `42` | Reproducible noise |
| `--trials` | `3` | Counted runs |
| `--warmup` | `1` | Discarded runs (Metal compile / cache) |
| `--cooldown` | `0` | Seconds between trials |
| `--json` | auto under Caches | Report path |
| `--output-dir` | Caches/Imarello/bench | Trial PNGs (`t2i`) |
| `--cache-limit` | MLX default | Optional `Memory.cacheLimit` bytes |
| `--vae-attn-chunk` | `64` | VAE mid-block query chunk. `0` = legacy `MLXFast` SDPA |
| `--eval-cache` | `product` | `product` (8 GB-safe) or `mid` (≥16 GB bench only; refused on tier L without `--force`) |
| `--with-quality` | off | Pixel score + color; I2I adds SSIM; identity adds face-crop SSIM |

### `imarello bench-compare BASE CANDIDATE`

Prints % deltas for e2e, denoise/step, encode, decode, peak RSS, peak MLX active. Lower is better.

---

## Metrics collected

### Timings (ms)

| Key | Source |
|-----|--------|
| `e2e` | Wall clock for the mode |
| `load_te` / `encode_te` / `unload_te` | PipelineTrace stages |
| `load_dit` / `denoise` / `unload_dit` | DiT residency |
| `denoise_steps[]` | Per Euler step (forward + eval + euler) |
| `load_vae` / `decode_vae` / `unload_vae` | VAE residency |
| `export_png` | ImageExport |

Aggregate: mean, stdev, min, max, p50, p95 across successful trials.

### Memory

At each `memorySample` label and at peaks:

| Field | API |
|-------|-----|
| `rss_bytes` | `task_info` resident size |
| `mlx_active_bytes` | `MLX.Memory.activeMemory` |
| `mlx_cache_bytes` | `MLX.Memory.cacheMemory` |
| `mlx_peak_bytes` | `MLX.Memory.peakMemory` (reset each trial) |

### System snapshot

Hostname, OS, physical RAM, processor count, Metal `recommendedMaxWorkingSetSize`, thermal state, MLX cache/memory limits, optional git SHA.

### Optional quality (not a speed metric)

With `--with-quality`: `technical_score`, `color_match` from `ImarelloEval` so you can refuse “faster but broken” regressions.

---

## Report schema

JSON schema version: **`1.0`** (`BenchReport.schemaVersion`).

Snake_case keys. Load with `BenchReportWriter.loadReport`. Fields: `label`, `created_at`, `system`, `config`, `trials[]`, `aggregate`.

---

## Instrumentation

`PipelineTrace` is optional on `ImarelloPipeline.generate` / `edit`:

- `stageBegin` / `stageEnd` for load/encode/denoise/decode/export  
- `denoiseStepBegin` / `denoiseStepEnd` per step  
- `memorySample(label:)` at residency boundaries  

Must **not** change numerics. BenchRunner builds `StageTimingsMs` from the trace + wall clock.

---

## Optimization backlog (data-driven)

Prioritize only after a **baseline** report exists on the target machine. IDs map to research notes.

### Memory (M)

| ID | Opportunity | How to measure |
|----|-------------|----------------|
| M1 | Staged policy already default — verify no dual residency | `mem-stages` peak RSS per module |
| M2 | `Memory.clearCache()` after each unload | delta peak_mlx_cache, peak_rss |
| M3 | Tune `Memory.cacheLimit` | `--cache-limit` sweeps |
| M4 | TE layer prune already (27 layers) — avoid full Qwen depth | te-only encode time/RAM |
| M5 | VAE decode-only path for pure T2I (skip encode weights if split) | vae-decode + peak during decode |
| M6 | Avoid retaining prompt embeds longer than needed | samples after_unload_te vs denoise |
| M7 | 3-bit preset trade quality vs RAM | `--with-quality` + peak_rss |
| M8 | Lower resolution for interactive preview | 256/384 vs 512 e2e |
| M9 | I2I: free image NCHW after encode | I2I trace samples (future mode) |
| M10 | Peak during first DiT step (activation) | denoise_step_0 memory sample |
| M11 | Process RSS vs MLX active gap (Metal heap) | both series in report |
| M12 | Resident policy only on high tiers | side-by-side staged vs resident |
| M13 | Snapshot mmap / avoid extra copies on load | load_* timings + peak during load |
| M14 | iOS: wired memory / jetsam budget | device benches (P7) |

### Speed (S)

| ID | Opportunity | How to measure |
|----|-------------|----------------|
| S1 | `MLX.compile` DiT step graph | denoise/step cold vs warm |
| S2 | Fuse eval boundaries (fewer `eval` syncs) | denoise/step |
| S3 | Ensure full metallib (no JIT thrash) | first trial cold vs second |
| S4 | SDPA / MLXFast already preferred | denoise/step vs baseline |
| S5 | RoPE precompute / cache | denoise/step |
| S6 | Parallel metadata load (not weights) | load_* only |
| S7 | Tokenize off Metal path | encode_te |
| S8 | Export PNG off critical path / lower bit depth preview | export_png |
| S9 | Avoid `item()` / host sync in hot path | denoise/step |
| S10 | Fewer steps (2–3) quality trade | e2e + `--with-quality` |
| S11 | Guidance path dead code removed | denoise/step |
| S12 | Batch = 1 fixed; no multi-image v1 | n/a |

### Process rules

1. One variable at a time (cache limit **or** compile **or** eval policy).  
2. Same machine, similar thermal (`system.thermal_state`).  
3. Report `label` must describe the change.  
4. Do not claim Tier L/M readiness without measured peak RSS.

---

## Measured results (keep updating)

**Machine:** Mac mini, Apple M2, 8 GB unified, thermal nominal  
**Build:** `swift build -c release` + `Scripts/ensure-metallib.sh`, **4-bit** weights  

### Fair A/B: 512² vs 1024² (current tree — Steel FA + cosine VAE tiles)

Protocol: **warmup 1 + trials 3**, seed 42, `--probe-density stages`, labels `fair-steel-512` / `fair-steel-1024`  
**Date:** 2026-08-11 · **Machine:** Mac mini Apple M2 8 GB · **Build:** release + full metallib · **Weights:** 4-bit

| Metric | **512²** | **1024²** | **1024 / 512** |
|--------|---------:|----------:|---------------:|
| **e2e (mean)** | **31.1 s** | **93.9 s** | **3.02×** |
| denoise total | 24.2 s | 80.9 s | 3.34× |
| denoise / step | 6.06 s | 20.2 s | 3.33× |
| encode TE | 2.15 s | 2.11 s | ~1.0× |
| decode VAE | 1.71 s | 7.93 s | 4.63× |
| **peak RSS** | **1.74 GiB** | **1.75 GiB** | **1.01×** |
| **peak MLX active** | **2.04 GiB** | **2.05 GiB** | **1.00×** |
| **peak MLX watermark** | **2.99 GiB** | **3.75 GiB** | **1.25×** |
| joint_seq_len | 1536 | 4608 | 3.0× |

**How to read RAM:** Live peak active is dominated by **DiT weights (~2 GB)** at both sizes. Watermark is higher at 1024² (activations + cosine-tile VAE accumulation) but still well under 8 GB. Prefer 512² for interactive speed.

### Current-tree 512² T2I (`hoist-512`, 2026-08-13)

SHA `7c331bb` + context-projection hoist. Protocol: warmup 1 + trials 3, seed 42, `--probe-density stages`, fox prompt. No `CONTAMINATED` tags (WindowServer/MTLCompiler treated as ambient).

| Metric | fair-steel-512 (2026-08-11) | **hoist-512** | Δ |
|--------|----------------------------:|--------------:|--:|
| e2e mean | 31.1 s | **27.5 s** | **−11.6%** |
| denoise / step | 6.06 s | **5.20 s** | **−14.2%** |
| peak MLX active | 2.04 GiB | **2.04 GiB** | 0 |
| peak MLX watermark | 2.99 GiB | **2.99 GiB** | 0 |
| peak RSS | 1.74 GiB | **1.75 GiB** | ~0 |

Trials were tight (e2e 27.44–27.53 s). This is a **cross-window** compare (tree also includes the 2026-08-11 performance pass). The hoist itself is one Linear 7680→3072 per generate instead of per step — expect a small slice of the denoise win, not the whole −14%. No same-day hoist-off A/B was run.

### Current-tree 1024² T2I (`hoist-1024`, 2026-08-13)

SHA `64dadfb`. Same protocol as `hoist-512` (W1 T3, seed 42, stages, fox prompt). All 3 trials OK, no new `.ips`.

| Metric | fair-steel-1024 (2026-08-11) | **hoist-1024** | Δ | hoist-1024 / hoist-512 |
|--------|-----------------------------:|---------------:|--:|-----------------------:|
| e2e mean | 93.9 s | **87.7 s** | **−6.7%** | **3.19×** |
| denoise / step | 20.2 s | **18.6 s** | **−7.7%** | **3.58×** |
| decode VAE | 7.93 s | **7.96 s** | ~0 | 4.75× |
| peak MLX active | 2.05 GiB | **2.05 GiB** | 0 | ~1.00× |
| peak MLX watermark | 3.75 GiB | **3.76 GiB** | ~0 | 1.26× |
| peak RSS | 1.75 GiB | **1.77 GiB** | ~0 | ~1.01× |

Trials 87.40–88.06 s. Product-default 1024² is healthy on this 8 GB mini (watermark well under 4 GiB). Identity 1024 **bench** still parked (2× image tokens); a single `i2i --identity` smoke at 1024 completed (~199 s wall, SSIM 0.739, pixel PASS).

### Identity I2I 512² (host-contention harness)

**Date:** 2026-08-13 · SHA `420bcdb` · M2 8 GB · release + full metallib · 4-bit  
**Mode:** `identity-i2i` · 512² · strength 0.9 · seed 7 · warmup 1 + trials 2 · `--with-quality`  
**Ref:** `outputs/demo/woman_t2i.png` (downscaled to 512)  
**Joint seq (true):** 2560 = 512 text + 1024 img + **1024 ref**

| Metric | Mean (n=2) |
|--------|-----------:|
| e2e | **45.3 s** |
| denoise / step | **9.45 s** |
| peak MLX active | **2.04 GiB** |
| peak MLX watermark | **3.90 GiB** |
| full-ref SSIM | 0.717 |
| face-crop SSIM | 0.522 |
| face-crop fidelity | 75.0 |
| faces detected | 1 / 1 |

**Host flags (as recorded):** both trials tagged `CONTAMINATED` because `WindowServer` ~16–17% and `MTLCompilerService` ~23% exceeded a 15% CPU rule. On this mini those processes are **ambient** (compositor + our own kernel compile) when only Ghostty + Grok are running. Later harness versions ignore them for the contaminated bit. Swap after DiT is recorded, not a run gate.

T2I 512² fair e2e was ~31 s / ~6.1 s per step; identity I2I is slower mainly because the DiT sequence is longer (ref tokens) (~9.5 s/step).

**vs prior fair A/B (chunked SDPA, hard VAE tiles):** 1024 e2e **96.0 s → 93.9 s** (−2%); denoise/step **21.0 s → 20.2 s** (−4%); watermark **3.29 → 3.75 GiB** (VAE blend path).

```bash
.build/release/imarello bench --label fair-steel-512  --width 512  --height 512  \
  --warmup 1 --trials 3 --probe-density stages --json /tmp/fair-steel-512.json
.build/release/imarello bench --label fair-steel-1024 --width 1024 --height 1024 \
  --warmup 1 --trials 3 --probe-density stages --json /tmp/fair-steel-1024.json
.build/release/imarello bench-compare /tmp/fair-steel-512.json /tmp/fair-steel-1024.json
```

### 2026-08-11 optimization pass

Phased speed/memory pass (P1 sync/cache, P2 token trim, P3 caching/session/compile
spike, P4 VAE, P5 micro-fusion). All deltas below are same-day A/Bs.

#### P1 sync/cache discipline — default path

Fresh same-day baseline (`base-583fcfb-*`, tree at P6c) vs Phase 1 candidate
(`p1-sync-*`). Protocol: W1/T3, seed 42, `--probe-density stages`, release + full metallib.
Absolute numbers run slower than the older `fair-steel-*` rows (machine state) — deltas
within the same day are the signal.

Phase 1 changes: per-block `Memory.clearCache()` gated on joint seq > 1536
(`AttentionTuning.blockCacheClearSeqThreshold` / `blockCacheClearInterval`; 512² skips
per-block clears, 1024² unchanged), dropped the duplicate QKV eval in `processQKV`
(JointAttention checkpoints after concat+RoPE), removed per-chunk `eval` in
`linearChunkedSequence` (single eval at concat), deleted the no-op
`ensureMatrixContiguous`.

| Metric | base 512² | P1 512² | Δ | base 1024² | P1 1024² | Δ |
|--------|----------:|--------:|---:|-----------:|---------:|---:|
| e2e (mean) | 35.9 s | 34.1 s | **−5.1%** | 108.5 s | 103.8 s | **−4.3%** |
| denoise / step | 7.09 s | 6.57 s | **−7.3%** | 23.3 s | 22.4 s | **−3.9%** |
| decode VAE | 1.97 s | 1.99 s | ~0 | 8.8 s | 8.6 s | −2.2% |
| peak MLX active | 2.19 GB | 2.19 GB | 0 | 2.19 GB | 2.20 GB | +0.1% |
| peak MLX watermark | 3.21 GB | 3.21 GB | 0 | 4.03 GB | 4.03 GB | 0 |
| peak RSS | 1.86 GB | 1.86 GB | 0 | 1.50 GB | 1.87 GB | +24%* |

\* 1024² baseline RSS was anomalously low (macOS compression / pool state); MLX
active + watermark — the gates that matter — are flat. P1 RSS matches the 512² profile.

Reports: `/tmp/base-512.json`, `/tmp/base-1024.json`, `/tmp/p1-512.json`, `/tmp/p1-1024.json`.

#### P4 + P5 (VAE stitch/encode-only + compiled RoPE/AdaLN) — default path

`p45-*` vs `p1-sync-*`, same W1/T3 protocol. Changes: slice-local tiled-VAE stitch with
cached cosine masks (was full-canvas pad+add ×2 per tile), `VAELoadMode.encodeOnly`
for I2I stage-0, identity `--ref-downsample`, compiled RoPE pair-mix
(`compiledRopeMix`) and compiled AdaLN (`ModulationOps.modApply`/`gateAdd`,
shapeless, shared across 25 blocks × steps).

| Metric | P1 512² | P4+5 512² | Δ | P1 1024² | P4+5 1024² | Δ |
|--------|--------:|----------:|---:|---------:|-----------:|---:|
| e2e (mean) | 34.1 s | 31.7 s | **−7.0%** | 103.8 s | 104.4 s | +0.6% (noise) |
| denoise / step | 6.57 s | 6.06 s | **−7.8%** | 22.4 s | 22.4 s | ≈ |
| decode VAE | 1.99 s | 1.78 s | −8.9% | 8.6 s | 9.0 s | +4.2% (noise) |
| peak MLX active / watermark | flat | flat | | flat | flat | |

The compiled elementwise ops pay off where attention doesn't dominate (512²); at
1024² the joint-seq attention cost swamps them. **Cumulative default-path result vs
same-day baseline: 512² e2e −11.8% (35.9 → 31.7 s), denoise/step −14.5%
(7.09 → 6.06 s); 1024² e2e −3.7% (108.5 → 104.4 s); MLX active + watermark flat.**
Quality gate: seed-42 512² smoke, pixel PASS + vision checklist clean (typical klein
two-handle quirk only). Reports: `/tmp/p45-512.json`, `/tmp/p45-1024.json`.

#### P2 `--text-tokens auto` (first-class opt-in; default stays 512)

Trims TE output + `txtIds` to the real (unpadded) token count instead of the full 512
padded window. Changes numerics vs the reference full-512 path (padding tokens
participate in attention in FLUX.2). Product default remains pad-512.

Quality close-out + recipes: [`Docs/TEXT_TOKENS.md`](TEXT_TOKENS.md) (2026-08-14).

- 512² bench (`p2-auto-512` vs `p45-512`, 12-word prompt → joint 1536 → ~1060):
  e2e **−32.4%** (31.7 → 21.4 s), denoise/step **−39.8%** (6.06 → 3.64 s),
  peak MLX active 2.19 → **2.04 GB**, watermark 3.21 → **2.99 GB**.
- 1024²: the full W1/T3 run (`p2-auto-1024`) was thermally contaminated (encode_te
  +27% on identical work after ~40 min sustained GPU load — treat back-to-back
  same-day runs with suspicion). Interleaved cold one-step A/B: auto **17.1–17.3 s/step**
  vs full512 **19.1 s/step** ≈ **−10%**, matching the seq-length ratio (4136/4608).
- **2026-08-14 quality A/B:** `T2I_EXTRA='--text-tokens auto' ./Scripts/eval-regression.sh`
  — **15/15 pixel PASS**. Vision (mug / fox / OPEN STUDIO + shop / fisherman): same
  subject and quality, texture/pose drift. Seed-7 “OPEN STUDIO” drops “OPEN” on
  **both** auto and pad-512 (Klein flake, not a trim regression). **Did not flip
  the default.**

#### P3a prompt-embed disk cache (default on)

`~/Library/Caches/Imarello/embeds/<sha256>.safetensors` keyed by
format-version | model id | TE bits | seq len | prompt (~7.9 MB per entry, f16).
Hit skips the whole TE stage (load + encode): **−4.0 s** at 512² (28.9 → 24.9 s
measured), byte-identical output PNG on same seed. `--no-embed-cache` opts out.

#### P3b warm session (`imarello session`, resident policy, ≥16 GB gate)

Repeat prompts keep modules resident; parity confirmed (same prompt+seed byte-identical
across staged-load and resident-reuse generations). On the 8 GB M2 mini with
`--force-resident`: repeat-gen 25.1 s ≈ staged (no win), and a cache-miss prompt
regressed to 35.6 s (TE encode with DiT+VAE co-resident → memory pressure) — the
16 GB RAM gate is doing its job; below it, staged + embed cache is strictly better.
Fixed along the way: `StageOrchestrator` exclusive loads are now idempotent
(second resident generation used to throw `moduleAlreadyLoaded`).

#### P3c block-level `MLX.compile` spike — NO-GO (S1 confirmed again)

`imarello dit-compile-spike`, 512² shapes, resident DiT, 6 iters: single-stream block
−3.1% vs product, double-stream −1.1%; compile first-call 649 / 209 ms. Blocks are
matmul/attention-bound; the elementwise fringes are already fused by the shipped
compiled AdaLN/RoPE helpers. Full-forward compile stays parked.

#### P5 f16 Q/K/V threshold retest (1024²)

Interleaved cold one-step A/B, f32 (`--attn-f16-threshold 99999`) vs default f16:
20.33 s vs **18.89 s** per step (f16 ~7% faster), identical peak active/watermark.
Threshold stayed 2048 until the 512² A/B below.

#### P5b f16 Q/K/V at 512² (2026-08-13)

`seq > threshold` → image seq at 512² is **1024**, so `--attn-f16-threshold 512` (not 1024) enables f16 image QKV. Same-day W1T3 staged T2I, seed 42, release, 8 GB M2:

| Label | e2e | denoise/step | peak active | watermark |
|-------|----:|-------------:|------------:|----------:|
| `f16-2048-512` (old default) | 28.18 s | 5.36 s | 2.04 GiB | 2.99 GiB |
| **`f16-512-512` (new default)** | **27.24 s (−3.3%)** | **5.13 s (−4.3%)** | 2.04 GiB | 2.99 GiB |

JSON: `/tmp/imarello-f16-2048-512.json` vs `/tmp/imarello-f16-512-512.json`.  
Eval: `T2I_EXTRA='--attn-f16-threshold 512' ./Scripts/eval-regression.sh` — **15/15 pixel PASS**. Spot vision: mug (terracotta), fox, “OPEN STUDIO” poster OK. **Shipped default `f16SeqThreshold = 512`.**

#### VAE D=512 query-chunked attention (2026-08-13)

Steel fused FA is D∈{64,80,128}; VAE mid-block is 1-head **D=512**. Ported query-chunked f32 SDPA (`VAEAttention`, default chunk 64, **`evalEachChunk = false`**). `bench --mode vae-decode` now decodes packed noise (was load-only). `--vae-attn-chunk 0` is the MLXFast fallback.

Same-day release + full metallib, 8 GB M2. 512: W1/T3. 1024: W1/T2 (XProtect remediator + swap ~1.5 GiB — treat as noisy).

| Label | decode mean | vs MLXFast | peak RSS |
|-------|------------:|-----------:|---------:|
| `vae-mlxfast-512` | 1.70 s | — | 54 MB |
| **`vae-chunk64-512` (default)** | **1.67 s** | **−1.8%** | 50 MB |
| `vae-mlxfast-1024` | 7.96 s | — | 25 MB |
| **`vae-chunk64-1024` (default)** | **8.01 s** | **+0.7%** | 23 MB |

JSON: `/tmp/imarello-vae-mlxfast-512.json` / `…-chunk64-512.json` / `…-1024.json`.  
Numeric: `IMARELLO_MLX_TESTS=1` tiny-tensor oracle, max abs err < 1e-4.  
**Ship chunked as default** — decode time is within noise; the bound is the score matrix (never S×S). Untiled 1024² encode is S=16384 (~1.1 GiB scores); chunk 64 keeps scores at `[Tq, S]`. Did **not** run T2I pixel/vision on this pass (host dirty; math is the same softmax). `--eval-cache mid` was **not** measured on this 8 GB host (refused without `--force`).

#### BFL Small Decoder (opt-in, 2026-08-14)

`--vae-variant small-decoder`. New module (`[96,192,384,384]`), CompVis→MLX remap, klein BN. **Default stays full AE.**

Same-day release, 512² `bench --mode vae-decode` W1/T3:

| Label | decode mean | vs full | peak RSS |
|-------|------------:|--------:|---------:|
| `vae-full-512` | 1.68 s | — | 57 MB |
| **`vae-small-512`** | **1.06 s** | **−37.3%** | 147 MB |

JSON: `/tmp/imarello-vae-full-512.json` / `/tmp/imarello-vae-small-512.json`. RSS is higher because the BFL file is F32 (~112 MB) vs the 4-bit-attn klein decode pack. Quality: fox + mug seed 42, pixel PASS (tech 96.4 / 85.3), vision near-identical to full AE. **Did not flip the default** (no 1024 A/B, no full eval-regression).

I2I (2026-08-15): klein `encodeOnly` + Small Decoder. Mug recolor s=0.8 seed 7: pixel PASS, SSIM 0.43, emerald glaze. Identity s=0.9 seed 7: pixel PASS, SSIM 0.73, face lock, emerald blouse; balcony scene mild (identity stack, not encode). Paths: `/tmp/imarello-smalldec-i2i/`. **No second encoder.**

### Historical snapshots (not for cross-size RAM A/B)

| Label | e2e mean | denoise/step | peak RSS | peak MLX active | peak MLX watermark | Notes |
|-------|----------|--------------|----------|-----------------|--------------------|-------|
| `baseline-rel` | 28.80 s | 5.54 s | 2.13 GB | 2.04 GB | 4.30 GB | Pre low-RAM path; 512 only |
| `opt-v3-scope` | 27.39 s | 5.24 s | 2.14 GB | 2.09 GB | 4.31 GB | Mid-series 512 |
| `probe-768-4bit` | 49.6 s (1 cold) | 10.3 s | 2.14 GB | 2.13 GB | 8.27 GB | Pre full checkpoint/tile path |
| `fair-512` / `fair-1024` | 30.6 / 96.0 s | 5.96 / 21.0 s | 1.73 / 1.77 GiB | 2.04 / 2.04 GiB | 2.99 / 3.29 GiB | Pre-Steel FA; hard VAE tiles |
| **`fair-steel-512` / `fair-steel-1024`** | **31.1 / 93.9 s** | **6.06 / 20.2 s** | **1.74 / 1.75 GiB** | **2.04 / 2.05 GiB** | **2.99 / 3.75 GiB** | **Current** (Steel FA + cosine VAE) |

**Shipped optimizations (cumulative in tree):**

| ID | Change | Effect |
|----|--------|--------|
| S9 | Remove DiT `item()` host sync; pass training-scale timesteps `[0,1000]` | Fewer GPU→CPU barriers per step |
| S2 | One `eval(latents)` per denoise step (not pred + latents) | Less sync / graph thrash |
| S5 | Precompute RoPE once per denoise session | Avoid 4-axis rope rebuild every step |
| S1 partial | `compile` Euler residual add | Small fused kernel |
| M2 | `Memory.clearCache()` after stage scopes + purge | Drop embeds/rope before VAE |
| S8 | Accelerate vDSP path for PNG float→u8 | Export ~50 ms warm (was slow mainly on cold debug) |
| TE | Drop redundant post-encode `eval` | Minor encode win |
| M2 | Cache RoPE ω, temb freqs, TE causal-512 mask | Less host/graph churn each step/encode |
| M2 | Precompute Euler `dt` arrays once per schedule | No per-step scalar `MLXArray` alloc |
| M5→M2 | **VAE decode-only load for T2I/I2I decode** | ~67 MB fewer weights; **load_vae ~115→69 ms**; peak watermark slightly lower |
| M5 tile | **VAE tiled decode** (default **overlap + cosine blend**, `VAETileConfig` tile=72/overlap=16) | Seams blended; cold 1024 decode ~8 s (was ~6.7 s hard 2×2); live VAE active still ~100 MB; e2e ~99 s cold |
| S4/M10 | **SDPA query chunk size 256→512** (`AttentionTuning`) | ~1% faster denoise/step @ 1024²; peak RAM unchanged |
| S4 FA | **Steel fused FA** (full-Q MLX SDPA; drop query-chunk for D=128) | denoise/step ~20.8→**20.2 s** @ 1024²; same peak RAM; + Imarello float4 fused Metal kernel for non-Steel D |
| Harness | Report `gpu=Apple M2 metal=Metal 4 neuralAccel=no` | Avoid confusing M5-only Metal 4 claims |
| S2 P1 | **Size-gated per-block cache clears + collapsed QKV evals + single-eval chunked Linear** | denoise/step **−7.3%** @ 512², **−3.9%** @ 1024²; watermark flat (see “2026-08-11 optimization pass”) |
| S5 P5 | Compiled RoPE pair-mix (`compiledRopeMix`, 50×/step) + compiled AdaLN `modApply`/`gateAdd` | e2e **−7.0%** @ 512² on top of P1 (with P4); flat @ 1024² (see “P4 + P5” subsection) |
| M5 P4 | VAE encode-only load for I2I stage-0 (`VAELoadMode.encodeOnly`, ~67 MB) | Decoder never resident during reference encode |
| M5 P4 | Tiled VAE stitch: slice-local accumulate + cached cosine masks (was full-canvas pad+add ×2 per tile) | decode −8.9% @ 512²; 1024² within noise (see “P4 + P5” subsection) |
| S6 P2 | `--text-tokens auto` trim (first-class opt-in; default stays 512; [`TEXT_TOKENS.md`](TEXT_TOKENS.md)) | e2e **−32%** @ 512², denoise/step ~−10% @ 1024²; watermark 3.21 → 2.99 GB @ 512²; 2026-08-14 eval 15/15 PASS |
| S7 P3a | Prompt-embed disk cache (default on; `--no-embed-cache`) | Hit skips TE stage: **−4 s** @ 512², byte-identical output |
| — P3b | `imarello session` warm mode (resident policy, ≥16 GB gate, `--force-resident`) | No win on 8 GB (gate validated); resident reuse byte-identical |

### Draw Things / PDF report mapping (2026-08-11)

External report claimed “1024 → OOM” and ranked tiled VAE as the unlock. **Grounded status in this tree:**

| PDF technique | Status | Notes |
|---------------|--------|-------|
| Tiled VAE (overlap + blend) | **Done** (default cosine) | `VAETileConfig` |
| Partial / JIT DiT **weights** | **Not done** | Activation checkpointing only; ~2 GiB DiT floor |
| Metal FlashAttention | **Done (Steel fused)** | Product path = MLX Steel simdgroup MMA FA for D=128 |
| Intermediate release / pacing | **Done** | Block eval + clearCache + cacheLimit |
| Further quant | **Out of product scope** | 4-bit lock |
| Pressure harness | **Done** | `pressure-map` / `res-ladder` / `dit-one-step` |

**Priority now:** iOS jetsam (optional DiT weight streaming) → speed (resident compile spike) — 1024 on 8 GB is green.

### Phase C — Attention knob sweep (2026-08-11, M2 8 GB)

Configurable via `AttentionTuning` / CLI: `--attn-chunk-size`, `--attn-chunk-threshold`, `--attn-f16-threshold`, `--attn-linear-chunk`, `--attn-linear-threshold`.

**Protocol A** (cold `dit-one-step`, 1 trial) — ranking only; first run cold-biased:

| Label | knobs | denoise/step |
|-------|-------|-------------:|
| baseline | q256 / t1536 / f16@2048 / lin512 | 21.52 s |
| **q512** | **chunk 512** | **20.68 s** |
| f16_1024 | f16@1024 | 20.93 s |
| t1024 | chunk threshold 1024 | 21.04 s |
| q128 | chunk 128 | 21.39 s |
| others | lin256/1024, t2048 | ~21.2–21.4 s |

**Protocol B** (warm `dit-steps` steps=1, warmup 1 + trials 2) — fair:

| Label | denoise/step mean | vs baseline |
|-------|------------------:|------------:|
| baseline / baseline_b | 21.08 / 21.12 s | — |
| **q512 / q512_b** | **20.82 / 20.84 s** | **≈ −1.1 %** |
| f16_1024 | 21.09 s | ~0 |
| combo q512+f16@1024 | 21.69 s (noisy) | worse |

Peak MLX active **2.05 GiB** and watermark **3.75 GiB** unchanged across all knobs.

**Shipped default change:** `AttentionTuning.queryChunkSize` **256 → 512**. Other defaults unchanged (threshold 1536, f16@2048, linear 512/1536).

### Custom Metal FlashAttention (fused Steel + Imarello kernels)

**Key finding (v5):** MLX’s `ScaledDotProductAttention` already dispatches **Steel Attention** — a fused Metal FA2 kernel using `simdgroup_matrix` MMA (BQ=32, BK=16/32, BD=64/80/128), online softmax, and threadgroup Q/K/V tiles. It is **not** limited to Tq=1 (that was outdated docs). Fallback to unfused matmul+softmax only when head dim ∉ {64,80,128} or training/logsumexp.

Older Imarello **query-chunking** (Tq=512) fought this path. Product default now uses **one full-Q Steel launch** for FLUX D=128.

| Item | Detail |
|------|--------|
| Product path | `AttentionUtils` → `MetalFlashAttention` → **MLX Steel fused FA** (D∈{64,80,128}) |
| Hybrid FA2 | Host-loop steel `matmul` tiles when D unsupported |
| Imarello fused Metal | `scaledDotProductAttentionFusedMetal` — float4 online softmax, BQ=BK=32 (research / non-Steel D) |
| Backend flag | `--attn-backend mlx\|metal-fa\|auto` (mlx default now = Steel for Klein) |
| Tests | 8 cases: steel D=128, fused float4, hybrid multi-tile, f16 |

**Measured (M2 8 GB, 1024², 1 denoise step warm):**

| Path | denoise/step | peak active / watermark |
|------|-------------:|-------------------------:|
| Pre-v5 **chunked** SDPA | ~20.8 s | 2.04 / 3.75 GiB |
| **v5 Steel fused (default)** | **~20.2 s** (−3%) | 2.05 / 3.75 GiB |
| Hybrid FA2 host tiles (v4) | ~26.3 s | same class |
| Pure scalar Metal FA (v0) | >10 min | — |

**Default = fused Steel.** Custom float4 Metal kernel is available for experiments; next win would be porting MFA-specific register schedules only if profiling shows Steel leaves gap on target silicon.

**Still dominant (~76% of e2e):** DiT denoise compute itself. M2 has Metal 4 **API** but **no Neural Accelerators** — further large speedups need silicon (M5) or algorithm (res/bits/steps), not custom MTL4 kernels.

**Quality gate (seed 42 fox prompt after opt-v3):** pixel overall ~95, color_match yes — no hard fails.

---

## Research deep-dive (2026-08-11)

Investigation of the three highest-effort candidates after the ~5% denoise pass. Sources: mlx-swift 0.31.6 docs (`Memory`, `compile`, `QuantizedLinear`), hub pack layout, Imarello pipeline, and **measured** `cacheLimit` sweeps on an 8 GB Mac mini.

### Weight budget (4-bit hub pack on disk)

| Module | Disk | Notes |
|--------|------|--------|
| Text encoder | ~2.1 GB | Dominates TE stage |
| DiT transformer | ~2.18 GB | `single_transformer_blocks` ~1.38 GB, `transformer_blocks` ~0.69 GB |
| VAE total | ~165 MB | Encoder **~67 MB**, decoder **~97 MB**, BN/post_quant negligible |

T2I **process peak** is still DiT weights + activations (~4.3 GB MLX watermark, ~2.1 GB RSS on this machine), not VAE.

---

### 1. Full DiT-step `MLX.compile` (S1)

#### What compile actually needs

From mlx-swift compilation docs:

- Compiles a **pure** graph of `MLXArray` ops; first call is slow (trace + fuse + Metal codegen), later calls reuse a cache keyed by shapes/dtypes.
- Mutable / captured state must go through `inputs:` / `outputs:` as `Updatable` (`Module` qualifies).
- Pattern for models:

```swift
let step = compile(inputs: [model]) { (latents: MLXArray, embeds: MLXArray, t: MLXArray, cos: MLXArray, sin: MLXArray) in
    model(hiddenStates: latents, encoderHiddenStates: embeds, timestep: t,
          imgIds: /* unused if rope passed */, txtIds: /* … */,
          guidance: nil, imageRotaryEmb: (cos, sin))
}
```

- **Shapeless** compile avoids recompile on shape change but not all graphs support it; resolution change (512→1024) otherwise forces recompile.
- Host ops inside the function (`item()`, Swift branches on array values, printing) **break** tracing. We already removed timestep `item()` syncs — a prerequisite for compile.

#### Affine quant is *not* the hard blocker

`QuantizedLinear.callAsFunction` is ordinary MLX ops:

```text
quantizedMM(x, weight, scales, biases, …) [+ bias]
```

Those are first-class graph nodes (same as training examples that compile modules with parameters). Affine mode is the default quant format; it is not a separate “non-compilable” IR. So “tricky with Affine quant” is **overstated** if it means “quant forbids compile.”

#### What *is* hard for a full Klein step

| Risk | Why it matters |
|------|----------------|
| **Graph size** | 5 double + 20 single blocks × (QKV, RoPE mix, SDPA, FFN, AdaLN) → very large compile graph |
| **First-call cost / RAM** | Compile can spike temporary memory; on **8 GB Tier L** this may jetsam or thrash before any steady-state win |
| **SDPA already fused** | `MLXFast.scaledDotProductAttention` is a single kernel; compile’s biggest wins are elementwise fusion (gelu-class). Attention-bound steps see smaller relative gains |
| **asType / reshape churn** | Many bf16↔fp32 casts and reshapes in attention/RoPE; compile may fuse some, not all Metal boundary costs |
| **API surface** | Optional `guidance`, optional rope, batch-1 only — keep signature fixed or recompile |
| **Module lifetime** | Compiled closure must outlive denoise loop; rebuild when DiT reloads after staged unload (every generation in staged mode → **recompile every T2I** unless we keep a resident compiled DiT) |

**Staged policy conflict:** default is load DiT → denoise 4 steps → unload. A compiled function is tied to a `Module` identity / parameter buffers. After unload, the next generate reloads new arrays → **new compile**. Amortization only works for:

- `MemoryPolicy.resident` (Tier H), or  
- “warm DiT” session mode (hold DiT across multiple prompts).

With **4 steps**, even a 20% step speedup saves ~4 s/gen, but a multi-second (or multi-minute) recompile each load can erase the win for single-shot CLI use.

#### Recommended experiment plan (do not ship until measured)

1. **Micro-compile** (safe): already have compiled Euler; optionally compile RoPE mix / AdaLN chunks.  
2. **Single-block compile**: wrap one `Flux2TransformerBlock` forward with `compile(inputs: [block])` and compare step time × 25.  
3. **Full forward compile** only if (2) shows ≥10% and peak RAM during compile is acceptable:  
   - Flag: `--compile-dit` on bench  
   - Warmup ≥1, trials ≥3, **resident DiT** for fair multi-step amortization  
   - Record first-call vs steady-state denoise/step  
4. **Abort criteria**: compile OOM, first-call > 2× one uncompiled generation, or quality drift (pixel gate).

#### Verdict (S1)

| Priority | Decision |
|----------|----------|
| Full DiT-step compile | **Parked for staged CLI** — reload kills amortization; graph risk on 8 GB |
| Partial / block compile | **Worth a spike** behind a flag if we add resident-DiT bench mode |
| Quant concern | **Not a fundamental ban** — use `inputs: [model]`; avoid host sync |

---

### 2. `Memory.cacheLimit` sweeps (M3)

#### API facts (mlx-swift `Memory`)

| Knob | Meaning |
|------|---------|
| `activeMemory` | Bytes in live `MLXArray`s |
| `cacheMemory` | Recyclable buffer pool (not returned to OS yet) |
| `peakMemory` | High-water of active (resettable) |
| `cacheLimit` | Cap on cache pool; excess freed on **next allocation** (not immediately) |
| `clearCache()` | Immediate pool drop |
| `memoryLimit` | Soft cap; malloc waits on tasks if exceeded (default ~1.5× Metal recommended working set) |

Docs note: unconstrained cache can grow large; many apps do fine with **small** limits (even ~2 MB class for other workloads). Limit applies on the **next** alloc unless you `clearCache()`.

#### Defaults on this 8 GB Mac mini (measured)

| Field | Value |
|-------|------:|
| Physical RAM | 8.00 GB |
| Metal recommended working set | ~5.33 GB |
| Default `cacheLimit` | **~7.60 GB** (≈ memory limit) |
| Default `memoryLimit` | **~7.60 GB** |

So “default” is already “cache may grow up to nearly all unified memory.”

#### Empirical sweep (release T2I 512² × 4, seed 42, W1/T1)

| `--cache-limit` | e2e ms | denoise/step ms | peak RSS | peak MLX active | peak MLX watermark |
|-----------------|-------:|----------------:|---------:|----------------:|-------------------:|
| 0 | 27638 | 5295 | 2.18 GB | 2.16 GB | 4.31 GB |
| 256 MiB | 27629 | 5304 | 2.18 GB | 2.16 GB | 4.31 GB |
| 1 GiB | 27434 | 5248 | 2.14 GB | 2.16 GB | 4.31 GB |
| 2 GiB | 27473 | 5256 | 2.14 GB | 2.09 GB | 4.31 GB |
| default (~8 GiB) | 27574 | 5268 | 2.13 GB | 2.09 GB | 4.31 GB |

Noise band for single-trial is ~1–2%. Differences are **within noise**.

#### Interpretation

1. **Peak watermark is active DiT + activations**, not the free-buffer pool. Capping cache cannot shrink weight tensors.  
2. Pipeline already **`clearCache()`** on module unload and after TE/DiT scopes — pool never accumulates across stages the way a long LLM session would.  
3. Per-step `eval(latents)` releases intermediate graphs; cache pressure during denoise is modest vs weights.  
4. `cacheLimit = 0` did **not** reduce `peak_mlx_peak` and was slightly slower (more alloc churn).

#### When cacheLimit *would* matter

- Multi-image **batch** or leaving modules **resident** without clear  
- Long-lived app process generating dozens of images without purge  
- iOS jetsam: lower cache reduces *idle* footprint after a run (pair with `clearCache` on background)

#### Verdict (M3)

| Priority | Decision |
|----------|----------|
| Aggressive cacheLimit for T2I speed | **No meaningful win** on current staged path |
| Default product setting | Leave MLX default **or** set ~1–2 GiB on iOS after gen + `clearCache` for jetsam hygiene |
| Harness | Keep `--cache-limit` for future resident/multi-gen scenarios |

---

### 3. VAE decode-only weight split for T2I (M5)

#### Structure today

`Flux2VAE` is one `Module`:

| Submodule | Role | ~Bytes (fp/conv pack) |
|-----------|------|------------------------:|
| `encoder` + `quant_conv` | I2I encode path | **~67 MB** |
| `decoder` + `post_quant_conv` | T2I/I2I decode | **~97 MB** |
| `bn` running stats | pack/unpack normalize | ~1 KB |

T2I only needs: **BN stats + post_quant_conv + decoder**.  
I2I needs full encode + decode (two VAE residencies already sequential).

#### What decode-only would save

| Scenario | Savings | Peak impact |
|----------|---------|-------------|
| T2I VAE stage | ~67 MB weights not loaded | Stage peak ↓ ~67 MB |
| T2I e2e peak | DiT still ~4.3 GB watermark | **≈ no change** to generation peak |
| I2I | Still load full VAE for encode; optional decode-only second load | Encode peak same; decode peak slightly lower |
| iOS jetsam after T2I | Slightly lower VAE resident set | Small but real |

Hub pack is a **single** `vae/0.safetensors`. Implementation options:

1. **Filter keys at load** (`decoder.*`, `post_quant_conv.*`, `bn.*`) into a `Flux2VAEDecoder` module — no new hub files.  
2. **Split shards offline** (maintainer tool) — faster mmap, clearer packaging.  
3. **Lazy submodules** — build full graph but only `update` decode keys (still allocates empty encoder params unless structure is split).

Option 1/3 requires a **decode-only Module structure** so encoder parameters are never allocated (filtered `update` alone still constructs empty encoder if `Flux2VAE()` builds both).

#### Implementation sketch

```text
Flux2VAEDecoder  // decoder + post_quant_conv + bn
Flux2VAEEncoder  // encoder + quant_conv (+ bn if encode path needs same stats)
VAEModule.load(mode: .decodeOnly | .encodeOnly | .full)
```

T2I: `load(.decodeOnly)` → `decodePacked`.  
I2I: encode with `.encodeOnly` or `.full`, unload, later `.decodeOnly` or `.full`.

Most of VAE wall time is **conv compute**, not the extra 67 MB load (~load_vae is already ~115 ms). Decode-only mainly helps **memory**, not speed.

#### Verdict (M5)

| Priority | Decision |
|----------|----------|
| T2I peak (macOS 8 GB) | **Low ROI** — DiT dominates |
| iOS / aggressive staging | **Medium ROI** — ~67 MB cleaner; implement when packaging iOS (P7) |
| Speed | **Negligible** vs denoise |
| Complexity | Medium (dual module + load filters + tests) |

Ship when: iOS host work starts, **or** measured VAE-stage RSS is a jetsam offender.

---

### Cross-cutting recommendation

| Bet | Expected e2e win | Expected peak win | Effort | Next action |
|-----|------------------|-------------------|--------|-------------|
| Full DiT `compile` (staged) | Uncertain; may lose on recompile | Neutral / worse at compile | High | Spike block-level compile + resident mode only |
| `cacheLimit` sweep | ~0% (measured) | ~0% peak (measured) | Low | **Done** — no product change |
| VAE decode-only | ~0% e2e | **Shipped for T2I/I2I decode** — load_vae −40%, ~67 MB | Done | Full VAE still for I2I encode |
| **Better next speed bets on M2** | | | | Lower preview res; 3-bit DiT; profile SDPA vs FFN; keep metallib full |

**Denoise remains the only large absolute time pool (~21 s).** Further speed work should either reduce DiT math (resolution/steps/bits) or successfully amortize a **resident** compiled forward—not chase VAE/cache micro-wins on staged T2I.

---

## Pressure probes (finding the 1024 cliff)

Report schema **1.1** includes a `pressure` object:

- `analytic` — `image_seq_len`, `joint_seq_len` (text 512 + packed H·W)
- `timeline` — every memory sample with Δactive vs previous
- `ranked_by_active` / `ranked_by_delta` — top pressure labels
- `phase_peaks` — max active in te / dit / vae
- `last_probe_before_failure` — last successful probe id if a trial errors

### Probe density

| Density | What is sampled |
|---------|-----------------|
| `off` | Nothing (product path) |
| `stages` | Load/unload/encode/export |
| `denoise` | + every denoise step begin/euler |
| `blocks` | + all 5 double blocks, singles 0/5/10/15/19, TE layer samples, embed/concat/proj |
| `max` | + every single block + every TE layer (slow) |

Block probes call `eval` so `Memory.activeMemory` reflects that region (**diagnosis only** — do not compare absolute step times to `density=off`).

### Example (512² pressure-map on 8 GB M2)

| Finding | Implication |
|---------|-------------|
| Largest **Δ** = `after_load_dit` (~+2.0 GB) | Weight residency fixed cost |
| Peak **active** during forward ≈ 2.08 GB (ws ~39%) | Live activations + weights co-resident |
| Peak **watermark** ≈ 4.25 GB | Temps + cache beyond live active |
| `phase_peaks`: te≈1.7 GB, dit≈2.1 GB, vae≈97 MB | Staged policy working; DiT is generation peak |
| joint_seq 512→1024: 1536 → **4608** (3×) | Expect superlinear attn temps → OOM on 8 GB |

Recipe: run `pressure-map` at 512 and 768, then `dit-one-step` / `res-ladder` toward 1024; use `last_probe` + `top_delta` to pick activation vs weight tactics.

### 1024² on 8 GB M2 — resolved (4-bit)

Probes showed **DiT completed** (progress `denoising step=4/4`) and OOM was in **VAE decode**. Fixes shipped:

1. **DiT activation checkpointing** — `eval` after each double/single block (stops full-graph peak ~8 GB watermark).  
2. **Chunked long-seq attention** — query-chunked SDPA + chunked Linear for fused QKV/MLP when L>1536.  
3. **f16 Q/K/V** when seq > 512.  
4. **Tiled VAE decode** — 2×2 latent tiles for unpatchified spatial ≥96 (1024² and large 768²).

**Measured (release, seed 42, 2 trials, 8 GB M2):** e2e **~96.5 s**, denoise/step **~21 s**, decode **~6.8 s**, peak RSS **~1.8 GB**, peak MLX active **~2.05 GB**, peak watermark **~3.3 GB**.

---

## Metal 4 (and what Imarello should / should not do)

Metal 4 (macOS 26 / iOS 26 era) adds first-class **ML tensors**, **quantized tensor formats + scales**, **Metal Performance Primitives (MPP) / TensorOps**, tighter ML↔graphics encoding, and (on **M5** and successors) **GPU Neural Accelerators** for high-throughput matmul. This host already reports **Metal Support: Metal 4** (e.g. M2 on macOS 26); that is **API generation**, not “has Neural Accelerators.”

### How Imarello touches Metal today

| Layer | Role |
|-------|------|
| **Imarello** | Swift graph: TE / DiT / VAE, staged residency, no hand-written MTLCommandBuffers |
| **mlx-swift (pinned)** | Array runtime, `QuantizedLinear` / `quantizedMM`, `MLXFast.scaledDotProductAttention`, `compile` |
| **Cmlx / metallib** | Prebuilt Metal kernels (`Scripts/ensure-metallib.sh` → ~130 MB full lib, not the 3 KB stub) |
| **GPU** | Executes fused matmul / SDPA / conv; unified memory |

There is **no separate “Metal 3 code path” in Imarello** to port. Optimization for Metal 4 is almost entirely **“ride a Metal‑4-aware MLX on a capable OS/GPU”**, not rewrite the DiT in raw `MTL4*` APIs.

### What Metal 4 could mean for FLUX.2-klein (in principle)

| Feature | Relevance to Imarello | Who owns it |
|---------|----------------------|-------------|
| Quantized tensor formats + scales | Matches Affine 4-bit DiT/TE packs; better native quant kernels | **MLX** backend |
| MPP / TensorOps | Faster matmul / fused ML ops under the hood | **MLX** |
| GPU Neural Accelerators (M5+) | Large win on **compute-bound** phases (DiT step ≈ huge GEMMs); Apple published **>3.8×** FLUX-dev-4bit 1024² M5 vs M4 with MLX | **Hardware + MLX** (macOS **≥ 26.2** for M5 accel) |
| Inline ML in shaders / graphics ML encoder | Neural rendering / games — **not** our staged diffusion CLI | Out of scope |
| MetalFX neural upscale | Could upscale 512→display cheaply as **UX** later; not a DiT replacement | Optional product later |
| Faster shader compile / encoding | Helps cold start if MLX JIT uses new pipelines | **MLX** + full metallib |

Apple’s ML research note: MLX uses TensorOps + Metal Performance Primitives for Neural Accelerators; on M5, **TTFT-style compute** benefits most; bandwidth-bound work scales closer to memory BW (~+20–30% M5 vs M4 in their LLM decode numbers). A DiT denoise step is **much closer to compute-bound matmul** than to autoregressive decode—so M5-class silicon is the hardware bet for “Metal 4 era” speed, not hand-tuned Imarello shaders.

### What we should *not* do

1. **Rewrite DiT/VAE as custom Metal 4 kernels in-repo** — duplicates MLX, breaks quant parity, huge maintenance, fights staged `actor` design.  
2. **Assume Metal 4 on the OS == free 2× on M2** — API support ≠ Neural Accelerators.  
3. **Bypass MLX for MPSGraph/Core ML “because Metal 4”** — different weight format, loses community 4-bit packs and Swift porting path.  
4. **Treat MetalFX as generation quality** — upscalers don’t replace 4-step distilled denoise math.

### What we *should* do (actionable)

| Action | Why | Effort |
|--------|-----|--------|
| **Keep full metallib** (`ensure-metallib.sh`) | Avoids JIT thrash / stub kernels; cold vs warm already a measured issue | Ongoing |
| **Prefer `MLXFast` / fused ops** (already SDPA) | Uses best available Metal kernels MLX ships | Already |
| **Stay on a deliberate mlx-swift pin; re-validate newer releases** | Metal 4 / M5 TensorOps land in **MLX**, not Imarello | Medium (when Apple bumps Cmlx) |
| **Record GPU chip + OS in bench `SystemSnapshot`** | Separate “M2 Metal 4 API” vs “M5 Neural Accel” baselines | Small harness tweak |
| **Optional: lower-res generate + MetalFX/display upscale** | Latency UX on Tier L; not a substitute for DiT opts | Product later |
| **Profile with Metal capture** (`MTL_CAPTURE_ENABLED`, MLX metal debugger) | See if denoise is ALU-bound vs bandwidth on this GPU | Occasional |

### Expected outcomes by hardware class

| Hardware | Metal 4 API | Neural Accelerators | Realistic Imarello lever |
|----------|:-----------:|:-------------------:|-------------------------|
| M1–M4 (this Mini: **M2**) | Yes (on macOS 26) | No | Software: eval/RoPE/stage opts (done); MLX upgrades; res/bits |
| **M5 / M5 Pro / Max** | Yes | Yes | **Same Imarello binary**, newer MLX + OS → large DiT speedup without Imarello Metal rewrites |
| iOS 26 + A19-class | Yes | Per SoC | Same staged core; metallib packaging (P7) |

### Verdict

**Optimizing “for Metal 4” in Imarello ≠ writing Metal 4 code.** It means:

1. Stay on the **MLX Metal backend** that adopts MPP / quant tensors / Neural Accelerators.  
2. Keep graphs **compile- and quant-friendly** (no host `item()` in hot paths — already fixed).  
3. Treat **M5+** as the big free win for denoise when available; on **M2**, Metal 4 is already the runtime—further gains are algorithmic / residency / model size, not a new Metal dialect.

Do **not** start a parallel custom-Metal DiT. Do **track mlx-swift upgrades** and add GPU model to bench labels when comparing machines.

---

## Recommended baseline protocol

```bash
# Cool machine, AC power, close other GPU apps
swift build -c release && ./Scripts/ensure-metallib.sh

.build/release/imarello bench \
  --label "baseline-release-512-s4" \
  --mode t2i --width 512 --height 512 --steps 4 \
  --warmup 1 --trials 5 --seed 42 \
  --json ~/Desktop/imarello-baseline.json

# Optional quality-coupled run
.build/release/imarello bench \
  --label "baseline-quality" \
  --trials 2 --with-quality \
  --json ~/Desktop/imarello-baseline-quality.json
```

After an optimization:

```bash
.build/release/imarello bench --label "opt-cache-2g" --cache-limit 2147483648 \
  --json ~/Desktop/imarello-opt.json
.build/release/imarello bench-compare ~/Desktop/imarello-baseline.json ~/Desktop/imarello-opt.json
```

---

## Modes detail

| Mode | What runs | Primary metrics |
|------|-----------|-----------------|
| `t2i` | Full staged generate + PNG | e2e, all stages, denoise/step, peaks |
| `dit-steps` | Same as t2i (focus report on denoise) | denoise/step |
| `te-only` | load TE + encodePrompt | load_te, encode_te, RSS |
| `vae-decode` | decode packed noise (decode-only VAE) | `decode_vae`, peaks |
| `load-only` | sequential TE, DiT, VAE load counts | load_* |
| `mem-stages` | orchestrator memory self-test | RSS samples per stage |

---

## Library API

```swift
import ImarelloBench
import ImarelloRuntime

let pipeline = ImarelloPipeline()
let config = BenchConfig(mode: .t2i, label: "api", trials: 3, warmup: 1)
let report = try await BenchRunner(pipeline: pipeline, config: config).run()
print(BenchReportWriter.textSummary(report))
try BenchReportWriter.write(report, to: url)
```

---

## Out of scope for this harness

- Instruments GPU counters (use Instruments / `xcprof` separately)  
- Multi-host CI leaderboards  
- Automatic opt application  
- Quality gates that block CI without snapshots (optional later)

See **ROADMAP P9** for tracking.
