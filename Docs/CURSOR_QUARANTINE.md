# `cursor-opt-quarantine` audit

**This file is a historical audit, not a work queue.** Leftovers 1–4 below shipped on `main` (VAE D=512 chunked SDPA with `evalEachChunk` off, Steel metallib check, `EvalCachePolicy.mid` bench-only, PromptEmbedCache / TextTokenMode tests). `--text-tokens auto` is the **product default**. Do not re-open the branch.

**Date:** 2026-08-13  
**Branch:** `cursor-opt-quarantine` @ `2ce2a8f` — **deleted** after the worth-doing items were reimplemented on `main`.  
**Merge-base with `main`:** `6c8b13c` (README I2I mug swap). Do **not** recreate or merge that tree. Retrieve the research note with `git show 2ce2a8f:Docs/OPTIMIZATION_RESEARCH_2026.md` while the object remains.

Isolated after WindowServer watchdog panics (Cursor + SwiftPM/Metal on 8 GB). See [`HOST_SAFETY.md`](HOST_SAFETY.md).

## What the branch is

One WIP commit (`2ce2a8f`, +3253/−230, 43 files) on top of `6c8b13c`. It is **Wave-0 research + experimental runtime**, not a reviewable PR.

Unique files (not on `main`):

| File | Role |
|------|------|
| `Docs/OPTIMIZATION_RESEARCH_2026.md` | 1244-line research note (useful) |
| `EvalCachePolicy.swift` | Process-wide eval/`clearCache` profiles including **`.high`** |
| `HardwareCapabilities.swift` | NAX eligibility (M5/A19 + OS 26.2+) |
| `MetallibArtifact.swift` | Steel vs NAX symbol scan of `mlx.metallib` |
| `VAEAttention.swift` | Query-chunked D=512 SDPA; default **`evalEachChunk = true`** |

Plus diffs into VAE decode, pipeline, bench, tokenizer, `ensure-metallib.sh`.

## Already on `main` (do not re-do)

The research doc’s Wave-0 “evidence lock” is mostly **already shipped** after quarantine:

| Quarantine idea | `main` now |
|-----------------|------------|
| Context projection hoist | Done (`hoist-512` / `hoist-1024`) |
| I2I / identity-i2i bench | Done |
| Host contention / ambient filter | Done |
| Face-region SSIM | Done |
| HostPreflight lock | Done |
| `--text-tokens auto` | Done (**product default** as of 2026-08-15) |
| Prompt-embed cache | Done (default on) |
| `imarello session` | Done (≥16 GB gate) |
| Steel FA D=128 | Done |
| f16 QKV @ 512² | Done (threshold 512) |
| P8 polish | Done |

Merging `2ce2a8f` would **rewind** those files to the `6c8b13c` shape.

## Verdicts

### Do not merge / do not port as written

| Item | Why |
|------|-----|
| **Whole branch** | Diverged; would clobber `main`. |
| **`EvalCachePolicy.high`** | Relaxes per-step `clearCache` and the 768-side clamp. Forbidden in `AGENTS.md`. This is the class of change that can grow the MLX watermark on 8 GB. |
| **VAE `evalEachChunk = true`** | Extra `eval` on every D=512 query chunk. Extra GPU sync on the VAE path that already produced `imarello-*.ips` Metal aborts. |
| **Auto-select cache policy from `DeviceTier`** | The quarantine type itself says never do this. Keep product constants. |
| TeaCache / TaylorSeer / FORA / ToCa as default Klein | Their own note: 4-step Klein has almost no safe skip. |
| FlowUpscaler / iRDM / Core ML backend / TensorOps W8A8 | New model or new stack. Out of v1. |
| NAX / custom Metal on this **M2 8 GB** | No Neural Accelerators. `neuralAccel=no`. |

### Worth pursuing (2026-08-13) — **all shipped on `main`**

Do not treat this section as a queue.

| # | Item | `main` now |
|---|------|------------|
| 1 | VAE D=512 query-chunked attention, `evalEachChunk = false` | `VAEAttention` + `VAEAttentionTests` |
| 2 | `imarello info` Steel metallib symbol check | `MetallibVerification` |
| 3 | `EvalCachePolicy.mid` as a ≥16 GB **bench flag only** | `--eval-cache mid`; never `.high` |
| 4 | PromptEmbedCache + TextTokenMode tests | `PromptEmbedCacheKeyTests`, `TextTokenModeTests` |

### Park until the right hardware / product decision

| Item | Wait for |
|------|----------|
| `HardwareCapabilities` + NAX dispatch | Physical M5/A19 + OS 26.2+ |
| Face-aware identity token keep vs pool | After VAE D=512; identity 512 bench first |
| Untiled VAE on high RAM | 16 GB+ host |
| MetalFX preview | P7 UI, not final 1024² |
| iRDM one-step student / FlowUpscaler | Explicit new-model decision |

## Status on `main` (2026-08-13)

Reimplemented from the ideas above — **not** cherry-picked from `2ce2a8f`.

| Item | Status |
|------|--------|
| Metallib Steel check | Done — `MetallibVerification` + `imarello info` + `Scripts/ensure-metallib.sh` fail-closed on stub / missing Steel |
| VAE D=512 chunked SDPA | Done — `VAEAttention`, `evalEachChunk = false`, `bench --mode vae-decode` actually decodes, `--vae-attn-chunk` A/B |
| Hygiene tests | Done — `PromptEmbedCacheKeyTests`, `TextTokenModeTests` |
| `EvalCachePolicy.mid` | Done as **bench flag only**. `--eval-cache mid` refused on `DeviceTier.low` unless `--force`. `named("high") == nil`. Product generate path stays on `product`. |
| `evalEachChunk = true` / `.high` | Still forbidden |

Branch deleted after this port. Park NAX / HardwareCapabilities / TeaCache / FlowUpscaler.

The research write-up on `2ce2a8f` is the valuable artifact. The runtime diffs were obsolete or unsafe on 8 GB.
