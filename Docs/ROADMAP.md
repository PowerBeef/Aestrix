# Aestrix roadmap

**Last updated:** 2026-08-13 (f16 QKV threshold 512; 512² −4.3% denoise/step)  
**Working tree focus:** macOS library + CLI is the shipping surface for now.  
**Remaining work** is parked here so agents and humans can resume without rediscovering context.  
**Experimental Cursor tree:** branch `cursor-opt-quarantine` — audit before merge. See [`Docs/HOST_SAFETY.md`](HOST_SAFETY.md).

---

## Done (v1 macOS core)

| ID | Item | Where / notes |
|----|------|----------------|
| P0 | Scaffold, SPM packages, staged memory policy, BFL skills | `Package.swift`, `Docs/MEMORY.md`, `.grok/skills/flux-best-practices/` |
| P1 | Pure math: RoPE, temb, modulation, flow-match scheduler | `AestrixCore` |
| P2 | MMDiT + 4-bit load | `AestrixDiT`, `aestrix load-dit` |
| P3 | VAE encode/decode + pack/BN path | `AestrixVAE`, `aestrix load-vae` |
| P4 | Qwen3 TE, chat template, BPE, 9/18/27 tap | `AestrixText`, `aestrix load-te` / `encode-prompt` |
| P5 | Staged T2I | `AestrixPipeline.generate`, `aestrix t2i` |
| P6 | Strength I2I (full N-step strength schedule) | `AestrixPipeline.edit`, `aestrix i2i` |
| P6b | Pixel + vision eval workflow | `AestrixEval`, `Docs/EVAL_WORKFLOW.md`, `--analyze` |
| P6c | Tier-B identity I2I | Ref latents (t=10), face mask, clean-pull, schedule curves; `--identity` |

**Resume baseline (macOS):**

```bash
swift build && ./Scripts/ensure-metallib.sh
# Snapshot: ~/Library/Caches/Aestrix/models/mlx-community--FLUX.2-Klein-4B-4bit
.build/debug/aestrix t2i "…" --output /tmp/out.png --analyze --vision-brief
```

---

## Parked — resume later

Status legend: `parked` = not started · `partial` = some code/docs · `blocked` = needs decision

### P7 — iOS host (next major phase)

| Field | Detail |
|-------|--------|
| **Status** | `parked` |
| **Goal** | Ship an iOS 26 app (or host) that links `AestrixRuntime` and runs staged T2I (+ I2I) on-device |
| **Depends on** | macOS path stable (done); full metallib packaging for app targets; device memory tiers |

**Acceptance criteria**

- [ ] Xcode app target (or multiplatform package consumer) links `AestrixRuntime` / MLX  
- [ ] Metallib / MLX resources packaged for iOS (not only SPM CLI bootstrap)  
- [ ] On-device snapshot path (app container / shared group) documented  
- [ ] Single 512² T2I smoke on a physical device or iOS simulator (if Metal allows)  
- [ ] Tier-aware max side / memory policy (`DeviceTier`, `MemoryPolicy.staged`)  
- [ ] Basic UI: prompt, generate, progress, save/share  
- [ ] Optional: I2I with photo picker + strength slider  
- [ ] Eval notes for device outputs (pixel harness runs on host Mac from exported PNG if needed)

**Resume notes**

- Reuse `AestrixPipeline` actor; do not reimplement TE/DiT/VAE in the app.  
- Product locks still apply: Klein 4B only, prequant only, staged default.  
- Start with Tier L (512², 4-step) before higher resolutions.  
- See `Docs/MEMORY.md` for peak budgets; `Docs/WEIGHTS.md` for pack layout.

**Suggested first PR**

1. Empty iOS app shell + SPM dependency on local `Aestrix` packages  
2. Wire metallib / MLX init parity with `MLXBootstrap`  
3. Call `AestrixPipeline.generate` from a button with hardcoded prompt  

---

### P8 — macOS polish / regression (optional before or after iOS)

| Field | Detail |
|-------|--------|
| **Status** | `partial` |
| **Goal** | Harden macOS CLI for regressions and measured memory |

**Backlog**

- [x] Scripted regression from `Docs/eval-prompts.md` with fixed seeds (42, 0, 7) — `Scripts/eval-regression.sh` (512² T2I)  
- [x] Performance harness (`aestrix bench` / `bench-compare`) — see **P9** / `Docs/PERF.md`  
- [x] Pin Hub `revision` SHA in config / docs (`WeightPreset.pin`, `Docs/hub-pins.json`, `aestrix info`)  
- [x] Golden metric floors in CI (pixel-only; `.github/workflows/eval-floors.yml` → `Scripts/ci-eval-floors.sh`; no image goldens; vision stays agent/human)  
- [ ] Document known I2I strength curves for color vs structure edits  

---

### P9 — Performance: measure then optimize

| Field | Detail |
|-------|--------|
| **Status** | `partial` — harness shipped; optimizations data-gated |
| **Goal** | Push generation speed and lower peak RAM without breaking quality or product locks |
| **Docs** | `Docs/PERF.md` |
| **Code** | `AestrixBench`, `PipelineTrace`, `aestrix bench`, `aestrix bench-compare` |

**Harness (done)**

- [x] Stage timings (TE / DiT / VAE load, encode, denoise steps, decode, export)  
- [x] RSS + MLX active / cache / peak samples  
- [x] Multi-trial aggregate (mean, stdev, p50, p95)  
- [x] JSON report + compare  
- [x] Modes: t2i, load-only, mem-stages, te-only, dit-steps, vae-decode  
- [x] Optional `--with-quality` pixel coupling  
- [x] Unit tests for aggregator / JSON  

**Optimization backlog (do not land without bench deltas)**

- [x] Baseline release report on primary Mac (`baseline-rel`, 512² s4, W1 T3)  
- [x] Reduce host sync / `eval` in denoise (S2, S9) — **~5% denoise/step**  
- [x] RoPE precompute across steps (S5)  
- [x] `clearCache` + stage scoping before VAE (M2)  
- [x] Compiled Euler residual (S1 partial)  
- [x] Faster PNG export path (S8)  
- [x] `Memory.cacheLimit` sweep (M3) — **no meaningful e2e/peak win** on staged T2I (see PERF research)  
- [x] Research full DiT-step `MLX.compile` (S1) — **Affine quant not the blocker**; staged unload prevents amortization; park full compile  
- [x] Research VAE decode-only — **shipped for T2I/I2I decode** (`VAELoadMode.decodeOnly`, ~67 MB less, load_vae ~−40%)  
- [x] M2-supported software opts: RoPE ω / temb freqs / TE causal cache, Euler `dt` precompute, GPU label in bench  
- [x] Pressure-map / res-ladder / dit-one-step harness  
- [x] 1024² low-RAM path (checkpointed DiT + chunked then **Steel FA** + decode-only + tiled VAE)  
- [x] VAE overlap + cosine blend tiles (`VAETileConfig`)  
- [x] AttentionTuning + SDPA query chunk sweep; **Steel fused FA** as product path for D=128  
- [x] Block-level compile spike under **resident DiT** (`aestrix dit-compile-spike`) — **NO-GO**: −3.1% single / −1.1% double block; full compile stays parked  
- [x] 2026-08-11 pass P1: size-gated per-block `clearCache` + collapsed QKV evals + single-eval chunked Linear — denoise/step **−7.3%** @ 512², **−3.9%** @ 1024²  
- [x] 2026-08-11 pass P4+P5: slice-local tiled-VAE stitch + cached masks, `VAELoadMode.encodeOnly` (I2I stage-0), identity `--ref-downsample`, compiled RoPE/AdaLN — cumulative **−11.8% e2e @ 512²**, **−3.7% @ 1024²**, watermark flat  
- [x] `--text-tokens auto` trim (opt-in, experimental; numerics differ from full-512) — e2e **−32%** @ 512², ~−10% denoise/step @ 1024²; eval gate seeds 42/0/7 pass  
- [x] Prompt-embed disk cache (default on) — TE stage skipped on hit (**−4 s** @ 512², byte-identical)  
- [x] `aestrix session` warm mode (resident policy, ≥16 GB RAM gate) — no win on 8 GB (gate validated); orchestrator loads made idempotent  
- [x] f16 Q/K/V threshold retest @ 1024² — f16 ~7% faster, same peaks  
- [x] f16 Q/K/V @ 512² (`f16SeqThreshold` **2048 → 512**) — denoise/step **−4.3%**, e2e **−3.3%**, watermark flat; eval-regression 15/15 pixel PASS |  
- [ ] DiT **weight** streaming for iOS jetsam (`stagedAggressive`)  
- [ ] Cold vs warm metallib / first-step cost (S3) documented separately  
- [ ] Document accepted quality trade for 2–3 steps if used (S10)  
- [x] Research: M2 compute dtype — **activations fp32**, only quant scales are bf16 (`load-dit --dump-dtypes`, 2026-08-13)  
- [x] Research: Klein AdaLN/modulation size vs DT split — **~4% of DiT**, shared (not per-block); unload not worth it |  

**Latest default-path A/B (8 GB Mac mini, release, W1T3, 2026-08-13 `hoist-*`):** 512² e2e **~27.5 s** / 1024² e2e **~87.7 s**; denoise/step **~5.20 s** / **~18.6 s**; peak active **~2.04 / 2.05 GiB**; watermark **~2.99 / 3.76 GiB**. Full tables: `Docs/PERF.md`.

**Acceptance for any “faster / leaner” claim**

1. `bench-compare` shows improvement on e2e and/or peak_rss  
2. denoise/step not regressed beyond noise without explanation  
3. Optional: `--with-quality` no hard pixel fails on fixed seed  

---

### Out of v1 scope (track only — do not start without product decision)

| Item | Why deferred | Notes if resumed |
|------|--------------|------------------|
| Klein 9B / FLUX.2 Dev | Non-Commercial / size | Different product locks |
| Multi-reference edit (>1 image) | Scope | Single-ref latents shipped (P6c); multi-ref still higher peak RAM |
| CFG / negative prompts | Distilled klein defaults | Not used on 4-step path |
| LoRA training / load | Scope | Separate product surface |
| User-facing bf16 | Product lock | Maintainer-only tools only |
| Prompt upsampling | Product lock | Optional later UX |
| VLM-in-CI vision review | Infra cost | Pixel gate first |

---

## How to resume (agents)

1. Read this file + `AGENTS.md` product locks.  
2. Pick the highest-priority `parked` ID (default **P7**).  
3. Confirm macOS smoke still works (`t2i` + `EVAL_WORKFLOW.md`).  
4. Open a focused branch; do not expand into “Out of v1” without an explicit ask.  
5. Update this roadmap (checkboxes + “Last updated”) when an item lands or is cancelled.

---

## Decision log (short)

| Date | Decision |
|------|----------|
| 2026-08 | Pre-quant only; default 4-bit; no user bf16 |
| 2026-08 | Staged TE→DiT→VAE is default memory policy |
| 2026-08 | Klein 4B only (Apache-2.0) |
| 2026-08 | I2I uses full N-step strength schedule (not truncated T2I slice) |
| 2026-08 | Gen quality = pixel harness + vision checklist (`EVAL_WORKFLOW.md`) |
| 2026-08 | Tier-B identity I2I: single-ref latents + face mask + clean-pull + schedule curves |
| 2026-08 | Park remaining work; macOS CLI is current focus surface |
| 2026-08-13 | 8 GB host: WindowServer watchdog from Cursor+Metal compile; quarantine Cursor opts; HostPreflight |
| 2026-08-13 | Promote safe Wave-0 slice: HostContention + i2i/identity-i2i bench + face-region SSIM |
| 2026-08-13 | Pin mlx-community Klein 4-bit Hub revision `1cebb9b45c21ece14a42615b16bf5fa4de9b56da`; CI eval floors on GitHub Actions |
