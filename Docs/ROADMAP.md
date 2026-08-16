# Imarello roadmap

**Last updated:** 2026-08-16 (P7 512² T2I + device harness; 1024² anatomy still open)  
**Working tree focus:** macOS library + CLI is the shipping surface. **Next product phase is P7 iOS.**  
**Backend / P9 leftovers are paused** — do not start TAEF2, ref-KV, Δ-DiT, `stagedAggressive`, or fused qmm+SwiGLU unless the user asks.  
**Agent workflow:** [`Docs/AGENT_WORKFLOW.md`](AGENT_WORKFLOW.md).  
**Experimental Cursor tree:** `cursor-opt-quarantine` **deleted** after the leftovers were ported. Audit: [`Docs/CURSOR_QUARANTINE.md`](CURSOR_QUARANTINE.md).

---

## Done (v1 macOS core)

| ID | Item | Where / notes |
|----|------|----------------|
| P0 | Scaffold, SPM packages, staged memory policy, BFL skills | `Package.swift`, `Docs/MEMORY.md`, `.claude/skills/flux-best-practices/` |
| P1 | Pure math: RoPE, temb, modulation, flow-match scheduler | `ImarelloCore` |
| P2 | MMDiT + 4-bit load | `ImarelloDiT`, `imarello load-dit` |
| P3 | VAE encode/decode + pack/BN path | `ImarelloVAE`, `imarello load-vae` |
| P4 | Qwen3 TE, chat template, BPE, 9/18/27 tap | `ImarelloText`, `imarello load-te` / `encode-prompt` |
| P5 | Staged T2I | `ImarelloPipeline.generate`, `imarello t2i` |
| P6 | Strength I2I (full N-step strength schedule) | `ImarelloPipeline.edit`, `imarello i2i` |
| P6b | Pixel + vision eval workflow | `ImarelloEval`, `Docs/EVAL_WORKFLOW.md`, `--analyze` |
| P6c | Tier-B identity I2I | Ref latents (t=10), face mask, clean-pull, schedule curves; `--identity` |

**Resume baseline (macOS):**

```bash
swift build -c release && ./Scripts/ensure-metallib.sh
# Snapshot: ~/Library/Caches/Imarello/models/mlx-community--FLUX.2-Klein-4B-4bit
# Small Decoder: ~/Library/Caches/Imarello/models/black-forest-labs--FLUX.2-small-decoder
.build/release/imarello t2i "…" --output /tmp/out.png --analyze --vision-brief
```

---

## Parked — resume later

Status legend: `parked` = not started · `partial` = some code/docs · `blocked` = needs decision

### P7 — iOS host (next major phase)

| Field | Detail |
|-------|--------|
| **Status** | `partial` — iOS 26 app runs staged Klein 4B on a physical iPhone. Simulator = UI only (no MLX). **512² T2I smoke passed** (fox, seed 42). Device harness can drop jobs without a UI tap. **1024² T2I** can fail vision anatomy (Klein 4-step / μ=1.15). Last-in-app I2I not eval’d |
| **Goal** | Ship an iOS 26 app (or host) that links `ImarelloRuntime` and runs staged T2I (+ I2I) on-device |
| **Depends on** | macOS path stable (done); full metallib packaging for app targets; device memory tiers |

**Acceptance criteria**

- [x] Xcode app target (`Apps/ImarelloIOS`) links `ImarelloRuntime`  
- [x] Metallib check via `MetallibVerification.resolveFromBundles()` (Xcode + mlx-swift resources)  
- [x] On-device snapshot path documented (`Docs/IOS.md`, app `Caches/Imarello/models`)  
- [x] Debug `iphoneos` build + `devicectl` install/launch on a physical iPhone  
- [x] Single 512² T2I smoke on that iPhone (fox, seed 42, 2026-08-15)  
- [x] Tier-aware config (`ImarelloConfig.autoDetectingTier()`, staged) — 512 / 1024 UI  
- [x] Basic UI: prompt, generate, progress, share/save  
- [x] In-app I2I of last result (no photo picker / import)  
- [x] Eval notes for device 512² T2I (`/tmp/imarello-ios-eval/t2i-1786850093.png`; pixel PASS, vision pass; `color_mismatch` waived)  
- [x] Device generate harness (`Scripts/ios-device-harness.sh` + `DeviceHarnessJob`)  
- [ ] 1024² T2I vision-clean on device (anatomy; same seed ≠ 512 upscale)  
- [ ] Last-in-app I2I eval on device

**Resume notes**

- Reuse `ImarelloPipeline` actor; do not reimplement TE/DiT/VAE in the app.  
- Product locks still apply: Klein 4B only, prequant only, staged default.  
- Start with Tier L (512², 4-step) before higher resolutions.  
- See `Docs/MEMORY.md` for peak budgets; `Docs/WEIGHTS.md` for pack layout.

**Suggested next step**

512² T2I works. Drive device generates with `Scripts/ios-device-harness.sh` (resync weights after install; new job `--id` on retry). 1024² quality is an open vision issue. Do not generate on the Simulator. Recipe: [`IOS.md`](IOS.md).  

---

### P8 — macOS polish / regression (optional before or after iOS)

| Field | Detail |
|-------|--------|
| **Status** | `done` |
| **Goal** | Harden macOS CLI for regressions and measured memory |

**Backlog**

- [x] Scripted regression from `Docs/eval-prompts.md` with fixed seeds (42, 0, 7) — `Scripts/eval-regression.sh` (512² T2I)  
- [x] Performance harness (`imarello bench` / `bench-compare`) — see **P9** / `Docs/PERF.md`  
- [x] Pin Hub `revision` SHA in config / docs (`WeightPreset.pin`, `Docs/hub-pins.json`, `imarello info`)  
- [x] Golden metric floors in CI (pixel-only; `.github/workflows/eval-floors.yml` → `Scripts/ci-eval-floors.sh`; no image goldens; vision stays agent/human)  
- [x] Document known I2I strength curves for color vs structure edits — [`Docs/I2I_STRENGTH.md`](I2I_STRENGTH.md) |  

---

### P9 — Performance: measure then optimize

| Field | Detail |
|-------|--------|
| **Status** | `done` — harness + product-path opts shipped; leftover slices **paused** (not an open queue) |
| **Goal** | Push generation speed and lower peak RAM without breaking quality or product locks |
| **Docs** | `Docs/PERF.md` |
| **Code** | `ImarelloBench`, `PipelineTrace`, `imarello bench`, `imarello bench-compare` |

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
- [x] Block-level compile spike under **resident DiT** (`imarello dit-compile-spike`) — **NO-GO**: −3.1% single / −1.1% double block; full compile stays parked  
- [x] 2026-08-11 pass P1: size-gated per-block `clearCache` + collapsed QKV evals + single-eval chunked Linear — denoise/step **−7.3%** @ 512², **−3.9%** @ 1024²  
- [x] 2026-08-11 pass P4+P5: slice-local tiled-VAE stitch + cached masks, `VAELoadMode.encodeOnly` (I2I stage-0), identity `--ref-downsample`, compiled RoPE/AdaLN — cumulative **−11.8% e2e @ 512²**, **−3.7% @ 1024²**, watermark flat  
- [x] `--text-tokens auto` trim — e2e **−32%** @ 512², ~−10% denoise/step @ 1024²  
- [x] **P9 Slice A (2026-08-15):** `--text-tokens auto` is the **product default** — eval 15/15 + identity I2I face lock. `--text-tokens 512` is the pad gallery path. [`Docs/TEXT_TOKENS.md`](TEXT_TOKENS.md)  
- [x] Prompt-embed disk cache (default on) — TE stage skipped on hit (**−4 s** @ 512², byte-identical)  
- [x] `imarello session` warm mode (resident policy, ≥16 GB RAM gate) — no win on 8 GB (gate validated); orchestrator loads made idempotent  
- [x] f16 Q/K/V threshold retest @ 1024² — f16 ~7% faster, same peaks  
- [x] f16 Q/K/V @ 512² (`f16SeqThreshold` **2048 → 512**) — denoise/step **−4.3%**, e2e **−3.3%**, watermark flat; eval-regression 15/15 pixel PASS  
- [ ] DiT **weight** streaming for iOS jetsam (`stagedAggressive`)  
- [ ] Cold vs warm metallib / first-step cost (S3) documented separately  
- [ ] Document accepted quality trade for 2–3 steps if used (S10)  
- [x] Research: M2 compute dtype — **activations fp32**, only quant scales are bf16 (`load-dit --dump-dtypes`, 2026-08-13)  
- [x] Research: Klein AdaLN/modulation size vs DT split — **~4% of DiT**, shared (not per-block); unload not worth it  
- [x] **P9 Slice B (2026-08-15):** BFL Small Decoder is the **product default** decode — 512 + 1024 quality PASS, decode **−37%**; `--vae-variant full` restores klein AE; I2I encode stays klein  
- [x] **P9 Slice C (2026-08-15):** Steel FA vs FFN vs `processQKV` glue — **park glue fusion**. Linear+FFN **87% / 75%** of a 512² / 1024² step; `qkv_rope` ~5%. `--op-profile` stays as a ranking tool.  
- [x] **P9 Linear/SwiGLU (2026-08-15):** f16 `quantizedMM` (cast bf16 scales; **×16** pre-scale — raw f16 is noise). 512² denoise **−6.5%**, e2e **−4.6%**. Compiled SwiGLU (same ops). `--attn-linear-compute f32` escape.  

**P9 leftover slices** (paused 2026-08-15 — do not start without an explicit ask):

| Slice | Status | Item | Notes |
|-------|--------|------|-------|
| **A** | **done** (default) | `--text-tokens auto` | Product default. Pad-512 via `--text-tokens 512`. [`TEXT_TOKENS.md`](TEXT_TOKENS.md). |
| **B** | **done** (default) | BFL **FLUX.2 Small Decoder** as product decode | 512 + **1024** T2I quality PASS. Decode **−37%**. Default **small-decoder**; `--vae-variant full` for klein. Encoder stays klein. |
| **C** | **done** (park glue) | Profile Steel FA vs FFN vs `processQKV` glue | 512² + 1024²: Linear+FFN own the step; `qkv_rope` ~5%. Do not fuse QK-Norm+RoPE. [`PERF.md`](PERF.md) Slice C. |
| — | `paused` | TAEF2 (or Small Decoder @ 256/384) `--preview` | Interactive only; never ship as export. |
| — | `paused` | Training-free **ref-KV** on 4B I2I | 9B-KV *schedule* on 4B; identity 512 first; kill if face-crop SSIM drops or watermark > ~4.2 GiB. |
| — | `paused` | **Δ-DiT** skip a subset of single blocks on **step 2 only** | High risk; kill on any vision-brief / pixel fail. |
| — | `paused` | `stagedAggressive` weight **drop** (not mmap) | iOS jetsam, not M2 speed. Expect slower e2e. |

**Latest default-path (8 GB Mac mini, release, auto + Small Decoder + f16 qmm ×16, 2026-08-15):** 512² W1T3 e2e **19.4 s** / denoise **3.34 s**; **1024² W1T3 e2e 74.0 s** / denoise **15.91 s** / decode **5.27 s**; peak active **2.04 / 2.05 GiB**; watermark **2.38 / 3.46 GiB**. Full tables: `Docs/PERF.md`.

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

1. Read this file + `CLAUDE.md` product locks + [`AGENT_WORKFLOW.md`](AGENT_WORKFLOW.md).  
2. Default next ID is **P7** (iOS host). P9 product path is shipped; leftover speed work is **paused**.  
3. Confirm macOS smoke still works (`swift build -c release` + `t2i` + `EVAL_WORKFLOW.md`).  
4. Open a focused branch; do not expand into “Out of v1” or resume P9 leftovers without an explicit ask.  
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
| 2026-08-13 | P8 I2I strength-curve recipes (`Docs/I2I_STRENGTH.md`); macOS polish backlog complete |
| 2026-08-13 | Port quarantine leftovers on main: Steel metallib check, VAE D=512 chunked SDPA (`evalEachChunk` off), `EvalCachePolicy.mid` bench-only |
| 2026-08-14 | P9 Slice A: `--text-tokens auto` first-class opt-in; pad-512 still default that day. *(Superseded 2026-08-15 — `auto` is the product default.)* |
| 2026-08-14 | P9 Slice B: BFL Small Decoder as `--vae-variant small-decoder`. 512² decode −37%. Full AE still default that day. *(Superseded 2026-08-15 — Small Decoder is the product default.)* |
| 2026-08-15 | I2I encoder lock: always klein `encodeOnly`. Do not load `full_encoder_small_decoder.safetensors`. 512² mug + identity smokes with Small Decoder decode PASS. |
| 2026-08-15 | Small Decoder 1024² T2I quality pass: 6/6 pixel PASS, vision match vs full AE, decode −37%. |
| 2026-08-15 | **Promote Small Decoder to product default.** `--vae-variant full` is the klein-pack escape hatch. Missing snapshot fails with `hf download` hint. |
| 2026-08-15 | `--text-tokens auto` identity I2I A/B (512², seed 7): face lock holds; outfit *cut* can drift. **Promoted to product default.** `--text-tokens 512` is the pad gallery path. |
| 2026-08-15 | P9 Slice C: Linear+FFN own 87% / 75% of a 512² / 1024² step; `qkv_rope` ~5%. **Park fused QK-Norm+RoPE / compile-glue.** |
| 2026-08-15 | **f16 scaled 4-bit Linear is the product default.** Raw f16 qmm → noise (pixel harness missed it). ×16 pre-scale restores images; 512² denoise **−6.5%**. `--attn-linear-compute f32` escape. |
| 2026-08-15 | Pixel harness **`unstructured_garbage` hard fail** (white noise + VAE-decoded rainbow speckle). Catches the unscaled-f16 miss. Schema 1.4. |
| 2026-08-15 | Current-stack 1024² W1T3: e2e **74.0 s**, denoise/step **15.91 s**, watermark **3.46 GiB**. Identity 512 recolor+replace face lock holds on f16 qmm. |
| 2026-08-15 | **4-bit is locked.** 3-bit is not a product path and is not parked for later. |
| 2026-08-15 | Official mark is the 3D cream/gold iris (`Docs/assets/readme/imarello-mark.{jpg,png}`); lockups updated. |
| 2026-08-15 | Docs freeze: [`AGENT_WORKFLOW.md`](AGENT_WORKFLOW.md) (skills / MCP / host-safe loops). **P9 leftover speed work paused.** Next product phase is **P7**. |
| 2026-08-15 | P7 started: `Apps/ImarelloIOS` iOS 26 demo. Simulator is UI-only (MLX has no Simulator Metal). No Catalyst. [`IOS.md`](IOS.md). |
| 2026-08-15 | P7 Debug build installed on a physical iPhone via `xcodebuild` + `devicectl`. Generate still blocked: weights not in the container; wildcard team profile strips kernel entitlements. |
