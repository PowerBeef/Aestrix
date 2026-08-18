# Imarello roadmap

**Last updated:** 2026-08-18 (two passes: the **audit + hardening pass** `9ad69cd`, then the **engine-uplift session** — mlx core **0.32.1 via fork** (encode_te −19–21%, 512² e2e **21.88 s**, watermarks flat, 15/15 V-eval), TE-splice shipped for conditioning (speed claim refuted — TE is dispatch-bound), `t2i --two-stage` = working opt-in rescue for the P7 1024² anatomy item (softens texture; SR stage before any default gate), ChipCapabilities probe + iOS floor 26.2 for the A19 NAX path. See the decision log's 2026-08-18 rows + PERF.md 2026-08-18.)  
**Working tree focus:** macOS library + CLI is the shipping surface. **Next product phase is P7 iOS.**  
**Engine-uplift track is ACTIVE** (user un-paused speed work 2026-08-18; plan in the session log + PERF.md 2026-08-18): next levers are **S4 fused qmm+SwiGLU full-f16 single block** and **N2/N3 A19 Pro NAX** (device must run iOS ≥ 26.2). Still parked without a further ask: TAEF2 preview, ref-KV, Δ-DiT, `stagedAggressive`.  
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
| **Status** | `partial` — iOS 26 app runs staged Klein 4B on a physical iPhone. Simulator = UI only (no MLX). **512² T2I and I2I both pass pixel + vision on device** (2026-08-16, on the rebuilt studio UI). Device harness can drop jobs without a UI tap. Only open item: **1024² T2I** can fail vision anatomy (Klein 4-step / μ=1.15) |
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
- [x] In-app I2I eval on device (`/tmp/imarello-ios-eval/i2i-rebuild-1/i2i-1786900586.png`; blue wool scarf on the fox at strength 0.8 — pixel PASS, vision PASS: subject, pose, anatomy and background preserved, edit applied)

**Resume notes**

- Reuse `ImarelloPipeline` actor; do not reimplement TE/DiT/VAE in the app.  
- Product locks still apply: Klein 4B only, prequant only, staged default.  
- Start with Tier L (512², 4-step) before higher resolutions.  
- See `Docs/MEMORY.md` for peak budgets; `Docs/WEIGHTS.md` for pack layout.

**Suggested next step**

512² T2I works. Drive device generates with `Scripts/ios-device-harness.sh` (resync weights after install; the default job id is timestamped and stale results are rejected by `startedAt`, so id reuse is safe). Since 2026-08-18 the harness honors `--steps` / `--text-tokens` / `--strength` on device. 1024² quality is an open vision issue. Do not generate on the Simulator. Recipe: [`IOS.md`](IOS.md).  

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
- [x] **P9 Slice A (2026-08-15):** `--text-tokens auto` promoted to default — eval 15/15 + identity I2I face lock. *(Reverted 2026-08-16 — vision regression; see decision log.)* [`Docs/TEXT_TOKENS.md`](TEXT_TOKENS.md)  
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
| **A** | **done** (opt-in) | `--text-tokens auto` | Opt-in speed path; default for one day, reverted 2026-08-16 (conditioning regression). [`TEXT_TOKENS.md`](TEXT_TOKENS.md). |
| **B** | **done** (default) | BFL **FLUX.2 Small Decoder** as product decode | 512 + **1024** T2I quality PASS. Decode **−37%**. Default **small-decoder**; `--vae-variant full` for klein. Encoder stays klein. |
| **C** | **done** (park glue) | Profile Steel FA vs FFN vs `processQKV` glue | 512² + 1024²: Linear+FFN own the step; `qkv_rope` ~5%. Do not fuse QK-Norm+RoPE. [`PERF.md`](PERF.md) Slice C. |
| — | `paused` | TAEF2 (or Small Decoder @ 256/384) `--preview` | Interactive only; never ship as export. |
| — | `paused` | Training-free **ref-KV** on 4B I2I | 9B-KV *schedule* on 4B; identity 512 first; kill if face-crop SSIM drops or watermark > ~4.2 GiB. |
| — | `paused` | **Δ-DiT** skip a subset of single blocks on **step 2 only** | High risk; kill on any vision-brief / pixel fail. |
| — | `paused` | `stagedAggressive` weight **drop** (not mmap) | iOS jetsam, not M2 speed. Expect slower e2e. |

**Latest default-path (8 GB Mac mini, release, pad-512 + f16 qmm + joint-f16 + Small Decoder, core 0.32.1, 2026-08-18):** 512² W1T3 e2e **21.9 s** / denoise **4.22 s** / encode_te **1.58 s**; **1024² e2e 67.5 s** / denoise **14.7 s** / decode **4.6 s**; identity-i2i 512² **36.2 s**; peak active ~2.05 GiB; watermark **2.57 / 3.00 GiB**. *(The 2026-08-15 auto-era row — 19.4 / 74.0 s — is historical; `auto` was reverted.)* Full tables: `Docs/PERF.md`.

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
| 2026-08-14 | P9 Slice A: `--text-tokens auto` first-class opt-in; pad-512 still default that day. |
| 2026-08-14 | P9 Slice B: BFL Small Decoder as `--vae-variant small-decoder`. 512² decode −37%. Full AE still default that day. *(Superseded 2026-08-15 — Small Decoder is the product default.)* |
| 2026-08-15 | I2I encoder lock: always klein `encodeOnly`. Do not load `full_encoder_small_decoder.safetensors`. 512² mug + identity smokes with Small Decoder decode PASS. |
| 2026-08-15 | Small Decoder 1024² T2I quality pass: 6/6 pixel PASS, vision match vs full AE, decode −37%. |
| 2026-08-15 | **Promote Small Decoder to product default.** `--vae-variant full` is the klein-pack escape hatch. Missing snapshot fails with `hf download` hint. |
| 2026-08-15 | `--text-tokens auto` identity I2I A/B (512², seed 7): face lock holds; outfit *cut* can drift. **Promoted to product default.** *(Reverted 2026-08-16.)* |
| 2026-08-15 | P9 Slice C: Linear+FFN own 87% / 75% of a 512² / 1024² step; `qkv_rope` ~5%. **Park fused QK-Norm+RoPE / compile-glue.** |
| 2026-08-15 | **f16 scaled 4-bit Linear is the product default.** Raw f16 qmm → noise (pixel harness missed it). ×16 pre-scale restores images; 512² denoise **−6.5%**. `--attn-linear-compute f32` escape. |
| 2026-08-15 | Pixel harness **`unstructured_garbage` hard fail** (white noise + VAE-decoded rainbow speckle). Catches the unscaled-f16 miss. Schema 1.4. |
| 2026-08-15 | Current-stack 1024² W1T3: e2e **74.0 s**, denoise/step **15.91 s**, watermark **3.46 GiB**. Identity 512 recolor+replace face lock holds on f16 qmm. |
| 2026-08-15 | **4-bit is locked.** 3-bit is not a product path and is not parked for later. |
| 2026-08-15 | Official mark is the 3D cream/gold iris (`Docs/assets/readme/imarello-mark.{jpg,png}`); lockups updated. |
| 2026-08-15 | Docs freeze: [`AGENT_WORKFLOW.md`](AGENT_WORKFLOW.md) (skills / MCP / host-safe loops). **P9 leftover speed work paused.** Next product phase is **P7**. |
| 2026-08-15 | P7 started: `Apps/ImarelloIOS` iOS 26 demo. Simulator is UI-only (MLX has no Simulator Metal). No Catalyst. [`IOS.md`](IOS.md). |
| 2026-08-15 | P7 Debug build installed on a physical iPhone via `xcodebuild` + `devicectl`. Generate still blocked: weights not in the container; wildcard team profile strips kernel entitlements. |
| 2026-08-16 | **`--text-tokens auto` default REVERTED** after user-reported quality degradation. Bisect (fisherman s42/s7, fox 512/1024, identity I2I s7) convicted `auto`: no text mask in joint attention → trimming the 512 pad weakens conditioning (illegible faces, averted subjects; f32 1024² fox was a headless chimera). f16 qmm and Small Decoder exonerated (byte-near-identical A/Bs). Pad-512 default restored across CLI, library requests, session, bench, iOS app, and device harness; byte-identity vs pad runs verified; eval-regression 15/15 + vision PASS. [`TEXT_TOKENS.md`](TEXT_TOKENS.md). |
| 2026-08-16 | Product-path numbers re-measured (pad-512 + f16 qmm + Small Decoder): 512² e2e **24.4 s** (W1T3), 1024² **79.0 s** (W0T1); watermark 2.54 / 3.63 GiB. Opt-in `auto` remains 19.4 / 74.0 s. |
| 2026-08-16 | Process gaps recorded: the auto/f16 promotions were vision-checked at 512² only (never 1024² or multi-seed humans), and bench JSON does not persist `textTokens` / `attnLinearCompute` / `vaeVariant` — provenance lived only in filename labels. |
| 2026-08-16 | P7 studio UI cleanup (Impeccable P0/P1s): gate story moved onto the stage (banner + raw Caches path removed; ready/missing-weights/Simulator empty states), elapsed timer in the status strip, Try Again on errors, 1024² duration caption, dynamic Edit a11y hint. Device scripts fixed: symlinked snapshot resolution, devicectl JSON-to-file, paired-device filter (idle tunnel reads `disconnected`). New Debug build installed on the iPhone; weights resynced; 512² fox smoke on the **pad-512 default** passed pixel + vision on device. |
| 2026-08-16 | **Engine research report** ([`ENGINE_RESEARCH.md`](ENGINE_RESEARCH.md)): full engine read + first pad-512 op-profiles (Linear+FFN 84%/74% of a step) + watermark diagnosed to two code lines + tiered optimization roadmap + external research (attention-sink literature backs the auto-revert mechanism; partial-pad experiment ladder designed; 4-step accel literature re-confirmed NO-GO; mlx-swift 0.31.6 is current, next core bump ships an 8–31%-class qmm win). Surfaced Tier-0 issues per the process rule: tokenizer fidelity gap (NFC/pre-tokenizer/merge order), PNG-export range-probe tone bug, Small-Decoder attention needlessly 4-bit quantized, unguarded VAE-encoder memory cliff (1024² I2I), `res-ladder` broken by the HostPreflight lock, `MemoryProbe` MLX fields never populated. |
| 2026-08-16 | **Partial-pad diagnostics (Tier 3, R3/R4)**: experimental `t2i` knobs (`--pad-content`, `--pad-keep`, `--pad-bias`; defaults byte-identical). LPIPS-calibrated verdicts — denominator compensation **refuted** (0.280→0.258 at k=32; pads are register capacity, not softmax mass), count-trimming **parked** (smooth drift curve: k=2→0.474 … k=32→0.26; tolerable budgets surrender the speed), **pad content swappable at equal quality** (clean-pad splice 0.15–0.22, vision-equal on fisherman + fox). Next lever: product **TE-splice** (real-token encode + cached clean-pad bank, ≈ −8% e2e @512², full promotion gate). |
| 2026-08-16 | **Tier 2 fully dispositioned** — NHWC end-to-end VAE shipped: bit-exact on all four pipeline paths, **speed-neutral** (the per-block transposes were lazy views that canceled; the research estimate assumed copies that never existed). Kept for layout clarity. Every Tier-2 lever is now shipped or NO-GO'd with data. |
| 2026-08-16 | **iOS studio rebuilt from the ground up** (Impeccable, code-led "Spread Deck"). Two full-bleed pages — Stage and Contact Sheet — replace the single-column form; `GenerationModel` is split into `GenerationEngine` / `HarnessService` (frozen contract) / `PrintStore` (persistent history) / `StudioModel`. Edit now runs from **any** print, not just the last. Ground moved espresso → darkroom black and the aperture glyph → a **Develop** pill after owner feedback; chrome capped at `accessibility2`. Finish review returned **fix-then-ship**; all seven material fixes applied. Device redeploy + resync, then fresh-id smokes: **512² T2I and I2I both pixel + vision PASS** — the last open P7 acceptance item (in-app I2I on device) is now closed. `Apps/ImarelloIOS/DESIGN.md` records the shipped system. |
| 2026-08-16 | **P7 device redeploy with the Tier-0/1/2 engine.** Rebuilt, entitlements verified, installed, weights resynced; fresh-id harness smoke: 512² fox on the iPhone 17 Pro in **11.6 s** (was 13.8 s this morning — **−16% on-device**), pixel + vision PASS. Note: the harness's default job id hit the documented stale-`done/{id}.json` trap on the reused container — always pass a fresh `--id` when re-smoking after a redeploy. |
| 2026-08-16 | **Tier 2 implemented on explicit ask.** Shipped: joint-f16 attention (all 25 blocks on f16 Steel; full promotion gate incl. the first formally-gated 1024² human portrait — clean), VAE tile threshold 96→128 (768² untiled, decode **−51%**, no seams; untiled 1024² measured to Metal-abort on 8 GB — tiling stays load-bearing there), relaxed product cache policy (interval-2 + 512 MiB; ~2% @1024² for an identical watermark post-streaming). NO-GO with data: asyncEval (+1% @1024², +0.27 GiB @512²); ÷16-fold (re-analysis: the pre-divide protects the f16 *input cast* — TV-static class). **Cumulative day: 512² 24.4→23.0 s, 1024² 79.0→71.0 s (−10%), watermark 3.63→3.00 GiB (−17%).** |
| 2026-08-18 | **Full-repo audit + hardening pass** (`9ad69cd`). Adversarially-verified audit (52 confirmed findings, 15+ refuted; report artifact linked from the session) followed by a same-day fix pass: all 22 Critical/High/Medium fixed. Highlights: iOS print history moved to **Application Support** (versioned, per-element-tolerant index; quarantine-not-wipe; Caches `outputs/` stays only as the frozen harness pull location); harness quarantines undecodable jobs + sweeps orphaned `running/` at launch + threads `--steps`/`--text-tokens`/`--strength` (pre-2026-08-18 device A/Bs over those knobs are **invalid data**) + timestamped default id with a `startedAt` stale-result guard; pipeline **cancellation** at every stage boundary and denoise step (iOS Stop is real); CGContext buffer-lifetime UB fixed in image I/O; embed cache atomic + validated + keyed on tap layers; CLI validates steps/trials/ref-downsample; **3-bit pin deleted** and `--weights` enforces the 4-bit lock; `ensure-metallib.sh` refuses partial kernel sets and stamps the mlx-swift revision; CI filter widened to 18 verified-MLX-free suites. Remaining Low-severity items tracked in the audit artifact's Fix pass panel. No perf or numerics changes on the product path. |
| 2026-08-16 | **Tier 0 + Tier 1 implemented on explicit ask** (user green-lit the report's recommended order). Tier-0: export range param (bright-corner tone crush fixed — fisherman pixel ~65→~83), Small-Decoder attention back to F32, VAE-encoder checkpointing, reference-faithful tokenizer (21-case HF differential, byte-for-byte; embed cache → v2), instrumentation + res-ladder + bench-provenance fixes. Tier-1 + chunk-streamed single blocks: **byte-identical** same-seed output, 512² e2e 24.4→**23.2 s**, 1024² e2e 79.0→**73.5 s** (−7%), 1024² watermark 3.63→**3.07 GiB** (−15%) — the pad-512 default now beats the reverted `auto` path at 1024². Remaining research tiers (partial-pad study, full-f16 single block, VAE untile/NHWC, asyncEval, mlx-swift bump) stay open in [`ENGINE_RESEARCH.md`](ENGINE_RESEARCH.md); P9 *parked* slices (TAEF2, ref-KV, Δ-DiT, `stagedAggressive`) remain parked. |
| 2026-08-18 | **Engine-uplift session** (plan approved same day; speed work un-paused on explicit ask). S1: dead query-chunked SDPA deleted; bench provenance schema 1.4; re-baselines (512² 22.26 s / RSS 1.73 GiB; identity 36.7 s / 2.56 GiB — stale 45.3 s / 3.90 GiB corrected); 512² watermark = step transients (streaming 512² measured NO-GO: −0.15 GiB for +8.2% e2e). Scheduler refine regime fixed (σ<1/N schedules; the audit-era clamp silently forced σ→1/N). **Q**: 1024² anatomy baseline 14/15 (dancer-s0 four arms; same seed clean at 512²); `t2i --two-stage` fixes it at every σ but softens texture 10–25 pts (interpolated upscale lacks high frequencies) → **opt-in anatomy rescue, not default**; SR-stage follow-up before re-gating. **S2**: TE-splice implemented (`--pad-content clean`: real-token encode + cached pad bank; LPIPS 0.194 in-band) but speed claim REFUTED — TE is dispatch-bound. **S3**: mlx core **0.32.1 via PowerBeef/mlx-swift fork** (nojit — 0.32.1's JIT affine_qmv strings are inconsistent; full metallib now mandatory): encode_te **−19–21% everywhere**, 512² e2e **21.88 s**, watermarks flat, 15/15 V-eval + vision PASS. N1 ChipCapabilities probe shipped (real recipe: arch gen ≥17/18 + OS 26.2); iOS floor → 26.2 for the A19 NAX path (N2/N3 next). |
