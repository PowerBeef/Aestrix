# Engine research — deep analysis & optimization roadmap

**Date:** 2026-08-16 · **Stack analyzed:** pad-512 + f16 scaled qmm + Small Decoder + Steel FA (the shipping product path) · **Machine of record:** 8 GB M2 Mac mini

> **Status update (same day):** Tiers 0, 1, **and 2** shipped on explicit request. Cumulative: 512² 24.4→**23.0 s**, 1024² 79.0→**71.0 s (−10%)**, 1024² watermark 3.63→**3.00 GiB (−17%)**, 768² decode **−51%** (untiled). Tier-2 verdicts: joint-f16 attention and the tile/cache-policy retunes shipped (full promotion gates); asyncEval measured NO-GO; §4.9's ÷16-fold **stands corrected** — the pre-divide also protects the f16 input cast, so folding into weights re-opens the TV-static class; untiled 1024² Metal-aborts on 8 GB (tiling is load-bearing there). Details: `Docs/PERF.md`. §4.11c's NHWC decoder also shipped — **bit-exact but speed-neutral**: the per-block transposes were lazy views that canceled, so the −10–25% estimate assumed copies that never existed (kept for layout clarity). Tier 2 is now fully dispositioned. **§5.1's R3/R4 diagnostics also ran** (same day): the denominator-compensation hypothesis is **refuted** (bias buys ~8%; pads are register *capacity*, not softmax mass), count-reduction is a poor trade (drift curve needs ≥256-token budgets), but **pad content is swappable at equal quality** — licensing a product **TE-splice** (real-token encode + cached clean-pad bank, ~−8% e2e @512² with the 512-key softmax intact) as the next lever, pending the full promotion gate. Numbers: `Docs/PERF.md`. Still open: the TE-splice implementation, full-f16 single block, mlx-swift bump watch, Tier 4 (iOS memory).

This document is the product of a full engine read (every file in `ImarelloDiT`, `ImarelloText`, `ImarelloVAE`, `ImarelloRuntime`, `ImarelloWeights`), fresh measurements on the pad-512 product path (`outputs/engine-research-2026-08-16/`, reproduction commands in §7), a compiled ledger of every experiment already run (§6), and external research: MLX upstream source/releases, the 2025-26 few-step-acceleration literature, and the attention-sink / padded-conditioning literature (§8).

---

## 1. Executive summary

The engine is already well-optimized along the axes it has explored: staged residency works exactly as designed (fresh `phase_peaks` @1024²: TE 1.66 GB → DiT 2.18 GB → VAE 0.12 GB, never co-resident), Steel fused attention runs on every product shape, weight loading is lazy with zero waste, and ~50 prior experiments (§6) closed most obvious doors. The remaining opportunity is real but specific:

1. **Quantized Linears + FFN still own the step** — fresh pad-512 op-profile: 84.4% @512², 73.9% @1024². Everything that touches the qmm chain outranks everything else.
2. **The memory watermark is two lines of code.** 512²'s 0.35 GiB of headroom above weights is the unfused `(x/16)→f16→qmm→f32→×16` rescale chain (`AttentionUtils.swift:181,195`); 1024²'s 1.43 GiB is `concatenated(parts)+eval` in `linearChunkedSequence` (`AttentionUtils.swift:223-225`), which holds all chunks *and* the concatenated result — defeating the chunking's stated purpose. Both reconcile with measured watermarks to ~10%.
3. **A basket of zero-numerics-risk changes** (Tier 1) is worth an estimated **−4–8% denoise and −0.3–0.6 GiB watermark** combined, plus ~1.4 s off every cache-miss generate from TE-side fixes.
4. **The single biggest speed lever is recovering the trimmed-text win safely.** `--text-tokens auto` measured **−32% e2e @512²** and was reverted for real conditioning damage. The mechanism is now backed by published literature (pads act as attention sinks/registers in FLUX-class joint attention), and §5.1 lays out a five-rung experiment ladder — including a cheap mechanistic differentiator — that could recover most of the win at a defensible quality bar.
5. **Four quality/correctness bugs were found during this analysis** (§3). Per the repo's own process rule, these surface before new phase work; three of them are quality *gains* with near-zero risk.
6. Honest limits: Klein 4-step already spent the big win. A 20% kernel-level improvement ≈ 4–5 s @512². Nothing in the 2025-26 literature changes that for a 4-step distilled model (§5.4) — the only surviving external family is per-call int8 attention, and its payoff on M2 is modest.

### Ranked recommendation table

| # | Item | Tier | Est. win | Quality risk | § |
|---|------|------|----------|--------------|---|
| 1 | Fix the 4 quality bugs (tokenizer fidelity, export range probe, Small-Decoder quant, VAE-encoder checkpointing) | 0 | quality ↑ | none (gains) | 3 |
| 2 | Fuse the qmm rescale chains (`compile`) | 1 | −2–5% denoise, −0.15–0.3 GiB | zero | 4.1 |
| 3 | Stream `linearChunkedSequence` chunks into the consumer | 1 | −0.4–0.6 GiB @1024² | zero | 4.2 |
| 4 | Move `mlpAct` above the attention call | 1 | −54–162 MiB live | zero | 4.3 |
| 5 | Pre-cast quant scales/biases to f16 at load | 1 | −0.3–1% denoise | zero (bit-identical) | 4.4 |
| 6 | Hoist `temb` + modulation triples out of the step loop | 1 | −0.2–0.7% denoise | zero | 4.5 |
| 7 | TE stage fixes (tokenizer lifecycle, `do_causal`, real-length for auto) | 1 | −0.5–1.5 s per cache miss | zero | 4.6 |
| 8 | Staging hygiene (cacheLimit restore, clear-interval consistency, I/O vectorization, dead code) | 1 | small ×7 | zero | 4.7 |
| 9 | Joint-seq attention dtype (or all-f32 double blocks) | 2 | −1–3% denoise, −60 MiB | low (gated) | 4.8 |
| 10 | Fold ÷16 into quantization scales at load | 2 | −1–2% denoise | low (gated) | 4.9 |
| 11 | `asyncEval` + periodic hard sync | 2 | −1–3% denoise | zero numerics; memory gated | 4.10 |
| 12 | VAE: untile ≤1024² / retune 768², NHWC end-to-end, clearCache interval | 2 | −20–50% decode ≥768² | gated (OOM risk only) | 4.11 |
| 13 | Re-A/B the 256 MiB denoise cache clamp at 1024² | 2 | unknown, plausibly several % | zero | 4.12 |
| 14 | **Partial-pad conditioning study** | 3 | up to −25–30% e2e | the study *is* the gate | 5.1 |
| 15 | Full-f16 single-stream block (fused qmm+SwiGLU) | 3 | −8–12% denoise, −0.6 GiB @1024² | high (TV-static adjacent) | 5.2 |
| 16 | mlx-swift pin watch (next core bump ships an 8–31%-class qmm win) | 3 | large, free, later | re-validate | 5.3 |
| 17 | iOS memory: `stagedAggressive` + encoder checkpointing | 4 | RAM/jetsam, not speed | gated | 5.5 |

---

## 2. Engine anatomy — where every second and GiB goes

### 2.1 The stage pipeline (measured, pad-512, this report's runs)

```
prompt ─► TE (Qwen3, 27 layers) ─► unload ─► DiT (5+20 blocks × 4 steps) ─► unload ─► VAE decode ─► PNG
   load 0.6–1.4 s · encode 2.1–2.7 s      load 1.3 s · 4.6 s/step @512²        load 0.1 s · 1.0 s @512²
   peak 1.66 GB                            peak 2.18 GB                          peak 0.12 GB
```

- Fresh `phase_peaks` (pressure-map @1024²): **te 1.66 GB · dit 2.18 GB · vae 119 MB** — the staged invariant holds perfectly; peak ≈ max(stage), never the sum.
- TE materializes **~1.66–1.75 GB**, not the 2.26 GB in `Docs/MEMORY.md` — 9 of 36 Qwen3 layers, `lm_head`, and final norm are structurally pruned and (because `MLX.loadArrays` is lazy over `pread`) **never read from disk**. Doc correction needed.
- The prompt-embed cache short-circuits *the entire TE stage including the weight load* (verified control flow: cache check precedes `loadTextEncoderExclusive`, `ImarelloPipeline.swift:257-299`) — worth ~4 s on a hit. No free win was hiding there.

### 2.2 One denoise step (product defaults)

Full op-by-op trace is in this session's analysis; the load-bearing facts:

- **Joint sequence** = image tokens + 512 text: 1536 @512², 4608 @1024². Every tuning threshold (`linearChunkThreshold=1536`, `blockCacheClearSeqThreshold=1536`, `f16SeqThreshold=512`) sits exactly *on* the 512² boundary — **512² and 1024² execute different code**: at 512² nothing chunks and no cache clears fire; at 1024² everything chunks and ~47 clears/step fire. Single-canvas A/Bs mislead; always measure both.
- **Fresh op-profile (pad-512, first ever on the product stack**; ranking only — profiler syncs inflate wall time):

| Bucket | 512² | 1024² | auto-era (stale) |
|---|---:|---:|---|
| `qkv_proj` (all quantized Linears) | **50.8%** | **43.5%** | 51.5 / 44.0 |
| `ffn` | **33.6%** | **30.4%** | 35.3 / 30.8 |
| `steel_fa` | 7.1% | 17.3% | 4.8 / 15.0 |
| `qkv_rope` (glue) | 4.9% | 5.2% | 4.7 / 5.8 |
| `other` | 3.6% | 3.6% | 3.8 / 4.4 |

  The ranking survived the pad-512 revert: **Linear+FFN own 84%/74% of a step**. Attention rose ~2 points with the longer sequence — still not the place to spend effort on M2 (and the glue-fusion NO-GO stands at ~5%).
- **Blocking sync count**: ~54 `eval` per step @512², ~124 @1024² (per-block checkpoints + QKV checkpoints + per-chunk evals). `asyncEval` exists in the pin and is used nowhere.
- **Per-step recomputation that is loop-invariant**: `temb` (`Transformer.swift:167`), the three modulation projections (`:177-178,:223`), AdaLN-continuous conditioning (`AdaLayerNormContinuous.swift:18`) — all functions of the (pre-known) timestep only; and `q.scales/biases.asType(.float16)` on **every call and every chunk** (§4.4). RoPE, context projection, ids, timesteps, `dt` are already hoisted.
- **The dtype seam**: `attnDType` is decided per *stream* (`AttentionUtils.swift:43-44`). The text stream (seq 512, `512 > 512` false) stays f32; the image stream casts f16; the joint `concatenate` then promotes everything **back to f32** (verified in `mlx/ops.cpp:1086-1090`), RoPE re-stores Q/K as f16, and SDPA promotes Q/K to f32 again (`fast.cpp:704-715`). Net: six wasted full-size casts per double block and **5 of 25 blocks silently run f32 Steel attention**. (Upstream kernel source confirms all accumulation is f32 regardless — so the f16 paths are accumulator-safe; the casts are pure waste.)

### 2.3 The memory watermark, line by line

Active peak ≈ DiT weights (2.03 GiB) at both canvases. The watermark headroom above that:

| Canvas | Headroom | Cause (verified arithmetic) |
|---|---|---|
| 512² (2.54 GiB) | ~0.5 GiB | Unchunked single-stream chain: fused proj f16 [81 MiB] → `.asType(.float32)` [162 MiB] → `× 16` [162 MiB] — the rescale epilogue at `AttentionUtils.swift:195` |
| 1024² (3.63 GiB) | ~1.6 GiB | `linearChunkedSequence`: 9 live chunk parts [486 MiB] + `concatenated` result [486 MiB] + `mlpHidden` held across attention [324 MiB] + h/QKV |

This is why the `cacheLimit` sweep measured 0% five times: the watermark is *live graph*, not pool. Tier-1 items 2–4 attack exactly these tensors.

### 2.4 Stage-level measurements (this report)

| Measurement | Result | Note |
|---|---|---|
| `encode_te` | 2.67 s (W1T3, this run; 2.11 s in the 08-15 table) | pure prompt-window cost, canvas-independent; ~93% of it is computing pad-token hidden states (which the pad-512 DiT path *does* consume — see §5.1) |
| `load_te` / `load_dit` / `load_vae` | 1.44 / 1.32 / 0.11 s cold | `load_te` includes re-parsing the 11.4 MB `tokenizer.json` on every load (§4.6) |
| VAE decode 512² / 768² / **1024²** | 0.99 / **4.78** / **4.79 s** | **768² costs the same as 1024²** — direct confirmation of the 2.25× tile redundancy at 768² (tile grid [0,24] on a 96² latent) vs 1.27× at 1024² |
| Denoise/step ladder (blocks density, cold): 512→4.79 · 640→7.35 · 768→10.0 · 896→13.1 · 1024→17.0 s | scales ≈ joint-seq^1.15 | first recorded ladder; res-ladder *mode* is broken (§3.6) |

---

## 3. Tier 0 — correctness & quality findings (fix before new phase work)

Per the repo's blocking process rule, these surfaced during analysis and precede optimization work. None is a speed item; three are free quality *improvements*.

### 3.1 Tokenizer fidelity gap (prompt-adherence correctness)
`QwenTokenizer` (`Sources/ImarelloText/QwenTokenizer.swift:131-242`) implements neither the **NFC normalizer** nor the **regex pre-tokenizer** in the model's `tokenizer.json` (verified against the shipped file: `Split(GPT2-style regex, Isolated)` + `ByteLevel(use_regex=false)`), and its BPE merges **one occurrence per pass** where the reference merges the leftmost occurrence of the lowest-ranked pair via a heap (observably: "merge all occurrences of the best pair, leftmost-first"). Token ids can therefore diverge from diffusers/BFL for prompts with mixed scripts, digits (Qwen splits *every digit separately*), contractions, or punctuation runs — silently changing embeddings and prompt adherence. The current `applyBPE` is also O(n²) with a String allocation per candidate pair — likely seconds of host time on long structured prompts (the FLUX.2 house style). A precise implementation spec with sources is in §8 (few-step/tokenizer brief): NFC (`precomposedStringWithCanonicalMapping`), the exact regex with an ICU `\s` caveat, byte-level mapping, and the linked-list+heap O(n log n) merge. **Acceptance test: byte-identical ids vs HF `tokenizers` over a stratified corpus.** Note Imarello's chat template and pad-key masking in the TE already match the references (verified) — the gap is only the tokenizer proper.

### 3.2 PNG export range heuristic (content-dependent tone crush)
`ImageExport.swift:30-36` samples ~85 top-left pixels to guess whether output is in [-1,1]; a bright top-left corner skips the rescale and `vDSP_vclip` crushes shadows to black. Both call sites are guaranteed [-1,1] — make the range an explicit parameter and delete the probe. Zero risk; add a regression test with a white-corner fixture.

### 3.3 Small Decoder attention needlessly quantized
`VAEWeights.loadSmallDecodeOnly` (`VAEWeights.swift:90-91`) 4-bit-quantizes BFL's F32 mid-block attention (4 × 384×384 Linears) after loading — the only *eager* quantization in the codebase, saving ~1.8 MB at pure quality cost on the product decode path. Skip for `.smallDecoder`.

### 3.4 VAE encoder is an unguarded OOM cliff (I2I @1024²)
`Flux2Encoder` has **zero** `eval`/`clearCache` (`VAEComponents.swift:191-196, 284-297`) while the decoder is heavily checkpointed for exactly this reason. A 1024² I2I materializes the whole encoder DAG at once — first down-block tensors are ~536 MB f32, mid-block attention S=16384. This is the most plausible hazard behind the open P7 "last-in-app I2I on device" item. Mirror the decoder's per-block checkpointing (encoder is not on the T2I path; cost is negligible).

### 3.5 Instrumentation defects
- `MemoryProbe` never populates `mlxActiveBytes`/`mlxCacheBytes` (`MemoryProbe.swift:34-43`) — **confirmed live**: every bench JSON in this report shows `peak_mlx_active: 0.0 MB` for probe-collected samples. `clearMLXCache()` is an empty function.
- `Transformer.swift:136-141` — `sample(forceEval:)` evals the *original input*, not the current `h`; the parameter is also dead.
- Bench JSON still lacks `textTokens`/`attnLinearCompute`/`vaeVariant` provenance (recorded 2026-08-16; §7).

### 3.6 `res-ladder` mode is structurally broken
Discovered attempting the first ladder run: the parent bench takes the `HostPreflight` lock, and every child rung refuses with "Another imarello process is running (parent PID)". Broken since the one-Metal-owner lock landed — which is why no ladder data was ever recorded. Fix: parent skips the lock (or passes a token to children) for `res-ladder` mode only.

---

## 4. Tier 1 & 2 — engineering recommendations

Ordered by win ÷ risk. Every item names its validation: **V-bench** = `bench` W1T3 at *both* canvases + `bench-compare`; **V-eval** = `eval-regression.sh` 15/15 + vision pass; **V-bit** = byte-identical PNG on same seed.

### Tier 1 — zero numerics risk

**4.1 Fuse the qmm rescale chains.** Wrap `(x / scale).asType(.float16)` and `y.asType(.float32) * scale` (`AttentionUtils.swift:181,195`) in `compile(shapeless: true)` closures — MLX fuses cast+multiply into one kernel, eliminating one full-size f32 temp and a full read/write per Linear call (~6.5 GB/step of traffic @512²). Directly attacks the 512² watermark line. *V-bit, V-bench.*

**4.2 Stream chunks instead of `concatenated(parts)+eval`.** Restructure `Flux2ParallelSelfAttention` so each 512-token chunk flows proj → split → norm/reshape → SwiGLU, keeping only f16 q/k/v and `mlpOut` alive; delete the concat (`AttentionUtils.swift:223-225`). Linear+elementwise ops are exactly sequence-separable. Removes the 1024² watermark line (~1 GiB of transients) and ~70 blocking evals/step. May allow relaxing the per-block clearCache gate (historically worth −3.9% @1024²). *V-bit, V-bench.*

**4.3 Move `mlpAct(mlpHidden)` above `computeAttention`** (`ParallelSelfAttention.swift:102` → before `:93`). SwiGLU halves 18432→9216, so the tensor held live across Steel FA drops 324→162 MiB @1024². Four lines. *V-bit.*

**4.4 Pre-cast quantized `scales`/`biases` once at load.** Today `q.scales.asType(.float16)` runs per call *and per chunk* (`AttentionUtils.swift:185-186`) — ~242 MB of scale/bias tensors re-cast at least once per step (~200–1500 extra dispatches). Upstream semantics validate the fix twice over: `quantized_matmul` promotes f16×bf16→f32 internally (`mlx/ops.cpp:4342-4360`, `dtype.cpp:45-48`), so the raw `linear()` call sites (modulation, temb, AdaLN, contextEmbedder) currently run their whole qmm in f32; and upstream's own (unreleased) fix dequantizes in f32 *because bf16 math is emulated pre-M5*. Rebuild `QuantizedLinear`s with f16 scales post-`update`. *V-bit (the identical cast, done once), V-bench.*

**4.5 Hoist `temb` + modulation triples + AdaLN conditioning to a pre-loop pass** (`Transformer.swift:167,177-178,223`; `AdaLayerNormContinuous.swift:18`). All depend only on the timestep; all 4 timesteps are known before the loop. ~170M params of GEMV/step removed; also enables folding `(1 + scale)` (`ModulationOps.swift:12`, 50×/step) into the precompute. *V-bit.*

**4.6 TE-stage fixes.**
(a) **Tokenizer lifecycle**: `QwenTokenizer.load` re-parses 11.4 MB of JSON and rebuilds ~450 K dictionary entries on *every* TE load and drops them on unload (`TextEncoderModule.swift:39,50`). Process-lifetime lazy singleton; tokenize before weights load. Est. −0.3–1.0 s per cache-miss (`bench --mode te-only` before/after).
(b) **`do_causal`**: the TE builds an additive causal+pad mask, which forces Steel's masked path (`do_causal=false`) — the kernel walks all KV blocks instead of skipping the upper triangle: ~2× attention work. For the (dominant) case where padding is only a tail, a pure `.causal` mask is equivalent for real-token outputs. *Verify with an embedding-diff oracle, then V-bit downstream.*
(c) **Real-length encode for the `auto` path**: with causal attention + tail padding, hidden states for real tokens are mathematically identical at L=real vs L=512; the auto path throws pad rows away anyway. Gate on `textTokens == .auto`, add L to the embed-cache key. `encode_te` 2.1–2.7 s → ~0.3–0.6 s on that path. **Note: the pad-512 *product* path genuinely needs the pad hidden states (they participate in joint attention — §5.1), so this cannot speed the default; it makes the opt-in path and any future partial-pad mode much cheaper.**
(d) `embed_tokens` reads+resides 219 MB to gather ≤512 rows — a custom safetensors range-read would cut ~0.2 s and 219 MB from cache-miss loads (medium effort; runner-up).

**4.7 Staging & I/O hygiene (small, all zero-risk).** Restore `Memory.cacheLimit` after denoise (`ImarelloPipeline.swift:48-54` clamps to 256 MB process-wide forever — the following VAE decode and every later `session` generation inherit it; also fires at 512² whenever `--ref-latents` is on). Make `ParallelSelfAttention.swift:88-90` honor `blockCacheClearInterval` (it ignores it; `--eval-cache mid` currently only halves 26 of 46 clears). Vectorize the two 3M-iteration scalar image-I/O loops with vDSP (`ImageImport.swift:39-48`, `ImageExport.swift:53-68`) and reuse one CGContext. Decode the identity-I2I source image once, not twice (`FaceIdentityMask.swift:27` + `ImarelloPipeline.swift:531`). Drop the no-op `/1.0` scale and `+0.0` shift latent passes. Delete the dead query-chunked SDPA path + stale comments (`AttentionUtils.swift:65-66,115-158`; `AttentionTuning.swift:49-51`). Move PNG encode off the pipeline actor (idea S8). Hoist ref-latent concat where loop-invariant (`ImarelloPipeline.swift:684-690`).

### Tier 2 — measured-risk levers (each individually gated)

**4.8 Attention dtype from the joint sequence.** Pass a caller-decided dtype so both streams match. Two directions: **(a) all-f16 double blocks** — removes 6 casts/block and moves 5 blocks to f16 Steel (−1–3% denoise, −60 MiB; accumulation is f32 in-kernel, so the risk is limited to stored Q/K/V precision — the same trade already shipped for 20 of 25 blocks); **(b) zero-risk variant: all-f32 double blocks** — skip the pointless img f16 casts, keep f32 attention: strictly *more* accurate than today and still removes 4 casts. Ship (b) immediately, gate (a) on V-eval + vision. 

**4.9 Fold ÷16 into the quantization scales at load** (`scales *= 1/16` once; delete the per-call `x / scale`). Mathematically exact — same accumulator magnitude, so the overflow rescue is preserved; only the f16 rounding point moves. Removes a kernel + a full-size f32 temp per Linear call and avoids pushing small activations toward f16 subnormals. *V-eval + vision (schema-1.4 `unstructured_garbage` gate).*

**4.10 `asyncEval` + periodic hard sync.** Replace most per-block blocking `eval`s with `asyncEval` and a hard `eval` every N blocks to cap graph depth. Numerics unchanged; the risk is watermark growth, so tune N against the memory gate at 1024². Attacks the ~54/~124 stalls/step. *V-bit, V-bench with watermark gate.*

**4.11 VAE decode program.** (a) **Untile ≤1024²**: with the Small Decoder (35% narrower) and DiT unloaded, an untiled 128² decode plausibly fits — measured decode-stage RSS is ~143 MB; expected −20% (~1 s) @1024² and −50%+ @768², plus zero seam risk. Tier-gate and measure with `bench --mode vae-decode` under host pressure before defaulting; keep tiling as fallback. (b) If tiling stays for any size: retune `tileSize/overlap` at 768² (2.25× redundancy today; mflux ships 512 px tiles + 64 px overlap + cosine as a working reference point) and expose `VAETileConfig` via the CLI (currently unreachable). (c) **NHWC end-to-end**: ~38 NCHW↔NHWC round-trips per decode (largest 127 MB each); MLX convs are NHWC-native and upstream's stated best practice is transpose-weights-once-at-load. Est. −10–25% decode; layout-only, but touches every block — gate with the VAE numeric oracle + seam check. (d) Interval-gate the ~45 `clearCache` drains per tiled decode (headroom exists: decode watermark contribution is small).

**4.12 Re-A/B the 256 MiB denoise cache clamp at 1024².** The only recorded `cacheLimit` sweep ran at 512², where the clamp never engages (min-side 768). At 1024² the pool cap is far below per-block transients, so every large buffer round-trips the allocator. Unknown outcome; the analogous gating change historically bought −3.9%. Pure allocator policy; *V-bench.*

---

## 5. Tier 3/4 — research programs

### 5.1 Partial-pad conditioning (the flagship; potential −25–30% e2e)

**What we know.** Trimming the 512-token pad (`auto`) measured −32.4% e2e @512² / ~−10%/step @1024² and was reverted: subjects avert, faces fall into shadow, 1024² anatomy breaks (`Docs/TEXT_TOKENS.md`). The mechanism is now literature-backed: in softmax attention, surplus mass needs somewhere to go (StreamingLLM; Gu et al. "key biases"); FLUX-class *joint* attention demonstrably uses pad tokens as registers/working memory, and removing them deletes image detail (**Padding Tone**, NAACL 2025 — effectively a published confirmation of our regression); template/pad sinks absorb ~6× more image-query attention per token than semantic tokens (arXiv 2607.19139, studies FLUX.2/Qwen-class); registers matter most at high noise where composition is set (arXiv 2605.16147). Ecosystem corroboration: ComfyUI hard-codes a 512 text floor **specifically for klein** (`min_length=512`) while FLUX.2-dev gets `min_length=1` — independent evidence the *distilled* model is length-sensitive; Chroma retrained with all pads masked *except one* and requires that one at inference. Every reference implementation runs the DiT unmasked over the full padded window (Imarello parity confirmed, including pad-key masking inside the TE).

**The experiment ladder** (all rungs: seeds 42/0/7 × prompts stratified by real-token count ~10/~60/~200 × both canvases; gates = the three named failure modes + LPIPS drift vs pad-512 baseline as the sensitive instrument + the 1024²-and-human-subject vision rule; keep ceil/8 alignment):

- **R0 anchors**: pad-512 and `auto` — establish the per-stratum drift scale.
- **R1 bucket ladder**: total text length ∈ {96,128,192,256,384}. Hypothesis: monotone recovery with a knee at 128–384 (ComfyUI's FLUX.1 precedent says test 256 hardest). Even a 256 knee ≈ −25% of the DiT text lattice and halves `encode_te`.
- **R2 keep-k sinks**: k ∈ {1,8,32,64} pads retained after real tokens (Chroma k=1; StreamingLLM k=4; DiT registers ~32).
- **R3 denominator compensation (the mechanistic differentiator — run this early)**: keep k pads but add ≈`log(n_removed/k)` to retained-pad attention logits, preserving total pad exp-mass per softmax. If quality holds at tiny k, pads are pure softmax denominator → near-full trim speed for a one-line bias. If not, pads carry distributed register *content* → count/capacity matters and R1's knee is the answer.
- **R4 content-vs-count at fixed 512**: swap pad embeddings for empty-prompt "clean pads" / zeros / mean-pad. If clean pads match baseline, cache one pad bank offline and run the TE on real tokens only — **TE speedup on the product path for free, DiT unchanged**.
- **R5 step-scheduled trim**: full 512 on step 1, trimmed on steps 2–4 (composition commits early) — ~75% of the trim savings if it holds.

Compounding: any winning budget also shrinks `encode_te` proportionally (§4.6c) and shortens every joint-attention/Linear call. Ship bar: beats the R0 `auto` anchor on all three failure modes across strata at both canvases, per the promotion gate.

### 5.2 Full-f16 single-stream block (parked "fused qmm+SwiGLU", quantified)

Keep proj output f16 → SwiGLU f16 → concat f16 → `to_out` input f16. Removes the f32 upcast epilogue and halves proj/mlpHidden/to_out transients: est. **−8–12% denoise, −0.6 GiB @1024²** — the largest single engineering lever. Also the riskiest: adjacent to the raw-f16 TV-static failure that the pixel harness missed. Prereqs: 4.1/4.2 land first (isolate variables); gate on schema-1.4 `unstructured_garbage` + full V-eval + vision at both canvases; consider per-tensor dynamic scale instead of the flat ÷16 ("better activation scale" is the ledger's other open lever).

### 5.3 mlx-swift pin: hold, and watch

0.31.6 **is the latest release** (core v0.31.1 vendored; mlx-swift `main` still vendors v0.31.1 — nothing to gain today). The next release that moves the core past 0.32.x ships material wins for exactly this stack: f32-dequant qmm (upstream measured 8–31% prefill on M1-class for 4-bit affine + bf16 scales — bf16 is emulated pre-M5), split-K qmm for small M (modulation/AdaLN), gemv/qmv_wide, conv-channel padding + tiled-unfold conv_transpose (VAE), `metal::set_metallib_path` (would simplify `ensure-metallib.sh`), compile correctness fixes. Known bump frictions: Package.swift restructure past v0.31.2 (jaccl split), contiguity/concat behavior shifts, qmm bit-exactness changes (f32 dequant) → full V-eval + golden-stage SSIMs on bump day. Action: subscribe to mlx-swift releases; budget one validation day when the core moves.

### 5.4 External acceleration families — verdicts with citations (§8)

Comprehensively surveyed for 4-step distilled applicability: token merging (ToMe/ToMA — validated only at 35–50 steps; NO-GO), region-adaptive (RAS needs ≥4 warmup steps, i.e. never activates; RALU reschedules the trajectory; NO-GO), sparse attention (SpargeAttn — per-model calibration + CUDA kernels + short-seq overhead; NO-GO now), linear-attention retrofits (training required; NO-GO), SVDQuant/Nunchaku (proven *on 4-step schnell* but needs requant + CUDA kernels; NO-GO under the 4-bit lock — the strongest evidence that per-call calibrated approximation suits 4-step models if the weight format ever reopens), layer skip (caching in disguise; NO-GO). **One conditional survivor: per-call int8 quantized attention** (SageAttention-class; validated on 4-step FLUX-schnell; open-source Apple-GPU existence proof in Draw Things' Metal Quantized Attention, BSD-3). Step-count-agnostic and training-free — but a custom-Metal-kernel project, best on M5-class silicon, and attention is only 7–17% of an Imarello step: expected e2e on M2 ≈ 2–5%. Verdict: **not now on M2; first candidate if an M5-class device becomes a target.** TeaCache-family stays NO-GO (an MLX port exists — for 20–50-step recipes only).

### 5.5 Tier 4 — iOS memory (not speed)

- **`stagedAggressive`** is today a declared-but-inert enum case (every policy check is `== .resident`/`!= .resident`). Design per docs: DiT block-group weight drop+reload (~0.7 GiB double / ~1.4 GiB single split) targeting jetsam headroom on the phone; expect *slower* e2e — measure watermark on-device via the harness, never ship on Mac.
- **VAE encoder checkpointing** (§3.4) is the cheap, high-value iOS item — it likely unblocks the P7 on-device I2I acceptance item.
- Small-canvas preview via TAEF2 stays parked (never for export); an MLX TAEF port for klein exists as a reference point.

---

## 6. Closed-experiment ledger (do not re-litigate)

Measured and closed — with the receipt: cacheLimit sweep 0/256M/1G/2G/default → 0% five ways (512²; see §4.12 for the open 1024² question) · full & block `MLX.compile` → −1–3%, NO-GO (staged unload kills amortization) · glue fusion (`qkv_rope` ~5%) → parked by Slice C · raw unscaled f16 qmm → TV static that passed 15/15 pixel · hybrid FA2 host tiles → +30% · scalar Metal FA → >10 min/step · custom MTL4/NAX on M2 → `neuralAccel=no` · MFA/s4nnc port → GPL + Steel-class already shipped · AdaLN split/unload → ~4% shared, below bar · 3-bit → cancelled · TeaCache/TaylorSeer/FORA/ToCa → no safe skip at 4 steps (§5.4 re-confirms with 2025-26 literature) · `EvalCachePolicy.high` / VAE `evalEachChunk` → forbidden crash class · resident on 8 GB → regression (35.6 s cache-miss) · `--text-tokens auto` as default → reverted with documented mechanism (recovery path is §5.1, not re-promotion) · SDPA chunk-size sweep → 512 shipped, rest noise · Klein AdaLN Qwen-style split → N/A (shared, small). Genuinely open per the docs and this report: fused qmm+SwiGLU (§5.2), better activation scale (§5.2), `--eval-cache mid` (≥16 GB hosts only), S3 cold-metallib cost, S10 2–3-step quality trade, and everything in §4–5.

---

## 7. Measurement appendix

All artifacts: `outputs/engine-research-2026-08-16/` (bench JSON schema 1.3; gitignored — regenerate with the commands below). Binary: release + full Steel metallib, pin match verified; pad-512 defaults; seed 42; thermal state nominal; one Metal owner throughout.

```bash
D=outputs/engine-research-2026-08-16
imarello bench --mode dit-one-step --width 512  --height 512  --op-profile --probe-density off --json $D/op-512.json
imarello bench --mode dit-one-step --width 1024 --height 1024 --op-profile --probe-density off --json $D/op-1024.json
imarello bench --mode te-only  --json $D/te-only.json
imarello bench --mode load-only --json $D/load-only.json
for S in 512 768 1024; do imarello bench --mode vae-decode --width $S --height $S --json $D/vae-$S.json; done
for S in 512 640 768 896 1024; do imarello bench --mode pressure-map --width $S --height $S --probe-density blocks --json $D/pressure-$S.json; done   # res-ladder mode is broken (§3.6)
imarello bench --mode mem-stages --json $D/mem-stages.json
```

Process gaps to close alongside any Tier-1 work: **(a)** persist `textTokens` / `attnLinearCompute` / `vaeVariant` (+ tile config) in `BenchConfig`/report JSON; **(b)** fix §3.5's `MemoryProbe` so orchestrator samples carry real MLX numbers; **(c)** fix §3.6 so ladders run; **(d)** the promotion gate (1024² + human subjects + multi-seed vision) stays mandatory for every default change proposed here.

Doc corrections found en route: `Docs/MEMORY.md` TE "2.26 GB" → ~1.7 GB materialized · `PromptEmbedCache.swift:7` "~16 MB f32" → 7.9 MB bf16 · `PERF.md:14` still quotes the auto-era 74 s for the product 1024² (now 79.0 s) · `--attn-chunk-size` help says 256, shipped default 512 · `AttentionTuning.swift:49-51` "default mlx (chunked SDPA)" describes a dead path.

---

## 8. Sources

**Padded-text conditioning:** Padding Tone (arXiv 2501.06751, NAACL 2025) · Template tokens as semantic registers, FLUX.2/Qwen-class (arXiv 2607.19139) · DiT attention-sink causal analysis (arXiv 2605.09313) · Registers for pixel-space DiTs (arXiv 2605.16147) · StreamingLLM (arXiv 2309.17453) · When Attention Sink Emerges (arXiv 2410.10781) · Why LLMs attend to the first token (arXiv 2504.02732) · ViT registers (arXiv 2309.16588) · Chroma keep-1-pad (huggingface.co/lodestones/Chroma1-Base; diffusers ChromaPipeline docs) · ComfyUI klein `min_length=512` vs dev `min_length=1` (`comfy/text_encoders/flux.py`, `comfy/model_base.py`) · diffusers/BFL flux & flux2 pipelines (unmasked DiT over padded window) · kohya #1488 / OneTrainer #950 ("padding trained into the model") · diffusers discussion #10177 (length 64–512 A/B).

**Few-step acceleration:** ToMe-SD (CVPR-W 2023) · ToMA (arXiv 2509.10918) · RAS (arXiv 2502.10389) · RALU (arXiv 2507.08422) · SageAttention2 (arXiv 2411.10958; validated on 4-step schnell) · SpargeAttn (arXiv 2502.18137) · PAROAttention (arXiv 2506.16054) · CLEAR (arXiv 2412.16112) · SVDQuant/Nunchaku (arXiv 2411.05007) · Draw Things Metal Quantized Attention + Metal FlashAttention v2/2.5 (releases.drawthings.ai; BSD-3 shaders in `ccv`).

**MLX upstream (verified in source at the v0.31.1 tag unless noted):** f32 accumulation in Steel attention/qmm/GEMM (`steel_attention.h`, `sdpa_vector.h`, `quantized.h`, `steel/gemm/mma.h`) · qmm dtype promotion (`ops.cpp:4342-4360`, `dtype.cpp:45-48`) · concat promotion (`ops.cpp:1086-1090`) · SDPA dispatch (`scaled_dot_product_attention.cpp:588-637`) · NHWC-native convs + transpose-at-load guidance (docs; discussion #724) · core v0.31.2 split-K qmm #3120 · v0.32.0 qmv_wide #3764, `set_metallib_path` #3597, compile fix #3720 · main: f32-dequant qmm #4241 (8–31% on M1-class), conv_transpose tiled unfold #3845, conv channel padding #3904 · mlx-swift: 0.31.6 latest; core-bump blocker #446; finalizer leak fix #448 · ecosystem: mflux tiling params (mflux#407), mlx-teacache, mlx-taef, flux-2-swift-mlx (external klein baselines).
