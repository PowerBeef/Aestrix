# Bare-Metal Klein — feasibility study

**Date:** 2026-08-18 · **Branch:** `research/bare-metal` (off `checkpoint-2026-08-18-engine-uplift`) · **Status:** exploration — no product decision implied.

**The question:** ditch MLX and engineer a from-scratch image-generation backend for Apple Silicon, tailored exclusively to FLUX.2 Klein 4B, to squeeze out every bit of performance.

## Verdict up front

Do not replace MLX in one move — the measured data says the DiT denoise loop is already running at **~⅔ of this GPU's theoretical peak**, so a perfect bespoke engine has at most ~1.5× of kernel headroom on M2, not 5×. But a bespoke engine is **not** a bad idea — it is three specific good ideas plus one bad one, and they separate cleanly:

| Bespoke thesis | Verdict on evidence |
|---|---|
| "MLX kernels are slow; ours would be faster" | **Mostly false on M2.** Steel GEMM/FA deliver ~65% of peak on our workload; world-class hand-tuned kernels reach 70–85%. Ceiling ≈ 1.2–1.3× on the big GEMMs. |
| "MLX's graph/dispatch overhead costs us" | **True where M is small.** The DiT loop shows near-zero overhead signature (identical utilization at both canvases), but the TE stage's time is almost invariant to sequence length — a fixed-cost anomaly worth ~1.5 s/generate that a recorded-command engine would crush. |
| "A static, whole-pipeline memory plan beats a general allocator" | **True and the strongest guaranteed win.** Every tensor shape is known before the first kernel for a fixed canvas. A planned engine turns the 3.00 GiB watermark into an engineered number (~2.3–2.5 GiB plausible) and makes 8 GB/iOS-jetsam robustness a design property instead of an empirical one. |
| "Fusions MLX won't do" | **True, moderate size.** qmm+scale+SwiGLU epilogue, RMSNorm+RoPE+split, AdaLN folded into GEMM epilogues — glue is ~5% of a step, boundary traffic maybe another 5–10%. Optimistically 10–20% of a step. |

**Recommended shape:** don't "ditch MLX" — *out-grow* it, stage by stage, from inside the working runtime, with the checkpoint as the safety net and the current engine as a byte-exact oracle. The ladder below ends with a fully bespoke engine *only if* the intermediate stages earn it.

---

## 1. What we already know from measurement

All numbers from this repo's benches on the 8 GB M2 machine of record (PERF.md / ENGINE_RESEARCH.md):

- **Op profile of a DiT step** (product stack): quantized Linears **50.8% / 43.5%** (512²/1024²), FFN **33.6% / 30.4%**, Steel flash attention **7.1% / 17.3%**, glue ~5%. Linear+FFN own **84% / 74%** of a step.
- **Blocking sync count:** ~54 `eval` per step @512², ~124 @1024².
- **Step times:** 4.22 s @512² (joint L=1536), 14.7 s @1024² (L=4608).
- **TE encode:** 1.58 s — and nearly **invariant to sequence length** (encoding ~30 real tokens instead of 512 was speed-neutral in the S2 A/B, on both core 0.31.1 and 0.32.1).
- **The 0.32.1 bump** moved encode_te −19–21% purely from upstream kernel improvements — evidence that staying on MLX keeps paying dividends for free.
- **Competitive check:** mflux (MLX Python) does Klein 1024² in 31.7 s on an M1 Max 32-core; per-GPU-core that's the same utilization class as us. Nobody in the MLX ecosystem is leaving a large multiple on the table.

## 2. The roofline result (new, this study)

FLOPs per DiT step (Klein: 5 double + 20 single blocks, inner 3072, fused single-stream proj 3072→27648, joint L = image+512):

| Canvas | Linear FLOPs | Attention FLOPs | Total | Measured step | Effective throughput |
|---|---:|---:|---:|---:|---:|
| 512² (L=1536) | 9.3 T | 0.7 T | **10.0 T** | 4.22 s | **2.37 TFLOPS** |
| 1024² (L=4608) | 27.8 T | 6.5 T | **34.4 T** | 14.7 s | **2.34 TFLOPS** |

M2 10-core peak ≈ 3.6 TFLOPS (fp16 = fp32 rate on Apple GPUs). Two conclusions:

1. **~65% of peak, through a 4-bit dequantizing GEMM path, is genuinely good.** The remaining kernel headroom to a hand-tuned ~80–85% is ≈ 1.2–1.3×.
2. **The utilization is identical at both canvases.** If MLX's lazy-graph build, encoder, or dispatch overhead were a material cost in the DiT loop, the smaller canvas (2.75× less work per step, same op count) would show visibly worse effective throughput. It doesn't. **The DiT loop is compute-bound; the MLX tax there is small.** (Stage 0 verifies this directly with a Metal trace.)

## 3. The TE anomaly — where the bespoke case is strongest

The TE encode contradicts the DiT result: cutting the encoded sequence from 512 to ~30 tokens (17× less compute) changed nothing.

**Stage 0 measured it (2026-08-18, `outputs/engine-uplift-2026-08-18/stage0/`) — three results, one correction.**

1. **Lazy weight I/O is exonerated.** A residency A/B (`encodePromptPairResident`: two different-prompt encodes under one TE residency) measures staged load+unload at ~150–200 ms and first-touch weight materialization at **~20 ms**. The steady-state, fully-warm encode is **~1.59 s** — the cost is real execution, not hidden loading. (An earlier read blaming `ParallelFileReader` I/O was a truncated-trace artifact: those reads were between-trial staged reloads. `encodePrompt` stages internally — load, encode, unload, every call — which the first probe missed.)
2. **Per-stage GPU busy (full-coverage Metal trace of one dit-one-step run @512²):** TE encode **74%** busy (77 kernels; two ~90–117 ms gaps at encode start), DiT step **84.5%** busy under tracing (322 kernels; many 40–59 ms eval-boundary gaps; tracing itself inflates the step 4.22→4.59 s, so the clean-run figure is ≈90%), VAE decode **77%** busy.
3. **The sequence-length invariance now has a coherent mechanism.** At M=512 the encode is ~1.2 s of genuine GPU compute (~2.0 TFLOPS — DiT-class utilization) plus ~0.4 s of gaps. Cut the tokens to ~30 and the compute floor drops to ~0.1 s — but the measured total barely moved, so the per-op/per-submission fixed costs that big kernels hide **expand to fill the stage at small M**. This is why the S2 TE-splice bought nothing: the splice removed the FLOPs but not the ~300-op submission overhead.

**What this means for the ladder:** the DiT loop is confirmed compute-bound (Stage 0 gate passes as predicted — Stage 2 is scoped to memory + fusions, with replay worth ~8–10% of a step). The TE is the opposite: a recorded-command path would let the splice's real win through (~1.6 s → ~0.3 s class), making the TE stage the first bespoke prototype target after all — for the overhead, not the I/O.

## 4. What a bespoke engine buys — honest estimates

| Lever | Mechanism | Estimated win (M2) |
|---|---|---|
| **Whole-step command replay** | All 4 steps share every shape. Record the step once (indirect command buffers, or Metal 4 command structures now that the floor is 26.2), replay 4×; per-step CPU/graph/dispatch cost → ~0 | DiT: 0–10% (compute-bound). TE/small-M stages: potentially large — this is what kills the §3 anomaly |
| **Static memory plan** | Shapes known at plan time → hand-placed buffers, double-buffered chunk streaming, zero allocator | Watermark 3.00 → ~2.3–2.5 GiB @1024²; deterministic 8 GB / iOS-jetsam behavior. The strongest *guaranteed* win |
| **Cross-op fusions** | qmm+÷16+SwiGLU single kernel; RMSNorm+RoPE+qkv-split; AdaLN scale/shift folded into GEMM epilogues | 10–20% of a step, optimistically |
| **Silicon-specific paths** | Metal 4 MPP tensor ops (NAX) on A19/M5; int8 quantized attention (SageAttention-class; Draw Things' BSD-3 MFA shaders as reference) | NAX: MLX's fork **already ships this for free** — bespoke adds little there. int8 attention: 2–5% e2e on M2, real on NAX hardware |

**Compound best case on M2:** denoise 58.7 → ~40–45 s @1024², TE 1.6 → ~0.3 s, decode roughly flat (conv/bandwidth-bound) → **e2e 67.5 → ~48–52 s (≈1.3–1.4×)**, plus ~0.5 GiB watermark. On 512²: 21.9 → ~15–17 s. **There is no engine that produces 1024² in 20 s on an M2 base — the arithmetic doesn't exist.** The strategic wins beyond the numbers are iOS: energy per image, jetsam headroom, and latency on A19-class silicon.

## 5. What it costs

- **~155 MB of battle-tested kernels.** Our own measurements prove Steel GEMM/FA are near-optimal for this workload. Every one of those kernels would need re-deriving, and this project's history shows how sharp the edges are (the raw-f16 qmm "TV static", the ÷16 input-cast overflow, the RoPE layout subtleties).
- **Free upstream wins stop.** The 0.32.1 bump handed us −19–21% encode for the cost of a pin move.
- **Months of engineering** and a Metal codebase to maintain across OS majors (Metal 4 migration is happening *now* — MLX absorbs that churn for us today).
- **Mitigation that makes this uniquely feasible here:** we own a byte-exact oracle (the current engine), a pixel+vision eval gate, golden metrics, and a bench harness with provenance. A bespoke kernel can be verified tensor-by-tensor against MLX output at every stage. Few projects attempting this rewrite have that.

## 6. The ladder — staged plan with gates

Everything on `research/bare-metal`; `main` stays at the checkpoint. Every stage passes the standard pixel+vision gates; host-safety rules apply (kernel source compiles are Metal compiles — serial with benches).

**Stage 0 — Measure the MLX tax. DONE (2026-08-18).**
Residency A/B + full-coverage Metal trace (§3): DiT step ≈90% busy clean (compute-bound confirmed — replay worth ~8–10%/step), TE 74% busy with per-op overhead that explodes at small M, decode 77%. Lazy weight I/O exonerated (~20 ms first-touch).
*Gate result:* Stage 2 scopes to **memory plan + fusions + small-M overhead**; the TE becomes the first recorded-command prototype target (it unlocks the splice's ~1.3 s).

**Stage 1 — Recover the overhead inside MLX. RUN AND LARGELY REFUTED (2026-08-18).**
Cheap levers measured before writing any Metal:
- `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER` (command-buffer batching): **flat** — the TE's gaps are not commit boundaries.
- `--attn-no-qkv-checkpoint` (new bench flag; removes ~25 of ~54 evals/step): **worse both ways** — 512² step +2.6%, watermark +160 MiB. The eval boundaries are not dead time; they keep the lazy graph and allocator tight.
This is the **fourth** independent experiment (asyncEval, streaming-512², S4 epilogue, qkv-checkpoint-off) with the same shape: the current engine sits at a local optimum of MLX's execution model — single-knob deviations lose in both directions. Elementwise-fusion traffic math (~1%-class on M2) plus the DiT's ≈90% busy means the original "≥5% per fusion" gate is unpassable without beating Steel's GEMMs, which §2 says is a ≤1.3× game. **The recoverable overhead is structural, and Stage 1 folds into Stage 2.**

**The direct-dispatch insight (what Stage 1's post-mortem bought).** The 155 MB metallib is a plain `MTLLibrary` of *named, complete* kernels — `affine_qmm_*`, `steel_attention_*`, `rms_norm`, RoPE — the exact binaries MLX executes. A bespoke engine therefore does **not** need to author competitive kernels: it can create pipeline states from MLX's own compiled functions and encode them into recorded command buffers over a static buffer plan. MLX's kernels without MLX's runtime. The kernel-ABI contract (argument order, template-name mangling) is readable in `quantized.cpp` / `nojit_kernels.cpp`, and every intermediate can be verified against the MLX oracle.

**Stage 2 — Direct-dispatch engine (rescoped 2026-08-18; the real prize).**
A standalone Metal module that dispatches **MLX's own metallib kernels** from recorded command buffers over a static buffer plan — no kernel authorship. First prototype: the **TE forward** (27 identical fixed-shape Qwen3 layers, M=512 — the simplest stage and the biggest measured prize: ~1.6 s → ~1.2 s compute floor at M=512, and ~0.3 s-class once the splice's M≈30 encode rides on it). Then the DiT step (memory-plan win + the ~8–10% eval-boundary gap). Weights land in MTLBuffers in the layout the kernels already expect; every layer's output verifies against the MLX oracle.
**Milestone A landed (2026-08-18, `imarello direct-spike`).** Layer-0 `q_proj` (M=512, K=2560, N=4096, 4-bit affine gs=64) dispatched directly against the metallib's `affine_qmm_t_float16_t_gs_64_b_4_alN_true_batch_0` from a hand-built command buffer: **100% bit-exact vs the MLX oracle** (2,097,152/2,097,152 values), and already −15% per call vs MLX's runtime path (4.17 vs 4.92 ms) despite a blocking wait per dispatch. ABI notes proven in code (`Sources/ImarelloDirect/`): host names use the Metal type spelling (`float16_t`); scalars are 32-bit; B=1 binds nothing past buffer 7; grid `((N+31)/32, (M+31)/32, 1)` × `(32,2,2)`. The metallib also carries `affine_qmm_t_nax_*` variants for the A19/M5 path. **Milestone B landed (same day, `direct-spike --stage layer`).** One full Qwen3 decoder layer (M=512) as a SINGLE command buffer, 18 dispatches: qmm ×7 + steel attention from the metallib, plus four bespoke glue kernels (`DirectGlueKernels`: f32-accumulate RMSNorm, non-traditional RoPE, SiLU·mul, add — Stage 1's fusion ideas landing where they belong). Stage-by-stage verification vs an identical-math MLX oracle: **cosine 1.0000000 at every checkpoint** (silu_mul 0.9999998, f16 rounding). Steel-attention ABI notes: function constants 200/201/300/301/302 (align/mask/causal/sinks), `AttnParams` is 56 bytes of scalars + 4×3 int64 strides with **no padding** (the one bug found), and its explicit strides let `[L, H·D]`-layout tensors flow straight through — every transpose/reshape in the attention block disappears. Per-layer wall: direct 40.0 ms vs MLX 41.5 ms *with* per-call re-encoding and a blocking wait — at M=512 both are compute-dominated, as Stage 0 predicted; the structural win is Milestone C's. **Milestone C landed (same day, `direct-spike --stage forward [--seq 512|30]`).** All 27 layers in ONE command buffer (486 dispatches + tap blits, ping-pong scratch — the static plan working). Tap cosines vs the identical-math oracle: 0.99997–0.99999 at both lengths. Results that rewrite the §3 story once more:

| | L=512 | L=30 (splice regime) |
|---|---|---|
| identical-math MLX loop (warm) | 1100 ms | **80.7 ms** |
| direct engine (single CB) | **1074 ms** | **79.3 ms** |
| real TE resident encode (today's probe) | 1594 ms | — |

Three conclusions. **(1)** The production TE's ~1.6 s was never an irreducible MLX per-op floor — a bare MLX loop of the same math runs 80 ms at L=30. The cost lives in the production module's *structure*: per-layer evals, f32-cast `[1,L,H,D]` intermediates, bf16 paths, staged reload, bank/cache plumbing. **(2)** The direct engine matches bare MLX at both lengths (1.02×) and beats the *real* resident encode **1.48×** at L=512 — it wins by deleting the runtime around the kernels, exactly the Stage-1 post-mortem's thesis. **(3)** The splice prize is confirmed and quantified: a direct (or even lean-MLX) spliced encode costs **~0.1 s**, vs ~1.6 s today — the full TE-stage prize is ~1.4–1.5 s per cache-miss generate. *Gate: 1.48× vs the resident-encode baseline with cosines ≥0.9999 —* **PASS** *(vs identical-math; f16-vs-bf16 drift against the real TE is Milestone D's check).* **Milestone D landed (same day, `t2i --te-engine direct`).** The complete production-shaped encode on the engine — chat-templated tokens, CPU 4-bit embedding dequant (**bit-exact** vs MLX `dequantized`), 27 masked layers, tap assembly to `[1, 512, 7680]` bf16 — wired into `t2i` by pre-seeding the embed cache (zero pipeline changes). Findings, in order:

1. **f16 cannot run this model.** Real Qwen activations reach |1.5e4| at the taps alone; the all-f16 engine produced 100% non-finite outputs on real inputs (Milestone B/C passed only because random test inputs were tame). The engine now runs **bf16 end-to-end like the product**; the metallib carries every kernel in bf16.
2. **Steel's mask ABI has a trap**: the kernel offsets the mask by `head × M_strides[1]`, so a single-head `[1,1,L,L]` mask must be passed with broadcast strides `{0, 0, L}` — the natural `{L·L, L·L, L}` reads out of bounds for heads ≥ 1.
3. **Fidelity vs the real TE**: real-token rows cosine **0.9999999**; pad rows 0.992 (484 near-identical rows iterating chaotically — far inside the envelope R4 proved vision-equal when swapping pad content entirely).
4. **End-to-end quality gate: PASS.** Fox 512² seed 42 on direct-TE conditioning: pixel **98.5** (product: 98.4), vision clean, image a near-twin of the product fox.
5. **A live product bug found and fixed en route**: the audit-era embed-cache validation required `.float32` while the TE emits bf16 — **every load failed, deleted the entry, and re-encoded**, so the cache never hit after `9ad69cd` and each generate silently re-paid the full TE stage. Fixed (bf16-or-f32 + dtype-aware size guard) and verified: repeat generates now leave the entry untouched and run ~2 s faster. **This fix is a hotfix candidate for `main`** — the checkpoint carries the dead cache.

**F1 landed (2026-08-18 evening, `direct-spike --stage dit-block`): the DiT-side track opens with a PASS.** One FLUX single-stream block (block 0, joint L=1536) as a single command buffer — metallib qmm ×2 + steel **full** (unmasked, non-causal) attention + seven fused glue kernels — verified against the REAL product block with shipped weights and the production ÷16/×16 protocol: **cosine 1.0000000** (maxAbs 0.0036, f16 rounding) and **9% faster per block** (152.2 vs 167.4 ms) before any recording gains. The scale-protocol folding is exact: ÷16 cancels inside q/k RMSNorm (rms is scale-invariant), ×16 folds into V-extraction and the SwiGLU input, ÷16 into the whole `to_out` input (one in-place pass over the attention half keeps the qmm input uniformly scaled); steel writes its output directly into the concat buffer via O-strides, so the concat itself costs nothing. **F2 landed (same session): the double-stream block PASSES** — one command buffer, 26 dispatches (10 qmm + steel joint attention + the F1 glue): image cosine 0.9999999, text 1.0000000, 9% faster than the product block (153.5 vs 168.5 ms). Joint assembly costs nothing: the per-stream RMSNorms write text/image rows straight into the joint Q/K/V buffers at offsets, and the split out-projections read the joint attention output through qmm pointer offsets.

**F3 landed (same session): the full 25-block step PASSES** — `DirectDiTStep`, the engine's DiT core: 5 doubles + 20 singles in ONE command buffer (~355 dispatches) over a static scratch plan, with Klein's shared modulation (15 vectors condition the whole step). Cosine vs the product block loop **0.9999996** (maxAbs 0.116 — 25 blocks of f16 rounding), **10.2% faster** (3859.5 vs 4299.1 ms) — precisely the eval-boundary gap Stage 0 measured, now recovered by the recorded structure. One composition bug found and fixed: the ping-pong originally wrote back into the caller's input buffers, corrupting re-runs (inputs now blit into internal ping-pongs).

**F4 landed (same session): the COMPLETE denoise stage PASSES at both canvases.** `x_embedder → 25 blocks → norm_out/proj_out → Euler`, one command buffer per step, product-computed per-step conditioning uploaded between steps (~50 KB). Verified against the full product DiT module end to end:

| | 512² | 1024² |
|---|---|---|
| final-latent cosine (4 steps) | 0.9999266 | 0.9998499 |
| product 4-step (warm) | 16.84 s | 60.37 s |
| direct 4-step | **15.21 s (1.11×)** | **52.31 s (1.15×)** |

The 1024² result lands **without chunk-streaming** — the engine materializes the full 255 MB single-proj and still beats the product's chunked path. **Gate status (≥1.25× or ≥0.5 GiB at ≤5%): not yet met on either arm, with the path clear.** Speed sits at 1.15×; memory: the engine's static plan currently sums to ≈1.46 GiB scratch + 1.93 GiB weights ≈ 3.4 GiB at 1024² vs the product's 3.00 GiB watermark, because the double-phase and single-phase scratch sets — disjoint in time — are allocated side by side instead of aliased over one slab (max ≈ 0.86 GiB instead of sum), rope buffers can rotate in place (−113 MB; each thread owns its pair), and the single-proj can run in fixed chunks (255 → 128 MB). That pass targets ≈ 2.6–2.7 GiB total, with the remaining distance to the −0.5 GiB gate coming from chunking the double-phase FF (−130 MB class). **The memory pass landed (same session) — THE DiT GATE IS MET.** Four mechanisms, all numerically transparent (cosines bit-identical at every stage): ① in-place RoPE (each thread owns its element pair — four buffers deleted); ② a **placement heap** with a persistent zone ordered `[hA][eA][hB][eB]` so the dead-at-blit halves are contiguous and back the single-phase ping-pongs by aliasing, plus a compute zone sized `max(double-phase, single-phase)` instead of their sum (the heap is untracked; every encoder joins an `MTLFence` chain); ③ the 27648-wide single-proj and the double-FF run in row-halves with chunk-local indexing (proj scratch halves); ④ the embedder/head scratch overlays the compute zone (pre-phase runs before block scratch lives, head phase after it dies). Result: engine-owned **2.44 GiB @1024² vs the product's 3.00 GiB (−0.56 GiB)** and 2.11 vs 2.57 @512² (−0.46), at **1.15×/1.11× faster** — the "≤5% slowdown" condition is exceeded in the wrong direction. Gate letter: 1024² memory arm **PASS**; 512² sits 46 MB short of −0.5 with named scraps remaining (txt-FF chunking, jointO reuse). The engine's footprint is now a printed, deterministic number instead of an empirical watermark.

**The first fully bespoke images landed (same session, `imarello direct-generate`).** The composed research command stages like the product: direct TE (engine freed) → conditioning on the product module (then unloaded — the engines never coexist) → direct DiT denoise → MLX small-decoder decode. Results:

- **Fox 512², seed 42**: clean image, pixel PASS + vision PASS. Total **23.1 s** including one-time engine builds (steady-state parts sum to ~16.5 s-class vs the product's 21.9 s).
- **Bikini anatomy probe 1024², seed 42**: pixel PASS + full anatomy-checklist vision PASS — navel on the midline at natural height, correct limb counts and proportions. Total **63.9 s vs the product's 67.5 s**, at a 2.43 GiB engine footprint.
- **Dancer 1024², seed 0**: pixel PASS, **vision FAIL — three arms**, exactly reproducing the product's documented dancer-s0 body-plan failure (Q1 baseline). The engine is faithful both ways: it reproduces the model's successes *and* its failures; the rescue remains the model-level `--two-stage`, orthogonal to the engine.

Stage 2's ladder is complete: both gates cleared, the full pipeline runs on bespoke TE + DiT with MLX reduced to conditioning math and the VAE, and every stage is oracle-verified. **Conditioning moved on-engine (same session): `DirectConditioner`.** Timestep sinusoid on CPU; the temb MLP, three modulation heads, and AdaLN-out on the metallib's **f32 qmm** (exact dtype parity at M=1); context projection on the bf16 qmm (same kernel family as the product's call); rope tables via the weight-free `Flux2PosEmbed`. Verified against `precomputeStepConditioning`/`projectContext`/`prepareRotaryEmbeddings`: **cosine 1.0000000 on every field, every step**. In `direct-generate` the conditioning stage drops **1394 → 150 ms** and the 1.9 GiB product-module load disappears; the regenerated fox is promotion-neutral vs the previous bespoke image (SSIM 0.9999, LPIPS-lite 0.001 — bit differences only because MLX dispatches qmv at M=1 where the engine uses qmm_t). **MLX's remaining role in `direct-generate` was the VAE decode alone — since closed by the V2–V5 decoder port below (pipeline swap pending the attention-GEMM speed fix).**

**The three productization items landed (2026-08-18, late session):**

1. **Persistent pipeline (`DirectPipeline`), honestly measured.** On the 8 GB host, every co-residency configuration loses to fresh-process staged generation (full residency: TE pages back from compressed swap, ~7 s encodes; DiT-resident + staged TE: the TE build evicts the DiT, 24–26 s denoises; even VAE-staged in-process trails by ~4 s/image from allocator churn). The class ships with a **memory-tiered policy**: <12 GB stages everything per generate (validating the product's staged-residency lock for the bespoke engine too); ≥12 GB (iPhone 17 Pro, larger Macs) keeps all engines resident — the configuration this class exists for, unverifiable on this host.
2. **Direct-VAE V1: the conv ABI is proven.** One decoder-class 3×3 conv (256→256 @64×64 NHWC) through the metallib's `implicit_gemm_conv_2d` kernel vs the MLX oracle: cosine **0.9999733** (not bit-exact — MLX dispatches Winograd for this shape; the direct decoder can force implicit-gemm for correctness first). `MLXConvParams<2>` = 17 int32 + pad-to-72 + 3×4 int64 strides + groups + flip; `ImplicitGemmConv2DParams` = 10 int32. The decoder port is de-risked.
3. **FULL promotion gate: PASS → the T2I default is now `--te-engine direct`.** Campaign: the 5-prompt regression set × seeds 42/0/7 (15/15 pixel PASS), four anatomy probes @512², and fox/bikini/dancer @1024² — **22/22 pixel, zero hard fails**, vision-clean on the human-figure probes (bikini 1024²: navel midline and proportions correct; dancer s42: two arms, two legs). `--te-engine mlx` remains the escape hatch; the i2i path is unchanged. Post-flip smoke on the bare default: fox 98.5, tests 86/86.

Timing honesty: at M=512 the bf16 direct encode is ≈ parity with the real resident TE (~1.6 s — bf16 qmm is slower than f16 on M2); the engine's speed prize remains the splice regime (M≈30 → ~0.1–0.2 s class) and everything the working cache now makes free. Next: the splice integration on the direct engine, then the DiT-side decision.

**The VAE decoder port landed (2026-08-18, V2–V5): the last MLX compute stage now runs on the direct engine.** The ladder, each rung oracle-verified against real Small Decoder weights (f32 end to end, NHWC):

| Rung | Scope | Cosine vs oracle | Max abs diff |
|---|---|---|---|
| V2 | mid resnet block (384ch @64×64) | 1.0000000 | 4.3e-5 |
| V3 | full mid block incl. single-head D=384 attention | 1.0000000 | 5.8e-5 |
| V4 | **whole Small Decoder untiled @512²** vs product `decodePacked` | 1.0000000 | 2.1e-5 |
| V5 | **tiled 1024²** — product cosine stitcher driving direct tile decode | 1.0000000 | 2.5e-5 |

Mechanics: convs run on the metallib's `implicit_gemm_conv_2d_float32_*` kernels with the host dispatch math mirrored from conv.cpp (all needed variants incl. the O=3 `bn8_wm4_wn1` config are in the library); everything else on seven small glue kernels (`dv_groupnorm_act` fused GN+SiLU, bias+SiLU, nearest-2× upsample, add, naive NT/NN matmuls, row softmax). The mapper's squeeze of the mid-attn 1×1 convs to dense `[C,C]` Linears means the attention projections ride the same matmul kernel. The entry (BN denorm + unpatchify) stays as trivial MLX ops; V5 reuses the product's `decodeLatentsTiled` stitcher with `DirectVAE.decodeTileNCHW` as the closure, so tiling behavior is byte-shared with the product.

Timing honesty (superseded same session — see V6): at first landing the port was correctness-proven but ~2× slow (V4 1.96 s vs ~0.93 s; V5 10.3 s vs ~4.5 s), and the pipeline swap was deliberately withheld.

**V6 (2026-08-18/19): the speed levers landed and the direct VAE went into the pipeline.** Two fixes, found by measurement, not guess:

1. **Steel GEMM ABI proven** (`steel_gemm_fused_{nt,nn}_float32_float32_*`): function constants 10/100/110 false + 200/201/202 align flags at PSO creation; GEMMParams (8×i32 + 3×i64 + 3×i32, pad 72) at buffer 4; A/B/D at 0/1/3 (2/5/6/7 unbound at B=1); M2 'g'-class tiles NT=bm64/bn32/bk32/wm2/wn2, NN=bm64/bn64/bk16/wm2/wn2; grid (tn,tm,1) × (32,wn,wn). All six attention matmuls rerouted: mid block 390 → 100 ms. But the per-stage profile then showed attention was never the whale —
2. **GroupNorm was**: the one-threadgroup-per-group kernel cost **138.7 ms per call @512²×192** (32 TGs, uncoalesced group-strided reads) — ~850 ms of V4's 1632 ms across the decoder's 22 GN calls. Rewritten as a three-pass coalesced reduction (per-(channel,chunk) partials with contiguous warp reads → per-group finalize → elementwise apply): **9.07 ms** (15×).

| | direct (naive) | + steel GEMM | + GN rewrite | product MLX |
|---|---|---|---|---|
| V4 untiled 512² | 1960 ms | 1632 ms | **808 ms** | ~932 ms |
| V5 tiled 1024² | 10.3 s | 8.3 s | **4.16 s** | ~4.5 s |

Cosine 1.0000000 at every step of the chase. **The direct decoder now beats the product decode on both canvases**, so the swap gate passed: `DirectPipeline`/`direct-generate` decode on `DirectVAE.decodePacked` (BN denorm + unpatchify as trivial MLX entry ops, UNet fully on-engine, product `decodeLatentsTiled` stitcher for 1024²). Validated end-to-end: fox 512² s42 pixel PASS + vision PASS; bikini 1024² s42 pixel PASS + full anatomy-checklist vision PASS (navel midline, proportions, no tile seams), total 64.8 s with the VAE stage at 4.9 s. **MLX's compute role in `direct-generate` is now entry math and glue only — every UNet/DiT/TE FLOP is bespoke.** **Product promotion (2026-08-19): `t2i --vae-engine direct` is the default.** The pipeline grew an injection seam (`PackedLatentDecoding` protocol + `ImarelloPipeline.setPackedDecoder`; the engine builds lazily inside the decode stage so staged residency is preserved, and the DiT still stages out first). Gate evidence, strongest first: same-seed outputs from the two engines differ by **at most 1 LSB in ≤0.0036% of subpixels at both canvases** (the f32 maxAbs 2.5e-5 sits ~370× below the 8-bit quantization step, so only rounding-boundary values flip) — which is why the anatomy re-sweep was waived; eval-regression 15/15 on the direct engine; fox + bikini 1024² vision PASS (tiled, no seams); W1T3 bench A/B: decode −11.6%/−11.7%, e2e 21.81→21.63 s @512² and 67.50→66.46 s @1024², MLX watermark flat, peak RSS +48–78 MB (1.90 GB, no 8 GB threat). Bench: `--vae-engine` knob + provenance (schema 1.5); `eval-anatomy.sh` now honors `T2I_EXTRA`. `--vae-engine mlx` is the escape hatch; the i2i decode path is unchanged.

**DiT promotion (2026-08-19): `t2i --dit-engine direct` is the product default — the product T2I path is now fully bespoke (TE + conditioning + DiT + VAE).** The pipeline grew a second seam (`PackedLatentDenoising` + `setPackedDenoiser`): the engine consumes the pipeline's own noise + embeds, fires an `onStep` hook before each step (cooperative cancellation + denoise-step trace events stay real — `encodeDenoiseStep` syncs per step), and loads/frees its 1.93 GiB inside the call, so staged residency holds. Injection is CLI-level, so the iOS app keeps the MLX DiT — deliberate, because the engine names Steel kernels explicitly and would bypass MLX's NAX dispatch on A19/M5 (N2/N3 interaction; NAX selection becomes engine work if iOS ever flips).

Unlike the VAE flip this is pixel-changing (final-latent cosine ≈ 0.9999, image drift mean 1.26/255 with 2.6% of pixels >8), so it ran the FULL gate on its own merits: eval-regression **15/15**; the full 12-probe anatomy set pixel-clean, **11/12 vision-clean — the twelfth (dancer-s42 @512², three arms) reproduced composition-identically on the MLX control**, i.e. a model-level failure faithfully rendered by both engines (its 1024² s42 counterpart is clean on both; the rescue remains `--two-stage`); fox/bikini/dancer 1024² pixel + vision PASS. Bench W1T3: **e2e 22.04→20.57 s @512² (−6.7%) and 66.28→60.34 s @1024² (−9.0%)**, denoise/step −12.4/−11.3%, **peak MLX active −20%** (2.22→1.77 GB — conditioning + DiT left MLX), peak RSS +5.7/+3.7% (≈1.99 GB — the RSS question resolved benignly; the engine's owned buffers largely replace, not stack on, the MLX pool). Constraints enforced by the CLI: 512 text tokens, no pad-keep/pad-bias (R3 diagnostics are MLX-path-only); i2i/identity remain MLX (unported); `--dit-engine mlx` is the escape hatch. Bench grew `--dit-engine` + provenance (schema 1.6).

**Conv tile fix (2026-08-19, post-promotion): the cheap experiment beat the Winograd plan.** The micro sweep (`vae-micro` now sweeps bm×bn on the up3 shapes) showed MLX's own host heuristic (bn=64 whenever bm=64) pads 25% of every tile row over the decoder's N=96 convs — bm64/bn32 runs the 192→96 conv **40% faster** (67→40.5 ms) and 96→96 20% faster. `encodeConv` now prefers the aligned tile whenever N%64≠0 but N%32=0: untiled 512² decode 808→**772 ms**, tiled 1024² 4.16→**3.94 s**, outputs **byte-identical** (tiling partitions outputs; K-accumulation order is unchanged — verified 0 differing pixels on the product smoke). The bespoke dispatch now beats MLX's dispatch of MLX's own kernel. **Winograd is parked with numbers**: the remaining conv gap is ~100–150 ms @512² (0.5% e2e) for three new transform ABIs — revisit only if the VAE becomes a bottleneck again. Per-stage profiling lives in `direct-spike --stage vae-profile` / `vae-micro`.



*Gate to continue to the DiT:* TE prototype ≥1.3× on the resident encode with embeddings matching the oracle (per-layer cosine ≥ 0.9999).
*DiT gate:* ≥1.25× on the DiT loop **or** ≥0.5 GiB watermark cut at ≤5% slowdown, both canvases.
*Kill:* under the bars → research artifact; the study's numbers still stand as the definitive MLX-tax measurement.

**Stage 3 — Full runtime (only if Stage 2 passes).**
Bespoke TE next (the §3 anomaly means the *relative* win is largest there), VAE last (conv/bandwidth-bound; least to gain). Tokenizer stays CPU/Swift. MLX exits the dependency graph only at the end of this stage — and only if every gate held.

**Platform strategy throughout (updated 2026-08-18, user decision):** floors raised to **macOS 26.2 / iOS 26.2** — every Apple Silicon Mac runs macOS 26, so no supported chip is dropped. The bespoke engine targets **Metal 4 as a single API surface** (MTLTensor, MPP `matmul2d` tensor ops, new command encoding) with no Metal 3 dual path; NAX execution remains runtime-gated to A19/M5-class GPUs, and MLX's own NAX host dispatch now compiles in everywhere (fork `079609a`, `MLX_METAL_NO_NAX` dropped).

## N-track: NAX on the direct engine (2026-08-19)

**The direct engine owns NAX now — and the A19 Pro question is answered: YES.** The engine dispatches `affine_qmm_t_nax_*` by name (ABI recovered from `quantized.cpp::qmm_nax`: byte-identical buffer layout to the Steel qmm — w/s/b/x/y at 0–4, K/N/M at 5–7 — with a 64-tile grid and the alN-by-N name; conditions K%64==0, non-f32, weights transposed). `DirectNAX.probe` mirrors core's `is_nax_available()` (OS ≥ 26.2, arch gen ≥ 17, 'p'-suffix parts ≥ 18); `DirectDiTStep(useNAXQmm:)` routes every DiT projection; `direct-generate --nax` and the adapter carry the flag, refused on ineligible GPUs.

**Device gate (iPhone 17 Pro, iOS 26.6, `ImarelloSpikes` runner): PASS.** `applegpu_g18p` gen 18 → eligible. NAX vs Steel at the DiT shape M=4096/K=3072/N=3072: **cosine 1.0000000, 99.5% bit-exact, max |Δ| 0.0039 (f16-ULP class)** — and **40.0 → 11.8 ms, 3.38×**, on a low-battery phone. qmm is 44–51% of a denoise step ⇒ projected ~1.4–1.5× device denoise once the direct DiT runs there. That 2026-08-19 gate qualified qmm only. Current fork inspection also finds packaged `sdpa_full_self_attention_nax` D=128 variants; V2 inventories their exact symbols and actual J families (1536/2816/4608), but none is promoted until ABI, tensor, full-consumer, and image qualification passes.

Hard-won facts from the device ladder:
- **PSO creation for NAX kernels SUCCEEDS on non-NAX GPUs** (M2 builds the pipeline fine; only execution needs the hardware) — the arch probe is the only valid gate, never a compile/creation check.
- **The iOS app's Cmlx-bundle metallib is the thin JIT-era set (4.5 MB vs 155 MB)** — Xcode auto-compiles the .metal sources without the nojit instantiations. Under nojit this is a **latent runtime breakage of the main iOS app** (last device-validated pre-nojit, 2026-08-16): any MLX op on device aborts on kernel lookup. The spike app ships a full `-sdk iphoneos` metallib as an app resource (155 MB, all 42 kernel files incl. NAX) and the engine takes it by URL; the spike's synthetic path is deliberately MLX-free so it cannot trip over MLX's own metallib. **The main app needs the same treatment (full metallib + pointing MLX at it) before any device generate.**
- Two more fork fixes shipped en route (`e5b42df`): 0.32.1's CPU `jit_compiler.cpp` calls `std::system` (absent on iOS) — excluded on Apple platforms with a four-symbol stub (`available()` = false → interpreted CPU fallback; GPU untouched; macOS byte-identical fox + 86/86 after the pin move).

Next rungs: device dit-step spike (whole 25-block step, NAX vs Steel), then a full on-device bespoke generate — both need weights reachable from the runner; then the iOS product decision.

## Metal Engine V2 foundation (2026-08-21; unpromoted)

V2 adds a reversible whole-T2I backend seam rather than treating `DirectPipeline` as an automatic product backend. Selection occurs before weight loading; unsupported requests stay on V1, while a V2 failure after start is surfaced without mid-run fallback. The internal `ImarelloPlan` graph now provides deterministic operations, lifetimes, placement, a digest, and independent overlap/alignment verification. Legacy placement reproduces 180/318.75/513 MiB raw scratch at 512/768/1024; this is a proof of the current layout, not a memory-improvement claim.

Production shader strings have moved into source-built `imarello-direct.metallib`, paired with a platform/hash/symbol/function-constant ABI manifest. The full ~155 MB MLX pack remains beside it for MLX editing and compatibility. Loader foundations include strict safetensors validation, page-rounded shared mappings, stage transforms, and explicit mapping/buffer lifetimes. The isolated Metal 4 executor owns argument tables, aligned constant uploads, explicit residency, allocators, and commit feedback; it is not selected by the product and has not yet run the required real qmm/attention qualification.

The worktree also contains V2-only atomic BF16 prompt caching, exact memory ledgers, multi-step conditioner batching, optional stage/trajectory capture manifests, OS 26.4 gating for native int4 tensor types, and signed-app artifact wiring. Focused MLX-free tests and the UI-only Simulator build pass. Full captures/tolerances, mapped native loading throughout, Tier B/C streaming, block/tile captures, Metal 4 performance gates, complete GPU-native handoffs, A/B corpus evidence, and physical-device qualification remain open. Therefore no V2 speed, memory, quality, reliability, or device-runtime claim is made here.

## 7. References

- Draw Things Metal FlashAttention v2.5 shaders — BSD-3, in `ccv` (permissive FA/GEMM reference, NAX existence proof on A19 Pro).
- MLX Steel kernel sources (MIT) — readable in the fork at `Source/Cmlx/mlx/mlx/backend/metal/kernels/steel/` (reference, and the bar to beat).
- Apple: Metal 4, MTLTensor, Metal Performance Primitives `matmul2d` (WWDC25); tzakharko Apple-GPU microbenchmarks (NAX fp16/int8 rates).
- This repo: `Docs/ENGINE_RESEARCH.md` §2.2 (op profile), `Docs/PERF.md` 2026-08-18 (baselines, S2/S4 refutations — both are *evidence* for where a bespoke engine should and should not spend effort).
