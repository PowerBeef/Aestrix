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
**Milestone A landed (2026-08-18, `imarello direct-spike`).** Layer-0 `q_proj` (M=512, K=2560, N=4096, 4-bit affine gs=64) dispatched directly against the metallib's `affine_qmm_t_float16_t_gs_64_b_4_alN_true_batch_0` from a hand-built command buffer: **100% bit-exact vs the MLX oracle** (2,097,152/2,097,152 values), and already −15% per call vs MLX's runtime path (4.17 vs 4.92 ms) despite a blocking wait per dispatch. ABI notes proven in code (`Sources/ImarelloDirect/`): host names use the Metal type spelling (`float16_t`); scalars are 32-bit; B=1 binds nothing past buffer 7; grid `((N+31)/32, (M+31)/32, 1)` × `(32,2,2)`. The metallib also carries `affine_qmm_t_nax_*` variants for the A19/M5 path. **Milestone B landed (same day, `direct-spike --stage layer`).** One full Qwen3 decoder layer (M=512) as a SINGLE command buffer, 18 dispatches: qmm ×7 + steel attention from the metallib, plus four bespoke glue kernels (`DirectGlueKernels`: f32-accumulate RMSNorm, non-traditional RoPE, SiLU·mul, add — Stage 1's fusion ideas landing where they belong). Stage-by-stage verification vs an identical-math MLX oracle: **cosine 1.0000000 at every checkpoint** (silu_mul 0.9999998, f16 rounding). Steel-attention ABI notes: function constants 200/201/300/301/302 (align/mask/causal/sinks), `AttnParams` is 56 bytes of scalars + 4×3 int64 strides with **no padding** (the one bug found), and its explicit strides let `[L, H·D]`-layout tensors flow straight through — every transpose/reshape in the attention block disappears. Per-layer wall: direct 40.0 ms vs MLX 41.5 ms *with* per-call re-encoding and a blocking wait — at M=512 both are compute-dominated, as Stage 0 predicted; the structural win is Milestone C's. Next: **Milestone C** — embed + 27 layers + final-norm taps in one (or few) command buffers, resident-encode A/B vs the 1.59 s baseline, then the M≈30 splice variant where MLX's per-op floor was the whole cost.

*Gate to continue to the DiT:* TE prototype ≥1.3× on the resident encode with embeddings matching the oracle (per-layer cosine ≥ 0.9999).
*DiT gate:* ≥1.25× on the DiT loop **or** ≥0.5 GiB watermark cut at ≤5% slowdown, both canvases.
*Kill:* under the bars → research artifact; the study's numbers still stand as the definitive MLX-tax measurement.

**Stage 3 — Full runtime (only if Stage 2 passes).**
Bespoke TE next (the §3 anomaly means the *relative* win is largest there), VAE last (conv/bandwidth-bound; least to gain). Tokenizer stays CPU/Swift. MLX exits the dependency graph only at the end of this stage — and only if every gate held.

**Platform strategy throughout (updated 2026-08-18, user decision):** floors raised to **macOS 26.2 / iOS 26.2** — every Apple Silicon Mac runs macOS 26, so no supported chip is dropped. The bespoke engine targets **Metal 4 as a single API surface** (MTLTensor, MPP `matmul2d` tensor ops, new command encoding) with no Metal 3 dual path; NAX execution remains runtime-gated to A19/M5-class GPUs, and MLX's own NAX host dispatch now compiles in everywhere (fork `079609a`, `MLX_METAL_NO_NAX` dropped).

## 7. References

- Draw Things Metal FlashAttention v2.5 shaders — BSD-3, in `ccv` (permissive FA/GEMM reference, NAX existence proof on A19 Pro).
- MLX Steel kernel sources (MIT) — readable in the fork at `Source/Cmlx/mlx/mlx/backend/metal/kernels/steel/` (reference, and the bar to beat).
- Apple: Metal 4, MTLTensor, Metal Performance Primitives `matmul2d` (WWDC25); tzakharko Apple-GPU microbenchmarks (NAX fp16/int8 rates).
- This repo: `Docs/ENGINE_RESEARCH.md` §2.2 (op profile), `Docs/PERF.md` 2026-08-18 (baselines, S2/S4 refutations — both are *evidence* for where a bespoke engine should and should not spend effort).
