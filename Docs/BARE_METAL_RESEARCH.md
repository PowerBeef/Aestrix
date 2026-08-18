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

The TE encode contradicts the DiT result: cutting the encoded sequence from 512 to ~30 tokens (17× less compute) changed nothing. Candidate explanations — per-layer kernel-launch floors at small M, weight-streaming bound qmv dispatches, MLX graph-build fixed cost — are unconfirmed; **napkin math rules none of them in cleanly** (weight traffic alone predicts ~15 ms, not 1.5 s). This is the single most suspicious MLX behavior we have measured, it costs ~1.5 s of every cache-miss generate, and the same signature likely floors the per-step modulation/AdaLN small-M work. **Stage 0's first target.**

## 4. What a bespoke engine buys — honest estimates

| Lever | Mechanism | Estimated win (M2) |
|---|---|---|
| **Whole-step command replay** | All 4 steps share every shape. Record the step once (Metal 3 indirect command buffers — available at our macOS 15 floor), replay 4×; per-step CPU/graph/dispatch cost → ~0 | DiT: 0–10% (compute-bound). TE/small-M stages: potentially large — this is what kills the §3 anomaly |
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

**Stage 0 — Measure the MLX tax (days).**
Metal System Trace (`xctrace record --template 'Metal System Trace'`) of: one 512² and one 1024² single-step DiT run, and one TE encode. Read: GPU busy vs gap %, encoder CPU time, per-launch overhead, and *what the TE stage actually does for 1.58 s*.
*Gate:* if DiT-loop GPU busy ≥ ~90% (predicted by §2), the bespoke case formally reduces to memory + fusions + small-M stages — Stage 2 gets scoped accordingly. The TE trace decides whether a recorded-command TE is the first prototype.

**Stage 1 — Custom kernels inside MLX (weeks).**
`MLXFast.metalKernel` (verified available in the fork) lets us author fused Metal kernels while MLX still owns tensors, scheduling, and memory:
1. fused qmm+÷16+SwiGLU epilogue (the #1 and #2 op buckets meet here);
2. RMSNorm+RoPE+split glue kernel (~5% bucket);
3. int8 attention spike (reference: Draw Things MFA, BSD-3).
*Gate:* each fusion must show ≥5% of a step in an A/B, quality-gated. *Kill:* if the fusions can't clear that individually, the "fusions" pillar of the full engine falls, and with it most of its M2 case.

**Stage 2 — The Klein kernel: a bespoke denoise engine (the real prize; 1–2 months).**
A standalone Metal module owning *only* the 4-step DiT loop: our own 4-bit weight layout (packed for the fused kernels), a static buffer plan, the whole step recorded into indirect command buffers and replayed 4×, Stage 1's fused kernels as the compute core. TE and VAE stay on MLX. Verified block-by-block against MLX activations, then the full gate.
*Gate to continue:* ≥1.25× on the DiT loop **or** ≥0.5 GiB watermark cut at ≤5% slowdown, at both canvases.
*Kill:* under both bars → the engine stays a research artifact; Stage 1's fused kernels are still keepable inside MLX.

**Stage 3 — Full runtime (only if Stage 2 passes).**
Bespoke TE next (the §3 anomaly means the *relative* win is largest there), VAE last (conv/bandwidth-bound; least to gain). Tokenizer stays CPU/Swift. MLX exits the dependency graph only at the end of this stage — and only if every gate held.

**Platform strategy throughout:** engine core targets Metal 3 features (ICBs, simdgroup matrices, function constants — macOS 15 floor holds, every supported chip keeps working); a Metal 4 / MPP-tensor-ops (NAX) path is a gated variant for 26.2+ on A19/M5, mirroring the existing NAX gating.

## 7. References

- Draw Things Metal FlashAttention v2.5 shaders — BSD-3, in `ccv` (permissive FA/GEMM reference, NAX existence proof on A19 Pro).
- MLX Steel kernel sources (MIT) — readable in the fork at `Source/Cmlx/mlx/mlx/backend/metal/kernels/steel/` (reference, and the bar to beat).
- Apple: Metal 4, MTLTensor, Metal Performance Primitives `matmul2d` (WWDC25); tzakharko Apple-GPU microbenchmarks (NAX fp16/int8 rates).
- This repo: `Docs/ENGINE_RESEARCH.md` §2.2 (op profile), `Docs/PERF.md` 2026-08-18 (baselines, S2/S4 refutations — both are *evidence* for where a bespoke engine should and should not spend effort).
