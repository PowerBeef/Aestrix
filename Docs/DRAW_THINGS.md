# Draw Things optimizations — implications for Aestrix

**Date:** 2026-08-13  
**Ask:** How Draw Things runs image models “impossibly” fast and in less RAM than the model “should” need, and what Aestrix should copy.

This is a **research note**, not a license to rewrite the runtime. Community Draw Things code is **GPL-v3**. Aestrix is MIT. Copy **principles**, not source.

## Sources

- Liu Liu, [*Stretch iPhone to its Limit*](https://liuliu.me/eyes/stretch-iphone-to-its-limit-a-2gib-model-that-can-draw-everything-in-your-pocket/) (2022-11)
- Engineering @ Draw Things: [MFA](https://engineering.drawthings.ai/p/integrating-metal-flashattention-accelerating-the-heart-of-image-generation-in-the-apple-ecosystem-16a86142eb18) (2023-08), [SD3 / s4nnc](https://engineering.drawthings.ai/p/from-iphone-ipad-to-mac-enabling-rapid-local-deployment-of-sd3-medium-with-s4nnc-324bd5e81cd5) (2024-06), [MFA 2.0](https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c) (2025-01), [BF16 / FP16](https://engineering.drawthings.ai/p/bf16-and-image-generation-models-803cf0515bee) (2025-04), [Qwen Image](https://engineering.drawthings.ai/p/optimizing-qwen-image-for-edge-devices) (2025-09), [ANE in a custom stack](https://engineering.drawthings.ai/p/making-apple-neural-engine-work-in) (2026-04)
- [draw-things-community](https://github.com/drawthingsai/draw-things-community), [s4nnc](https://github.com/liuliu/s4nnc), [ccv MFA](https://github.com/liuliu/ccv/tree/unstable/lib/nnc/mfa), [metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
- Aestrix: `Docs/MEMORY.md`, `Docs/PERF.md`, `Docs/ROADMAP.md`

---

## What Draw Things is

Offline iPhone / iPad / Mac app (Liu Liu). The speed/RAM claims are real **versus naive stacks** (full-graph Core ML, PyTorch+MPSGraph, ggml/GGUF, unfused attention), not versus physics.

The engine is **not** MLX and **not** end-to-end Core ML:

| Layer | Role |
|-------|------|
| **ccv / libnnc** | C runtime: tensors, alloc, CUDA/Metal kernels |
| **s4nnc** | Swift binding |
| **SwiftDiffusion** | Model ports (FLUX.1, SD3, Qwen Image, Klein, …) |
| **Metal FlashAttention** | Fused SDPA + GEMM (Philip Turner + Liu) |
| **Optional Core ML** | **Only** int8 matmul on ANE (2026); does not own the graph |

They ship Klein in-app (e.g. `flux_2_klein_4b_q6p.ckpt`). They often prefer **6-bit / 8-bit S** for quality. Aestrix’s lock is Klein **4B**, **4-bit default**, no user bf16.

---

## Four principles

### P1 — Own intermediates

2022 iPhone post: SD 1.5 FP16 UNet ~1.6 GiB weights. Naive MPSGraph op-by-op peaked at **~6 GiB**. They walked it to **~2 GiB** without changing math:

1. Pace Metal submissions (≈8 ops in flight).
2. Explicit layout (MPSGraph permute hid a 500 MiB transpose).
3. In-place softmax / strided GEMM (MPSGraph allocated a second 500 MiB even when aliased).
4. Never materialize attention `N×N` (FlashAttention).

**Aestrix:** MLX + Steel FA already delete the N×N matrix. The remaining fat allocation is **DiT weights (~2.0 GiB active)**, not softmax scratch. Watermark ~2.99 / 3.76 GiB @ 512 / 1024 is weights + MLX cache.

### P2 — Peak = max(resident subgraph), not sum(model)

- **SD3:** AdaLN ~670 M params, tiny FLOPs, no activation dependency. Precompute **all timesteps**, then sample a slimmer transformer. Quantized peak ≈ **2.2 GiB**.
- **Qwen Image (20B):** ~**7 B** params are timestep-only AdaLN. Cache `1001×718×3072` instead of loading 7 B.
- TE / denoise / VAE never co-reside.

**Aestrix:** Module staging and `projectContext` hoist are done. Klein’s modulation is **shared**, not per-block (see count below) — DT’s “unload 7 B AdaLN” does **not** transfer.

### P3 — Fused attention + quantized GEMM are throughput

MFA 2.0 vs other Apple stacks (their numbers): up to **20%** FLUX.1/SD3 on M3/M4; ~**2%** on older SoCs for FLUX.1; up to **25%** vs mflux / **94%** vs ggml on M2 Ultra.

**Aestrix:** Product path is already **MLX Steel fused FA** (D=128). Same *class* of win. Full DiT `MLX.compile` is a measured NO-GO.

### P4 — Numerics are a speed feature on M1/M2

MMDiT activations grow with depth. BF16 has range; **M1/M2 BF16 is emulated and ~50% slower than FP16** (DT 2025-04). Their recipe: FP32 residual / last LN; FP16 block body; scale late layers (FLUX.1: ×8 on last double blocks); apply 1/√d **before** FA.

**Aestrix (measured 2026-08-13, `aestrix load-dit --dump-dtypes --width 512`):**

| Site | dtype |
|------|--------|
| 4-bit packed weights | `uint32` |
| Quantization **scales** | **`bfloat16`** (tiny: e.g. `[3072, 48]`) |
| Latents, AdaLN, residual, FFN, Q/K/V, Steel FA @ 512² | **`float32`** |

The M2 “bf16 emu on activations” tax **does not apply**. Scales are bf16 but negligible. A follow-up A/B lowered `f16SeqThreshold` **2048 → 512** so 512² image QKV matches the 1024² f16 path: denoise/step **−4.3%**, e2e **−3.3%**, watermark flat, eval-regression 15/15. Residual/FFN stay fp32.

---

## Klein AdaLN count (read-only)

FLUX.2 [klein] does **not** use per-block AdaLN like SD3/Qwen. Three **shared** `Flux2Modulation` Linears plus a small time embed and `norm_out`:

| Tensor | Shape (no bias) | Params |
|--------|-----------------|-------:|
| `double_stream_modulation_img` | 3072 × (3072×3×2) | 56.6 M |
| `double_stream_modulation_txt` | 3072 × (3072×3×2) | 56.6 M |
| `single_stream_modulation` | 3072 × (3072×3×1) | 28.3 M |
| `time_guidance_embed` (no guidance) | 256×3072 + 3072×3072 | 10.2 M |
| `norm_out` AdaLN | 3072 × 6144 | 18.9 M |
| **Modulation family** | | **~171 M** |

DiT 4-bit pack is ~2.18 GiB ≈ **~4.4 B** 4-bit weights. Modulation is **~4%** of DiT — below the 10–15% bar for “split and unload.” Precomputing 4-step modulation is still a tiny hoist (milliseconds), not a RAM strategy.

`context_embedder` (7680→3072, ~23.6 M) is already hoisted once per generate.

---

## Catalog vs Aestrix

| Draw Things technique | Aestrix | Rec |
|----------------------|---------|-----|
| Staged TE / denoise / VAE | Done | Keep |
| Flash / Steel fused FA | Done (Steel, D=128) | Do not port MFA |
| VAE decode-only + tiles | Done | Keep |
| Block checkpoint + `clearCache` | Done | Keep |
| Context hoist | Done | Keep |
| Prompt-embed cache | Done | Keep |
| AdaLN precompute / unload | Shared mods ~4% of Klein DiT | **Deprioritize** as RAM work |
| DiT weight streaming | Parked `stagedAggressive` | iOS / watermark only |
| 4/6/8-bit palletization | 4-bit default; 3/6/8 exist | 3-bit = honest RAM SKU |
| 8-bit S + ANE int8 | Not in stack | Park (M3+ / iOS 26) |
| FP16 body + FP32 residual | Acts are fp32; **QKV f16@512 shipped** | No full-body f16 |
| Full-graph Core ML | Never used | Do not start |
| Full DiT compile | NO-GO | Stay parked |
| Rewrite in s4nnc | GPL + years | Do not |

---

## Honest limits

1. Klein **4-step** already spent the big latency win. A 20% kernel win is ~3.7 s @ 512 or ~15 s @ 1024, not another 2×.
2. Live peak is **DiT weights (~2.04 GiB)** at both 512 and 1024. Streaming cuts watermark/jetsam; it usually **slows** e2e.
3. ANE **1.8×** is M3/M4 + int8 + macOS 26, not M2 + 4-bit MLX.
4. Do not vendor GPL Draw Things / s4nnc.

---

## Recommendations

| ID | Item | Priority | Why |
|----|------|----------|-----|
| **R2** | Log compute dtypes; trial FP16 body if hot path is bf16 | **Done (probe)** | Activations are **fp32**; scales only are bf16. No emu-fix to ship |
| **R3** | DiT weight streaming as `stagedAggressive` | P7 / 8 GB comfort | RAM, not speed |
| **R6** | Quality-gate 3-bit as optional Tier-L | If RAM SKU wanted | No architecture |
| **R1** | 4-step modulation batch | Low | ~4% of weights; tiny compute |
| **R4** | Port MFA / s4nnc | No | Steel FA + MIT |
| **R5** | ANE / 8-bit S | Park | Wrong SoC and stack |

**Measured:** activations fp32; QKV f16@512 is the product default (2026-08-13). Streaming / ANE still parked.

---

## Bottom line

Draw Things is fast and lean because they **control allocation, fuse attention, quantize with a quality metric, and split graphs so parameter-heavy AdaLN is not resident in the hot loop** — and they **refuse to let Core ML or MPSGraph own the model**.

Aestrix already has the MLX equivalents that matter: staged residency, Steel FA, tiled/decode-only VAE, checkpointing, context hoist. Klein’s shared modulation is **not** Qwen’s 7 B AdaLN. Remaining DT-shaped work is **M2 compute dtype**, then **weight streaming for iOS**. The “2× faster / half the RAM” story is mostly already spent on 4-step + 4-bit + Steel FA.
