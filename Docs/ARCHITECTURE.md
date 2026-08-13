# Aestrix architecture

## Goal

Native Swift/MLX runtime for **FLUX.2-klein-4B** on Apple Silicon: low-RAM-first staged inference, prequant weights, T2I + single-image I2I.

## Layers

```text
AestrixCLI / host apps
        │
AestrixRuntime   (Pipeline actor, StageOrchestrator, MemoryPolicy)
        │
   ┌────┴────┬────────────┐
AestrixText  AestrixDiT   AestrixVAE
 (Qwen3 tap) (MMDiT 5+20) (FLUX.2 AE)
        │
AestrixWeights  (Hub snapshot load, module dirs)
        │
mlx-swift
```

## Model constants (Klein 4B distilled)

| Constant | Value |
|----------|-------|
| Double-stream blocks | 5 |
| Single-stream blocks | 20 |
| Attention heads | 24 |
| Head dim | 128 → inner **3072** |
| Joint attention dim | **7680** (3×2560 Qwen layers 9/18/27) |
| Max text length | **512** (full pad) |
| Latent channels (VAE) | 32 → 128 after unshuffle |
| RoPE axes | (32,32,32,32), θ=**2000** |
| Default steps / guidance | **4** / **1.0** |

## Correctness footguns

1. Qwen chat template (empty think block with `enable_thinking=False`)  
2. Full 512 pad to DiT (not trimmed tokens)  
3. Timestep ×1000 when max(t) ≤ 1  
4. Exact scheduler sigmas vs mflux/diffusers  
5. TE unload before DiT load  

## Package modules

| Target | Role |
|--------|------|
| `AestrixCore` | Types, errors, tier, memory probe, Hub pins (`WeightPreset.pin`), **pure math** (RoPE, timestep emb, modulation layout, scheduler) |
| `AestrixWeights` | Resolve snapshot paths; read `hf download` metadata SHA |
| `AestrixText` | Tokenizer + Qwen3 3-layer tap |
| `AestrixDiT` | MMDiT + `MetalFlashAttention` / `AttentionTuning` |
| `AestrixVAE` | Encode / decode-only / tiled cosine blend |
| `AestrixRuntime` | Orchestrator + public pipeline API |
| `AestrixBench` | Multi-trial timings, pressure maps, attn knobs |
| `AestrixEval` | Pixel quality (no Metal) |
| `AestrixCLI` | Executable |

## Phase 1 pure math (no Metal required)

| API | File | Notes |
|-----|------|-------|
| `Flux2RoPE` | `Flux2RoPE.swift` | 4-axis θ=2000, axes 32×4; grid/text ids |
| `Flux2TimestepEmbedding` | `Flux2TimestepEmbedding.swift` | sin/cos emb + ×1000 if max(t)≤1 |
| `Flux2ModulationMath` | `Flux2ModulationMath.swift` | split shift/scale/gate; apply; SiLU |
| `Flux2Scheduler` | `Flux2Scheduler.swift` | linear-μ time-shift Euler; image_seq_len = (H/16)(W/16) |

Scheduler primary path: `linspace(1, 1/N, N)` → exponential time-shift with μ from linear map (0.5@256 → 1.15@4096) → append 0; step `x + (σ_{t+1}-σ_t)·e`.

## Phase 2 MMDiT (`AestrixDiT`)

| Type | Role |
|------|------|
| `Flux2Transformer` | Full Klein-4B graph (5 double + 20 single) |
| `Flux2Attention` / `Flux2ParallelSelfAttention` | Joint + fused single-stream attn |
| `MetalFlashAttention` / `AttentionTuning` | Steel fused FA (product), hybrid FA2, float4 Metal research path |
| `Flux2FeedForward` / `Flux2SwiGLU` | mlp_ratio 3.0 |
| `Flux2Modulation` / `AdaLayerNormContinuous` | AdaLN conditioning |
| `Flux2PosEmbed` / `Flux2TimestepGuidanceEmbeddings` | RoPE + temb |
| `TransformerWeights.loadQuantized` | 4-bit affine load from `transformer/*.safetensors` |
| `DiTModule` | Staged loadable wrapper |

Weight keys match mlx-community 4-bit packs (`x_embedder`, `transformer_blocks.N.attn.to_q`, … + `.scales`/`.biases` after `quantize`).

## Snapshot wiring

`AestrixPipeline` resolves `ModelPaths.resolveIfPresent(config)` and passes the snapshot into `DiTModule` / `VAEModule` / `TextEncoderModule`.

| CLI | Action |
|-----|--------|
| `aestrix info` | Shows `snapshot_ready` + path |
| `aestrix load-dit` | Staged quant DiT load + param leaf count |
| `aestrix load-vae` | Staged VAE load + param leaf count |
| `aestrix load-te` | Staged quant Qwen3 TE load + param leaf count |
| `aestrix encode-prompt` | Staged TE encode → `[1,512,7680]` embeds |
| `aestrix mem-selftest` | Dry residency only (forces empty models dir) |

## Phase 4 Text encoder (`AestrixText`)

| Type | Role |
|------|------|
| `Qwen3TextEncoder` | Pruned Qwen3 (layers 0…26) with HF hidden_states taps 9/18/27 → concat 7680 |
| `Qwen3Attention` / `Qwen3MLP` / `Qwen3DecoderLayer` | GQA + Q/K RMSNorm + SwiGLU + RoPE θ=1e6 |
| `QwenChatTemplate` | Fixed empty-think chat string (`enable_thinking=False`) |
| `QwenTokenizer` | Byte-level BPE from snapshot `tokenizer/tokenizer.json` |
| `TextEncoderWeights.loadQuantized` | 4-bit affine load; drops unused layers 27–35 + norm + rotary_emb |
| `TextEncoderModule` | Staged loadable wrapper + `encode(prompt)` |

## Phase 3 VAE (`AestrixVAE`)

| Type | Role |
|------|------|
| `Flux2VAE` | encode / decode / decodePackedLatents |
| `Flux2Encoder` / `Flux2Decoder` | UNet-style residual stacks |
| `VAEWeights.load` | Quantize mid-block attention Linears only; load hub shards |
| `VAEModule` | Staged loadable wrapper |

Requires Metal metallib for real load (same as DiT).

## Phase 5 Staged T2I (`AestrixRuntime`)

| Type | Role |
|------|------|
| `LatentOps` | Packed noise `[B,H/16·W/16,128]`, pack/unpack, text/img RoPE ids, Euler step |
| `ImageExport` | NCHW float → PNG via ImageIO |
| `DiTModule.projectContext` | Hoist `[B,T,7680]→[B,T,3072]` once per generate (not per step) |
| `AestrixPipeline.generate` | TE → unload → DiT Euler → unload → VAE decode → PNG |
| `aestrix t2i` | CLI with `--output`, `--seed`, `--steps` |

Text RoPE ids: `[0,0,0,token_i]` (diffusers `_prepare_text_ids`). Distilled path: **no CFG**, guidance `nil`.

## Phase 6 Strength I2I (+ Tier B identity)

| Type | Role |
|------|------|
| `ImageImport` | Disk image → NCHW `[-1,1]`, align to 16, cover-crop; `loadCGImage` for Vision |
| `Flux2VAE.encodePackedForDiT` | Encode → patchify → BN normalize (inverse of decodePacked) |
| `Flux2Scheduler.strengthSchedule` | Full N-step schedule; `StrengthScheduleCurve` (color / identity / linear) |
| `LatentOps.scaleNoise` | `(1−σ)·x₀ + σ·ε` at start sigma |
| `LatentOps` ref helpers | `referenceImageIds` (`t=10…`), concat denoise+refs, slice pred, clean-pull |
| `FaceIdentityMask` | Vision face rect → soft packed mask for regional σ + pull |
| `IdentityPreserveConfig` | Tier-B knobs; `.identityPreset` / `.disabled` |
| `AestrixPipeline.edit` | VAE encode → TE → DiT (strength + optional ref tokens) → VAE decode |
| `aestrix i2i` | `--image`, `--strength`, `--identity`, `--ref-latents`, `--face-preserve`, … |

**Default path:** strength-only (color curve).  
**Identity path (`--identity`):** clean reference latents concatenated into DiT (`t=10` RoPE), face-regional start-σ, post-Euler clean-latent pull, milder schedule. Prefer strength **≥ 0.85** for wardrobe/scene changes. Recipes: [`I2I_STRENGTH.md`](I2I_STRENGTH.md). Multi-reference (>1 image) remains out of v1.

Memory order: **VAE encode unload → TE unload → DiT unload → VAE decode unload**.

## Image analysis harness (`AestrixEval`)

No-Metal quality/accuracy feedback for agents and CI.

**Procedure (mandatory after gen):** [`Docs/EVAL_WORKFLOW.md`](EVAL_WORKFLOW.md)  
**Metrics schema:** [`Docs/IMAGE_ANALYSIS.md`](IMAGE_ANALYSIS.md)

| Type | Role |
|------|------|
| `ImageAnalyzer` | Orchestrates load → technical + optional ref + prompt |
| `TechnicalQuality` | Sharpness, contrast, clip, hue histogram, entropy, score |
| `ReferenceCompare` | SSIM, PSNR, histogram correlation, fidelity score |
| `PromptAlignment` | Color-word heuristics, style length, unverifiable keywords |
| `VisionReview` | Checklist + agent brief + `Assessment` merge |
| `ImageAnalysisReport` | Codable JSON + findings (`fail`/`warn`/`info`) + optional vision |
| `aestrix analyze-image` | CLI; exit 2 on hard failures |
| `t2i`/`i2i --analyze --vision-brief` | Generate then write `.eval.json` + `.vision-brief.md` |
| `Scripts/eval-generation.sh` | Eval existing PNG |

## References

- Plan (session): low-RAM / prequant / BFL skills  
- Official prompts: `.grok/skills/flux-best-practices/`  
- Weights: `Docs/WEIGHTS.md`  
- Memory: `Docs/MEMORY.md`  
- **Eval procedure:** `Docs/EVAL_WORKFLOW.md`  
- Eval metrics: `Docs/IMAGE_ANALYSIS.md`  
- Eval prompts: `Docs/eval-prompts.md`  
- **Roadmap (parked work):** `Docs/ROADMAP.md`  


