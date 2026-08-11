# Aestrix weights

## Product policy

- Users download **pre-quantized MLX packages only**.
- **Default: 4-bit.** Optional: 3-bit (Tier L), 6/8-bit (quality).
- **No bf16** in runtime defaults or App Store flows.
- Cache: `~/Library/Caches/Aestrix/models/` (macOS); app container on iOS.

## Canonical package (Phase 0 decision)

| Role | Hugging Face ID | Notes |
|------|-----------------|-------|
| **Primary (default)** | [`mlx-community/FLUX.2-Klein-4B-4bit`](https://huggingface.co/mlx-community/FLUX.2-Klein-4B-4bit) | Module-split; Apache-2.0 base |
| Equivalent layout | [`Runpod/FLUX.2-klein-4B-mflux-4bit`](https://huggingface.co/Runpod/FLUX.2-klein-4B-mflux-4bit) | Same tree sizes (~mflux q4); alternate source |
| Tier L optional | [`mlx-community/FLUX.2-Klein-4B-3bit`](https://huggingface.co/mlx-community/FLUX.2-Klein-4B-3bit) | Same layout family |
| Quality optional | `mlx-community/FLUX.2-Klein-4B-5bit` / `6bit` / `flux2-klein-4b-8bit` | Same family when needed |

### On-disk layout (already staged-friendly)

Measured on Hub API (2026-08-10):

| Component | Path | Approx size (4-bit) |
|-----------|------|---------------------|
| Text encoder (Qwen3-4B) | `text_encoder/*.safetensors` | **~2.26 GB** |
| Transformer (MMDiT) | `transformer/*.safetensors` | **~2.18 GB** |
| VAE | `vae/*.safetensors` | **~0.17 GB** |
| Tokenizer | `tokenizer/*` | ~11 MB |

**Loader implication:** Community packs are already **module-split**. Aestrix can load `text_encoder/`, `transformer/`, `vae/` independently for staged residency **without** a mandatory re-shard import step. Optional Aestrix `manifest.json` may still record bits, group size, and revision SHA.

**Total download ~4.6 GB** class — not ~20 GB bf16.

## Loader format

1. **Runtime loads** Hugging Face snapshot directories directly (prefer primary ID above).
2. **Pin** `revision` in config when shipping.
3. **Integrity**: size checks after download; optional SHA later.
4. Maintainer-only bf16/oracle tools stay under `Tools/` and are never product defaults.

## Related

- Model card: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B  
- BFL prompting skill: `.grok/skills/flux-best-practices/`  
