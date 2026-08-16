# Imarello weights

## Product policy

- Users download **pre-quantized MLX packages only**.
- **4-bit is locked.** That is the only shipping SKU.
- 6/8-bit pins exist in `hub-pins.json` but are **not** the product path.
- **No 3-bit product path.** Do not document or resume 3-bit as a Tier L option.
- **No bf16** in runtime defaults or App Store flows.
- Cache: `~/Library/Caches/Imarello/models/` (macOS); app container on iOS. Existing `~/Library/Caches/Aestrix/models/` snapshots are still resolved.
- **Pinned revision:** each product preset ships against a Hugging Face **commit SHA** (`WeightPreset.pin`, [`hub-pins.json`](hub-pins.json)). `imarello info` prints `model_revision` and compares it to local `hf download` metadata when present.

## Canonical package (Phase 0 decision)

| Role | Hugging Face ID | Pinned revision | Notes |
|------|-----------------|-----------------|-------|
| **Primary (default)** | [`mlx-community/FLUX.2-Klein-4B-4bit`](https://huggingface.co/mlx-community/FLUX.2-Klein-4B-4bit) | `1cebb9b45c21ece14a42615b16bf5fa4de9b56da` (2026-05-29) | Module-split; Apache-2.0 base |
| Equivalent layout | [`Runpod/FLUX.2-klein-4B-mflux-4bit`](https://huggingface.co/Runpod/FLUX.2-klein-4B-mflux-4bit) | — | Same tree sizes (~mflux q4); alternate source, **not** the product pin |
| Non-product pins | 6-bit / 8-bit mlx-community packs | see [`hub-pins.json`](hub-pins.json) | Not the shipping path. **No 3-bit SKU.** |

Machine-readable pins: [`hub-pins.json`](hub-pins.json) (must match `WeightPreset`). A 5-bit community pack exists but is **not** a product preset.

### Default decode: BFL Small Decoder

Not a swap into the klein `vae/` pack. Narrower channels `[96, 192, 384, 384]` vs `[128, 256, 512, 512]`. Product default is **`--vae-variant small-decoder`**. Use `--vae-variant full` for the klein-pack decoder.

**I2I encoder lock:** stage-0 encode is always the **klein pack** (`VAELoadMode.encodeOnly`, ~67 MB). BFL did not ship a small encoder. Do **not** load `full_encoder_small_decoder.safetensors` — that bundle is a ComfyUI convenience file (full AE encoder + Small Decoder). Imarello already splits encode/decode; mixing it would duplicate weights, force a PyTorch remap, and can drift BN / `quant_conv` vs the mlx-community pack the DiT was paired with. Identity ref latents, face mask, and clean-pull assume klein encode + klein BN. TAEF2 stays preview-only (not export I2I encode).

| Role | Hugging Face ID | Pinned revision | File |
|------|-----------------|-----------------|------|
| **Default decode** | [`black-forest-labs/FLUX.2-small-decoder`](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) | `a3efc24f613ef42d9428af62fdbd6f5fd8856c4a` | `small_decoder.safetensors` (~112 MB, Apache-2.0) |

```bash
hf download black-forest-labs/FLUX.2-small-decoder \
  --revision a3efc24f613ef42d9428af62fdbd6f5fd8856c4a \
  --include small_decoder.safetensors --include config.json \
  --local-dir ~/Library/Caches/Imarello/models/black-forest-labs--FLUX.2-small-decoder

.build/release/imarello t2i "…" --width 512 --height 512 \
  --output /tmp/out.png --analyze --vision-brief
```

Packed-latent BN stats still come from the klein pack. Missing Small Decoder snapshot → `t2i` / `i2i` refuse with this `hf download` hint.

**1024² T2I quality pass (2026-08-15):** mug / OPEN STUDIO / fox seed 42, full vs small — **6/6 pixel PASS**, vision match, tile seams clean (score 1.2–1.7). Decode **−37%** (8.08 → 5.12 s). **Promoted to product default** (2026-08-15). `--vae-variant full` is the klein-pack escape hatch.

I2I (2026-08-15, klein `encodeOnly` + Small Decoder, 512²):

```bash
.build/release/imarello i2i "the same ceramic mug but emerald green glaze #0B5F4B, same morning light" \
  --image Docs/assets/readme/i2i-mug-before.jpg --strength 0.8 --seed 7 \
  --output /tmp/edit.png --analyze --vision-brief
```

Mug recolor: pixel PASS, SSIM 0.43, emerald glaze, handle/table held. Identity (`--identity` 0.9, `identity-ref.jpg`): pixel PASS (warn `color_mismatch` waived — blouse is emerald; hair/skin dominate the hue bucket), SSIM 0.73, face lock, scene balcony only mild. Encoder stays klein.

### On-disk layout (already staged-friendly)

Measured on Hub API (2026-08-10):

| Component | Path | Approx size (4-bit) |
|-----------|------|---------------------|
| Text encoder (Qwen3-4B) | `text_encoder/*.safetensors` | **~2.26 GB** on disk (~1.7 GB materialized — 27 of 36 layers loaded) |
| Transformer (MMDiT) | `transformer/*.safetensors` | **~2.18 GB** |
| VAE | `vae/*.safetensors` | **~0.17 GB** |
| Tokenizer | `tokenizer/*` | ~11 MB |

**Loader implication:** Community packs are already **module-split**. Imarello can load `text_encoder/`, `transformer/`, `vae/` independently for staged residency **without** a mandatory re-shard import step. Optional Imarello `manifest.json` may still record bits, group size, and revision SHA.

**Total download ~4.6 GB** class — not ~20 GB bf16.

## Loader format

1. **Runtime loads** Hugging Face snapshot directories directly (prefer primary ID above).
2. **Pin** `ImarelloConfig.revision` / `WeightPreset.pinnedRevision` (shipped; see table).
3. **Integrity**: `hf download` metadata SHA is compared to the pin by `imarello info` (`snapshot_revision_match`). Size checks after download; content-hash of shards is still optional.
4. Maintainer-only bf16/oracle tools stay under `Tools/` and are never product defaults.

```bash
hf download mlx-community/FLUX.2-Klein-4B-4bit \
  --revision 1cebb9b45c21ece14a42615b16bf5fa4de9b56da \
  --local-dir ~/Library/Caches/Imarello/models/mlx-community--FLUX.2-Klein-4B-4bit
```

`imarello info` prints this command as `snapshot_hint` when the cache is empty.

### Bumping a pin

1. `hf models info <repo> --format json` → `sha` / `lastModified`.
2. Update `Docs/hub-pins.json` and `WeightPreset.pin`.
3. `swift test --filter HubPinTests` (must stay in lockstep with WEIGHTS.md + README).

## Related

- Model card: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B  
- BFL prompting skill: `.claude/skills/flux-best-practices/`  
