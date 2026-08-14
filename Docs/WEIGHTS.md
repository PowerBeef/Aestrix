# Imarello weights

## Product policy

- Users download **pre-quantized MLX packages only**.
- **Default: 4-bit.** Optional: 3-bit (Tier L), 6/8-bit (quality).
- **No bf16** in runtime defaults or App Store flows.
- Cache: `~/Library/Caches/Imarello/models/` (macOS); app container on iOS. Existing `~/Library/Caches/Aestrix/models/` snapshots are still resolved.
- **Pinned revision:** each product preset ships against a Hugging Face **commit SHA** (`WeightPreset.pin`, [`hub-pins.json`](hub-pins.json)). `imarello info` prints `model_revision` and compares it to local `hf download` metadata when present.

## Canonical package (Phase 0 decision)

| Role | Hugging Face ID | Pinned revision | Notes |
|------|-----------------|-----------------|-------|
| **Primary (default)** | [`mlx-community/FLUX.2-Klein-4B-4bit`](https://huggingface.co/mlx-community/FLUX.2-Klein-4B-4bit) | `1cebb9b45c21ece14a42615b16bf5fa4de9b56da` (2026-05-29) | Module-split; Apache-2.0 base |
| Equivalent layout | [`Runpod/FLUX.2-klein-4B-mflux-4bit`](https://huggingface.co/Runpod/FLUX.2-klein-4B-mflux-4bit) | — | Same tree sizes (~mflux q4); alternate source, **not** the product pin |
| Tier L optional | [`mlx-community/FLUX.2-Klein-4B-3bit`](https://huggingface.co/mlx-community/FLUX.2-Klein-4B-3bit) | `246946064c7218227b1e99509245392cdcedc9d3` (2026-05-29) | Same layout family |
| Quality optional | [`mlx-community/FLUX.2-Klein-4B-6bit`](https://huggingface.co/mlx-community/FLUX.2-Klein-4B-6bit) / [`flux2-klein-4b-8bit`](https://huggingface.co/mlx-community/flux2-klein-4b-8bit) | `76fd8a876cb61126fb1fdce97eb9464eab063ff5` / `9beac1a3ad296d9e5e3f8845674e6577fa8654ec` | Same family when needed |

Machine-readable pins: [`hub-pins.json`](hub-pins.json) (must match `WeightPreset`). A 5-bit community pack exists but is **not** a product preset.

### Optional: BFL Small Decoder (decode only)

Not a swap into the klein `vae/` pack. Narrower channels `[96, 192, 384, 384]` vs `[128, 256, 512, 512]`. Encoder (I2I) stays the klein AE. Product default remains **full**.

| Role | Hugging Face ID | Pinned revision | File |
|------|-----------------|-----------------|------|
| Opt-in decode | [`black-forest-labs/FLUX.2-small-decoder`](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder) | `a3efc24f613ef42d9428af62fdbd6f5fd8856c4a` | `small_decoder.safetensors` (~112 MB, Apache-2.0) |

```bash
hf download black-forest-labs/FLUX.2-small-decoder \
  --revision a3efc24f613ef42d9428af62fdbd6f5fd8856c4a \
  --include small_decoder.safetensors --include config.json \
  --local-dir ~/Library/Caches/Imarello/models/black-forest-labs--FLUX.2-small-decoder

.build/release/imarello t2i "…" --width 512 --height 512 --vae-variant small-decoder \
  --output /tmp/small.png --analyze --vision-brief
```

Packed-latent BN stats still come from the klein pack. Flip the default only after a 512 + 1024 pixel/vision A/B.

### On-disk layout (already staged-friendly)

Measured on Hub API (2026-08-10):

| Component | Path | Approx size (4-bit) |
|-----------|------|---------------------|
| Text encoder (Qwen3-4B) | `text_encoder/*.safetensors` | **~2.26 GB** |
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
- BFL prompting skill: `.grok/skills/flux-best-practices/`  
