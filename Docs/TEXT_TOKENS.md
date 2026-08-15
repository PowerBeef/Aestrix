# Text tokens (`512` vs `auto`)

**P9 Slice A.** Product default is **`--text-tokens auto`** (2026-08-15). `--text-tokens 512` is the byte-stable gallery path.

Code: `TextTokenMode`, `ImarelloPipeline.trimTextTokens`. CLI: `imarello t2i|i2i|session|bench --text-tokens 512|auto`.

## Why this exists

FLUX.2 joint attention has **no text mask**. The Qwen3 TE always encodes a **512-token** padded window; those pad tokens participate in DiT attention. That is the byte-stable reference path (`--text-tokens 512`).

`--text-tokens auto` slices the padded embeds (and rebuilds `txtIds`) to the **real prompt length, rounded up to a multiple of 8** before the DiT:

```
trimmed = min(512, max(8, ceil(realTokens / 8) * 8))
```

Qwen attention is causal, so the real-token embeddings themselves are identical. Only the DiT joint sequence shortens. **Numerics change** — same seed does **not** produce the same PNG.

Embed-cache entries stay the full `[1, 512, 7680]` tensor (key includes `len=512`). Auto trims **after** a cache hit, so `512` and `auto` share one TE encode.

## When to use which

| Intent | Flag | Why |
|--------|------|-----|
| Product `t2i` / `i2i` (default) | **`auto`** | −32% e2e @ 512²; quality + identity face lock passed |
| Gallery, “same seed → same pixels” | **`512`** | Pad participates in joint attn; byte-stable vs old PNGs |
| Official pad-512 eval | `T2I_EXTRA='--text-tokens 512'` | Compare against the old reference |

Do **not** claim a pad-512 PNG is reproducible under `auto`.

## Speed (already measured)

From `Docs/PERF.md` P2 (8 GB M2, release, 12-word prompt, joint 1536 → ~1060):

| Canvas | Effect |
|--------|--------|
| 512² | e2e **−32.4%** (31.7 → 21.4 s), denoise/step **−39.8%**, watermark 3.21 → 2.99 GiB |
| 1024² | denoise/step **~−10%** (seq 4608 → ~4136); e2e share is smaller because image tokens dominate |

Shorter prompts win more. A 512-token prompt is a no-op trim.

## Token counts (eval prompts, 2026-08-14)

`imarello encode-prompt` → `real_tokens`. Image seq at 512² is **1024**.

| Prompt | real | auto trim | joint auto | joint 512 |
|--------|-----:|----------:|-----------:|----------:|
| 1 coffee shop | 63 | 64 | 1088 | 1536 |
| 2 fisherman | 78 | 80 | 1104 | 1536 |
| 3 mug | 60 | 64 | 1088 | 1536 |
| 4 OPEN STUDIO | 54 | 56 | 1080 | 1536 |
| 5 fox | 26 | 32 | 1056 | 1536 |

## Quality A/B (2026-08-14)

512² · 4-step · 4-bit · release + full metallib · seeds **42 / 0 / 7**.

Auto: `T2I_EXTRA='--text-tokens auto' OUT_DIR=/tmp/imarello-eval-regression-auto ./Scripts/eval-regression.sh`  
Pad spots: `/tmp/imarello-eval-regression-pad/p{3,4,5}_s42.png` and `p4_s7.png`.

### Pixel gate

**Auto 15/15 PASS** (no hard fails). Technical scores:

| id | s42 | s0 | s7 |
|----|----:|---:|---:|
| p1 coffee | 84.9 | 89.2 | 89.1 |
| p2 fisherman | 61.2 | 64.9 | 65.0 |
| p3 mug | 78.8 | 91.9 | 76.6 |
| p4 poster | 67.1 | 70.2 | 74.9 |
| p5 fox | 95.8 | 97.8 | 96.6 |

Fisherman ~61–65 is silhouette / dark navy under rim light, not a gate fail. Poster `highlight_clip` + `low_semantic_alignment` are the usual white-type / vision-proxy warnings.

Pad seed-42 spots (same prompts): mug tech **85.3**, poster **63.3**, fox **96.1** — same band as auto.

### Vision (required spots)

| Image | Verdict |
|-------|---------|
| auto p3_s42 mug | Terracotta matte mug on linen, window from the left. **Two handles** — known Klein quirk, **also on pad s42**. Color matches `#C45C26`. |
| auto p4_s42 poster | **"OPEN STUDIO"** correct, condensed white sans, deep indigo, centered. |
| auto p4_s0 poster | Same headline, slight 3-D / drop-shadow treatment. Pass. |
| auto p4_s7 poster | **"OPEN" dropped** — stylized S + “TUDIO”. |
| pad p4_s7 poster | **"OPEN" dropped** — clean “STUDIO” only. **Same class of miss; not auto-specific.** |
| auto p5_s42 / s7 fox | Photoreal red fox, snow, sunrise. Anatomy OK. Pad s42 is a closer walk-in; auto sits / crouches. Same subject, pose drift. |
| auto p1_s42 shop | Brick, leather, steam, tall windows. Pass. |
| auto p2_s42 fisherman | Weathered profile, salt-and-pepper beard, golden-hour rim, harbor bokeh. Pass. |

Expected under auto: **texture / pose / type-treatment drift**, not a different subject. Klein 4-step typography is already seed-fragile on **both** paths.

### Identity I2I (2026-08-15)

Character-consistency is the harder test: ref latents (`t=10`) share joint attention with the prompt. Auto changes text length next to those tokens; it does not change klein encode, face mask, or clean-pull.

512² · `--identity` · strength **0.9** · seed **7** · Small Decoder · ref `Docs/assets/readme/identity-ref.jpg`.

| Intent | tokens | pixel | SSIM vs ref | Vision |
|--------|--------|-------|------------:|--------|
| Recolor same cut `#0B5F4B` | 512 | PASS | 0.743 | Same woman; same collar/sleeves; ivory → emerald. |
| Recolor same cut | **auto** | PASS | 0.751 | Same face/hair/pose; emerald; cut still a collared blouse (collar a bit more open). |
| Replace wrap top + balcony | 512 | PASS* | 0.727 | Face lock. Emerald long-sleeve blouse (lapel-ish). **No balcony.** Cut change mild. |
| Replace wrap top + balcony | **auto** | PASS* | 0.748 | Face lock. Emerald **sleeveless** silk (stronger cut change). **No balcony.** |

\* `color_mismatch` warn waived: blouse is green; hair/skin dominate the hue bucket.

Paths: `/tmp/imarello-auto-id-i2i/recolor-{512,auto}.png`, `replace-{512,auto}.png`.

**Read-out:** auto does **not** drop the person. Recolor quality matches pad-512. On “replace outfit,” auto changed sleeves more than pad-512; neither delivered the outdoor balcony (identity-stack limit, both paths). Auto can change *how* a wardrobe edit lands, not just texture.

## Decision (product default is `auto`)

**`--text-tokens auto` is the default** (2026-08-15). Speed at 512² is large; T2I eval + identity I2I face lock passed. Same seed is **not** the same PNG — use `--text-tokens 512` when you need the old pad-512 pixels.

Outfit-edit *cut* can drift vs pad-512 on identity replace. That is accepted for the default; pin `512` for a locked recipe.

## Re-run

```bash
# Product path (auto is CLI default)
OUT_DIR=/tmp/imarello-eval-regression-auto \
  IMARELLO=.build/release/imarello \
  ./Scripts/eval-regression.sh

# Pad-512 gallery path
T2I_EXTRA='--text-tokens 512' \
  OUT_DIR=/tmp/imarello-eval-regression-pad \
  IMARELLO=.build/release/imarello \
  ./Scripts/eval-regression.sh

# Identity, pad-512 control
.build/release/imarello i2i "$PROMPT" --image Docs/assets/readme/identity-ref.jpg \
  --strength 0.9 --identity --seed 7 --width 512 --height 512 \
  --text-tokens 512 --output /tmp/id-512.png --analyze --vision-brief

# Token count for any prompt
.build/release/imarello encode-prompt "YOUR PROMPT"

# Bench the speed path
.build/release/imarello bench --width 512 --height 512 --text-tokens auto \
  --json /tmp/imarello-auto-512.json
```

Eval procedure: [`EVAL_WORKFLOW.md`](EVAL_WORKFLOW.md). Speed tables: [`PERF.md`](PERF.md).
