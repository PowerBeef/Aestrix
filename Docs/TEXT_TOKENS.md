# Text tokens (`512` vs `auto`)

**P9 Slice A close-out (2026-08-14).** Product default stays **full 512 pad**. `--text-tokens auto` is a first-class, documented opt-in.

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
| Gallery, eval, “same seed → same pixels” | **`512`** (default) | Matches `AGENTS.md` / `ARCHITECTURE.md` pad lock |
| Interactive 512², prompt iteration | **`auto`** | Largest measured unused speedup in-tree |
| Official `Scripts/eval-regression.sh` | omit (512) | Pixel floors are on the pad-512 path |
| Re-run this A/B | `T2I_EXTRA='--text-tokens auto'` | See below |

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

## Decision (do not flip the default)

Keep **`--text-tokens 512`** as the product default.

Reasons:

1. `AGENTS.md` / `ARCHITECTURE.md` lock: full 512 pad to DiT.  
2. Padding participates in attention — auto is a different sampler, not a free lunch.  
3. Vision is comparable, but PNGs are not byte-identical; gallery / eval must stay on one path.  
4. Seed-7 “OPEN STUDIO” already flakes on pad-512. Flipping the default would not fix typography and would invalidate every stored pad-512 seed.

`auto` is **first-class**: documented here, listed in `imarello info` / `--help`, safe for interactive 512². Changing the default needs a new product decision plus a pad-512 → auto eval-floor migration.

## Re-run

```bash
# Auto quality loop (512² only on 8 GB)
T2I_EXTRA='--text-tokens auto' \
  OUT_DIR=/tmp/imarello-eval-regression-auto \
  IMARELLO=.build/release/imarello \
  ./Scripts/eval-regression.sh

# Pad-512 reference (script default)
OUT_DIR=/tmp/imarello-eval-regression-pad \
  IMARELLO=.build/release/imarello \
  ./Scripts/eval-regression.sh

# Token count for any prompt
.build/release/imarello encode-prompt "YOUR PROMPT"

# Bench the speed path
.build/release/imarello bench --width 512 --height 512 --text-tokens auto \
  --json /tmp/imarello-auto-512.json
```

Eval procedure: [`EVAL_WORKFLOW.md`](EVAL_WORKFLOW.md). Speed tables: [`PERF.md`](PERF.md).
