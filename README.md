<div align="center">

# Aestrix

**Native Swift + [MLX](https://github.com/ml-explore/mlx-swift) runtime for [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) on Apple Silicon**

[![Swift 6](https://img.shields.io/badge/Swift-6-F05138?logo=swift&logoColor=white)](https://swift.org)
[![macOS 15+](https://img.shields.io/badge/macOS-15%2B-000000?logo=apple&logoColor=white)](#requirements)
[![MLX](https://img.shields.io/badge/MLX-0.31.6-blue)](https://github.com/ml-explore/mlx-swift)
[![Weights Apache-2.0](https://img.shields.io/badge/weights-Apache--2.0-green)](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

Low-RAM-first inference: text encoder, DiT, and VAE **never share memory**.
Pre-quantized **4-bit** weights, **1024²** default canvas, runs on an **8 GB** Mac.

<img src="Docs/assets/readme/hero-coffee-1024.jpg" width="640" alt="Cozy coffee shop interior, warm afternoon light — Aestrix T2I, 1024², 4 steps, seed 42">

<sub>*“A cozy coffee shop interior bathed in warm afternoon light, steam rising lazily from ceramic cups…”* — 1024², 4 steps, seed 42, 4-bit, generated on an 8 GB M2 Mac mini.</sub>

</div>

```text
prompt ──► Qwen3 TE ──► unload ──► MMDiT (4 steps) ──► unload ──► VAE decode ──► PNG
              ~2 GB                  ~2 GB                     decode-only / tiled
```

## Output samples

All images below are unedited Aestrix outputs (release build, 4-bit weights, 4 steps, fixed seeds).

### Text-to-image

| | |
|:--:|:--:|
| <img src="Docs/assets/readme/fisherman-1024.jpg" width="380"> | <img src="Docs/assets/readme/fox-512.jpg" width="380"> |
| <sub>*“A weathered fisherman in his 70s … golden hour rim lighting on his profile.”* — 1024², seed 42</sub> | <sub>*“A red fox in a snowy forest at sunrise, photorealistic.”* — 512², seed 7</sub> |

### Image-to-image (strength edit)

Color/style edits recolor the subject while keeping its shape, framing, and lighting.

| Source (T2I) | `i2i --strength 0.8` |
|:--:|:--:|
| <img src="Docs/assets/readme/i2i-mug-before.jpg" width="320"> | <img src="Docs/assets/readme/i2i-mug-after.jpg" width="320"> |
| <sub>*“A cobalt blue ceramic mug with a single curved handle … morning light”* — seed 42</sub> | <sub>*“the same ceramic mug but emerald green glaze, keeping the same bright morning light …”* — seed 7</sub> |

### Identity-preserving edit (`--identity`)

Big wardrobe + lighting change at strength 0.9 — face shape, eyes, freckles, hair, and framing stay locked.

| Reference (T2I) | `i2i --strength 0.9 --identity` |
|:--:|:--:|
| <img src="Docs/assets/readme/identity-ref.jpg" width="320"> | <img src="Docs/assets/readme/identity-edit.jpg" width="320"> |
| <sub>Ivory blouse, studio light — seed 42</sub> | <sub>*“Same woman, identical face … golden hour, deep emerald green silk top.”* — seed 7</sub> |

## Features

| | |
|--|--|
| **Model** | FLUX.2-klein-4B only (Apache-2.0) — not 9B / Dev |
| **Text-to-image** | Distilled defaults: **1024²**, 4 steps, guidance 1.0 |
| **Image-to-image** | Strength edits + optional **identity** stack (ref latents, face mask, clean-pull) |
| **Memory** | Staged TE → DiT → VAE; DiT block checkpointing; **MLX Steel fused FA** (simdgroup MMA); **tiled VAE** (overlap + cosine blend) |
| **Weights** | Hub 4-bit packs only — no bf16 product path |
| **Speed** | Prompt-embed disk cache (default on), warm `session` mode (≥16 GB), opt-in `--text-tokens auto` trim |
| **Eval** | Pixel quality harness + vision-review workflow for agents |
| **Bench** | Multi-trial timings, MLX/RSS memory, **pressure-map** block probes |

## Performance

Measured on an Apple M2 **8 GB** Mac mini (release, 4-bit, warmup 1 + 3 trials, seed 42, Steel fused FA + cosine tiled VAE + compiled RoPE/AdaLN — same-day A/B, 2026-08-11 optimization pass):

| Canvas | Time to image | Denoise / step | Peak MLX active | Peak MLX watermark | Peak RSS |
|--------|--------------:|---------------:|----------------:|-------------------:|---------:|
| **512²** | **~31.7 s** (−11.8% vs pre-pass) | ~6.1 s | 2.04 GiB | 2.99–3.21 GiB | 1.74 GiB |
| **1024²** | **~104 s** (−3.7% vs pre-pass) | ~22.4 s | 2.05 GiB | ~4.0 GiB | 1.75 GiB |
| **512²** + `--text-tokens auto` (opt-in) | **~21.4 s** | ~3.6 s | 2.04 GiB | 2.99 GiB | 1.73 GiB |

- Repeat prompts skip the text-encoder stage via the on-disk prompt-embed cache (default on, byte-identical output, −4 s at 512²).
- `--text-tokens auto` trims padding tokens — a large win on short prompts, but numerics differ from the full-512 reference (experimental).
- Live peak is dominated by DiT weights (~2 GiB) at both sizes; 1024² is slower, not dramatically hungrier on peak RSS/active.
- Absolute times vary a few percent with machine/thermal state — compare same-day A/Bs only. Full tables: [Docs/PERF.md](Docs/PERF.md).

## Requirements

- Apple Silicon Mac, macOS 15+
- Xcode 16+ / Swift 6
- ~5 GB free for the 4-bit snapshot; **~8 GB unified** runs 1024² with the low-RAM path

## Quick start

### 1. Clone and build

```bash
git clone https://github.com/PowerBeef/Aestrix.git
cd Aestrix
swift build -c release
./Scripts/ensure-metallib.sh   # full Metal library for MLX kernels (~130 MB)
```

### 2. Download weights

```bash
hf download mlx-community/FLUX.2-Klein-4B-4bit \
  --local-dir ~/Library/Caches/Aestrix/models/mlx-community--FLUX.2-Klein-4B-4bit
```

Canonical package: [`mlx-community/FLUX.2-Klein-4B-4bit`](https://huggingface.co/mlx-community/FLUX.2-Klein-4B-4bit) (module-split `text_encoder/` `transformer/` `vae/` `tokenizer/`). Details: [Docs/WEIGHTS.md](Docs/WEIGHTS.md).

### 3. Generate

```bash
.build/release/aestrix info

# Text-to-image (defaults: 1024×1024, 4 steps, 4-bit)
.build/release/aestrix t2i \
  "A weathered fisherman at the helm of a wooden boat, golden hour rim light, shallow depth of field." \
  --seed 42 --output out.png

# Faster smoke at 512²
.build/release/aestrix t2i "A red fox in a snowy forest at sunrise, photorealistic." \
  --width 512 --height 512 --seed 42 --output out512.png

# Image-to-image (color / style edits often need strength ≥ 0.8)
.build/release/aestrix i2i \
  "the same scene at blue hour, cooler tones" \
  --image out.png --strength 0.8 --seed 7 --output edit.png

# Identity-preserving edit (Tier B: ref latents + face mask + milder schedule)
.build/release/aestrix i2i \
  "Same person, exact face and pose; outdoor golden hour, emerald silk top" \
  --image portrait.png --strength 0.9 --identity --seed 7 --output edit-id.png

# Warm multi-prompt session (keeps modules resident; ≥16 GB RAM)
.build/release/aestrix session --width 1024 --height 1024

# Generate + quality report + vision checklist
.build/release/aestrix t2i "A red fox in a snowy forest at sunrise, photorealistic." \
  --output out.png --analyze --vision-brief
```

### 4. Benchmark & pressure map

```bash
# Timings + memory peaks
.build/release/aestrix bench --width 512 --height 512 --warmup 1 --trials 3 \
  --json /tmp/bench-512.json

# DiT block-level memory probes (find OOM / peak sites)
.build/release/aestrix bench --mode pressure-map --width 768 --height 768 \
  --probe-density blocks --json /tmp/pressure.json

# Compare two runs
.build/release/aestrix bench-compare /tmp/bench-a.json /tmp/bench-b.json
```

### 5. Tests (no model required)

```bash
swift test
.build/release/aestrix mem-selftest
```

## CLI

| Command | Description |
|---------|-------------|
| `aestrix info` | Tier, memory policy, snapshot path |
| `aestrix t2i <prompt>` | Text-to-image (default **1024²**) |
| `aestrix i2i <prompt> --image PATH` | Strength I2I; `--identity` for ref latents + face preserve |
| `aestrix session` | Warm multi-prompt loop, modules resident (≥16 GB gate) |
| `aestrix analyze-image PATH` | Pixel quality / accuracy report |
| `aestrix bench` | Performance + pressure harness |
| `aestrix bench-compare A B` | Compare two bench JSON reports |
| `aestrix load-te` / `load-dit` / `load-vae` | Module load smoke tests |
| `aestrix mem-selftest` | Dry staged load/unload |
| `aestrix schedule` | Print flow-match sigmas / timesteps |

Useful flags: `--width` `--height` `--steps` `--seed` `--output` `--analyze` `--vision-brief` `--fail-on-pixel-gate` `--text-tokens 512|auto` `--no-embed-cache`
I2I identity: `--identity` `--ref-latents` `--ref-downsample` `--face-preserve` `--face-strength-scale` `--clean-pull` `--schedule color|identity|linear`
Bench: `--mode pressure-map|dit-one-step|res-ladder` `--probe-density off|stages|denoise|blocks|max` `--attn-backend mlx|metal-fa|auto`

Performance: [Docs/PERF.md](Docs/PERF.md) · Quality: [Docs/EVAL_WORKFLOW.md](Docs/EVAL_WORKFLOW.md)

## Architecture

| Module | Role |
|--------|------|
| `AestrixCore` | Config, scheduler, RoPE math, `PipelineTrace` / probe density |
| `AestrixText` | Qwen3 3-layer-tap encoder + tokenizer |
| `AestrixDiT` | FLUX.2 MMDiT (checkpointed blocks, **Steel fused FA**, `AttentionTuning`) |
| `AestrixVAE` | Encode-only / decode-only loads / **tiled cosine-blend** large decode |
| `AestrixRuntime` | Staged pipeline actor; I2I + identity preserve (Vision face mask); prompt-embed cache |
| `AestrixEval` | Image quality harness (no Metal) |
| `AestrixBench` | Multi-trial metrics, pressure reports |
| `aestrix` | CLI |

Design notes: [Docs/ARCHITECTURE.md](Docs/ARCHITECTURE.md) · [Docs/MEMORY.md](Docs/MEMORY.md) · [Docs/PERF.md](Docs/PERF.md)

## Prompting

Follow Black Forest Labs **klein** guidance: narrative prose, subject first, strong lighting, **no negative prompts**.

Vendored skill: [`.grok/skills/flux-best-practices`](.grok/skills/flux-best-practices) · Eval prompts: [Docs/eval-prompts.md](Docs/eval-prompts.md)

## Quality workflow

```bash
.build/release/aestrix t2i "…" --output out.png --analyze --vision-brief
# or
./Scripts/eval-generation.sh out.png --prompt "…" --mode t2i
```

Full procedure: [Docs/EVAL_WORKFLOW.md](Docs/EVAL_WORKFLOW.md) · metrics: [Docs/IMAGE_ANALYSIS.md](Docs/IMAGE_ANALYSIS.md)

## Status

| Area | State |
|------|--------|
| macOS library + CLI | **Working** (T2I, I2I, identity I2I, eval) |
| 4-bit staged load | **Working** |
| 1024² on 8 GB class Macs | **Working** (~104 s e2e; Steel FA + tiled VAE) |
| Identity I2I (Tier B) | **Working** (`--identity`: ref latents + face mask + clean-pull) |
| Performance harness | **Working** (`bench`, pressure-map, ladder, attn backends) |
| MLX Steel fused FA | **Working** (product default for D=128) |
| Prompt-embed cache / warm session | **Working** (cache default on; `session` ≥16 GB) |
| iOS host | **Parked** — [Docs/ROADMAP.md](Docs/ROADMAP.md) |
| Multi-reference (>1) / CFG / LoRA | Out of v1 (tracked on roadmap) |

## Documentation

| Doc | Topic |
|-----|--------|
| [AGENTS.md](AGENTS.md) | Product locks & agent conventions |
| [Docs/ROADMAP.md](Docs/ROADMAP.md) | Done vs parked backlog |
| [Docs/PERF.md](Docs/PERF.md) | Benchmarks, pressure probes, 1024 path |
| [Docs/ARCHITECTURE.md](Docs/ARCHITECTURE.md) | Module map |
| [Docs/WEIGHTS.md](Docs/WEIGHTS.md) | Hub packs & cache layout |
| [Docs/MEMORY.md](Docs/MEMORY.md) | Staged residency & tiers |
| [Docs/EVAL_WORKFLOW.md](Docs/EVAL_WORKFLOW.md) | Generation quality gates |
| [Docs/HOST_SAFETY.md](Docs/HOST_SAFETY.md) | 8 GB / one-Metal-owner rules (watchdog) |

## License

| Component | License |
|-----------|---------|
| Aestrix source | MIT (see repository) |
| FLUX.2-klein-4B weights | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Black Forest Labs) |
| Vendored BFL prompting skill | MIT |

Not affiliated with Black Forest Labs or Apple. FLUX is a trademark of Black Forest Labs.
