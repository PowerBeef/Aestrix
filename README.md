# Aestrix

**Native Swift + [MLX](https://github.com/ml-explore/mlx-swift) runtime for [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) on Apple Silicon.**

Low-RAM-first inference: the text encoder, DiT, and VAE never share memory. Weights stay **pre-quantized** (default 4-bit). macOS library + CLI today; iOS host next.

```text
prompt ──► Qwen3 TE ──► unload ──► MMDiT (4 steps) ──► unload ──► VAE ──► PNG
              ~2.3 GB                  ~2.2 GB                    ~0.2 GB
```

## Features

| | |
|--|--|
| **Model** | FLUX.2-klein-4B only (Apache-2.0) — not 9B / Dev |
| **Text-to-image** | Distilled defaults: 4 steps, guidance 1.0 |
| **Image-to-image** | Strength-based edits (encode → re-noise → denoise) |
| **Memory** | Staged TE → DiT → VAE residency by default |
| **Weights** | Hub 4-bit packs only — no bf16 download path |
| **Eval** | Pixel quality harness + vision-review workflow for agents |

## Requirements

- Apple Silicon Mac  
- macOS 15+  
- Xcode 16+ / Swift 6  
- ~5 GB free for the 4-bit snapshot (plus room for activations)

## Quick start

### 1. Clone and build

```bash
git clone https://github.com/PowerBeef/Aestrix.git
cd Aestrix
swift build
./Scripts/ensure-metallib.sh   # full Metal library for MLX kernels
```

### 2. Download weights

Use any Hugging Face client. Target layout:

```text
~/Library/Caches/Aestrix/models/mlx-community--FLUX.2-Klein-4B-4bit/
  text_encoder/
  transformer/
  vae/
  tokenizer/
```

Example with `hf`:

```bash
hf download mlx-community/FLUX.2-Klein-4B-4bit \
  --local-dir ~/Library/Caches/Aestrix/models/mlx-community--FLUX.2-Klein-4B-4bit
```

Canonical package: [`mlx-community/FLUX.2-Klein-4B-4bit`](https://huggingface.co/mlx-community/FLUX.2-Klein-4B-4bit) (~4.6 GB total). Details: [Docs/WEIGHTS.md](Docs/WEIGHTS.md).

### 3. Generate

```bash
# Check snapshot
.build/debug/aestrix info

# Text-to-image
.build/debug/aestrix t2i \
  "A weathered fisherman at the helm of a wooden boat, golden hour rim light, shallow depth of field." \
  --width 512 --height 512 --steps 4 --seed 42 \
  --output out.png

# Image-to-image (color / style edits often need strength ≥ 0.8)
.build/debug/aestrix i2i \
  "the same scene at blue hour, cooler tones" \
  --image out.png --strength 0.8 --seed 7 \
  --output edit.png

# Generate + quality report + vision checklist
.build/debug/aestrix t2i "A red fox in a snowy forest at sunrise, photorealistic." \
  --output out.png --analyze --vision-brief
```

### 4. Tests (no model required)

```bash
swift test
.build/debug/aestrix mem-selftest   # staged residency smoke
```

## CLI

| Command | Description |
|---------|-------------|
| `aestrix info` | Tier, memory policy, snapshot path |
| `aestrix t2i <prompt>` | Text-to-image |
| `aestrix i2i <prompt> --image PATH` | Strength-based image-to-image |
| `aestrix analyze-image PATH` | Pixel quality / accuracy report |
| `aestrix load-te` / `load-dit` / `load-vae` | Module load smoke tests |
| `aestrix mem-selftest` | Dry staged load/unload |
| `aestrix schedule` | Print flow-match sigmas / timesteps |

Useful flags: `--width` `--height` `--steps` `--seed` `--output` `--analyze` `--vision-brief` `--fail-on-pixel-gate`.

## Architecture

Swift packages (SwiftPM):

| Module | Role |
|--------|------|
| `AestrixCore` | Config, scheduler, RoPE math, memory probe |
| `AestrixText` | Qwen3 3-layer-tap encoder + tokenizer |
| `AestrixDiT` | FLUX.2 MMDiT (5 double + 20 single blocks) |
| `AestrixVAE` | Encode / decode / packed latents |
| `AestrixRuntime` | Staged pipeline actor |
| `AestrixEval` | Image quality harness (no Metal) |
| `aestrix` | CLI |

Design notes: [Docs/ARCHITECTURE.md](Docs/ARCHITECTURE.md) · [Docs/MEMORY.md](Docs/MEMORY.md)

## Prompting

Follow Black Forest Labs **klein** guidance: narrative prose, subject first, strong lighting, **no negative prompts**.  

Vendored skill: [`.grok/skills/flux-best-practices`](.grok/skills/flux-best-practices)  
Eval prompts: [Docs/eval-prompts.md](Docs/eval-prompts.md)

## Quality workflow

After generations you care about, run the pixel + vision eval procedure:

```bash
.build/debug/aestrix t2i "…" --output out.png --analyze --vision-brief
# or
./Scripts/eval-generation.sh out.png --prompt "…" --mode t2i
```

Full procedure: [Docs/EVAL_WORKFLOW.md](Docs/EVAL_WORKFLOW.md) · metrics: [Docs/IMAGE_ANALYSIS.md](Docs/IMAGE_ANALYSIS.md)

## Status

| Area | State |
|------|--------|
| macOS library + CLI | **Working** (T2I, I2I, eval) |
| 4-bit staged load | **Working** |
| iOS host | **Parked** — see [Docs/ROADMAP.md](Docs/ROADMAP.md) |
| Multi-reference / CFG / LoRA | Out of v1 (tracked on roadmap) |

Remaining work is intentionally deferred and tracked in **[Docs/ROADMAP.md](Docs/ROADMAP.md)** so it can be resumed without losing context.

## Documentation

| Doc | Topic |
|-----|--------|
| [AGENTS.md](AGENTS.md) | Product locks & agent conventions |
| [Docs/ROADMAP.md](Docs/ROADMAP.md) | **Done vs parked backlog** |
| [Docs/ARCHITECTURE.md](Docs/ARCHITECTURE.md) | Module map |
| [Docs/WEIGHTS.md](Docs/WEIGHTS.md) | Hub packs & cache layout |
| [Docs/MEMORY.md](Docs/MEMORY.md) | Staged residency & tiers |
| [Docs/EVAL_WORKFLOW.md](Docs/EVAL_WORKFLOW.md) | Generation quality gates |

## License

| Component | License |
|-----------|---------|
| Aestrix source | MIT (see repository) |
| FLUX.2-klein-4B weights | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Black Forest Labs) |
| Vendored BFL prompting skill | MIT |

Not affiliated with Black Forest Labs or Apple. FLUX is a trademark of Black Forest Labs.
