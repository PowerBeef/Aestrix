<div align="center">

<img src="Docs/assets/readme/imarello-lockup-wide.png" width="560" alt="Imarello">

**Turn words into photographs — entirely on your Mac.**

A native Swift + [MLX](https://github.com/ml-explore/mlx-swift) runtime for Black Forest Labs'
[FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), written from scratch for Apple Silicon.
No cloud, no accounts, no data leaving your machine.

[![Swift 6](https://img.shields.io/badge/Swift-6-F05138?logo=swift&logoColor=white)](https://swift.org)
[![macOS 26.2+](https://img.shields.io/badge/macOS-26.2%2B-000000?logo=apple&logoColor=white)](#1-what-you-need)
[![MLX](https://img.shields.io/badge/MLX-0.32.1%20(fork)-blue)](https://github.com/PowerBeef/mlx-swift)
[![Weights Apache-2.0](https://img.shields.io/badge/weights-Apache--2.0-green)](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
[![Eval floors](https://github.com/PowerBeef/Imarello/actions/workflows/eval-floors.yml/badge.svg)](https://github.com/PowerBeef/Imarello/actions/workflows/eval-floors.yml)

**512² in ~22 s · 1024² in ~68 s — on an 8 GB M2 Mac mini.** Also runs on iPhone.

<img src="Docs/assets/readme/hero-coffee-1024.jpg" width="640" alt="Cozy coffee shop interior, warm afternoon light — Imarello T2I, 1024², 4 steps, seed 42">

<sub>1024² · 4 steps · seed 42 · generated on the 8 GB Mac mini above</sub>

</div>

## What Imarello does

| | You type / give it | You get |
|---|---|---|
| **Generate** | A text prompt | A 1024² photograph-quality image in about a minute |
| **Edit** | An image + what should change | The same scene, recolored / restyled / relit |
| **Identity edit** | A portrait + a wardrobe or lighting change | The same person — face locked — in the new look |

Everything runs locally. The model is a 4-bit build of FLUX.2 [klein] (about a 4.8 GB one-time download), tuned so the whole pipeline fits comfortably in 8 GB of unified memory.

## Gallery

Unedited outputs, fixed seeds.

<table>
<tr>
<td align="center" width="50%">
<img src="Docs/assets/readme/fisherman-1024.jpg" width="380" alt="Weathered fisherman, golden hour">
<br><sub>Generate · 1024² · seed 42</sub>
</td>
<td align="center" width="50%">
<img src="Docs/assets/readme/fox-512.jpg" width="380" alt="Red fox in snow at sunrise">
<br><sub>Generate · 512² · seed 7</sub>
</td>
</tr>
<tr>
<td align="center">
<img src="Docs/assets/readme/i2i-mug-before.jpg" width="280" alt="Cobalt ceramic mug">
<br><sub>Original</sub>
</td>
<td align="center">
<img src="Docs/assets/readme/i2i-mug-after.jpg" width="280" alt="Emerald ceramic mug">
<br><sub>Edit: “emerald green glaze” — everything else stays</sub>
</td>
</tr>
<tr>
<td align="center">
<img src="Docs/assets/readme/identity-ref.jpg" width="280" alt="Woman, ivory blouse, studio light">
<br><sub>Original</sub>
</td>
<td align="center">
<img src="Docs/assets/readme/identity-edit.jpg" width="280" alt="Same woman, emerald silk, golden hour">
<br><sub>Identity edit: new outfit and light — same face</sub>
</td>
</tr>
</table>

## Get started

### 1. What you need

- A Mac with Apple Silicon (M1 or newer) running **macOS 26.2+** — 8 GB of memory is enough
- **Xcode 26** (Swift 6)
- ~5.1 GB of free disk space
- The [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`hf`) for the model download

### 2. Build it

```bash
git clone https://github.com/PowerBeef/Imarello.git && cd Imarello
swift build -c release
./Scripts/ensure-metallib.sh   # builds the GPU kernel library (~155 MB) — required
```

> The second script is not optional: Imarello loads its Metal kernels from this library at runtime.

### 3. Download the model (one time, ~4.8 GB)

```bash
hf download mlx-community/FLUX.2-Klein-4B-4bit \
  --revision 1cebb9b45c21ece14a42615b16bf5fa4de9b56da \
  --local-dir ~/Library/Caches/Imarello/models/mlx-community--FLUX.2-Klein-4B-4bit

hf download black-forest-labs/FLUX.2-small-decoder \
  --revision a3efc24f613ef42d9428af62fdbd6f5fd8856c4a \
  --include small_decoder.safetensors --include config.json \
  --local-dir ~/Library/Caches/Imarello/models/black-forest-labs--FLUX.2-small-decoder

.build/release/imarello info   # everything should read “ready”
```

### 4. Make your first image

```bash
.build/release/imarello t2i \
  "A weathered fisherman at the helm of a wooden boat, golden hour rim light, shallow depth of field." \
  --seed 42 --output my-first.png
```

That's a 1024² image in about a minute. For a quick ~22-second test, add `--width 512 --height 512`.

To **edit** an image:

```bash
# Recolor / restyle — describe the change AND what stays
.build/release/imarello i2i "the same ceramic mug, emerald green glaze, same morning light" \
  --image my-first.png --strength 0.8 --output edited.png

# People — lock the face while changing wardrobe or lighting
.build/release/imarello i2i "Same person, exact face and pose; outdoor golden hour, emerald silk top" \
  --image portrait.png --strength 0.9 --identity --output edited.png
```

## Getting better images

**Write prompts like captions, not keyword lists.** Klein responds best to narrative prose with the subject first and the lighting spelled out. Don't list what you *don't* want — there are no negative prompts; just describe what you do want. Exact colors (`#C45C26`) and quoted text (`"OPEN STUDIO"`) work. Full recipes: [flux-best-practices](.claude/skills/flux-best-practices).

**Editing strength matters.** `--strength 0.8` or higher for color and material changes; people need `0.85–0.9` *plus* `--identity`. Guide: [Docs/I2I_STRENGTH.md](Docs/I2I_STRENGTH.md).

**Full-body subjects at 1024².** The 4-step model can occasionally scramble limbs on complex full-body poses at 1024². If that happens, add `--two-stage`: it composes the image at 512² (where anatomy is reliable) and refines it at 1024² — trading a little texture sharpness for a correct body.

**Repeat prompts are faster.** Prompt understanding is cached on disk, so re-running a prompt with a new seed skips a whole pipeline stage. (Disable with `--no-embed-cache`.)

**Lots of prompts in a row?** On Macs with 16 GB or more, `imarello session` keeps the model warm between images.

Every command and flag: `imarello --help`.

## The iPhone studio

`Apps/ImarelloIOS` is a full photo-studio app on the same engine — generation happens **on the phone** (iPhone 17-class recommended; ~12 s for a 512² image on an iPhone 17 Pro).

- **Stage** — your print fills the screen; one glass status row tells you everything (progress, errors, a working Stop); a gold **Develop** button runs it.
- **Contact Sheet** — every print you've made. Tap one to zoom, share, save to Photos, delete — or **Edit**, which develops a new print from that one.

Your prints are stored durably (they survive iOS cache cleanups and app reinstalls). Building and installing it requires Xcode and a paired iPhone on iOS 26.2+ — the full recipe, including how weights get onto the device, is in [Docs/IOS.md](Docs/IOS.md).

---

# For developers

## How it stays inside 8 GB

Only one heavy model is ever resident. Peak memory is roughly **max(stage)**, never the sum:

```text
prompt ──► Qwen3 text encoder ──► unload ──► MMDiT, 4 steps ──► unload ──► VAE decode ──► PNG
                ~1.7 GB                          ~2.1 GB                     ~0.1 GB (tiled at 1024²)
```

The DiT is the watermark: ~2.06 GiB live, 2.57 GiB (512²) / 3.00 GiB (1024²) peak MLX allocations.

## Performance

8 GB M2 Mac mini · release build + full metallib · product defaults · warmup 1, trials 3:

| Canvas | End-to-end | Denoise / step | Decode | Peak MLX |
|--------|-----------:|---------------:|-------:|---------:|
| 512² | **21.9 s** | 4.22 s | 0.94 s | 2.57 GiB |
| 1024² | **67.5 s** | 14.69 s | 4.58 s | 3.00 GiB |
| Identity edit 512² | **36.2 s** | 7.57 s | 0.94 s | 2.56 GiB |

What makes it fast, in one paragraph: 4-step distilled sampling · staged residency · MLX **Steel fused flash attention** (D=128, joint-f16) · scaled-f16 4-bit GEMMs · chunk-streamed transformer blocks at 1024² · BFL **Small Decoder** (−37% decode) · per-prompt embedding cache · and, since 2026-08-18, **mlx core 0.32.1** via a [maintained fork](https://github.com/PowerBeef/mlx-swift) (upstream mlx-swift is pre-0.32) whose split-K/`gemv_wide` kernels cut text-encoding 19–21%. Kernels are served from the prebuilt metallib (nojit), which is why `ensure-metallib.sh` is mandatory. The full A/B history with every promotion and refutation lives in [Docs/PERF.md](Docs/PERF.md); the research ledger is [Docs/ENGINE_RESEARCH.md](Docs/ENGINE_RESEARCH.md).

Quality is gated twice on every change: a pixel harness (with hard fails for garbage output) *and* a vision review — [Docs/EVAL_WORKFLOW.md](Docs/EVAL_WORKFLOW.md).

## Use it as a Swift library

```swift
import ImarelloCore
import ImarelloRuntime

let pipeline = ImarelloPipeline()   // an actor — all MLX state lives inside
let url = try await pipeline.generate(
    T2IRequest(prompt: "A red fox in a snowy forest at sunrise, photorealistic.", seed: 42)
)
```

`edit(_:)` takes an `I2IRequest` (strength, optional identity stack). Both calls are cancellable mid-denoise and validate their inputs up front.

| Module | Owns |
|--------|------|
| `ImarelloRuntime` | The staged pipeline, I2I + identity, embed cache |
| `ImarelloText` | Qwen3 encoder (layer taps 9/18/27 → 7680) |
| `ImarelloDiT` | MMDiT (5 double + 20 single blocks), Steel FA, quantized GEMMs |
| `ImarelloVAE` | Small Decoder default, klein encoder, tiled decode |
| `ImarelloEval` / `ImarelloBench` | Pixel quality gates · multi-trial benchmark harness |
| `ImarelloCore` | Pins, scheduler, RoPE, config — MLX-free |

## Developing

```bash
# Unit tests — ALWAYS filtered; unfiltered `swift test` can hang on Metal FA suites
swift test --filter 'HostPreflight|GoldenMetric|Flux2Math|IdentityPreserve|ImarelloBench|HubPin|Metallib|EvalCachePolicy|TextTokenMode|PromptEmbedCacheKey|VAEAttention|DiTOpProfile|DeviceHarness|Qwen'

# Benchmarks (any perf claim needs these)
.build/release/imarello bench --width 512 --height 512 --json /tmp/a.json
.build/release/imarello bench-compare /tmp/a.json /tmp/b.json

# Quality regression: 5 prompts × 3 seeds, pixel-gated
IMARELLO=.build/release/imarello ./Scripts/eval-regression.sh
```

| Doc | What's in it |
|-----|--------------|
| [ARCHITECTURE.md](Docs/ARCHITECTURE.md) · [MEMORY.md](Docs/MEMORY.md) | Layer design · the staged-residency model |
| [PERF.md](Docs/PERF.md) · [ENGINE_RESEARCH.md](Docs/ENGINE_RESEARCH.md) | A/B ledger · optimization research and verdicts |
| [WEIGHTS.md](Docs/WEIGHTS.md) | Model packs and pinned revisions |
| [EVAL_WORKFLOW.md](Docs/EVAL_WORKFLOW.md) | The pixel + vision quality gate |
| [I2I_STRENGTH.md](Docs/I2I_STRENGTH.md) · [TEXT_TOKENS.md](Docs/TEXT_TOKENS.md) | Edit-strength recipes · the pad-512 conditioning decision |
| [IOS.md](Docs/IOS.md) | iOS build, install, device weight sync, headless device harness |
| [ROADMAP.md](Docs/ROADMAP.md) · [CLAUDE.md](CLAUDE.md) | Backlog + decision log · agent/contributor ground rules |

Working on this repo with an AI agent? [CLAUDE.md](CLAUDE.md) is the contract — host safety on 8 GB machines, quality gates, and product locks live there.

## Status

| Shipping today | In progress / next |
|----------------|--------------------|
| macOS library + CLI: generate, edit, identity edit — 1024² on 8 GB | A19 Pro / M5 **Neural Accelerator** path (probe + iOS 26.2 floor shipped; blocked on a device running iOS ≥ 26.2) |
| iOS studio: 512² generate + edit on device, durable print history | 1024² anatomy: `--two-stage` rescue works, texture re-gate awaits an SR-stage decision |
| mlx core 0.32.1 (fork) · hardened scripts/CI · 18-suite CI gate | **Bare-metal direct-dispatch engine** (mainline direction): TE landed opt-in (`--te-engine direct`, splice 1.6 s → 0.2 s); DiT/VAE next through gates |

Not in scope for v1: Klein 9B / FLUX.2 Dev, multi-reference editing, LoRA, CFG, bf16 paths.

## License

| | |
|--|--|
| Imarello source | MIT |
| FLUX.2 [klein] 4B weights | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) · Black Forest Labs |
| Vendored prompting skill | MIT · [black-forest-labs/skills](https://github.com/black-forest-labs/skills) |

Not affiliated with Black Forest Labs or Apple. FLUX is a trademark of Black Forest Labs.
