<div align="center">

<img src="Docs/assets/readme/imarello-lockup-wide.png" width="560" alt="Imarello">

**Native Swift + [MLX](https://github.com/ml-explore/mlx-swift) runtime for [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)**

From-scratch on Apple Silicon — not a wrapper around another port. Formerly **Aestrix**.

[![Swift 6](https://img.shields.io/badge/Swift-6-F05138?logo=swift&logoColor=white)](https://swift.org)
[![macOS 15+](https://img.shields.io/badge/macOS-15%2B-000000?logo=apple&logoColor=white)](#quick-start)
[![MLX](https://img.shields.io/badge/MLX-0.31.6-blue)](https://github.com/ml-explore/mlx-swift)
[![Weights Apache-2.0](https://img.shields.io/badge/weights-Apache--2.0-green)](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
[![Eval floors](https://github.com/PowerBeef/Imarello/actions/workflows/eval-floors.yml/badge.svg)](https://github.com/PowerBeef/Imarello/actions/workflows/eval-floors.yml)

Text-to-image and single-image edit · **4-bit** weights · **1024²** default · fits an **8 GB** Mac

<img src="Docs/assets/readme/hero-coffee-1024.jpg" width="640" alt="Cozy coffee shop interior, warm afternoon light — Imarello T2I, 1024², 4 steps, seed 42">

<sub>1024² · 4 steps · seed 42 · 4-bit · 8 GB M2 Mac mini</sub>

</div>

```text
prompt ──► Qwen3 TE ──► unload ──► MMDiT (4 steps) ──► unload ──► VAE decode ──► PNG
              ~2 GB                  ~2 GB                  decode-only / tiled
```

Only one heavy module is resident at a time. Peak is roughly **max(TE, DiT, VAE)**, not the sum.

| Generate | Default | Memory | License |
|:--------:|:-------:|:------:|:-------:|
| T2I + I2I + identity | 1024² · 4 steps · guidance 1.0 | Staged · ~2 GiB live | MIT + Apache-2.0 weights |

---

## Samples

Unedited release outputs, 4-bit, 4 steps, fixed seeds.

<table>
<tr>
<td align="center" width="50%">
<img src="Docs/assets/readme/fisherman-1024.jpg" width="380" alt="Weathered fisherman, golden hour">
<br><sub>T2I · 1024² · seed 42</sub>
</td>
<td align="center" width="50%">
<img src="Docs/assets/readme/fox-512.jpg" width="380" alt="Red fox in snow at sunrise">
<br><sub>T2I · 512² · seed 7</sub>
</td>
</tr>
<tr>
<td align="center">
<img src="Docs/assets/readme/i2i-mug-before.jpg" width="280" alt="Cobalt ceramic mug">
<br><sub>Source</sub>
</td>
<td align="center">
<img src="Docs/assets/readme/i2i-mug-after.jpg" width="280" alt="Emerald ceramic mug">
<br><sub><code>i2i --strength 0.8</code> · cobalt → emerald</sub>
</td>
</tr>
<tr>
<td align="center">
<img src="Docs/assets/readme/identity-ref.jpg" width="280" alt="Woman, ivory blouse, studio light">
<br><sub>Reference</sub>
</td>
<td align="center">
<img src="Docs/assets/readme/identity-edit.jpg" width="280" alt="Same woman, emerald silk, golden hour">
<br><sub><code>i2i --strength 0.9 --identity</code> · wardrobe + light; face stays</sub>
</td>
</tr>
</table>

---

## Quick start

**Needs:** Apple Silicon, macOS 15+, Xcode 16 / Swift 6, ~5.1 GB disk, [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`hf`).

```bash
git clone https://github.com/PowerBeef/Imarello.git && cd Imarello
# Old clone URLs (`PowerBeef/Aestrix`) redirect here.
swift build -c release
./Scripts/ensure-metallib.sh          # full MLX Metal lib (~130 MB), not the 3 KB stub

hf download mlx-community/FLUX.2-Klein-4B-4bit \
  --revision 1cebb9b45c21ece14a42615b16bf5fa4de9b56da \
  --local-dir ~/Library/Caches/Imarello/models/mlx-community--FLUX.2-Klein-4B-4bit

hf download black-forest-labs/FLUX.2-small-decoder \
  --revision a3efc24f613ef42d9428af62fdbd6f5fd8856c4a \
  --include small_decoder.safetensors --include config.json \
  --local-dir ~/Library/Caches/Imarello/models/black-forest-labs--FLUX.2-small-decoder

.build/release/imarello info           # tier, pin, snapshot, Steel metallib, Small Decoder

.build/release/imarello t2i \
  "A weathered fisherman at the helm of a wooden boat, golden hour rim light, shallow depth of field." \
  --seed 42 --output out.png
```

Default canvas is **1024²**. For a faster smoke: `--width 512 --height 512` (~19 s on an 8 GB M2).

```bash
# Recolor / style — strength ≥ 0.8 for object color
.build/release/imarello i2i "the same ceramic mug, emerald green glaze, same morning light" \
  --image out.png --strength 0.8 --seed 7 --output edit.png

# People — higher strength + identity stack (ref latents, face mask, milder schedule)
.build/release/imarello i2i "Same person, exact face and pose; outdoor golden hour, emerald silk top" \
  --image portrait.png --strength 0.9 --identity --seed 7 --output edit-id.png
```

Add `--analyze --vision-brief` to any generate for the pixel report + agent checklist.

---

## Use it well

**Prompting (klein).** Narrative prose, subject first, lighting spelled out. No negative prompts — say what you want. Hex colors (`#C45C26`) and quoted text (`"OPEN STUDIO"`) work. Recipes: [`.grok/skills/flux-best-practices`](.grok/skills/flux-best-practices).

**I2I.** Strength-only is for color/style. People + scene changes want `--identity` at **0.85–0.9**. See [Docs/I2I_STRENGTH.md](Docs/I2I_STRENGTH.md).

**Repeats.** Prompt-embed cache is **on** (`~/Library/Caches/Imarello/embeds`). A hit skips the text encoder (~4 s at 512²), byte-identical. Opt out with `--no-embed-cache`.

**Many prompts.** `imarello session` keeps modules warm on **≥16 GB**. 8 GB stays staged.

**Not in v1:** Klein 9B / FLUX.2 Dev, multi-reference, CFG, LoRA, user-facing bf16.

---

## Performance

Apple M2 **8 GB** Mac mini · release + full metallib · 4-bit staged · W1/T3 · seed 42 · fox prompt. Product defaults: `--text-tokens auto`, BFL Small Decoder, scaled f16 4-bit Linear (`÷16`). Live peak is the DiT (~2 GiB) at both sizes — 1024² is slower, not much hungrier.

| Canvas | Time | Denoise / step | Decode | MLX active | Watermark |
|--------|-----:|---------------:|-------:|-----------:|----------:|
| **512²** | **19.4 s** | 3.34 s | 1.05 s | 2.04 GiB | 2.38 GiB |
| **1024²** | **74.0 s** | 15.91 s | 5.27 s | 2.05 GiB | 3.46 GiB |

`--text-tokens auto` trims pad tokens. Same seed is not the same PNG as `--text-tokens 512` (gallery / old recipes). See [Docs/TEXT_TOKENS.md](Docs/TEXT_TOKENS.md).

BFL **Small Decoder** is the default decode (−37% vs the klein AE). `--vae-variant full` restores klein. Pin: [Docs/WEIGHTS.md](Docs/WEIGHTS.md).

`--attn-linear-compute f32` is the old f32 GEMM. Raw unscaled f16 overflows to noise — do not use it.

Same-day A/Bs and older trees: [Docs/PERF.md](Docs/PERF.md).

---

## CLI

| Command | |
|---------|--|
| `t2i` / `i2i` | Generate. `--identity` on I2I for people. |
| `session` | Warm multi-prompt loop (≥16 GB). |
| `info` | Tier, Hub pin, snapshot, Steel metallib. |
| `analyze-image` | Pixel quality on an existing PNG. |
| `bench` / `bench-compare` | Timings, memory, pressure-map, I2I. |
| `load-te` `load-dit` `load-vae` | Staged load smoke. |
| `encode-prompt` `schedule` `mem-selftest` | TE-only, sigmas, dry residency. |

`imarello --help` and `imarello t2i --help` list flags.

---

## Library

Public API is `ImarelloPipeline` (an `actor` — all MLX state lives there).

```swift
import ImarelloCore
import ImarelloRuntime

let pipeline = ImarelloPipeline()
let url = try await pipeline.generate(
    T2IRequest(
        prompt: "A red fox in a snowy forest at sunrise, photorealistic.",
        seed: 42
    )
)
```

`edit(_:)` takes `I2IRequest` (`strength`, optional `identity`). Modules:

| | |
|--|--|
| `ImarelloRuntime` | Staged pipeline, I2I / identity, embed cache |
| `ImarelloText` | Qwen3 tap (layers 9/18/27 → 7680) |
| `ImarelloDiT` | MMDiT 5+20, Steel fused FA, f16 QKV, scaled f16 4-bit Linear |
| `ImarelloVAE` | Small Decoder default, klein encode-only, tiled cosine blend |
| `ImarelloEval` / `ImarelloBench` | Pixel gates (incl. unstructured-garbage fail) · multi-trial harness |
| `ImarelloCore` | Pins, scheduler, RoPE, policy |

Design: [Docs/ARCHITECTURE.md](Docs/ARCHITECTURE.md) · [Docs/MEMORY.md](Docs/MEMORY.md).

```bash
# Filtered unit tests (no weights). Skip unfiltered `swift test` after a GPU abort.
swift test --filter 'HostPreflight|GoldenMetric|Flux2Math|HubPin|Metallib|EvalCachePolicy|TextTokenMode'

.build/release/imarello bench --width 512 --height 512 --warmup 1 --trials 3 \
  --json /tmp/bench.json
IMARELLO=.build/release/imarello ./Scripts/eval-regression.sh   # 512² pixel loop
```

---

## Status

| Shipping | Parked |
|----------|--------|
| macOS library + CLI | iOS host (same staged core) |
| 1024² on 8 GB · 4-bit staged | Multi-ref, CFG, LoRA, bf16 |
| T2I, strength I2I, `--identity` | |
| Steel FA · Small Decoder · auto tokens · f16 qmm · embed cache · Hub pin + CI floors | |

Backlog: [Docs/ROADMAP.md](Docs/ROADMAP.md). Agent rules: [AGENTS.md](AGENTS.md).

| Doc | |
|-----|--|
| [WEIGHTS.md](Docs/WEIGHTS.md) | Hub packs and pinned revisions |
| [PERF.md](Docs/PERF.md) | Benchmarks and pressure probes |
| [EVAL_WORKFLOW.md](Docs/EVAL_WORKFLOW.md) | Pixel + vision quality gate |
| [I2I_STRENGTH.md](Docs/I2I_STRENGTH.md) | Color vs identity strength |
| [TEXT_TOKENS.md](Docs/TEXT_TOKENS.md) | `--text-tokens auto` vs pad-512 |

---

## License

| | |
|--|--|
| Imarello source | MIT |
| FLUX.2 [klein] 4B weights | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) · Black Forest Labs |
| Vendored prompting skill | MIT · [black-forest-labs/skills](https://github.com/black-forest-labs/skills) |

Not affiliated with Black Forest Labs or Apple. FLUX is a trademark of Black Forest Labs.
