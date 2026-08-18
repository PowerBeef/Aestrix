<div align="center">

<img src="Docs/assets/readme/imarello-lockup-wide.png" width="560" alt="Imarello">

**Native Swift + [MLX](https://github.com/ml-explore/mlx-swift) runtime for [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)**

From-scratch on Apple Silicon — not a wrapper around another port. Formerly **Aestrix**.

[![Swift 6](https://img.shields.io/badge/Swift-6-F05138?logo=swift&logoColor=white)](https://swift.org)
[![macOS 15+](https://img.shields.io/badge/macOS-15%2B-000000?logo=apple&logoColor=white)](#quick-start)
[![MLX](https://img.shields.io/badge/MLX-0.32.1%20(fork)-blue)](https://github.com/PowerBeef/mlx-swift)
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

**Needs:** Apple Silicon, macOS 15+, Xcode 26 / Swift 6 (CI runs Xcode 26.6), ~5.1 GB disk, [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`hf`).

```bash
git clone https://github.com/PowerBeef/Imarello.git && cd Imarello
# Old clone URLs (`PowerBeef/Aestrix`) redirect here.
swift build -c release
./Scripts/ensure-metallib.sh          # full MLX Metal lib (~155 MB) — REQUIRED: kernels load by name (nojit)

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

Default canvas is **1024²**. For a faster smoke: `--width 512 --height 512` (~22 s on an 8 GB M2).

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

**Prompting (klein).** Narrative prose, subject first, lighting spelled out. No negative prompts — say what you want. Hex colors (`#C45C26`) and quoted text (`"OPEN STUDIO"`) work. Recipes: [`.claude/skills/flux-best-practices`](.claude/skills/flux-best-practices).

**I2I.** Strength-only is for color/style. People + scene changes want `--identity` at **0.85–0.9**. See [Docs/I2I_STRENGTH.md](Docs/I2I_STRENGTH.md).

**1024² full-body subjects.** Klein's 4-step path can occasionally break body plans at 1024² (multi-limb). `t2i --two-stage` composes at 512² (where anatomy is reliable) and refines at 1024² — an opt-in rescue that trades a little texture sharpness for correct anatomy.

**Repeats.** Prompt-embed cache is **on** (`~/Library/Caches/Imarello/embeds`). A hit skips the text encoder (~4 s at 512²), byte-identical. Entries are written atomically and validated on load; corrupt ones self-delete. Opt out with `--no-embed-cache`.

**Many prompts.** `imarello session` keeps modules warm on **≥16 GB**. 8 GB stays staged.

**Not in v1:** Klein 9B / FLUX.2 Dev, multi-reference, CFG, LoRA, user-facing bf16.

---

## Performance

Apple M2 **8 GB** Mac mini · release + full metallib · 4-bit staged · W1/T3 · seed 42 · fox prompt. Product defaults: `--text-tokens 512` (pad), BFL Small Decoder, scaled f16 4-bit Linear (`÷16`). Live peak is the DiT (~2 GiB) at both sizes — 1024² is slower, not much hungrier.

| Canvas | Time | Denoise / step | Decode | MLX active | Watermark |
|--------|-----:|---------------:|-------:|-----------:|----------:|
| **512²** | **21.9 s** | 4.22 s | 0.94 s | 2.06 GiB | 2.57 GiB |
| **1024²** | **67.5 s** | 14.69 s | 4.58 s | 2.05 GiB | 3.00 GiB |

The 2026-08-16 Tier-1/2 engine work (fused qmm rescale, hoisted step conditioning, chunk-streamed single blocks, joint-f16 attention, untiled 768² decode, relaxed cache policy) took 1024² from 79.0 → 71.0 s (−10%) and its watermark from 3.63 → **3.00 GiB (−17%)**. The **2026-08-18 mlx core 0.32.1 bump** (via a [maintained fork](https://github.com/PowerBeef/mlx-swift) — upstream mlx-swift is pre-0.32; kernels are served from the prebuilt metallib) added split-K/`gemv_wide` small-M kernels: text-encode **−19–21%** in every mode, 512² now **21.9 s** and identity-I2I **36.2 s**, memory identical, full quality gate (15/15 pixel + vision) passed. 768² decode is untiled (no seams). `--text-tokens auto` (opt-in) still trims for speed at 512² but weakens prompt conditioning; same seed under `auto` is not the same PNG. See [Docs/TEXT_TOKENS.md](Docs/TEXT_TOKENS.md).

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

`edit(_:)` takes `I2IRequest` (`strength`, optional `identity`). Both calls are **cancellable** — the pipeline checks `Task` cancellation at every stage boundary and denoise step, unloading the resident stage before rethrowing. Inputs are validated up front (`steps`, canvas, `--ref-downsample` grids) instead of trapping mid-pipeline. Modules:

| | |
|--|--|
| `ImarelloRuntime` | Staged pipeline, I2I / identity, embed cache |
| `ImarelloText` | Qwen3 tap (layers 9/18/27 → 7680) |
| `ImarelloDiT` | MMDiT 5+20, Steel fused FA (joint-f16), scaled f16 4-bit Linear, chunk-streamed single blocks, hoisted step conditioning |
| `ImarelloVAE` | Small Decoder default, klein encode-only, NHWC end-to-end, untiled ≤768² / tiled cosine blend at 1024² |
| `ImarelloEval` / `ImarelloBench` | Pixel gates (incl. unstructured-garbage fail) · multi-trial harness |
| `ImarelloCore` | Pins, scheduler, RoPE, policy |

Design: [Docs/ARCHITECTURE.md](Docs/ARCHITECTURE.md) · [Docs/MEMORY.md](Docs/MEMORY.md).

```bash
# Filtered unit tests (no weights). Never run unfiltered `swift test` (Metal FA tests can hang).
swift test --filter 'HostPreflight|GoldenMetric|Flux2Math|IdentityPreserve|ImarelloBench|HubPin|Metallib|EvalCachePolicy|TextTokenMode|PromptEmbedCacheKey|VAEAttention|DiTOpProfile|DeviceHarness|Qwen'

.build/release/imarello bench --width 512 --height 512 --warmup 1 --trials 3 \
  --json /tmp/bench.json
IMARELLO=.build/release/imarello ./Scripts/eval-regression.sh   # 512² pixel loop
```

---

## Status

| Shipping | In progress / parked |
|----------|----------------------|
| macOS library + CLI | iOS 26 studio (`Apps/ImarelloIOS`) — **512² T2I and I2I on device**, rebuilt two-page UI; 1024² anatomy has a working opt-in rescue (`--two-stage`), default fix pending an SR stage |
| 1024² on 8 GB · 4-bit staged (enforced: `--weights` is 4-bit only) | Multi-ref, CFG, LoRA, bf16 |
| T2I, strength I2I, `--identity` · cancellable runs · validated CLI inputs | `--text-tokens auto` speed path (opt-in) |
| Steel FA · joint-f16 attention · f16 qmm · Small Decoder · untiled ≤768² decode · crash-safe embed cache · Hub pin + CI (18 no-Metal suites) | Partial-pad conditioning study ([Docs/ENGINE_RESEARCH.md](Docs/ENGINE_RESEARCH.md) Tier 3) |

**2026-08-18 hardening pass:** a full-repo adversarial audit (52 verified findings) was fixed the same day — durable iOS print history, self-healing device-harness queue, real cancellation, atomic caches, input validation, and a metallib build that refuses partial kernel sets and tracks the mlx-swift pin. **Same-day engine uplift:** mlx core bumped to 0.32.1 via a maintained fork (text-encode −19–21%, 512² 21.9 s, memory flat, full quality gate passed), a two-stage anatomy rescue for 1024², and a GPU capability probe readying the M5/A19 Neural-Accelerator path. Details in the [roadmap decision log](Docs/ROADMAP.md).

Backlog: [Docs/ROADMAP.md](Docs/ROADMAP.md). Agent rules: [CLAUDE.md](CLAUDE.md). Workflow (skills / MCP): [Docs/AGENT_WORKFLOW.md](Docs/AGENT_WORKFLOW.md).

| Doc | |
|-----|--|
| [WEIGHTS.md](Docs/WEIGHTS.md) | Hub packs and pinned revisions |
| [PERF.md](Docs/PERF.md) | Benchmarks and pressure probes |
| [EVAL_WORKFLOW.md](Docs/EVAL_WORKFLOW.md) | Pixel + vision quality gate |
| [AGENT_WORKFLOW.md](Docs/AGENT_WORKFLOW.md) | Skills, MCP servers, host-safe loops |
| [IOS.md](Docs/IOS.md) | iOS 26 demo — Simulator UI, `xcodebuild` + `devicectl`, device generate harness |
| [I2I_STRENGTH.md](Docs/I2I_STRENGTH.md) | Color vs identity strength |
| [TEXT_TOKENS.md](Docs/TEXT_TOKENS.md) | `--text-tokens auto` vs pad-512 |
| [ENGINE_RESEARCH.md](Docs/ENGINE_RESEARCH.md) | Engine analysis, optimization ledger, open research tiers |

---

## iOS 26 studio

`Apps/ImarelloIOS` is a phone studio on the same staged `ImarelloRuntime`, rebuilt as **two full-bleed pages** you swipe between:

- **Stage** — your print fills the screen; a plate chip carries `512² · seed 42`, one glass status row speaks for every state (gate, progress + elapsed, staged edit, error + Try Again), and a gold **Develop** pill runs it. **Stop actually stops** — the pipeline honors cancellation mid-denoise.
- **Contact Sheet** — every print you have made, in a film grid (with the same status row). Tap one to open the viewer: zoom, swipe, Share, Save, Delete — and **Edit**, which develops a new print from *that* one at strength 0.8.

Prints are **durable**: a versioned index and canonical PNGs live in Application Support, so history survives OS cache purges and app reinstall container swaps. Design system: [Apps/ImarelloIOS/DESIGN.md](Apps/ImarelloIOS/DESIGN.md).

- **Simulator** is UI only — MLX has no Simulator Metal. Develop is a no-op.
- **Physical iPhone** is the generate host. Build with `xcodebuild` (`-skipPackagePluginValidation`) and install with `devicectl`. No Mac Catalyst.
- Weights stay out of the bundle (`Caches/Imarello/models/`). Resync after every install (`Scripts/sync-ios-device-weights.sh`).
- Drive a device generate from the Mac with `Scripts/ios-device-harness.sh` (default 512²; `--steps`, `--text-tokens`, and `--strength` apply on device, bad jobs are quarantined instead of wedging the queue, and stale results are rejected automatically). Do not tap Develop from an agent. See [Docs/IOS.md](Docs/IOS.md).

## License

| | |
|--|--|
| Imarello source | MIT |
| FLUX.2 [klein] 4B weights | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) · Black Forest Labs |
| Vendored prompting skill | MIT · [black-forest-labs/skills](https://github.com/black-forest-labs/skills) |

Not affiliated with Black Forest Labs or Apple. FLUX is a trademark of Black Forest Labs.
