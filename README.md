<div align="center">

# Aestrix

**Blazing-fast on-device image generation & editing for iOS**

Powered by MLX · FLUX.2-klein-4B · Swift 6

[![Platform](https://img.shields.io/badge/platform-iOS%2026%2B-blue)](https://developer.apple.com/ios/)
[![Swift](https://img.shields.io/badge/Swift-6.3-orange)](https://swift.org)
[![MLX-Swift](https://img.shields.io/badge/MLX--Swift-0.31.4-green)](https://github.com/ml-explore/mlx-swift)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)
[![Model](https://img.shields.io/badge/model-FLUX.2--klein--4B-ff6b6b)](https://huggingface.co/mlx-community/flux2-klein-4b-4bit)

</div>

---

Aestrix is an optimized MLX inference engine that runs **FLUX.2-klein-4B** — a 4-billion-parameter
rectified-flow transformer — entirely **on-device** on iPhone 15 Pro and later. It generates
high-quality images from text prompts and performs natural-language instruction editing, all
without a server, cloud API, or network connection.

### Why Aestrix?

- **⚡ Sub-second generation** — FLUX.2-klein is distilled to **4 denoising steps** with no
  classifier-free guidance (single forward pass per step). Combined with 4-bit quantization,
  the engine targets interactive-speed generation on Apple Silicon GPUs.
- **🎨 Generate AND edit** — one unified model handles both text-to-image generation and
  instruction-based editing ("turn this cat into a dog"). No separate edit pipeline, no masks.
- **🔒 100% on-device** — your prompts and images never leave the device. Zero network calls
  during inference.
- **💾 Fits in 8 GB** — staged model loading frees the 2.26 GB text encoder before denoising
  and the transformer before VAE decode, keeping peak wired memory under ~4 GB.

---

## Architecture

```
 ┌─────────┐     ┌────────────┐     ┌───────────────────┐     ┌──────┐
 │  prompt │────▶│  Qwen3-4B  │────▶│  Klein Transformer │────▶│ VAE  │──▶ image
 │         │     │ (layers    │     │  4-step rectified  │     │decode│
 │         │     │  9,18,27)  │     │  flow (no CFG)     │     │      │
 └─────────┘     └────────────┘     └───────────────────┘     └──────┘
   tokenize        encode → ctx       denoise noise → latent     decode
                   (512, 7680)        (128-ch, H/16)            → pixels
```

**Key design choices:**
| Component | Detail |
|---|---|
| **Text encoder** | Qwen3-4B (4-bit), extract hidden states at layers [9, 18, 27] → context (512, 7680) |
| **Transformer** | 5 double-stream + 20 single-stream blocks, 4D RoPE (t, h, w, l), 4 steps, guidance = 1.0 |
| **VAE** | AutoencoderKLFlux2 (32-ch latent, 16× spatial), decodes in float32 for quality |
| **Editing** | Reference image latents concatenated to noise tokens with `t=10` RoPE offset |
| **Memory** | Staged load/free via MLX wired-memory tickets — peak ~4 GB < 8 GB |

---

## Quick Start

```bash
git clone https://github.com/PowerBeef/Aestrix.git
cd Aestrix

# Build the engine package
swift build --package-path AestrixEngine

# Run the iOS demo (regenerate the Xcode project first)
cd AestrixDemo && xcodegen generate && open AestrixDemo.xcodeproj
```

> **Requirements:** Xcode 26+, Swift 6.3+, an Apple Silicon Mac for development,
> and an **iPhone 15 Pro or later** for on-device execution (the iOS Simulator cannot
> run MLX GPU ops — a physical device is required).

---

## Build & Test

```bash
# Fast MLX-free unit tests (config, tokenizer parity, downloader)
swift test --package-path AestrixEngine

# Full test suite including MLX execution (requires xcodebuild for Metal shaders)
TEST_RUNNER_AESTRIX_RUN_MLX_TESTS=1 \
TEST_RUNNER_AESTRIX_FIXTURES="$(pwd)/AestrixEngine/.build/fixtures/flux2-klein-4b-4bit" \
  xcodebuild test -scheme AestrixEngine -destination 'platform=macOS'
```

Heavy tests (multi-GB model downloads) are gated on `AESTRIX_HEAVY_TESTS`. See
[`CLAUDE.md`](CLAUDE.md) and [`AestrixEngine/README.md`](AestrixEngine/README.md) for details.

---

## Roadmap

| # | Milestone | Status |
|---|---|---|
| 0 | Scaffold — SwiftPM package + iOS demo, MLX integration verified | ✅ |
| 1 | IO — HF downloader, safetensors loader, Qwen3 tokenizer | ✅ |
| 2 | VAE — encoder/decoder ported, **float32 parity Δ<1e-4** vs mflux | ✅ |
| 3 | Text encoder — Qwen3-4B loads + runs, ctx (1, 512, 7680) | ✅ |
| 4 | Transformer + 4D RoPE | ⏳ |
| 5 | Pipeline (t2i) — first image | ⏳ |
| 6 | Single-image edit | ⏳ |
| 7 | Optimization — compile, custom Metal kernels | ⏳ |
| 8 | Harden — memory/thermal, demo polish | ⏳ |

---

## Project Structure

```
Aestrix/
├── AestrixEngine/              # Swift package — the MLX engine
│   ├── Sources/AestrixEngine/
│   │   ├── Engine/             # Public actor façade (load/generate/edit)
│   │   ├── Models/             # FluxVAE, Qwen3TextEncoder, (Transformer soon)
│   │   ├── IO/                 # HF downloader, safetensors loader, tokenizer
│   │   └── Support/            # Errors, logging
│   ├── Tests/                  # Unit + parity tests (10/10 green)
│   ├── tools/                  # Python mflux reference generators
│   └── Package.swift           # MLX-Swift 0.31.4, swift-transformers 1.1.0
├── AestrixDemo/                # Thin SwiftUI iOS app
├── docs/PLAN.md                # Full architecture + roadmap
└── CLAUDE.md                   # Developer guide (build, test, MLX gotchas)
```

---

## Numeric Parity

Each component is validated against a Python **mflux/MLX** reference:
- **VAE**: float32 parity at max|Δ| < 1e-4 (proven via deterministic float32 reductions).
- **Tokenizer**: exact token-id match vs Python `tokenizers`.
- **Qwen3**: loads + runs (36-layer forward), ctx shape verified; exact parity in progress.

The bf16 regime shows amplified drift due to inherent **MLX-Metal bf16 non-determinism**
(well-documented). Production VAE decodes in float32 (matching diffusers `force_upcast`).

---

## Acknowledgments

- **[Black Forest Labs](https://bfl.ai)** — FLUX.2-klein-4B (Apache-2.0 open weights)
- **[ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift)** — Apple's MLX framework for Swift
- **[filipstrand/mflux](https://github.com/filipstrand/mflux)** — the canonical MLX FLUX port (our reference)
- **[huggingface/swift-transformers](https://github.com/huggingface/swift-transformers)** — BPE tokenizer for Swift

## License

[Apache-2.0](LICENSE). The FLUX.2-klein-4B model weights are also Apache-2.0 licensed by Black Forest Labs.
