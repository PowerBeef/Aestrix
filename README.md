<div align="center">

# Aestrix

On-device FLUX.2-klein-4B image generation and editing for iOS, built on MLX-Swift.

[![Platform](https://img.shields.io/badge/platform-iOS%2026%2B-blue)](https://developer.apple.com/ios/)
[![Swift](https://img.shields.io/badge/Swift-6.3-orange)](https://swift.org)
[![MLX-Swift](https://img.shields.io/badge/MLX--Swift-0.31.4-green)](https://github.com/ml-explore/mlx-swift)
[![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)

</div>

---

## Overview

Aestrix is an MLX inference engine for running [FLUX.2-klein-4B](https://huggingface.co/mlx-community/flux2-klein-4b-4bit)
— a 4-billion-parameter rectified-flow transformer from Black Forest Labs — on iOS devices.
The engine performs text-to-image generation and natural-language instruction editing
entirely on-device, with no server or network dependency during inference.

The project ports the model's three components (Qwen3 text encoder, Klein diffusion
transformer, FLUX.2 VAE) from Python/MLX to Swift/MLX-Swift, using 4-bit quantized
weights to fit within the 8 GB memory budget of an iPhone 15 Pro. Staged model loading
frees each submodel after use, keeping peak wired memory tractable.

**Status:** Components M0–M4 are ported and load/run correctly. The end-to-end pipeline
(M5) is the next milestone. On-device performance has not yet been measured — the sub-second
target referenced in the FLUX.2-klein model card is aspirational, not validated.

---

## Architecture

```
prompt → Qwen3-4B [layers 9,18,27] → ctx (512, 7680)
                                            ↓
noise (128ch, H/16×W/16) → Klein Transformer (4 steps, guidance=1.0)
                                            ↓
              latent → VAE.decode (float32) → image
```

| Component | Implementation | Verified |
|---|---|---|
| **Tokenizer** | Qwen3 BPE via swift-transformers, chat template, 512-token pad | Exact token-id parity vs Python |
| **Text encoder** | Qwen3-4B (4-bit QuantizedLinear + QuantizedEmbedding), 36-layer GQA | Loads + runs, ctx shape confirmed |
| **Transformer** | 5 double-stream + 20 single-stream blocks, inner_dim 3072, 4D RoPE (t,h,w,l) | Loads + runs, forward output confirmed |
| **VAE** | AutoencoderKLFlux2, 32-ch latent, float32 decode (diffusers `force_upcast`) | float32 parity Δ<1e-4 vs mflux |
| **Scheduler** | Rectified-flow Euler, Karras power schedule, 4 steps | Implemented (M5) |
| **Editing** | Reference latents concatenated with t=10 RoPE offset | Planned (M6) |

---

## Build

```bash
git clone https://github.com/PowerBeef/Aestrix.git
cd Aestrix

# Build the engine
swift build --package-path AestrixEngine

# Build the iOS demo app
cd AestrixDemo && xcodegen generate
```

**Requirements:** Xcode 26+, Swift 6.3+. Development on Apple Silicon; on-device execution
requires an iPhone 15 Pro or later (the iOS Simulator cannot run MLX GPU operations).

---

## Testing

```bash
# MLX-free tests (tokenizer parity, config, downloader)
swift test --package-path AestrixEngine

# Full suite including MLX execution (xcodebuild compiles Metal shaders)
TEST_RUNNER_AESTRIX_HEAVY_TESTS=1 \
TEST_RUNNER_AESTRIX_FIXTURES="$(pwd)/AestrixEngine/.build/fixtures/flux2-klein-4b-4bit" \
  xcodebuild test -scheme AestrixEngine -destination 'platform=macOS'
```

Tests requiring multi-GB model downloads are gated on `AESTRIX_HEAVY_TESTS`. The parity
harness uses a Python mflux/MLX reference (`tools/vae_reference.py`) to validate each
component numerically. See [`CLAUDE.md`](CLAUDE.md) for MLX-specific build constraints.

---

## Roadmap

| # | Milestone | Status |
|---|---|---|
| 0 | Scaffold — SwiftPM, iOS demo, MLX verified | ✅ |
| 1 | IO — downloader, safetensors loader, tokenizer | ✅ |
| 2 | VAE — ported, float32 parity Δ<1e-4 | ✅ |
| 3 | Text encoder — Qwen3-4B loads, ctx (512, 7680) | ✅ |
| 4 | Transformer — Klein DiT forward, 4D RoPE | ✅ |
| 5 | Pipeline — wire components, first image | Next |
| 6 | Single-image instruction editing | Planned |
| 7 | Optimization — compile, Metal kernels | Planned |
| 8 | Production hardening — memory, thermal, error handling | Planned |

---

## Repository

```
Aestrix/
├── AestrixEngine/              # Swift package
│   ├── Sources/AestrixEngine/
│   │   ├── Models/             # FluxVAE, Qwen3TextEncoder, KleinTransformer
│   │   ├── IO/                 # HF downloader, safetensors loader, tokenizer
│   │   └── Support/            # Errors, logging
│   ├── Tests/                  # 11 tests across 7 suites
│   ├── tools/                  # Python reference generators
│   └── Package.swift
├── AestrixDemo/                # SwiftUI app (xcodegen project)
├── docs/PLAN.md                # Architecture document
└── CLAUDE.md                   # Developer guide
```

---

## Parity Testing

Each component is compared against a Python (mflux/MLX) reference to verify the port is
structurally correct:

- **VAE**: float32 reductions are deterministic — Swift matches mflux at max|Δ| < 1e-4.
- **Tokenizer**: exact token-id match against Python `tokenizers`.
- **Qwen3 / Transformer**: load and run with correct output shapes; exact numerical parity
  pending full reference harness.

MLX-Metal bf16 reductions are not bit-deterministic across runs (a known platform property).
The VAE decodes in float32 to avoid this, matching diffusers' `force_upcast = true`.

---

## Dependencies

- [FLUX.2-klein-4B](https://huggingface.co/mlx-community/flux2-klein-4b-4bit) — model weights (Apache-2.0, Black Forest Labs)
- [MLX-Swift](https://github.com/ml-explore/mlx-swift) 0.31.4 — Apple's ML framework for Swift
- [swift-transformers](https://github.com/huggingface/swift-transformers) 1.1.0 — BPE tokenizer
- [mflux](https://github.com/filipstrand/mflux) — Python MLX reference implementation used for parity testing

## License

[Apache-2.0](LICENSE)
