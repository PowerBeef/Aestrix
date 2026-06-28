# Aestrix

![Swift](https://img.shields.io/badge/Swift-6.0-orange.svg)
![iOS](https://img.shields.io/badge/iOS-26+-blue.svg)
![macOS](https://img.shields.io/badge/macOS-14+-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)

Aestrix is a Swift inference engine that runs [FLUX.2-klein-4B](https://huggingface.co/mlx-community/flux2-klein-4b-4bit) on iOS devices using Apple's [MLX](https://github.com/ml-explore/mlx-swift) framework.

It performs text-to-image generation — and eventually single-image instruction editing — entirely on-device. The 4-billion-parameter rectified-flow transformer runs with 4-bit quantized weights (~4.6 GB total) and a staged loading strategy that keeps peak memory within the budget of an iPhone 15 Pro.

## Pipeline

```
prompt → Qwen3-4B → ctx (512, 7680)
                        ↓
noise → Klein Transformer (4 Euler steps, no CFG) → latent
                                                    ↓
                                        VAE decode (float32) → image
```

FLUX.2-klein uses a distilled 4-step rectified-flow schedule with guidance fixed at `1.0`, so each denoising step needs only a single transformer forward pass. Image editing reuses the same transformer: reference-image latents are concatenated to the noise tokens with a distinct `t` coordinate in the 4D RoPE, requiring no separate edit model or mask.

## Features

- **On-device inference** — no network call during generation; all models run locally on Apple Silicon.
- **Staged memory loading** — Qwen3 text encoder, Klein transformer, and FLUX.2 VAE are loaded and freed in sequence to stay under ~4 GB of wired memory at peak.
- **4-bit quantization** — uses MLX fused dequant-GEMM for the bulk of the model weights.
- **Float32 VAE decode** — avoids amplified bf16 reduction drift, matching diffusers' `force_upcast` behavior.
- **Numerical parity** — every ported component is validated against a Python mflux/MLX reference.

## Component Status

| Component | Implementation | Verification |
|---|---|---|
| Tokenizer | Qwen3 BPE (swift-transformers), chat template, 512-token pad | Exact token-id match vs Python |
| Text encoder | Qwen3-4B, 36-layer GQA, 4-bit `QuantizedLinear` + `QuantizedEmbedding` | Loads and runs; ctx shape confirmed |
| Transformer | 5 double-stream + 20 single-stream blocks, inner_dim 3072, 4D RoPE | Loads and runs; output shape confirmed |
| VAE | `AutoencoderKLFlux2` (32-ch latent), float32 decode | float32 parity max\|Δ\| < 1e-4 vs mflux |
| Scheduler | Rectified-flow Euler, Karras power schedule | Implemented |
| Editing | Reference latents concatenated with `t=10` RoPE offset | Planned |

## Getting Started

Requires **Xcode 26+**, **Swift 6.3+**, and an **Apple Silicon Mac**. The iOS Simulator cannot run MLX GPU operations, so on-device execution requires an **iPhone 15 Pro or later**.

Build the engine package:

```bash
swift build --package-path AestrixEngine
```

Run the MLX-free unit tests:

```bash
swift test --package-path AestrixEngine
```

Run the full suite (requires `xcodebuild` to compile Metal shaders):

```bash
cd AestrixEngine
TEST_RUNNER_AESTRIX_RUN_MLX_TESTS=1 \
TEST_RUNNER_AESTRIX_FIXTURES="$(pwd)/.build/fixtures/flux2-klein-4b-4bit" \
  xcodebuild test -scheme AestrixEngine -destination 'platform=macOS'
```

Heavy tests that download multi-GB model shards are gated on `AESTRIX_HEAVY_TESTS=1`. Generated test images are written to `AestrixEngine/outputs/`; override with `AESTRIX_OUTPUTS=<path>`.

Generate the demo app project:

```bash
cd AestrixDemo && xcodegen generate
```

## Project Structure

```
Aestrix/
├── AestrixEngine/          Swift package — inference engine, models, pipeline, IO
│   ├── Sources/AestrixEngine/
│   ├── Tests/AestrixEngineTests/
│   ├── tools/vae_reference.py
│   └── Package.swift
├── AestrixDemo/            Thin SwiftUI iOS demo app
└── docs/PLAN.md            Architecture and milestone roadmap
```

## Roadmap

| # | Milestone | Status |
|---|---|---|
| 0 | Package scaffold, MLX integration | Done |
| 1 | Downloader, safetensors loader, tokenizer | Done |
| 2 | VAE (float32 parity proven) | Done |
| 3 | Qwen3 text encoder | Done |
| 4 | Klein transformer + 4D RoPE | Done |
| 5 | End-to-end pipeline, first verified image | In progress |
| 6 | Image editing | Planned |
| 7 | Optimization (compile, Metal kernels) | Planned |
| 8 | Memory/thermal hardening | Planned |

## Dependencies

- [FLUX.2-klein-4B](https://huggingface.co/mlx-community/flux2-klein-4b-4bit) — 4-bit MLX weights (Apache-2.0)
- [MLX-Swift](https://github.com/ml-explore/mlx-swift) 0.31.4 — Apple ML framework
- [swift-transformers](https://github.com/huggingface/swift-transformers) 1.1.0 — tokenizer
- [mflux](https://github.com/filipstrand/mflux) — Python reference for parity testing

## Agent Guidance

Build constraints, Kimi Code CLI tooling notes, and project conventions are documented in [`AGENTS.md`](AGENTS.md).

## License

[Apache-2.0](LICENSE)
