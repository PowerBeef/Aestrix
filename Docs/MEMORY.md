# Imarello memory model

## North star

> Never require simultaneous residency of text encoder + DiT + VAE.

Peak ≈ `max(TE, DiT+acts, VAE)` + context + latents + OS slack.

## Device tiers

| Tier | Hardware | Peak budget (ballpark) | Default side | Weights |
|------|----------|------------------------|--------------|---------|
| L | ~8 GB unified | ≤ ~5.5–6.5 GB process | **1024** default (tight; measure with `bench`) | 4-bit + staged |
| M | ~16 GB | ≤ ~10–12 GB | 1024 | 4-bit + staged |
| H | 24 GB+ | optional resident | 1024 | **4-bit** + staged; resident opt-in |

## Stages (T2I)

1. Load TE → encode → keep `EmbeddingBundle` (~8 MB) → **unload TE** + clearCache  
2. Load DiT → 4 denoise steps (eval each step) → **unload DiT** + clearCache  
3. Load VAE → decode → **unload VAE** + clearCache  

I2I adds VAE encode (then unload) before TE/DiT.

## Policies

| Policy | Behavior |
|--------|----------|
| `staged` (default) | One heavy module at a time; DiT **activation** checkpointing; **MLX Steel fused FA** (D=128); **Small Decoder** T2I/I2I decode + klein encode-only + **tiled cosine blend** |
| `stagedAggressive` | Reserved for tighter devices (optional DiT **weight** streaming — not default; measure first) |
| `resident` | Keep quant modules warm (Tier H only) |

## Low-RAM path (shipped)

| Technique | Status |
|-----------|--------|
| Staged TE → DiT → VAE | Done (default) |
| DiT per-block `eval` + `clearCache` | Done |
| **MLX Steel fused FA** (full Q, D∈{64,80,128}) | Done (product default for Klein) |
| f16 QKV when seq > 512 | Done (512² image tokens; 2026-08-13 A/B) |
| f16 scaled 4-bit Linear (`÷16`) | Done (product default 2026-08-15; `--attn-linear-compute f32` escape) |
| Query-chunked SDPA | Fallback only (unsupported head dims) |
| VAE D=512 query-chunked attention | Done (`VAEAttention`; `evalEachChunk` **off**; `--vae-attn-chunk 0` = MLXFast A/B) |
| `EvalCachePolicy.product` | Done (default). `mid` is ≥16 GB **bench only** (`--eval-cache mid`). **No `.high`.** |
| VAE decode-only for T2I | Done |
| BFL **Small Decoder** (product decode) | Done (default; `--vae-variant full` = klein AE). I2I encode stays klein |
| VAE tiled decode (overlap + cosine blend) | Done |
| DiT **weight** streaming (block JIT load) | Not default — iOS headroom spike |
| Draw Things–style AdaLN split | **N/A for Klein** — shared modulation is ~4% of DiT (`Docs/DRAW_THINGS.md`) |
| Imarello float4 fused Metal FA | Research / non-Steel D (`MetalFlashAttention`) |

## Measurement

- CLI: `imarello bench` (`pressure-map`, `dit-one-step`, `res-ladder`, `mem-stages`)  
- Instrument with `Memory.activeMemory` / process footprint around each stage  
- Tests must assert TE and DiT are never both loaded  

## Component sizes (4-bit Hub packs)

| Module | ~Disk / load |
|--------|----------------|
| TE | 2.26 GB |
| DiT | 2.18 GB |
| VAE (klein pack) | 0.17 GB (encode-only ~67 MB) |
| Small Decoder (default decode) | ~112 MB F32 (`FLUX.2-small-decoder`) |

On 8 GB M2 after the product path (auto + Small Decoder + f16 qmm, 2026-08-15), **live peak active ≈ DiT weights (~2.04 GiB)** at both 512² and 1024²; watermark **2.38 GiB (512)** / **3.46 GiB (1024)**. Prefer **512² for interactive speed**, not because 1024² OOMs. See [`PERF.md`](PERF.md).
