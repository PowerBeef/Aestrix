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
| f16 QKV from the **joint** sequence (all 25 blocks) | Done (2026-08-16 Tier-2; was per-stream `seq > 512`) |
| f16 scaled 4-bit Linear (`÷16`, scales pre-cast f16 at load) | Done (product default 2026-08-15/16; `--attn-linear-compute f32` escape) |
| Chunk-streamed single-stream blocks (proj/mlpHidden/to_out never full-length) | Done (2026-08-16; 1024² watermark 3.63 → 3.07 GiB) |
| Timestep conditioning hoisted per generate (temb + modulations + AdaLN-out) | Done (2026-08-16) |
| Untiled decode ≤ 768² (tile threshold 128) | Done (2026-08-16; decode −51% @768². **Untiled 1024² Metal-aborts** — keep tiling) |
| NHWC end-to-end VAE (one transpose per boundary) | Done (2026-08-16; bit-exact, speed-neutral — old transposes were lazy views) |
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
| TE | 2.26 GB disk · **~1.7 GB materialized** (27 of 36 layers loaded; pruned tail never read — lazy safetensors) |
| DiT | 2.18 GB |
| VAE (klein pack) | 0.17 GB (encode-only ~67 MB) |
| Small Decoder (default decode) | ~112 MB F32 (`FLUX.2-small-decoder`) |

On 8 GB M2 after the post-Tier-2 product path (pad-512 + f16 qmm + joint-f16 + Small Decoder, 2026-08-16), **live peak active ≈ DiT weights (~2.05 GiB)** at both 512² and 1024²; watermark **2.57 GiB (512)** / **3.00 GiB (1024)**. Prefer **512² for interactive speed**, not because 1024² OOMs. See [`PERF.md`](PERF.md).
