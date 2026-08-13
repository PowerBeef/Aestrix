# Aestrix memory model

## North star

> Never require simultaneous residency of text encoder + DiT + VAE.

Peak ≈ `max(TE, DiT+acts, VAE)` + context + latents + OS slack.

## Device tiers

| Tier | Hardware | Peak budget (ballpark) | Default side | Weights |
|------|----------|------------------------|--------------|---------|
| L | ~8 GB unified | ≤ ~5.5–6.5 GB process | **1024** default (tight; measure with `bench`) | 4-bit + staged |
| M | ~16 GB | ≤ ~10–12 GB | 1024 | 4-bit + staged |
| H | 24 GB+ | optional resident | 1024 | 4/8-bit; resident opt-in |

## Stages (T2I)

1. Load TE → encode → keep `EmbeddingBundle` (~8 MB) → **unload TE** + clearCache  
2. Load DiT → 4 denoise steps (eval each step) → **unload DiT** + clearCache  
3. Load VAE → decode → **unload VAE** + clearCache  

I2I adds VAE encode (then unload) before TE/DiT.

## Policies

| Policy | Behavior |
|--------|----------|
| `staged` (default) | One heavy module at a time; DiT **activation** checkpointing; **MLX Steel fused FA** (D=128); VAE decode-only + **tiled cosine blend** |
| `stagedAggressive` | Reserved for tighter devices (optional DiT **weight** streaming — not default; measure first) |
| `resident` | Keep quant modules warm (Tier H only) |

## Low-RAM path (shipped)

| Technique | Status |
|-----------|--------|
| Staged TE → DiT → VAE | Done (default) |
| DiT per-block `eval` + `clearCache` | Done |
| **MLX Steel fused FA** (full Q, D∈{64,80,128}) | Done (product default for Klein) |
| f16 QKV when seq > 512 | Done (512² image tokens; 2026-08-13 A/B) |
| Query-chunked SDPA | Fallback only (unsupported head dims) |
| VAE decode-only for T2I | Done |
| VAE tiled decode (overlap + cosine blend) | Done |
| DiT **weight** streaming (block JIT load) | Not default — iOS headroom spike |
| Draw Things–style AdaLN split | **N/A for Klein** — shared modulation is ~4% of DiT (`Docs/DRAW_THINGS.md`) |
| Aestrix float4 fused Metal FA | Research / non-Steel D (`MetalFlashAttention`) |

## Measurement

- CLI: `aestrix bench` (`pressure-map`, `dit-one-step`, `res-ladder`, `mem-stages`)  
- Instrument with `Memory.activeMemory` / process footprint around each stage  
- Tests must assert TE and DiT are never both loaded  

## Component sizes (4-bit Hub packs)

| Module | ~Disk / load |
|--------|----------------|
| TE | 2.26 GB |
| DiT | 2.18 GB |
| VAE | 0.17 GB (decode-only ~97 MB) |

On 8 GB M2 after the low-RAM path, **live peak active ≈ DiT weights (~2.0 GiB)** at both 512² and 1024²; watermark ~**3.0 GiB (512)** / ~**3.75 GiB (1024)**. Prefer **512² for interactive speed**, not because 1024² OOMs.
