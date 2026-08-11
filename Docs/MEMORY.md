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
| `staged` (default) | One heavy module at a time |
| `stagedAggressive` | + lower res + optional DiT block streaming |
| `resident` | Keep quant modules warm (Tier H only) |

## Measurement

- CLI: `aestrix mem-selftest` (fake stages) and later `aestrix bench-mem`  
- Instrument with `Memory.activeMemory` / process footprint around each stage  
- Tests must assert TE and DiT are never both loaded  

## Component sizes (4-bit Hub packs)

| Module | ~Disk / load |
|--------|----------------|
| TE | 2.26 GB |
| DiT | 2.18 GB |
| VAE | 0.17 GB |

Activations at 1024² dominate mid-tier peaks; prefer 512² on Tier L.
