# CLAUDE.md — Aestrix

Instructions for Claude Code (and similar agents) working in this repository.

**Authoritative product locks & agent rules:** [`AGENTS.md`](AGENTS.md)  
**Backlog / parked work:** [`Docs/ROADMAP.md`](Docs/ROADMAP.md)  
**Performance truth:** [`Docs/PERF.md`](Docs/PERF.md)

## What this is

From-scratch **Swift 6 + MLX** runtime for **FLUX.2 [klein] 4B** (Apache-2.0) on Apple Silicon. **Not** a fork of public Swift ports (those are oracles only).

## Non-negotiables

| Lock | Value |
|------|--------|
| Model | Klein **4B only** (no 9B / Dev) |
| Weights | **Prequant 4-bit** default (`mlx-community/FLUX.2-Klein-4B-4bit`) — no user bf16 |
| Memory | **Staged** TE → DiT → VAE (never co-resident) |
| Defaults | **1024²**, 4 steps, guidance 1.0, no negatives |
| Verify | Measure with `aestrix bench` before “faster/leaner” claims |

## Build / run

```bash
swift build -c release && ./Scripts/ensure-metallib.sh   # full ~130 MB metallib, not the stub
.build/release/aestrix t2i "…" --seed 42 --output out.png
.build/release/aestrix bench --width 1024 --height 1024 --warmup 1 --trials 3 --json /tmp/b.json
swift test
```

Snapshot path: `~/Library/Caches/Aestrix/models/mlx-community--FLUX.2-Klein-4B-4bit`

## Architecture (short)

1. **TE** (Qwen3): chat template; layers 9/18/27 → 7680; 512 pad.  
2. **DiT** (MMDiT 5+20): block `eval`+`clearCache`; **MLX Steel fused FA** for D=128 (full Q, simdgroup MMA); f16 QKV when seq>2048.  
3. **VAE**: decode-only for T2I; **tiled decode** with cosine blend (`VAETileConfig`).  
4. Public API: `AestrixRuntime`; MLX state in actors.

## Performance (M2 8 GB, release, 4-bit, fair W1T3)

| Canvas | e2e | denoise/step | peak active | watermark | RSS |
|--------|----:|-------------:|------------:|----------:|----:|
| 512² | ~31 s | ~6.1 s | 2.04 GiB | 2.99 GiB | 1.74 GiB |
| 1024² | ~94 s | ~20.2 s | 2.05 GiB | 3.75 GiB | 1.75 GiB |

Full tables and opt history: `Docs/PERF.md`.

## Agent workflow

- Product/prompting → skill `flux-best-practices` (project `.grok/skills/`)  
- After generation quality claims → `Docs/EVAL_WORKFLOW.md` + vision  
- After perf claims → `bench` / `bench-compare` + update `Docs/PERF.md`  
- Process: fix broken foundation before stacking the next phase  

## Out of scope (v1)

Klein 9B / Dev, multi-ref, CFG/negatives, LoRA, user bf16, full DiT compile on staged CLI.
