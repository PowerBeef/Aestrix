# Aestrix

**Low-RAM-first** native **Swift + MLX** runtime for [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) on Apple Silicon.

- **Model**: Klein 4B only (Apache-2.0)
- **Weights**: pre-quantized **4-bit** default (no user-facing bf16)
- **Memory**: staged pipeline — TE → unload → DiT → unload → VAE (never all co-resident)
- **v1**: text-to-image + single-image I2I (implementation in progress)
- **Platforms**: macOS library + CLI first, then iOS 26

## Status

| Phase | Status |
|-------|--------|
| 0 Scaffold + BFL skills + staged memory | **Done** |
| 1 Pure math (RoPE, temb, modulation layout, scheduler) | **Done** |
| 2 DiT MMDiT (MLX) | **Done** (structure + quant load path) |
| 3 VAE | **Done** (architecture + load path) |
| 4 Qwen3 TE | Done (`load-te`, `encode-prompt`) |
| 5 Staged T2I | Done (`aestrix t2i … --output out.png`) |
| 6 I2I strength | Done (`aestrix i2i … --image ref.png --strength 0.8`) |
| Eval workflow | Done — pixel + vision ([Docs/EVAL_WORKFLOW.md](Docs/EVAL_WORKFLOW.md)) |

```bash
swift test
swift build
./Scripts/ensure-metallib.sh           # required for generate after clean build
swift run aestrix info
swift run aestrix mem-selftest

# Generate + mandatory-style eval kickoff (pixel report + vision brief):
.build/debug/aestrix t2i "A red fox in a snowy forest at sunrise, photorealistic." \
  --width 512 --height 512 --steps 4 --seed 42 \
  --output /tmp/out.png --analyze --vision-brief

# Eval an existing PNG:
./Scripts/eval-generation.sh /tmp/out.png --prompt "…" --mode t2i

# Then open the PNG with a vision-capable agent and complete the checklist
# (see Docs/EVAL_WORKFLOW.md).
```

**Model cache:**  
`~/Library/Caches/Aestrix/models/mlx-community--FLUX.2-Klein-4B-4bit`

**Metal note:** run `./Scripts/ensure-metallib.sh` so MLX has a full ~130MB metallib (not the SPM stub).

## Requirements

- Apple Silicon Mac
- macOS 15+
- Xcode 16+ / Swift 6

## Build

```bash
swift build
swift test
swift run aestrix info
swift run aestrix mem-selftest
```

## Default weights

[`mlx-community/FLUX.2-Klein-4B-4bit`](https://huggingface.co/mlx-community/FLUX.2-Klein-4B-4bit)  
(~2.26 GB TE + ~2.18 GB DiT + ~0.17 GB VAE, already module-split).

See [Docs/WEIGHTS.md](Docs/WEIGHTS.md) and [Docs/MEMORY.md](Docs/MEMORY.md).

## Agent config

- [AGENTS.md](AGENTS.md) — product locks + skill routing + eval contract  
- [Docs/EVAL_WORKFLOW.md](Docs/EVAL_WORKFLOW.md) — **generation quality procedure** (pixel + vision)  
- [Docs/IMAGE_ANALYSIS.md](Docs/IMAGE_ANALYSIS.md) — metrics schema  
- [.grok/skills/flux-best-practices](.grok/skills/flux-best-practices) — official BFL FLUX prompting (vendored)  
- Engineering: `mlx-swift`, `mlx-swift-lm`, Context7, hf-cli, XcodeBuildMCP, Axiom  

## Prompting & eval

Follow BFL **FLUX.2 [klein]** guidance: narrative prose, subject first, strong lighting, no negative prompts.  
Prompts: [Docs/eval-prompts.md](Docs/eval-prompts.md).  
After every judged generation: [Docs/EVAL_WORKFLOW.md](Docs/EVAL_WORKFLOW.md).

## License

- Runtime code: MIT (to be finalized)  
- Model weights: Apache-2.0 (Black Forest Labs Klein 4B)  
- Vendored BFL skills: MIT (Black Forest Labs)  
