# Imarello runtime rules (short)

- **Model**: FLUX.2 klein **4B** only (Apache-2.0).
- **Weights**: **4-bit only** (`mlx-community/FLUX.2-Klein-4B-4bit`). No 3-bit product path. Never bf16 for users.
- **Memory**: `MemoryPolicy.staged` default — max one of {TE, DiT, VAE} resident.
- **Sampling**: 4 steps, guidance 1.0, no negatives, no upsampling by default.
- **Defaults**: `--text-tokens auto`; BFL Small Decoder; scaled f16 4-bit Linear (`÷16`).
- **Prompts**: follow project skill `flux-best-practices` (klein narrative prose, lighting, front-load subject).
- **MLX**: load `mlx-swift` skill; TE porting uses `mlx-swift-lm` patterns.
- **Workflow**: [`Docs/AGENT_WORKFLOW.md`](../../Docs/AGENT_WORKFLOW.md).
- **Verify**: measure peak memory before claiming Tier L/M readiness. Pixel **and** vision after any generate used to judge quality.
