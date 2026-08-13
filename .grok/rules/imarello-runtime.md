# Imarello runtime rules (short)

- **Model**: FLUX.2 klein **4B** only (Apache-2.0).
- **Weights**: prequant **4-bit** default (`mlx-community/FLUX.2-Klein-4B-4bit`); optional 3/6/8-bit. Never bf16 for users.
- **Memory**: `MemoryPolicy.staged` default — max one of {TE, DiT, VAE} resident.
- **Sampling**: 4 steps, guidance 1.0, no negatives, no upsampling by default.
- **Prompts**: follow project skill `flux-best-practices` (klein narrative prose, lighting, front-load subject).
- **MLX**: load `mlx-swift` skill; TE porting uses `mlx-swift-lm` patterns.
- **Verify**: measure peak memory before claiming Tier L/M readiness.
