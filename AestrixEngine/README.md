# AestrixEngine

An optimized MLX inference engine for **FLUX.2-klein-4B** on iOS — text-to-image
generation and single-image instruction editing on-device, targeting iPhone 15 Pro / iOS 26.

## Status

| Milestone | Status |
|---|---|
| M0 — scaffold (build green) | done |
| M1 — IO (downloader, weights, tokenizer) | done |
| M2 — VAE | done — port proven correct (float32 parity Δ<1e-4 vs mflux); float32 decode (diffusers `force_upcast`) |
| M3 — Qwen3 text encoder | done — ported (loads 2.26 GB 4-bit, ctx (1, 512, 7680)); exact mflux parity pending |
| M4 — Klein transformer + RoPE4D | done |
| M5 — text-to-image pipeline (first image) | in progress (`generate(prompt:config:)` wired; verified first image pending) |
| M6 — single-image instruction editing | pending |
| M7 — optimization (custom Metal kernels) | pending |
| M8 — harden | pending |

See `../docs/PLAN.md` for the full plan.

## Build & test

```bash
# Compile the package (host)
swift build

# Fast, MLX-free unit tests (config invariants)
swift test
```

The MLX-free suite currently has **18 tests across 10 suites**.

Generated test outputs (e.g. images from `PipelineTests`) are written to `outputs/` at the package root; override with `AESTRIX_OUTPUTS=<path>`.

### MLX execution tests (require xcodebuild, NOT `swift test`)

MLX's Metal kernels (`default.metallib`) can only be built by Xcode/xcodebuild — the
SwiftPM CLI (`swift test`) cannot build Metal shaders, so any MLX-execution test is
skipped under `swift test`. Run them via xcodebuild:

```bash
cd AestrixEngine
TEST_RUNNER_AESTRIX_RUN_MLX_TESTS=1 \
  xcodebuild test -scheme AestrixEngine -destination 'platform=macOS'
```

(`TEST_RUNNER_<NAME>=1` is how xcodebuild injects an env var into the test runner; the
`mlxExecutes` test is gated on `AESTRIX_RUN_MLX_TESTS`.)

The VAE weight-load test (`loadVAEWeights`) also needs xcodebuild (it calls MLX) and
additionally downloads a 165 MB shard, so it is gated on `AESTRIX_HEAVY_TESTS`:

```bash
TEST_RUNNER_AESTRIX_RUN_MLX_TESTS=1 TEST_RUNNER_AESTRIX_HEAVY_TESTS=1 \
  xcodebuild test -scheme AestrixEngine -destination 'platform=macOS'
```

The remaining tests (downloader, tokenizer parity, index parsing) are MLX-free and run
under plain `swift test`.

### Numeric parity vs Python (mflux)

`tools/vae_reference.py` generates an MLX reference (encode/decode + intermediate stages)
for the VAE, saved as safetensors under the fixtures dir. Generate it, then run the parity
suite (same `TEST_RUNNER_AESTRIX_HEAVY_TESTS=1 … xcodebuild test` command):

```bash
.build/venv/bin/python tools/vae_reference.py   # one-time: install mlx+mflux into .build/venv
```

`FluxVAE(precision: .float32)` (the default) matches the float32 reference to **max|Δ| < 1e-4**
(latent 3.9e-6, decoded 8.7e-5) — proving the port has no structural bug. In bf16 the decoded
image drifts (~0.5) due to inherent MLX-Metal bf16 reduction non-determinism amplified through
the decoder (diffusers `AutoencoderKLFlux2.force_upcast=true` decodes in float32 for this reason).
`float32Parity` is the strict gate; `bf16Amplification` documents the phenomenon.

> ⚠️ **iOS Simulator limitation:** MLX cannot run its GPU ops on the iOS Simulator (it
> requires a physical Apple Silicon device). Host-side `xcodebuild test` on macOS proves
> execution; on-device verification needs a real iPhone 15 Pro.
