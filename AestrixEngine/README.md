# AestrixEngine

An optimized MLX inference engine for **FLUX.2-klein-4B** on iOS — text-to-image
generation and single-image instruction editing on-device, targeting iPhone 15 Pro / iOS 26.

## Status

| Milestone | Status |
|---|---|
| M0 — scaffold + build green | ✅ done |
| M1 — IO (downloader, weights, tokenizer) | ✅ done |
| M2 — VAE | ✅ ported (round-trip runs; exact Python parity pending) |
| M3 — Qwen3 text encoder | pending |
| M4 — Klein transformer + RoPE4D | pending |
| M5 — t2i pipeline (first image) | pending |
| M6 — image edit | pending |
| M7 — optimization (custom Metal kernels) | pending |
| M8 — harden | pending |

See `/Users/patricedery/.claude/plans/i-want-to-research-unified-horizon.md` for the full plan.

## Build & test

```bash
# Compile the package (host)
swift build

# Fast, MLX-free unit tests (config invariants)
swift test
```

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

> ⚠️ **iOS Simulator limitation:** MLX cannot run its GPU ops on the iOS Simulator (it
> requires a physical Apple Silicon device). Host-side `xcodebuild test` on macOS proves
> execution; on-device verification needs a real iPhone 15 Pro.
