# CLAUDE.md — Aestrix

On-device image generation & editing for iOS via **MLX-Swift** and **FLUX.2-klein-4B**.
Target: iPhone 15 Pro (8 GB) / iOS 26. The full design + roadmap lives in `docs/PLAN.md`.

## Layout

```
Aestrix/
├── AestrixEngine/      Swift package — the MLX engine (generation + edit API)
│   ├── Sources/AestrixEngine/{Engine,Models,Pipeline,IO,Image,Support}/
│   ├── Tests/AestrixEngineTests/
│   ├── tools/vae_reference.py   # Python (mflux/MLX) parity-reference generator
│   └── Package.swift            # MLX-Swift 0.31.4, swift-transformers 1.1.0 (pinned)
├── AestrixDemo/        Thin iOS app (xcodegen-generated .xcodeproj; iOS 26)
└── docs/PLAN.md        Full architecture + M0–M8 roadmap
```

## Build & test (read carefully — MLX has sharp edges)

```bash
# Compile + run MLX-free unit tests (config, JSON parse, tokenizer, downloader)
swift test --package-path AestrixEngine

# MLX execution REQUIRES xcodebuild (SwiftPM CLI can't build the Metal shaders /
# default.metallib → MLX ops crash under plain `swift test`). macOS host proves execution;
# the iOS Simulator cannot run MLX (needs a physical Apple-Silicon device).
TEST_RUNNER_AESTRIX_RUN_MLX_TESTS=1 \
TEST_RUNNER_AESTRIX_FIXTURES="$(pwd)/AestrixEngine/.build/fixtures/flux2-klein-4b-4bit" \
  xcodebuild test -scheme AestrixEngine -destination 'platform=macOS'
```

- **Heavy tests** (download MB–GB shards / run full MLX models) are gated on `AESTRIX_HEAVY_TESTS` → `TEST_RUNNER_AESTRIX_HEAVY_TESTS=1`.
- `TEST_RUNNER_<NAME>=1` is how xcodebuild injects an env var into the test runner; tests read `<NAME>`.
- iOS demo: `cd AestrixDemo && xcodegen generate`, then build via Xcode/xcodebuild (MLX metal shaders are compiled by the Xcode build).

## MLX-Swift gotchas (all learned the hard way — see memory + `flux-vae-mlx-porting-notes`)

- **Conv2d is channels-last (NHWC)**; weight shape `[out, kh, kw, in]`. mflux saves convs in this layout → load directly, no permute. Operate NHWC throughout.
- **`@ModuleInfo(key: "snake_name")`** + assign via `self._prop.wrappedValue = value`. The key is NOT auto snake_cased from the property label. MLXNN represents `[Module]` as `NestedItem.array([...])` — the flat→nested loader (`NestedDictionary(flatWeights:)`) must convert integer-string keys to `.array` or `update(parameters:)` aborts.
- **`MLX.loadArrays` also needs the metallib** → weight-loading tests run via xcodebuild, not `swift test`.
- dtype type is `MLX.DType` (capital T); `DType`/`dtype` won't resolve (Swift 6 member-import-visibility).
- **MLX-Metal bf16 reductions are not bit-deterministic.** Run parity checks in **float32** (deterministic) to prove a port is structurally correct; bf16 will show amplified drift. The VAE decodes in float32 (`FluxVAE(precision: .float32)`, matching diffusers `force_upcast`).

## Architecture (FLUX.2 Klein pipeline)

```
prompt → Qwen3-4B [layers 9,18,27 → concat (512,7680)] → ctx
noise (128,H/16,W/16) → img tokens (+ ref-image latents @ t=10 for edit)
4-step rectified-flow Euler (guidance=1.0, no CFG) over the Klein transformer
latent → VAE.decode (float32) → image
```

- **Text encoder = Qwen3-4B** (not T5); VAE = `AutoencoderKLFlux2` (32-ch latent; `bn`+pixel-unshuffle → 128-ch transformer latent is pipeline-level, M5).
- **Memory strategy:** staged load/free via MLX wired-memory tickets — free the 2.26 GB text encoder before denoising, free the transformer before decode. Peak ~4 GB wired < 8 GB.

## Conventions

- **Public surface:** `public actor AestrixEngine` (`load / downloadModel / generate / edit / unload`). All MLX state stays behind the actor (MLXArray is not Sendable).
- **Parity harness pattern (M2–M5):** `tools/<comp>_reference.py` (mflux/MLX, `--fp32` for float32) → safetensors of inputs + per-stage outputs; Swift test loads via `loadArrays`, runs, compares `max|Δ| = (a-b).abs().max().item(Float.self)`. Install refs: `.build/venv/bin/pip install mlx mflux` (pure-MLX, no torch).
- **Commit messages** end with `Co-Authored-By: Claude <noreply@anthropic.com>`. Don't commit on the user's behalf unless asked.
- `.gitignore` excludes `.build/`, `DerivedData/`, `*.xcodeproj`, `*.safetensors`, `.impeccable/`.

## Status

M0 (scaffold), M1 (IO: downloader/weights/tokenizer), M2 (VAE, port proven via float32 parity Δ<1e-4) ✅.
Next: **M3 — Qwen3 text encoder** (load 4-bit Qwen3-4B, extract hidden states at layers [9,18,27], concat → (512,7680), parity vs mflux). See `AestrixEngine/README.md` milestone table.
