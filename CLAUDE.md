# CLAUDE.md — Aestrix

On-device FLUX.2-klein-4B image generation & editing for iOS via **MLX-Swift**.
Target: iPhone 15 Pro (8 GB) / iOS 26. Full design + roadmap in `docs/PLAN.md`.

## Layout

```
Aestrix/
├── AestrixEngine/                     Swift package
│   ├── Sources/AestrixEngine/
│   │   ├── AestrixEngine.swift        public actor (load/generate/edit/unload), GenConfig, AestrixImage
│   │   ├── Models/
│   │   │   ├── FluxVAE.swift          AutoencoderKLFlux2 (encoder/decoder, 4-bit attn, float32 decode)
│   │   │   ├── Qwen3TextEncoder.swift  36-layer GQA, 4-bit, extracts layers [9,18,27] → ctx (512,7680)
│   │   │   └── KleinTransformer.swift  5 double-stream + 20 single-stream blocks, 4D RoPE, 4-bit
│   │   ├── IO/                        ModelDownloader, WeightLoader, NestedWeights, Tokenizer
│   │   └── Support/                   Errors, Logging
│   ├── Tests/AestrixEngineTests/      11 tests, 7 suites (config, IO, VAE parity, Qwen3, transformer)
│   ├── tools/vae_reference.py         Python mflux/MLX parity-reference generator (--fp32 for float32)
│   └── Package.swift                  MLX-Swift 0.31.4, swift-transformers 1.1.0
├── AestrixDemo/                       Thin SwiftUI iOS app (xcodegen-generated .xcodeproj)
└── docs/PLAN.md                       Full architecture + M0–M8 roadmap
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
- iOS demo: `cd AestrixDemo && xcodegen generate`, then build via Xcode/xcodebuild.

## MLX-Swift gotchas (all learned the hard way)

**Weight loading:**
- **Conv2d is channels-last (NHWC)**; weight shape `[out, kh, kw, in]`. mflux saves convs in this layout → load directly, no permute.
- **`@ModuleInfo(key: "snake_name")`** + assign via `self._prop.wrappedValue = value`. The key is NOT auto snake_cased. MLXNN represents `[Module]` as `NestedItem.array([...])` — the flat→nested loader (`NestedDictionary(flatWeights:)`) must convert integer-string keys to `.array` or `update(parameters:)` aborts with `incompatibleItems`.
- **embed_tokens is 4-bit** → use `QuantizedEmbedding`, not `Embedding`. Check the safetensors dtype (U32 = quantized).
- **LayerNorm with `affine: false`** has no weights in the checkpoint — don't expect `norm1.weight` keys.
- dtype type is `MLX.DType` (capital T); `DType`/`dtype` won't resolve (Swift 6 member-import-visibility).

**API naming:**
- `MLX.loadArrays` also needs the metallib → weight-loading tests run via xcodebuild, not `swift test`.
- `arange` uses `step:` not `stride:`. `MLX.arange(0, n, step: 2)`.
- `repeated(_:, count:, axis:)` = `mx.repeat()` (interleave, not tile). `broadcast(_, to:)` is a free function, not a method.
- `.ellipsis` indexing on a 3D tensor slices the **last** axis. To slice the middle axis, transpose first, slice, transpose back.
- `scaledDotProductAttention(queries:keys:values:scale:mask:)` is a free function in the MLX module (mask: `MLXArray?`, pass nil for no mask).
- `silu(_:)`, `softmax(_:axis:)` are free functions in MLXNN.

**Float literals:**
- `MLXArray(1000.0)` / `MLXArray([1.0])` create **float64** (Swift Double default). Metal doesn't support float64 → crash. Use `MLXArray(Float(1000.0))` / `MLXArray([Float(1.0)])`.

**Numerical:**
- **MLX-Metal bf16 reductions are not bit-deterministic.** Run parity checks in **float32** (deterministic) to prove a port is structurally correct. The VAE decodes in float32 (`FluxVAE(precision: .float32)`, matching diffusers `force_upcast`).
- `WeightLoader.dequantized(_:groupSize:bits:)` folds 4-bit weight+scales+biases into a full-precision weight for plain `Linear` modules (used for the VAE attention, which is tiny).

## Architecture (FLUX.2 Klein pipeline)

```
prompt → Qwen3-4B [layers 9,18,27 → concat (512,7680)] → ctx
noise (128,H/16,W/16) → img tokens (+ ref-image latents @ t=10 for edit)
4-step rectified-flow Euler (guidance=1.0, no CFG) over the Klein transformer
latent → VAE.decode (float32) → image
```

- **Text encoder = Qwen3-4B** (not T5); VAE = `AutoencoderKLFlux2` (32-ch latent; `bn`+pixel-unshuffle → 128-ch transformer latent is pipeline-level, M5).
- **Memory strategy:** staged load/free via MLX wired-memory tickets — free the 2.26 GB text encoder before denoising, free the transformer before decode. Peak ~4 GB wired < 8 GB.

## Component-porting workflow (M2–M5)

1. Read mflux Python source (`mflux/models/flux2/...`) for the exact architecture.
2. Dump the component's safetensors shapes (Python: parse header — no MLX needed).
3. Port to MLX-Swift (`QuantizedLinear` for 4-bit projs, `LayerNorm(affine:false)`, `RMSNorm`).
4. Write a structural test (load weights, forward pass, assert shape + finite) — gated on `AESTRIX_HEAVY_TESTS`.
5. Write a parity harness (`tools/<comp>_reference.py` using mflux) → compare in float32.

## Conventions

- **Public surface:** `public actor AestrixEngine` (`load / downloadModel / generate / edit / unload`). All MLX state stays behind the actor (MLXArray is not Sendable).
- **Commit messages** end with `Co-Authored-By: Claude <noreply@anthropic.com>`. Don't commit on the user's behalf unless asked.
- `.gitignore` excludes `.build/`, `DerivedData/`, `*.xcodeproj`, `*.safetensors`, `.impeccable/`.

## Status

M0 (scaffold), M1 (IO), M2 (VAE — float32 parity Δ<1e-4), M3 (Qwen3 text encoder), M4 (Klein transformer) ✅.
Next: **M5 — end-to-end pipeline** (wire Qwen3 → transformer 4-step denoise → VAE decode → first image). See `docs/PLAN.md`.
