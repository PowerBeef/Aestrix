# AGENTS.md — Aestrix

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
│   │   ├── Pipeline/
│   │   │   ├── RectifiedFlowScheduler.swift   rectified-flow Euler + Karras schedule
│   │   │   ├── CoordinateBuilder.swift        txt_ids/img_ids/ref_ids builders
│   │   │   └── LatentTransforms.swift         pack/unpack, patchify/unpatchify, bn denorm
│   │   ├── IO/                        ModelDownloader, WeightLoader, NestedWeights, Tokenizer
│   │   ├── Image/
│   │   │   └── ImageConversion.swift  MLXArray ↔ AestrixImage / normalization helpers
│   │   └── Support/                   Errors, Logging
│   ├── Tests/AestrixEngineTests/      18 tests, 10 suites (config, IO, scheduler, VAE parity, Qwen3, transformer, pipeline, integration, output paths)
│   ├── tools/vae_reference.py         Python mflux/MLX parity-reference generator (--fp32 for float32)
│   ├── Package.swift                  MLX-Swift 0.31.4, swift-transformers 1.1.0
│   └── README.md                      Engine-specific build/parity notes
├── AestrixDemo/                       Thin SwiftUI iOS app (xcodegen-generated .xcodeproj)
└── docs/PLAN.md                       Full architecture + M0–M8 roadmap
```

- `.gitignore` excludes `.build/`, `.swiftpm/`, `DerivedData/`, `xcuserdata/`, `*.xcodeproj`, `Package.resolved`, `*.safetensors`, `AestrixDemo/Models/`, `outputs/`, `.impeccable/`, `.DS_Store`.

## Agent/tooling guide (Kimi Code CLI)

This project is developed with **Kimi Code CLI**. When you work here, prefer Kimi-native tools over generic shell work.

- **Skills first.** Before any non-trivial task, invoke the relevant skill with the `Skill` tool. Likely skills for this codebase:
  - `using-superpowers` — skill discovery and workflow.
  - `writing-plans` — for multi-step planning.
  - `subagent-driven-development` or `executing-plans` — for implementation.
  - `verification-before-completion` — before claiming work is done.
  - `xcodebuildmcp` — for iOS/macOS build, test, simulator, and UI automation.
  - `axiom` — for profiling, log capture, and crash symbolication.
  - `systematic-debugging` / `test-driven-development` as appropriate.
- **Subagents.** Use the `Agent` tool:
  - `subagent_type: "explore"` for read-only codebase investigation.
  - `subagent_type: "plan"` for architecture/implementation planning.
  - `subagent_type: "coder"` for implementation, review, or spec tasks.
- **Build & test.** See the `## Build & test` section below for exact commands. Use XcodeBuildMCP tools (`build_run_sim`, `launch_app_sim`, etc.) for the demo app after configuring session defaults.
- **Profiling & diagnostics.** Use Axiom MCP tools (`axiom_xcprof_record`, `axiom_xclog_launch`, `axiom_xcsym_crash`) for on-device performance, logs, and crash symbolication.
- **Environment-gated tests.** Heavy tests (multi-GB downloads / full MLX models) are opt-in only. Never run them without `AESTRIX_HEAVY_TESTS=1` / `TEST_RUNNER_AESTRIX_HEAVY_TESTS=1`.
- **Don't touch tool caches.** `.impeccable/` is a local tool cache and is already gitignored; do not edit it manually.

## Build & test

```bash
# Compile + run MLX-free unit tests (config, JSON parse, tokenizer, downloader)
swift test --package-path AestrixEngine

# MLX execution REQUIRES xcodebuild because plain swift test cannot build the Metal shaders / default.metallib.
cd AestrixEngine
TEST_RUNNER_AESTRIX_RUN_MLX_TESTS=1 \
TEST_RUNNER_AESTRIX_FIXTURES="$(pwd)/.build/fixtures/flux2-klein-4b-4bit" \
  xcodebuild test -scheme AestrixEngine -destination 'platform=macOS'
```

- **Heavy tests** (download MB–GB shards / run full MLX models) are gated on `AESTRIX_HEAVY_TESTS` → `TEST_RUNNER_AESTRIX_HEAVY_TESTS=1`.
- `TEST_RUNNER_<NAME>=1` is how xcodebuild injects an env var into the test runner; tests read `<NAME>`.
- iOS demo: `cd AestrixDemo && xcodegen generate`, then build via Xcode/xcodebuild.

## MLX-Swift gotchas

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
- **Do not run git mutations unless explicitly asked.** `git commit`, `git push`, `git reset`, `git rebase`, and similar operations require user confirmation each time.
- **Do not commit on the user's behalf.** If the user asks for a commit, confirm the message and scope first.
- **Keep this file current.** If you change project conventions, build commands, or tool usage, update `AGENTS.md` so the next agent sees the ground truth.

## Status

M0 (scaffold), M1 (IO), M2 (VAE — float32 parity Δ<1e-4), M3 (Qwen3 text encoder), M4 (Klein transformer) — done.
**M5 — end-to-end pipeline** is in progress (`generate(prompt:config:)` is wired; the remaining milestone goal is to produce a verified first image). See `docs/PLAN.md`.
M6 (image edit), M7 (optimization), M8 (harden) — pending.
