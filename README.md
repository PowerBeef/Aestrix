# Aestrix

On-device image generation & editing for iOS, powered by **MLX** and
**FLUX.2-klein-4B**. Target: iPhone 15 Pro, iOS 26.

```
Aestrix/
├── AestrixEngine/   Swift package — the optimized MLX engine (generation + edit API)
└── AestrixDemo/     Thin iOS app exercising the engine (xcodegen-generated project)
```

## What's here

- **`AestrixEngine`** — `public actor AestrixEngine` with `load / generate / edit / unload`,
  plus `verifyIntegration()` (proves MLX executes on-device). Depends on
  [`ml-explore/mlx-swift`](https://github.com/ml-explore/mlx-swift) `0.31.4`.
- **`AestrixDemo`** — SwiftUI app (iOS 26) that imports the engine. Regenerate the Xcode
  project with `cd AestrixDemo && xcodegen generate`.

## Verified (M0)

- Package compiles against MLX-Swift 0.31.4 (`swift build` ✅); MLX-free unit tests pass.
- **MLX executes on Apple Silicon** — a 512×512 matmul ran on the GPU via `xcodebuild test`.
- Demo app builds via xcodebuild (iOS 26.5, MLX C++ + Metal shaders, zero errors) and
  launches cleanly on the iPhone 17 Pro simulator; `default.metallib` is bundled.
- ⚠️ MLX cannot run on the **iOS Simulator** (needs a physical device); on-device
  execution on iPhone 15 Pro is the remaining hardware-gated check.

See `AestrixEngine/README.md` for build/test commands and the full plan at
`~/.claude/plans/i-want-to-research-unified-horizon.md`.
