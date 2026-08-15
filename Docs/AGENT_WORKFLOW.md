# Agent workflow

How to work on Imarello: which docs to read, which skills and MCP servers to load, and the host-safe loops for build / test / generate / eval / bench.

**Product locks and host rules live in [`AGENTS.md`](../AGENTS.md).** This file is the operational map. Backlog and pause state: [`ROADMAP.md`](ROADMAP.md).

**Pause (2026-08-15):** backend / P9 leftover slices (TAEF2 preview, ref-KV, Δ-DiT, `stagedAggressive`, fused qmm+SwiGLU) are **paused**. Next product phase is **P7 iOS**. Do not start speed work unless the user asks.

---

## Session start

1. Read [`AGENTS.md`](../AGENTS.md) — product goal, locks, host safety.
2. Read [`ROADMAP.md`](ROADMAP.md) — what is done, parked, or paused.
3. Load **this file** plus only the skills for the current task (table below).
4. Confirm one Metal owner before any `imarello` generate, bench, or Metal compile.

Do not dump the whole skill catalog into context.

---

## Skills (by task)

| Task | Load |
|------|------|
| Prompts, CLI examples, I2I UX, eval wording | Project **`flux-best-practices`** (`.grok/skills/flux-best-practices/`). Klein rules: `flux2-models.md`, `t2i-prompting.md`, `i2i-prompting.md`, `negative-prompt-alternatives.md`, `core-principles.md`. **Do not** install flux-3 video skills. |
| MLX arrays, quant, `eval()`, Metal, wired memory | **`mlx-swift`** |
| Qwen3 TE port, tokenizer, chat template | **`mlx-swift-lm`** |
| Swift 6 actors / `MLXArray` is not `Sendable` | **`axiom-concurrency`** |
| Build / SPM / Xcode / simulator failures | **`axiom-build`**, **`axiom-xcode-mcp`** |
| Unit tests, Swift Testing vs XCTest | **`axiom-testing`** |
| App-side memory / jank | **`axiom-performance`** — **speed claims** still go through [`PERF.md`](PERF.md) + `imarello bench` |
| Hub download, pin, inspect | **`hf-cli`** (`hf`) |
| Library API truth (mlx-swift, SwiftPM, Hugging Face) | **Context7** skill + MCP |
| Apple framework APIs (P7 later) | **`axiom-apple-docs`**, **`axiom-macos`**, **`axiom-swiftui`** |
| Quality “done” | [`EVAL_WORKFLOW.md`](EVAL_WORKFLOW.md) + open the PNG with vision |

Do **not** load Vercel, Chrome DevTools, game-asset, or Hugging Face Spaces / SageMaker skills for this repo.

BFL skills cover **prompting and product behavior**, not DiT / VAE math. MLX skills cover **implementation**.

---

## MCP servers

| Server | Use for this repo | Do not use for |
|--------|-------------------|----------------|
| **Context7** | Current mlx-swift / Swift / Hugging Face API docs | Inventing Imarello internals |
| **XcodeBuildMCP** | P7 iOS app target, simulator, Xcode project | Daily SPM CLI generate / bench — use `swift build -c release` |
| **GitHub** | CI (`eval-floors`), PRs, issues | Local Metal work |
| **sosumi** | Apple doc pages when axiom-apple-docs is thin | Kernel math |

Ignore for Imarello: `chrome-devtools`, `browser-use`, `vercel`, `gmail`, `google_drive`, `tasks`, `voice`.

Do not spawn Metal-owning subagents in parallel on the 8 GB host.

---

## Product path (defaults)

| Knob | Default | Escape hatch |
|------|---------|--------------|
| Weights | **4-bit** Klein 4B only | None for users. No 3-bit product path. No user bf16. |
| Text tokens | `--text-tokens auto` | `--text-tokens 512` (byte-stable gallery) |
| VAE decode | BFL **Small Decoder** | `--vae-variant full` (klein AE decoder) |
| I2I encode | Always klein `encodeOnly` | Do not load `full_encoder_small_decoder.safetensors` |
| 4-bit Linear | Scaled f16 `quantizedMM` (`÷16`) | `--attn-linear-compute f32` |
| Canvas | **1024²** | `--width` / `--height` (512 for smokes) |
| Steps / guidance | 4 / 1.0 | Distilled path; no negatives |

Pins and download: [`WEIGHTS.md`](WEIGHTS.md). Numbers: [`PERF.md`](PERF.md) (8 GB M2, 2026-08-15: 512² **19.4 s** / 1024² **74.0 s**).

---

## Build / test / generate / eval / bench

Always **release** for generation and benches. `swift build` alone often leaves a **stub** metallib (~3 KB). Forward kernels need the **full** library (~130 MB).

```bash
swift build -c release && ./Scripts/ensure-metallib.sh
.build/release/imarello info
```

### Filtered tests (no weights)

Do not assume unfiltered `swift test` is safe (Metal FA tests have hung after GPU aborts).

```bash
swift test --filter 'HostPreflight|GoldenMetric|Flux2Math|IdentityPreserve|ImarelloBench|HubPin|Metallib|EvalCachePolicy|TextTokenMode|PromptEmbedCacheKey|VAEAttention|DiTOpProfile'
```

### Generate + quality (mandatory after any PNG used to judge quality)

```bash
# T2I — 512 smoke; omit --width/--height for product 1024²
.build/release/imarello t2i "$PROMPT" --width 512 --height 512 --steps 4 --seed 42 \
  --output /tmp/out.png --analyze --vision-brief

# 512² eval-prompts × seeds 42/0/7 (pixel gate)
IMARELLO=.build/release/imarello ./Scripts/eval-regression.sh

# I2I color/style
.build/release/imarello i2i "$PROMPT" --image "$REF" --strength 0.8 \
  --output /tmp/edit.png --analyze --vision-brief

# I2I identity (people)
.build/release/imarello i2i "$PROMPT" --image "$REF" --strength 0.9 --identity \
  --output /tmp/edit-id.png --analyze --vision-brief
```

Then **open the PNG** with vision. Pixel metrics alone are not enough. Schema **1.4** includes `unstructured_garbage` (TV-static / f16-overflow speckle). Procedure: [`EVAL_WORKFLOW.md`](EVAL_WORKFLOW.md).

### Performance (mandatory before “faster / leaner”)

```bash
.build/release/imarello bench --width 512 --height 512 --warmup 1 --trials 3 \
  --json /tmp/bench.json

.build/release/imarello bench --mode identity-i2i --image "$REF" \
  --width 512 --height 512 --strength 0.9 --with-quality \
  --json /tmp/id-i2i.json
```

1024² T2I bench is OK on this host (~74 s, watermark 3.46 GiB). Do **not** start a 4-trial `identity-i2i` at 1024 unless the user asks.

---

## Host safety (8 GB — blocking)

This machine is 8 GB unified (`Mac14,3`). Cursor + `swift build` / Metal compile + DiT starved WindowServer and triggered a watchdog panic.

1. **One Metal owner.** Do not run `imarello` generate / bench / compile-spike while Xcode, another `imarello`, or a second IDE is compiling Metal.
2. Default to filtered unit tests and **512²** smokes.
3. Never `MLX.compile` the full DiT. Never `dit-compile-spike` without `--force` on an idle machine.
4. Never `EvalCachePolicy.high` or VAE D=512 `evalEachChunk`.
5. After a reboot, hang, or new `imarello-*.ips`, stop and inspect DiagnosticReports.
6. Ambient ≠ contaminated: `WindowServer`, Ghostty, `grok`, `MTLCompilerService`. Cursor / `swift-package` still mark trials dirty.

`HostPreflight` takes `~/Library/Caches/Imarello/imarello.lock`. Details: [`HOST_SAFETY.md`](HOST_SAFETY.md).

---

## Definition of done

| Claim | Required |
|-------|----------|
| Generation / quality | PNG + pixel eval + vision checklist + paths/scores in the summary |
| Faster / leaner | `imarello bench` + `bench-compare` on this host; peak RAM not worse in a way that threatens 8 GB |
| Docs | Defaults and benches match [`PERF.md`](PERF.md) product-path table; ROADMAP “Last updated” bumped |

Do not claim “blue mug works” from metrics alone without opening the image.
