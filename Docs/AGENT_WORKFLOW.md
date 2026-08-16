# Agent workflow (Claude Code)

How to work on Imarello with Claude Code: which docs to read, which skills, MCP servers, and subagents to use, and the host-safe loops for build / test / generate / eval / bench.

**Product locks and host rules live in [`CLAUDE.md`](../CLAUDE.md)** (loaded automatically every session). This file is the operational map. Backlog and pause state: [`ROADMAP.md`](ROADMAP.md).

**Pause (2026-08-15):** backend / P9 leftover slices (TAEF2 preview, ref-KV, Δ-DiT, `stagedAggressive`, fused qmm+SwiGLU) are **paused**. Next product phase is **P7 iOS**. Do not start speed work unless the user asks.

---

## Session start

1. `CLAUDE.md` is already in context — product goal, locks, host safety.
2. Read [`ROADMAP.md`](ROADMAP.md) — what is done, parked, or paused.
3. Load only the skills for the current task (table below). Do not dump the whole catalog into context.
4. Confirm one Metal owner before any `imarello` generate, bench, or Metal compile (no Xcode/`xcodebuild` build in flight, no second `imarello`).

---

## Skills (by task)

| Task | Load |
|------|------|
| Prompts, CLI examples, I2I UX, eval wording | Project skill **`flux-best-practices`** (`.claude/skills/flux-best-practices/`, auto-discovered). Klein rules: `rules/flux2-models.md`, `t2i-prompting.md`, `i2i-prompting.md`, `negative-prompt-alternatives.md`, `core-principles.md`. **Do not** install flux-3 video skills. |
| MLX arrays, quant, `eval()`, Metal, wired memory | **`mlx-swift`** |
| Qwen3 TE port, tokenizer, chat template | **`mlx-swift-lm`** |
| Swift 6 actors / `MLXArray` is not `Sendable` | **`axiom:axiom-concurrency`** |
| Apple framework APIs, Swift diagnostics | **`axiom:axiom-apple-docs`** (Xcode-bundled for-LLM docs) |
| iOS 26 UI, Liquid Glass, HIG | **`axiom:axiom-swiftui`**, **`axiom:axiom-design`**; UI critique/polish passes: **`impeccable:impeccable`** |
| Unit tests, Swift Testing patterns | **`axiom:axiom-testing`** |
| App-side memory / jank | **`axiom:axiom-performance`** — **speed claims** still go through [`PERF.md`](PERF.md) + `imarello bench` |
| Hub download, pin, inspect | **`huggingface-skills:hf-cli`** (`hf`) |
| Library API truth (mlx-swift, SwiftPM, HF) | **Context7** (`context7-mcp` skill + MCP server) |
| Quality "done" | [`EVAL_WORKFLOW.md`](EVAL_WORKFLOW.md) + Read the PNG (multimodal) |

Do **not** load Vercel, Chrome DevTools / claude-in-chrome, HF Spaces / SageMaker / Gradio, or dataviz skills for this repo.

BFL skills cover **prompting and product behavior**, not DiT / VAE math. MLX skills cover **implementation**.

---

## Subagents

Delegate to specialized agents when the task matches; keep the conclusion, not the file dumps.

| Situation | Agent |
|-----------|-------|
| Xcode / SPM build failure, "No such module", stale-code runs | `axiom:build-fixer` |
| New `imarello-*.ips` / DiagnosticReports crash (host-safety rule) | `axiom:crash-analyzer` |
| Simulator UI drive, screenshots, a11y verification | `axiom:simulator-tester` (never sleep-and-rescreenshot) |
| Swift 6 concurrency audit before landing actor/`Sendable` changes | `axiom:concurrency-auditor` |
| Swift perf anti-pattern scan (ARC, copies, generics) | `axiom:swift-performance-analyzer` |
| Failing/flaky test loop | `axiom:test-debugger` |
| SPM resolution conflicts (mind the exact 0.31.6 mlx-swift pin) | `axiom:spm-conflict-resolver` |

**Hard constraint (8 GB host): at most one Metal-owning process at a time.** Never fan out subagents, workflows, or background jobs that each run `imarello` generate/bench, `swift test` with MLX tests, or a Metal compile. Read-only audit agents (Glob/Grep/Read) may run in parallel; anything that touches the GPU is strictly serial.

---

## MCP servers

| Server | Use for this repo | Do not use for |
|--------|-------------------|----------------|
| **Context7** | Current mlx-swift / Swift / Hugging Face API docs | Inventing Imarello internals |
| **XcodeBuildMCP** | P7 **Simulator** UI (build_run_sim, screenshot, snapshot_ui); call `session_show_defaults` first | Physical device install (device workflow **not enabled** — use `xcodebuild` + `devicectl`). Daily SPM generate / bench — use `swift build -c release`. **Do not** call `ImarelloPipeline` on the Simulator. No Catalyst. |
| **GitHub** (MCP or `gh`) | CI (`eval-floors`), PRs, issues on `PowerBeef/Imarello` | Local Metal work |
| **sosumi** | Apple doc pages when `axiom-apple-docs` is thin | Kernel math |

Ignore for Imarello: `claude-in-chrome`, `chrome-devtools`, `vercel`, `gmail`, `Excalidraw`, `Uber Eats`, HF Spaces/SageMaker tooling.

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
swift test --filter 'HostPreflight|GoldenMetric|Flux2Math|IdentityPreserve|ImarelloBench|HubPin|Metallib|EvalCachePolicy|TextTokenMode|PromptEmbedCacheKey|VAEAttention|DiTOpProfile|DeviceHarness'
```

MLX-gated numerics (Metal owner rules apply): `IMARELLO_MLX_TESTS=1 swift test --filter ImarelloDiTTests`.

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

Then **Read the PNG** (multimodal Read tool) and complete the Phase-B checklist in [`EVAL_WORKFLOW.md`](EVAL_WORKFLOW.md). Pixel metrics alone are not enough. Schema **1.4** includes `unstructured_garbage` (TV-static / f16-overflow speckle).

### Performance (mandatory before "faster / leaner")

```bash
.build/release/imarello bench --width 512 --height 512 --warmup 1 --trials 3 \
  --json /tmp/bench.json

.build/release/imarello bench --mode identity-i2i --image "$REF" \
  --width 512 --height 512 --strength 0.9 --with-quality \
  --json /tmp/id-i2i.json

.build/release/imarello bench-compare /tmp/baseline.json /tmp/candidate.json
```

1024² T2I bench is OK on this host (~74 s, watermark 3.46 GiB). Do **not** start a 4-trial `identity-i2i` at 1024 unless the user asks.

---

## Host safety (8 GB — blocking)

This machine is 8 GB unified (`Mac14,3`). An IDE agent + `swift build` / Metal compile + DiT starved WindowServer and triggered a watchdog panic.

1. **One Metal owner.** No `imarello` generate / bench / compile-spike while Xcode, another `imarello`, or a second IDE is compiling Metal. No parallel Metal-owning subagents.
2. Default to filtered unit tests and **512²** smokes.
3. Never `MLX.compile` the full DiT. Never `dit-compile-spike` without `--force` on an idle machine.
4. Never `EvalCachePolicy.high` or VAE D=512 `evalEachChunk`.
5. After a reboot, hang, or new `imarello-*.ips`: stop, inspect DiagnosticReports (`axiom:crash-analyzer`).
6. Ambient ≠ contaminated: `WindowServer`, Ghostty, `MTLCompilerService`. Another IDE / `swift-package` still mark trials dirty.

`HostPreflight` takes `~/Library/Caches/Imarello/imarello.lock`. Details: [`HOST_SAFETY.md`](HOST_SAFETY.md).

---

## P7 iOS (device)

Full recipe: [`IOS.md`](IOS.md). Short version:

1. Simulator = UI only. `Generate` is a no-op. Do not fake Klein. Verify UI with `axiom:simulator-tester` / XcodeBuildMCP sim tools.
2. After `project.yml` edits: `./Scripts/generate-ios-project.sh`.
3. Device build needs `-skipPackagePluginValidation`, `-allowProvisioningUpdates`, team `FK2D8X36G2`.
4. Install / launch with `xcrun devicectl device install app` / `process launch`. **Resync weights after every install** (new data container).
5. Profile is **`iOS Team Provisioning Profile: app.imarello.demo`** (both kernel entitlements). Do not hand-resign extra keys (`0xe8008015`).
6. Weights stay on disk (`Caches/Imarello/models/`), never in the bundle. `Scripts/sync-ios-device-weights.sh` — copy the real snapshot dirs, not a host symlink.
7. Drive generate from the Mac with `./Scripts/ios-device-harness.sh --eval` (default 512²). Do not ask the user to tap Generate. `--width 1024` needs `--allow-1024`. New job id on every retry.
8. First generate is **512²**. Eval the PNG on the Mac (`EVAL_WORKFLOW.md`). 1024² same seed is a different sample; vision-check anatomy.
9. If Generate says `weightsNotFound` but there is no gate banner, the pipeline was created before the copy — recreate it (`hasLocalSnapshot`).

---

## Definition of done

| Claim | Required |
|-------|----------|
| Generation / quality | PNG + pixel eval + vision checklist (Read the image) + paths/scores in the summary |
| Faster / leaner | `imarello bench` + `bench-compare` on this host; peak RAM not worse in a way that threatens 8 GB |
| Docs | Defaults and benches match [`PERF.md`](PERF.md) product-path table; ROADMAP "Last updated" bumped |

Do not claim "blue mug works" from metrics alone without opening the image.
