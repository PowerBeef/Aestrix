# Agent workflow (Codex)

This is the operational playbook for repository work. The durable safety and product contract lives in [`../AGENTS.md`](../AGENTS.md); current phase, blockers, and backlog live in [`ROADMAP.md`](ROADMAP.md).

## Current operating state

- FLUX.2 Klein 4B, prequantized 4-bit only.
- Text-to-image defaults to the bespoke direct text encoder, direct DiT, and small direct VAE decoder.
- Identity/image-to-image still uses the MLX DiT path.
- Default generation is 1024², four steps, guidance 1.0, staged residency.
- The package uses the pinned mlx-swift 0.32.1 fork. Inspect `../mlx-swift-fork` first for fork-specific API/ABI truth when that sibling checkout is present, and confirm its revision against `Package.resolved` before relying on it.
- Reference direct-path timings are 20.6 seconds at 512² and 60.3 seconds at 1024²; identity image-to-image is approximately 36.2 seconds. Memory references are approximately 2 GB RSS and a 1.77 GiB MLX watermark.
- The iOS app cannot currently perform device generation: its embedded thin JIT-era `Cmlx` metallib is incompatible with the no-JIT runtime. `ImarelloSpikes` proves the full `iphoneos` metallib recipe, but integration and device verification remain blocked work.

Treat timings as A/B baselines, not evergreen claims. Read [`PERF.md`](PERF.md), [`MEMORY.md`](MEMORY.md), and [`ROADMAP.md`](ROADMAP.md) before performance work.

## Session startup

1. Read [`../AGENTS.md`](../AGENTS.md), then the active section of [`ROADMAP.md`](ROADMAP.md).
2. Read the specialized authority for the task: architecture, performance, evaluation, weights, iOS, product, or design.
3. Run `git status --short`; preserve unrelated user changes.
4. Select the installed skill route below and read its `SKILL.md` before acting.
5. If a command could initialize MLX, compile Metal, run Xcode, generate, or benchmark, confirm there is no other Metal owner.
6. State the validation scope. Documentation-only work stops at documentation/structural checks.

## Installed skill routes

Use the smallest set that fully covers the task. Skill names below are exact.

| Work | Route |
|---|---|
| MLX tensors, modules, quantization, graph behavior | `$swift-mlx` |
| Qwen/text-model architecture, local inference, MLX LM behavior | `$swift-mlx-lm` |
| Async/await, actors, `Sendable`, data races | `$axiom-concurrency` |
| Profiling, memory growth, retain cycles, CPU/GPU performance | `$axiom-performance` |
| Modern Swift implementation and review | `$axiom-swift` |
| SwiftUI state, navigation, layout, performance | `$axiom-swiftui` |
| Product/HIG decisions and app structure | `$axiom-design` |
| VoiceOver, Dynamic Type, contrast, touch targets | `$axiom-accessibility` |
| Swift Testing/XCTest design and failure triage | `$axiom-testing` |
| Build failures, hangs, dependency and environment diagnosis | `$axiom-build` |
| Apple APIs, compiler diagnostics, Xcode-bundled documentation | `$axiom-apple-docs` |
| Simulator build/run/UI/log debugging | `$build-ios-apps:ios-debugger-agent` plus XcodeBuildMCP |
| iOS 26+ Liquid Glass implementation/review | `$build-ios-apps:swiftui-liquid-glass` |
| SwiftUI navigation, state, layout, controls, screen composition | `$build-ios-apps:swiftui-ui-patterns` |
| Large-view structure and Observation ownership refactors | `$build-ios-apps:swiftui-view-refactor` |
| SwiftUI render/update performance audit | `$build-ios-apps:swiftui-performance-audit` |
| Simulator UI/accessibility, logs, crashes, and trace tooling | `$axiom-tools` (`xcui`, `xclog`, `xcsym`, `xcprof`) |
| GitHub repository, issue, and PR orientation | `$github:github` |
| Unresolved PR review threads and requested changes | `$github:gh-address-comments` |
| Failing GitHub Actions checks | `$github:gh-fix-ci` |
| Commit, push, and draft PR publication | `$github:yeet`, only after an explicit publication request |
| Hugging Face downloads, verification, and repository operations | `$hugging-face:hf-cli` |
| BFL FLUX prompting and product behavior | repo-local `$flux-best-practices` |

The prompt skill is at [`../.agents/skills/flux-best-practices`](../.agents/skills/flux-best-practices). Its scope is BFL prompt construction, model-facing prompt behavior, and product semantics. It is not authority for DiT/VAE mathematics, Swift/MLX implementation, kernel design, weight conversion, or performance.

When multiple Apple skills apply, begin with the narrowest specialist. For UI feature work, a common order is `$axiom-design`, `$axiom-swiftui`, `$axiom-accessibility`, then `$axiom-testing`. Add `$axiom-concurrency` whenever async or isolated state is touched.

## Documentation and capability routing

### MLX and library truth

1. Inspect the pinned sibling `../mlx-swift-fork` first when it exists. Confirm the exact commit from `Package.resolved`.
2. Inspect the checked-out dependency source/build artifacts when necessary.
3. Use Context7 for upstream library documentation and examples.
4. Do not let upstream docs override behavior in the pinned fork.

### Apple truth

Use `$axiom-apple-docs` to search Xcode-bundled Apple documentation and compiler material first. Use the Sosumi MCP server as the fallback for Apple documentation unavailable locally. Do not guess current APIs from memory when availability, signatures, entitlements, or compiler behavior matters.

### Simulator and XcodeBuildMCP

For Simulator work, use `$build-ios-apps:ios-debugger-agent` and XcodeBuildMCP. Before the first build, run, test, or UI action in a task:

1. Call `session_show_defaults`.
2. If missing, configure the project `Apps/ImarelloIOS/ImarelloIOS.xcodeproj`, the intended scheme, and an available Simulator device.
3. Build/run with the configured session, inspect logs/UI, and keep every operation serial.
4. Never initialize the MLX pipeline in Simulator. Simulator verifies UI behavior only.

The current tools may combine boot, build, and launch operations; use their live schemas rather than memorized command names. Do not launch Xcode merely to inspect source or project settings that can be read directly.

Use `$axiom-tools` for the narrow diagnostic artifact: `xcui` for scripted Simulator UI/accessibility validation, `xclog` for console capture, `xcsym` for `.ips`/MetricKit/`.crash` symbolication, and `xcprof` for structured trace analysis. Read the routed reference and follow its procedure inline. These tools remain subject to the one-owner rule whenever they launch, attach to, or profile Apple runtime work.

### GitHub

Prefer the connected GitHub app for structured repository, issue, PR, review, and check context. Use the skill matching the task. Before any `gh` CLI workflow involving Actions, review data, push, or publication, run:

```bash
gh auth status
```

Stop if authentication is invalid. Reading GitHub context does not authorize edits, comments, labels, commits, pushes, or PR creation. `$github:yeet` is permitted only when the user explicitly asks to publish local changes.

### Hugging Face

Use `$hugging-face:hf-cli` and the `hf` command, never the deprecated CLI name. Every model or weight download/verification must use the exact repository and pinned revision from [`WEIGHTS.md`](WEIGHTS.md), [`hub-pins.json`](hub-pins.json), or the package configuration. Check authentication and intended destination before any upload or remote mutation. Downloads and uploads require explicit task scope; this workflow does not authorize them by itself.

### Image validation

Use the local image viewer to open generated PNGs directly. AI image-generation tools, captions, and text-only reports are not visual evaluators. Quality completion always combines pixel metrics with human-visible PNG inspection as defined in [`EVAL_WORKFLOW.md`](EVAL_WORKFLOW.md).

### Optional plugins

Recommended but uninstalled plugins, including Codex Security and Figma, may be proposed for future tasks but are not workflow dependencies. Do not install a plugin or rewrite the workflow around one without user authorization.

## Read-only parallel audits

The primary agent may authorize up to three bounded read-only subagents for independent repository audits or research. Each assignment must state a concrete question and scoped paths. A subagent may search/read and return file-referenced findings only.

Subagents cannot edit, build, test, run scripts, resolve packages, launch Xcode/Simulator/apps, generate or evaluate images, benchmark, profile, initialize MLX/Metal, touch external state, or delegate again. The primary agent owns all edits and all verification, performed serially. See [`../AGENTS.md`](../AGENTS.md) for the blocking policy.

## One-owner preflight

Before MLX, Metal, Xcode, generation, benchmark, or eligible test work:

- inspect relevant active processes;
- confirm no other agent/user task owns Metal;
- run exactly one operation in the foreground;
- wait for cleanup before the next operation;
- never retry a quiet operation concurrently.

The 8 GB host has a history of watchdog failures and kernel instability under overlapping workloads. This constraint applies even to commands that are usually harmless on larger machines.

## Commands

Commands below are recipes, not permission to run them. Select only the command required by the task and keep it isolated.

### Build and runtime preflight

```bash
swift build -c release
./Scripts/ensure-metallib.sh
.build/release/imarello info
```

Runtime/generation work requires the full no-JIT metallib, approximately 155 MB. Never accept the thin JIT-era artifact as equivalent.

### Filtered tests only

```bash
swift test --filter 'HostPreflight|GoldenMetric|Flux2Math|IdentityPreserve|ImarelloBench|HubPin|Metallib|EvalCachePolicy|TextTokenMode|PromptEmbedCacheKey|VAEAttention|DiTOpProfile|DeviceHarness|Qwen'
```

```bash
./Scripts/ci-eval-floors.sh
```

```bash
IMARELLO_MLX_TESTS=1 swift test --filter ImarelloDiTTests
```

Unfiltered `swift test` is prohibited. Run the CI wrapper alone; if it builds and then produces no test output, follow `$axiom-build` environment-first triage: capture the exact command and last output, inspect the test process and DiagnosticReports, distinguish compilation from test execution, and avoid concurrent retries. Do not change code until the failing layer is identified.

### Generate, evaluate, and inspect

Use current CLI help and [`EVAL_WORKFLOW.md`](EVAL_WORKFLOW.md) for exact flags; do not copy historical flags from old reports. Record model revision, prompt, seed, dimensions, steps, guidance, token mode, engines, decoder, and output path.

The required sequence is:

1. Generate one declared case.
2. Run the pixel evaluator for that exact output/case.
3. Open the PNG with the local image viewer.
4. Record metric and visible findings together.

Do not parallelize cases on the 8 GB host.

### Performance

Follow [`PERF.md`](PERF.md). Capture a before case, make the change, capture the identical after case, and compare with the repository benchmark comparison tooling. Retain wall-clock stage timings, peak RSS, MLX active/cache/watermark values, and quality evidence. A single post-change timing is not an A/B result.

## iOS workflow

Read [`IOS.md`](IOS.md), [`../PRODUCT.md`](../PRODUCT.md), and [`../Apps/ImarelloIOS/DESIGN.md`](../Apps/ImarelloIOS/DESIGN.md).

- Preserve the frozen `HarnessService` paths, request/result JSON, output naming, launch arguments, and completion behavior.
- Use Simulator only for UI, layout, navigation, accessibility, and non-MLX state flow.
- Do not claim model or device-runtime success from Simulator.
- Do not enable JIT or substitute the thin metallib to bypass the current device blocker.
- For device-runtime work, first integrate and verify the full `iphoneos` no-JIT metallib using the proven spike recipe; keep this serial and evidence-backed.
- Use `$build-ios-apps:swiftui-liquid-glass` for native iOS 26+ glass decisions, `$build-ios-apps:swiftui-ui-patterns` for screen composition, `$build-ios-apps:swiftui-view-refactor` for large-view/data-flow cleanup, and `$build-ios-apps:swiftui-performance-audit` for code-first render-performance work.
- Preserve product and design authority when applying those patterns. The specialized skills do not override `PRODUCT.md` or `DESIGN.md`.

## Definition of done

| Change | Required evidence |
|---|---|
| Documentation only | `git diff --check`, stale-reference/consistency checks, changed-link validation; no Swift/Metal run |
| Swift logic without MLX | Narrow filtered test(s), run alone |
| MLX/model behavior | Narrow MLX test or declared runtime case, memory evidence, one-owner compliance |
| Output quality | Exact generation record, pixel evaluation, direct PNG inspection |
| Performance/memory | Comparable benchmark A/B, stage timing, RSS and MLX memory metrics, quality check |
| SwiftUI/iOS UI | Simulator UI inspection, accessibility validation, narrow tests where applicable; no MLX in Simulator |
| Device runtime | Full `iphoneos` metallib verification plus physical-device evidence |
| GitHub review fix | Addressed thread/check evidence and local validation; no publication unless requested |

Update the authoritative documentation whenever behavior, baselines, blockers, pins, commands, or schemas actually change. Report unrun validation explicitly.
