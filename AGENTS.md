# Imarello — Codex Development Contract

This file is the durable repository-wide contract for Codex and any other development agent. It applies from the repository root unless a more specific nested `AGENTS.md` adds compatible local guidance.

Imarello is a native Swift + MLX implementation of FLUX.2 Klein 4B for Apple silicon. It includes a Swift package, command-line tools, evaluation and benchmark infrastructure, a bespoke Metal execution path, and an iOS app. The primary development host is an 8 GB M2 Mac mini; host safety is a product requirement, not an optimization preference.

## Authority and documentation routing

Use the narrowest authoritative source for the question:

| Source | Authority |
|---|---|
| `AGENTS.md` | Durable product locks, host-safety rules, evidence requirements, review policy, and development boundaries |
| [`Docs/ROADMAP.md`](Docs/ROADMAP.md) | Current phase, accepted decisions, blockers, and prioritized backlog |
| [`Docs/AGENT_WORKFLOW.md`](Docs/AGENT_WORKFLOW.md) | Commands, installed skill and tool routing, and task-specific definition of done |
| [`PRODUCT.md`](PRODUCT.md) | Product behavior and user-facing intent |
| [`Docs/ARCHITECTURE.md`](Docs/ARCHITECTURE.md) | Module boundaries, execution paths, and architectural invariants |
| [`Docs/PERF.md`](Docs/PERF.md), [`Docs/MEMORY.md`](Docs/MEMORY.md) | Performance baselines, profiling, and memory constraints |
| [`Docs/EVAL_WORKFLOW.md`](Docs/EVAL_WORKFLOW.md), [`Docs/IMAGE_ANALYSIS.md`](Docs/IMAGE_ANALYSIS.md) | Quality gates, metrics, prompt corpus, and image inspection |
| [`Docs/IOS.md`](Docs/IOS.md), [`Apps/ImarelloIOS/DESIGN.md`](Apps/ImarelloIOS/DESIGN.md) | iOS harness/runtime contract and app design authority |
| [`Docs/WEIGHTS.md`](Docs/WEIGHTS.md), [`Docs/TEXT_TOKENS.md`](Docs/TEXT_TOKENS.md), [`Docs/I2I_STRENGTH.md`](Docs/I2I_STRENGTH.md) | Weight, token, and image-to-image semantics |

Do not silently choose between conflicting code and documentation. Confirm actual behavior, update the authoritative document in the same change, and call out any unresolved mismatch. Historical research documents provide context but do not override this contract, the roadmap, or current code.

## Start every task this way

1. Read this file, then the current phase and blockers in [`Docs/ROADMAP.md`](Docs/ROADMAP.md).
2. Read the narrow technical authority for the files in scope.
3. Check `git status --short` and preserve unrelated user changes.
4. Select applicable installed skills and tools from [`Docs/AGENT_WORKFLOW.md`](Docs/AGENT_WORKFLOW.md).
5. Before any MLX, Metal, Xcode build, runtime, test, generation, or benchmark command, establish that no competing Metal owner is active.
6. State the intended validation before editing. Run only the validation permitted by the safety rules below.

Documentation-only changes do not justify Swift, MLX, Metal, generation, benchmark, Xcode, or Simulator validation.

## Non-negotiable product locks

These values are intentional. Changing one requires explicit user authorization, a roadmap decision, and evidence appropriate to the change.

| Area | Locked behavior |
|---|---|
| Model | FLUX.2 Klein 4B |
| Quantization | Prequantized 4-bit weights only |
| Residency | Staged component residency; do not make all major components resident together |
| Default output | 1024 × 1024 |
| Sampling | Four steps, guidance `1.0`, no negative prompt |
| Primary path | Direct-engine text encoder + DiT + VAE for text-to-image |
| Image-to-image | Identity/image-to-image DiT remains on the MLX path until the roadmap changes it |
| Host | Must remain safe on the 8 GB Apple-silicon development host |
| Runtime metallib | Full no-JIT MLX metallib, approximately 155 MB; never substitute the thin JIT-era artifact |

Current measured references are 20.6 seconds at 512² and 60.3 seconds at 1024² for the direct-engine path, with approximately 2 GB RSS and a 1.77 GiB MLX watermark. Identity image-to-image is approximately 36.2 seconds. These are comparison baselines, not promises that may be restated after code changes without fresh measurements.

## One Metal owner

Only one Metal/MLX/Xcode workload may exist at a time on this host.

Never overlap any of the following:

- image generation;
- benchmark or profiling runs;
- any MLX-enabled test;
- `swift build` or other work that compiles Metal;
- `xcodebuild`, XcodeBuildMCP build/test/run, or Xcode compilation;
- metallib generation or validation;
- another agent performing any item above.

Run these operations serially and in the foreground. Do not start a second workload because the first appears quiet. If ownership is uncertain, inspect running processes and stop before proceeding. Do not kill an unknown process without confirming ownership.

The primary agent is the sole Metal owner and the sole agent permitted to edit or run verification. Subagents are read-only under the policy below.

## Build, test, and runtime safety

### Build and metallib

The normal release path is:

```bash
swift build -c release
./Scripts/ensure-metallib.sh
.build/release/imarello info
```

Run each command alone. Runtime work requires the full metallib produced or verified by `Scripts/ensure-metallib.sh`. A successful Swift build does not prove that the runtime metallib is suitable for no-JIT execution.

### Tests

Unfiltered `swift test` is prohibited. Use the narrowest relevant test filter. The broad MLX-free regression gate is still filtered:

```bash
swift test --filter 'HostPreflight|GoldenMetric|Flux2Math|IdentityPreserve|ImarelloBench|HubPin|Metallib|EvalCachePolicy|TextTokenMode|PromptEmbedCacheKey|VAEAttention|DiTOpProfile|DeviceHarness|Qwen'
```

The CI wrapper is:

```bash
./Scripts/ci-eval-floors.sh
```

Run it alone and investigate silence or a stall before retrying. MLX tests are opt-in, isolated, and never concurrent with another Metal owner:

```bash
IMARELLO_MLX_TESTS=1 swift test --filter ImarelloDiTTests
```

Do not broaden a test filter merely to save time selecting tests.

### Generation and evaluation

Generation, pixel evaluation, visual inspection, and benchmarking are separate gates:

1. Generate the exact declared prompt/seed/settings case.
2. Run pixel-based evaluation and record the produced report.
3. Open every relevant PNG with the local image viewer and inspect it directly.
4. Record failures and artifacts; metric success alone is not a quality claim.

Image-generation tools and model-generated descriptions are not evaluators. A quality claim requires both pixel evidence and direct PNG inspection. Preserve the exact prompt, seed, dimensions, steps, guidance, model revision, weight mode, and output paths.

### Performance claims

A performance or memory claim requires:

- a relevant before/after benchmark A/B under the same inputs and environment;
- wall-clock stage timing, not only a total;
- peak RSS and MLX active/cache/watermark measurements;
- quality comparison when numerical behavior or scheduling can affect output;
- a retained benchmark comparison artifact or clearly reproducible command record.

Do not describe an optimization as faster, lower-memory, neutral-quality, or regression-free from code inspection alone.

## Architecture invariants

- Public orchestration belongs in the runtime layer; keep lower-level targets focused on their owned representation and kernels.
- The direct text-to-image path is the default. Do not accidentally route it back through the MLX DiT or full VAE decoder.
- Qwen3 hidden-state taps are 9, 18, and 27 and concatenate to width 7680. The default text-token mode pads to 512; `auto` remains an explicit weaker opt-in mode.
- The DiT topology is 5 double-stream plus 20 single-stream blocks. Preserve the established f16 quantized-matmul scaling convention and validate any kernel-level numerical change.
- The small direct VAE decoder is the default. The full decoder is an explicit escape path; untiled 1024² VAE decode is prohibited.
- Do not wrap the full DiT in `MLX.compile`.
- `EvalCachePolicy.high` is prohibited on the 8 GB host. Do not use D=512 `evalEachChunk` execution.
- Prompt-embedding cache behavior and keys are part of reproducibility. Do not silently alter cache identity or enable cross-configuration reuse.
- Cancellation must remain cooperative and must release staged resources.
- Instrumentation must default off and must not alter numerical results when enabled.
- No cloud inference API may replace or mask the local pipeline.

Read [`Docs/ARCHITECTURE.md`](Docs/ARCHITECTURE.md) and the relevant implementation before modifying any of these areas.

## Swift and MLX implementation rules

- Use Swift 6 concurrency deliberately. Treat MLX arrays and model state as actor-isolated unless the implementation proves a safer boundary.
- Do not move non-`Sendable` MLX state across isolation domains, hide warnings with unchecked conformance, or use detached tasks to bypass ownership.
- Keep the UI responsive by moving suitable work off the main actor, but do not create competing command buffers or concurrent generation.
- Prefer small, testable changes that preserve target boundaries and public behavior.
- Match local naming, access control, formatting, and error-handling conventions.
- Add or update narrow tests for behavior changes, but obey the filtered-test and one-owner rules.
- Consult the pinned sibling `../mlx-swift-fork` first for fork-specific API/ABI truth when it is present. Treat upstream documentation as secondary for fork behavior.

## iOS contract

The Simulator is UI-only. Never load the model, initialize MLX generation, or claim device-runtime validity from Simulator behavior. There is no Catalyst fallback.

The iOS harness contract is frozen unless the user explicitly authorizes a coordinated schema migration. Preserve its paths, JSON request/result schema, output conventions, launch arguments, and completion semantics as documented in [`Docs/IOS.md`](Docs/IOS.md).

Device generation is currently blocked: the main app embeds a thin JIT-era `Cmlx` metallib that is incompatible with the no-JIT runtime. `ImarelloSpikes` demonstrates the full `iphoneos` metallib recipe, but the blocker remains until that full artifact is integrated and verified in the app. Do not report device generation as working, and do not workaround the blocker by enabling runtime JIT.

For UI work, preserve [`PRODUCT.md`](PRODUCT.md) and [`Apps/ImarelloIOS/DESIGN.md`](Apps/ImarelloIOS/DESIGN.md) as product/design authority. Accessibility is part of completion, including VoiceOver labels, Dynamic Type, contrast, and touch targets.

## Installed capability routing

Use the exact installed routes documented in [`Docs/AGENT_WORKFLOW.md`](Docs/AGENT_WORKFLOW.md):

- MLX/model work: `$swift-mlx`, `$swift-mlx-lm`, `$axiom-concurrency`, `$axiom-performance`.
- Swift/iOS work: `$axiom-swift`, `$axiom-swiftui`, `$axiom-design`, `$axiom-accessibility`, `$axiom-testing`, `$axiom-build`, `$axiom-apple-docs`.
- SwiftUI specialization: `$build-ios-apps:swiftui-liquid-glass`, `$build-ios-apps:swiftui-ui-patterns`, `$build-ios-apps:swiftui-view-refactor`, `$build-ios-apps:swiftui-performance-audit`.
- Simulator: `$build-ios-apps:ios-debugger-agent` with XcodeBuildMCP, beginning with `session_show_defaults`.
- Apple diagnostics: `$axiom-tools` for `xclog`, `xcui`, `xcsym`, and `xcprof` routing.
- GitHub: `$github:github`, `$github:gh-address-comments`, `$github:gh-fix-ci`; `$github:yeet` only after an explicit publication request.
- Hugging Face: `$hugging-face:hf-cli` with exact pinned revisions.
- BFL prompt behavior: repository-local `$flux-best-practices` from `.agents/skills/flux-best-practices`.

Recommended but uninstalled plugins are optional future additions. Do not make Codex Security, Figma, or another unavailable plugin a prerequisite for repository work.

## Read-only subagent policy

The primary agent may use at most three bounded subagents for independent audits or research. Each assignment must name a concrete question and scoped paths.

Subagents may only search and read. They must return concise findings with repository file references. They may not:

- edit, create, move, or delete files;
- run builds, tests, scripts, package resolution, generation, evaluation, benchmarks, profiling, or Metal compilation;
- launch Xcode, a Simulator, an app, or a model;
- access or mutate GitHub, Hugging Face, or other external state;
- generate or evaluate images;
- delegate to another agent.

The primary agent owns every mutation, resolves conflicting findings, and performs all verification serially. Read-only delegation never transfers Metal ownership.

## Code Review Rules

Treat the following as blocking review findings unless the change explicitly carries the required authorization and evidence:

1. **Product-lock violation:** different model, quantization, default dimensions, steps, guidance, staged residency, or local-only behavior.
2. **Residency regression:** broader simultaneous residency, retained command buffers/arrays, unsafe cache policy, untiled 1024² decode, or unexplained memory growth.
3. **Unsafe concurrency:** overlapping Metal/MLX/Xcode work, multiple generation owners, actor-isolation bypass, or non-`Sendable` MLX state crossing domains.
4. **Runtime metallib regression:** thin/JIT-era artifact, missing full-metallib verification, or a JIT workaround for iOS.
5. **Harness incompatibility:** iOS request/result schema, paths, launch arguments, output, or completion behavior changed without an authorized coordinated migration.
6. **Missing quality evidence:** an output-quality claim without pixel evaluation and direct inspection of the generated PNGs.
7. **Missing performance evidence:** a speed or memory claim without comparable A/B benchmarks and memory measurements.
8. **Simulator overclaim:** model/runtime conclusions drawn from the UI-only Simulator.
9. **Test-scope violation:** unfiltered tests, concurrent validation, or a supposedly MLX-free test that initializes MLX.
10. **Documentation drift:** behavior, blockers, baselines, commands, or pins changed without updating the authoritative document.

Review for correctness and host safety before style. Give findings a severity, exact file/line location, consequence, and a concrete correction. If no blocking findings remain, state any residual validation gap instead of implying unrun verification.

## Scope discipline

- Do not install plugins, alter global Codex configuration, commit, push, open a pull request, publish artifacts, download models, or mutate remote state unless the user explicitly requests it.
- Do not alter public Swift APIs, CLI flags, model or harness schemas, package dependencies, weight pins, product behavior, or runtime code as incidental cleanup.
- Preserve unrelated worktree changes. Never use destructive Git commands to make the tree look clean.
- Keep the roadmap and specialized documents current when behavior actually changes; do not duplicate a long historical narrative here.
