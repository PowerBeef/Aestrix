# Imarello — Claude Code Instructions

Imarello is a **from-scratch** native **Swift + MLX** runtime for **Black Forest Labs FLUX.2 [klein] 4B** (Apache-2.0) on Apple Silicon. It is **not** a fork of existing Swift ports; public MLX ports (mflux, diffusers) are oracles only. The repo directory is still `Aestrix` (pre-rename); the product, targets, binary (`imarello`), caches (`~/Library/Caches/Imarello`, legacy `Aestrix` still read), and remote (`PowerBeef/Imarello`) all say **Imarello**.

## Product goal

Make **plain** `t2i` / `i2i` **faster** without **noticeable quality loss**, at **minimal RAM** (8 GB unified is the machine of record). That is the bar for defaults, not a research side path.

| Ship as the **default** when | Leave **opt-in** when |
|------------------------------|------------------------|
| Faster e2e and/or denoise/step | Quality fails pixel or vision vs the current default |
| Peak e2e RAM (DiT watermark / RSS) does not threaten 8 GB | Unmeasured, or only a decode-only RSS blip that does not move the e2e peak |
| `Docs/EVAL_WORKFLOW.md` (pixel **and** vision) shows no noticeable degradation | — |

An extra Hub file or a longer first-run `hf download` is **not** a reason to hide a proven win behind a flag — document the pin and refuse with the download hint when missing. `--vae-variant full` is the old reference decode path. `--text-tokens auto` is an **opt-in** speed path — it was default for one day (2026-08-15) and reverted after vision showed real conditioning degradation (`Docs/TEXT_TOKENS.md`). Same-seed PNG drift is acceptable only when vision says the subject and edit still land. Do not trade staged residency or the 4-bit lock for speed.

## Product locks

| Rule | Detail |
|------|--------|
| Model | **Klein 4B only** — do not ship Klein 9B (Non-Commercial) or FLUX.2 Dev (32B) |
| Weights | **4-bit only** (pre-quantized), enforced: `--weights` rejects non-4bit and the 3-bit pin was deleted (2026-08-18). No user-facing bf16 download or runtime quantize-from-bf16 |
| Memory | **Staged pipeline is the default**: never co-reside text encoder + DiT + VAE |
| Speed / quality | Defaults are the fastest path that passes quality + RAM (see Product goal) |
| Platforms | macOS library + CLI first; **floors are macOS 26.2 / iOS 26.2** (raised 2026-08-18 — Metal 4 single target + NAX host paths everywhere; every Apple Silicon Mac runs macOS 26, so no chip is dropped; NAX stays runtime-gated to A19/M5) |
| v1 features | Text-to-image + single-image I2I (strength + optional **identity** stack) |
| Guidance | Distilled defaults: **4 steps**, **guidance = 1.0**, **no negative prompts**, no prompt-upsampling by default |
| Default canvas | **1024²** (4-bit staged). Lower sizes via `--width` / `--height` |

## Host safety (8 GB Mac mini — blocking)

This machine is **8 GB unified** (`Mac14,3`). Agents + `swift build`/`swift test` + `MTLCompilerService` + a resident DiT (~2 GiB) starved **WindowServer** for 127 s and triggered a **watchdog kernel panic** (2026-08-13). `imarello` Metal command-buffer aborts (VAE decode) are process crashes that can worsen compositor stalls.

1. **One Metal owner.** Do not run `imarello` generate/bench/compile-spike while Xcode, `xcodebuild` (including an iphoneos mlx-swift compile), another `imarello`, or a second IDE is compiling Metal. **Never fan out Metal-owning subagents, workflows, or background Bash jobs in parallel** — generation, benches, and Metal compiles are strictly serial on this host. This includes `swift test`: a filtered run left in the background while Simulator builds ran **hung at 0% CPU for 25 min** (2026-08-16); it passed cleanly on a quiet machine. Run the suite alone.
2. **Default to filtered unit tests and 512² smokes.** 1024² T2I bench is OK when asked (~67 s, watermark 3.00 GiB). Do **not** start a 4-trial `identity-i2i` at 1024 (joint ~8704) unless the user explicitly wants it. **Never decode 1024² untiled** — measured Metal abort on this host.
3. **Never** `MLX.compile` the full DiT; **never** `dit-compile-spike` on this host without `--force` on an idle machine.
4. **Never** reintroduce VAE D=512 `evalEachChunk` or `EvalCachePolicy.high` (deliberately absent from this tree). `EvalCachePolicy.mid` is a ≥16 GB bench flag only; this host stays on `product`.
5. After any reboot, hang, or new `imarello-*.ips`, **stop and inspect** `~/Library/Logs/DiagnosticReports` before retrying — use the `axiom:crash-analyzer` agent on the `.ips`.
6. **Ambient ≠ contaminated:** `WindowServer`, Ghostty, `MTLCompilerService` do not mark bench trials dirty. Another IDE agent or `swift-package` still do.

CLI: `HostPreflight` takes `~/Library/Caches/Imarello/imarello.lock` and refuses a second instance. Swap is not a run gate. Details: `Docs/HOST_SAFETY.md`.

**Tests — never run unfiltered `swift test`** (Metal FA tests have hung after GPU aborts):

```bash
swift test --filter 'HostPreflight|GoldenMetric|Flux2Math|IdentityPreserve|ImarelloBench|HubPin|Metallib|EvalCachePolicy|TextTokenMode|PromptEmbedCacheKey|VAEAttention|DiTOpProfile|DeviceHarness|Qwen'
```

MLX-gated numeric tests are opt-in: `IMARELLO_MLX_TESTS=1 swift test --filter ImarelloDiTTests` (Metal owner rules apply).

## Build & Metal (SPM CLI)

```bash
swift build -c release && ./Scripts/ensure-metallib.sh   # always release for generate/bench
.build/release/imarello info                             # tier, pin, snapshot, Steel metallib check
```

- `swift build` alone leaves a **stub** `default.metallib` (~3 KB); since the nojit switch (2026-08-18) MLX loads kernels **by name from the library**, so a stub is a hard runtime failure, not a slow path — the **full** ~155 MB library is mandatory. Run `Scripts/ensure-metallib.sh` after every build/resolve/clean checkout (it is gitignored). The script refuses partial kernel sets, stamps the mlx-swift revision in `Tools/Metal/mlx.metallib.rev`, and auto-rebuilds when the pin moves.
- `ensure-metallib.sh` is `-sdk macosx` only — **never run it for the iOS app** (Xcode embeds its own `mlx-swift_Cmlx.bundle/default.metallib`).
- Smoke without a full generate: `imarello load-te` / `load-dit` / `load-vae` / `mem-selftest`.
- Weights: pre-quantized MLX packs only, pinned in `Docs/hub-pins.json` + `WeightPreset.pin` (kept in lockstep by `HubPinTests`). Download recipes: `Docs/WEIGHTS.md`.

## Generation eval gate (mandatory)

After **every** `t2i` / `i2i` used to judge quality: run pixel eval **and** a vision pass. For Claude the vision pass means **Read the PNG with the multimodal Read tool** and complete the Phase-B checklist in `Docs/EVAL_WORKFLOW.md` (subject, real color, text, hands/anatomy, artifacts, edit applied). Pixel metrics alone are never enough — do not claim "blue mug works" without looking at the image. The pixel harness has a hard `unstructured_garbage` fail (added after a broken f16 qmm path produced pure noise that metrics missed).

```bash
.build/release/imarello t2i "$PROMPT" --width 512 --height 512 --steps 4 --seed 42 \
  --output /tmp/out.png --analyze --vision-brief          # 512 smoke; omit sizes for product 1024²
IMARELLO=.build/release/imarello ./Scripts/eval-regression.sh   # 512² prompts × seeds 42/0/7, pixel gate
```

Definition of done for any generation claim: PNG written · pixel eval run · vision checklist completed · fails fixed or waived with reason · paths + scores in the summary.

**Perf claims:** no "faster / leaner" without `imarello bench` + `bench-compare` deltas on this host, per `Docs/PERF.md` (the A/B authority). Do not claim Tier L/M or 1024² readiness without measured peak RSS / MLX watermark.

## Architecture reminders

1. **Serial residency**: TE encode → unload → DiT denoise → unload → VAE decode → unload; `Memory.clearCache()` after unload and between large-canvas stages. `Docs/MEMORY.md`.
2. **Qwen3 TE**: chat template required; hidden-state taps at layers **9/18/27** concat → **7680**. TE encodes a 512 pad; DiT default is **`--text-tokens 512`** (pad participates in joint attention — the distillation regime). `--text-tokens auto` (trim) is opt-in: faster but weaker conditioning; reverted as default 2026-08-16. `Docs/TEXT_TOKENS.md`.
3. **DiT**: MMDiT **5 double + 20 single** blocks; 4-axis RoPE θ=2000; inner dim 3072. Block checkpointing; **MLX Steel fused FA** (D=128); **f16 Q/K/V decided from the joint sequence** (all 25 blocks; 2026-08-16). **4-bit Linear GEMMs run in f16 with ÷16/×16 scaling** (scales pre-cast f16 at load, `context_embedder` excluded) — raw f16 is noise, and the pre-divide also protects the input cast (never fold ÷16 into the weights); `--attn-linear-compute f32` restores the reference GEMM. Long single-stream sequences are **chunk-streamed** (proj/mlpHidden/to_out never materialize at full length). Prompt context and all timestep conditioning (temb, modulations, AdaLN-out) are computed **once per generate**, not per step.
4. **Scheduler**: match mflux/diffusers (time-shift / sigma); training-scale timesteps **[0, 1000]** passed from the pipeline (no host `item()` sync).
5. **VAE**: decode uses **decode-only** weights; default is BFL **Small Decoder** (`--vae-variant full` = klein AE decoder). Large canvases use tiled decode. **I2I encode is always the klein AE** — never load `full_encoder_small_decoder.safetensors`.
6. **Canonical weights**: `mlx-community/FLUX.2-Klein-4B-4bit` @ `1cebb9b45c21ece14a42615b16bf5fa4de9b56da` (module-split TE/DiT/VAE).
7. **Text RoPE ids** (FLUX.2): `[t,h,w,l] = [0,0,0,token_i]`. Latents are packed `[B, H/16·W/16, 128]`; decode = BN denorm + unpatchify.
8. **I2I strength**: full N-step schedule, color curve default, color/object edits **≥ 0.8**. Identity (Tier B): `--identity` = ref latents (`t=10`) + face mask + clean-pull + `identity` curve; people **0.85–0.9**. `Docs/I2I_STRENGTH.md`.
9. **Prompt-embed disk cache is default on** (`~/Library/Caches/Imarello/embeds`); a hit skips the TE stage byte-identically; `--no-embed-cache` opts out. Writes are atomic (temp+rename), loads validate shape/dtype/size and delete corrupt entries, and the key includes the Qwen tap layers (2026-08-18).
10. **Cancellation is cooperative**: `generate`/`edit` check `Task.checkCancellation()` at every stage boundary and denoise step, unwinding the resident stage before rethrowing. The iOS Stop button relies on this ("Stopping" state until the run unwinds).
10. **Perf reference** (8 GB M2, 2026-08-18 post-0.32.1-bump, product path = pad-512 + f16 qmm + joint-f16 attention + Small Decoder): 512² e2e **21.9 s** / 1024² **67.5 s**; encode_te **1.58 s** (−19–21% from the bump's split-K/gemv_wide small-M kernels); peak active ~2.05 GiB; watermark 2.57 / **3.00 GiB**; 768² decode untiled (**never** untile 1024² — measured Metal abort on 8 GB). Opt-in `--text-tokens auto`: ~19 s @512² (1024² is faster on the default). Opt-in `t2i --two-stage`: 1024² anatomy rescue (compose@512² + low-σ refine) — fixes body-plan failures but softens texture; not the default. `Docs/PERF.md`.

Correctness footguns and layer diagram: `Docs/ARCHITECTURE.md`.

## Phase status & process rule

Authoritative backlog / pause state: `Docs/ROADMAP.md`.

- **Done**: P0–P6c (macOS library + CLI: T2I, I2I, identity I2I, eval workflow), P8 (polish, CI floors), P9 (perf harness; Small Decoder + f16 scaled qmm are product defaults; `--text-tokens auto` was reverted to opt-in 2026-08-16 after a vision regression), the **2026-08-16 engine pass** (`Docs/ENGINE_RESEARCH.md` Tiers 0–2: quality fixes, reference-faithful tokenizer, byte-identical optimizations, joint-f16 attention, untiled 768² decode — 512² 23.0 s / 1024² 71.0 s / watermark 3.00 GiB), the **2026-08-18 audit + hardening pass** (`9ad69cd`: all 22 Critical/High/Medium findings fixed — durable print history, harness quarantine/sweep/param-threading, pipeline cancellation, atomic embed cache, CLI validation, 3-bit retirement, metallib gating, CI widened to 18 suites; remaining Low items listed in the audit artifact), and the **2026-08-18 engine-uplift session** (`8b9f51a`+`1a2cbfd`: mlx core 0.32.1 via fork — encode_te −19–21%, 512² 21.9 s; TE-splice as `--pad-content clean` (conditioning-only; TE is dispatch-bound — speed claim refuted); refine-regime scheduler; `t2i --two-stage` opt-in anatomy rescue; ChipCapabilities probe; iOS floor 26.2). The **2026-08-18 uplift continuation** (`1b5c397`) closed the batch: S4 (full-f16 single-stream epilogue + dynamic amax scale) **measured and closed** (+1.9–2.8% e2e, only win −150 MiB watermark @512², quality clean — `--attn-f16-full-epilogue` / `--attn-dynamic-scale` stay opt-in experiments); Lanczos two-stage upscale **refuted** (band-limited like bicubic — the re-gate needs an SR stage); fork thread-local-stream fix (86/86 tests); upstream filings mlx#4350 + mlx-swift#457. **That state is tagged `checkpoint-2026-08-18-engine-uplift`** — the validated restore point; work after it is experimental research until it re-passes the gates (ROADMAP Checkpoint section). **Next engine levers**: N2/N3 A19 Pro NAX build + device A/B (iPhone must be on iOS ≥ 26.2); two-stage texture re-gate needs an **SR stage** (Lanczos measured 2026-08-18: within noise of bicubic — interpolation is not the bottleneck; `--refine-upscale lanczos` kept as an option).
- **P9 leftover slices are paused** (TAEF2 preview, ref-KV, Δ-DiT, `stagedAggressive`, fused qmm+SwiGLU). **Do not resume speed work unless asked.**
- **Current phase: P7 iOS** (partial): the studio was **rebuilt from the ground up 2026-08-16** (two-page spread; services split out of `GenerationModel`; persistent print history; edit from any print). 512² T2I **and** I2I both pass pixel + vision on a physical iPhone on that build — the in-app I2I acceptance item is closed. Only open item: **1024² vision-clean anatomy**.
- **Blocking process rule**: always address issues that surface (metallib, load failures, parity fails, failed eval gates, unexplained OOM) **before** starting the next phase.

## Tooling

**Canonical map: `Docs/AGENT_WORKFLOW.md`** — session start, skills by task, MCP servers, subagents, host-safe loops. Load skills per task; do not dump the catalog.

- BFL prompting/product behavior: project skill **`flux-best-practices`** at `.claude/skills/flux-best-practices/` (vendored from black-forest-labs/skills, pinned in `.upstream-sha`). Do not install flux-3 video skills.
- MLX implementation: `mlx-swift` / `mlx-swift-lm` skills. Library API truth: Context7. Apple APIs: `axiom-apple-docs` / sosumi.
- BFL skills cover prompting, not DiT/VAE math; MLX skills cover implementation.

## CLI map

| Command | Role |
|---------|------|
| `imarello info` | Tier, policy, snapshot, pinned Hub revision, metallib Steel check |
| `imarello t2i` / `i2i … --analyze --vision-brief` | Generate/edit + eval kickoff |
| `imarello analyze-image` | Pixel eval / vision brief for an existing PNG |
| `imarello bench` / `bench-compare A B` | Timings, memory, pressure (modes: `t2i` \| `i2i` \| `identity-i2i` \| `pressure-map` \| `dit-one-step` \| `res-ladder` \| `mem-stages` \| `te-only` \| `dit-steps` \| `vae-decode` \| `load-only`); percent deltas |
| `imarello load-te` / `load-dit` / `load-vae` | Staged load smoke; `load-dit --dump-dtypes` = compute-dtype probe |
| `imarello encode-prompt` / `schedule` / `mem-selftest` | TE-only shape check · sigma print · dry residency (no weights) |
| `imarello session` | Warm multi-prompt loop (≥16 GB gate, `--force-resident`) |
| `imarello dit-compile-spike` | Research only; NO-GO; refused on 8 GB without `--force` |
| `Scripts/ensure-metallib.sh` | Full MLX metallib (SPM CLI only — never the iOS app) |
| `Scripts/eval-generation.sh` / `eval-regression.sh` | Eval a PNG · 512² regression + pixel gate |
| `Scripts/ci-eval-floors.sh` | CI gate: Hub pins + golden floors + 18 MLX-free suites (no weights) |
| `Scripts/generate-ios-project.sh` | XcodeGen → `Apps/ImarelloIOS/ImarelloIOS.xcodeproj` |
| `Scripts/sync-ios-device-weights.sh` / `ios-device-harness.sh` | Copy snapshots to the iPhone · headless device generate + eval |

## P7 iOS (blocking facts)

Full recipe: `Docs/IOS.md`. Product locks unchanged.

- **Simulator is UI-only** (MLX does not run; Develop is a chrome no-op — never fake Klein there). **Mac Catalyst is forbidden.** Physical iPhone runs T2I + I2I (both pass pixel + vision on device, 2026-08-16).
- **App shape** (rebuilt 2026-08-16; hardened 2026-08-18): two full-bleed pages — Stage and Contact Sheet — over four services: `GenerationEngine` (pipeline), `HarnessService` (**frozen** harness contract — do not change paths or the job/result schema), `PrintStore` (durable history: versioned `prints-index.json` + canonical PNGs in **Application Support** — the Caches `outputs/` copy exists only for the harness pull; a corrupt index is renamed aside, never wiped), `StudioModel` (UI state). Views are `StudioRootView` / `StudioPage` / `ContactSheetPage` / `PrintViewer` / `StatusRow` / `PromptBar` / `PlateSheet`. Design system: `Apps/ImarelloIOS/DESIGN.md`; product record: `PRODUCT.md`. Edit runs from **any** print in history at the locked strength 0.8. One status row carries every state (mirrored on the Contact Sheet) — do not add a second instrument voice.
- Device build needs `-skipPackagePluginValidation` (mlx-swift ships a CUDA package plugin), `-allowProvisioningUpdates`, team `FK2D8X36G2`. Install/launch with `xcrun devicectl`. XcodeBuildMCP's device workflow is **not enabled** on this host — Simulator tools only.
- Entitlements `increased-memory-limit` + `extended-virtual-addressing` are pinned via `project.yml` `SystemCapabilities`; after `project.yml` edits run `./Scripts/generate-ios-project.sh`. The wildcard profile silently strips them; hand-`codesign` extra keys fails install with `0xe8008015`.
- Weights are never bundled. **Resync after every install** (`Scripts/sync-ios-device-weights.sh` — resolves symlinked host snapshots and detects any paired iPhone itself) — a new `devicectl install` usually creates a new data container. If Generate throws `weightsNotFound` while the stage looks ready, the pipeline predates the copy — recreate it (`GenerationEngine.ensureReady`).
- Drive device generates from the Mac with `./Scripts/ios-device-harness.sh --eval` (default 512² fox seed 42; `--mode i2i` edits the last print; `--width 1024` needs `--allow-1024`). Since 2026-08-18 the harness honors `--steps` / `--text-tokens` / `--strength` on device, timestamps its default job id, and rejects stale `done/` results by `startedAt` — device A/B data over those knobs from before that date ran the defaults and is invalid. Undecodable jobs are quarantined with a `failed` result; orphaned `running/` jobs are swept at app launch. **Never ask the user to tap Develop.**
- 1024² same-seed is a **new noise tensor** (μ 1.15 vs 0.63); Klein 4-step can assemble a broken body plan while pixel PASSes — **vision anatomy is the gate**.

## Coding conventions

- Swift 6, `actor` isolation for all MLX state (`MLXArray` is not `Sendable`).
- `mlx-swift` is pinned to the **PowerBeef/mlx-swift fork** (branch `imarello/core-0.32.1`, revision-pinned; core v0.32.1, **nojit** Metal kernels — the full metallib is mandatory at runtime, a stub is fatal not slow; the companion mlx-c fork carries a compile-cache compat patch **and returns process-wide thread-unsafe default streams** — core ≥0.31.2 thread-local CommandEncoders otherwise abort cross-thread evals against mlx-swift's global `Stream.gpu` static). Tests that eval MLX need the full metallib cloned into the xctest bundle (`Tests/ImarelloCoreTests/TestMetallib.swift` does this; a fresh checkout still needs `ensure-metallib.sh` first). Re-sync to upstream when ml-explore ships a 0.32.x mlx-swift release (upstream refs: mlx#4350, mlx-swift#457, #446, PR#450); any pin move = full bump-day validation (V-bench both canvases + 15/15 V-eval + vision).
- Prefer `MLXFast.scaledDotProductAttention`.
- Public API lives in `ImarelloRuntime`; modules stay independently loadable.
- Instrumentation (`PipelineTrace` / `ProbeDensity`) must not change numerics when density is `.off`.
- Do not add the BFL cloud API as a runtime dependency.
- No lint/format config exists; match the surrounding code.

## Out of scope (v1)

Klein 9B / FLUX.2 Dev · multi-reference edit (>1 image) + KV-cache · LoRA training · base CFG 28-step path · bf16 product path · 3-bit SKU (4-bit is locked; 6/8-bit pins exist but are not product paths).
