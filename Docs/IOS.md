# Imarello iOS 26 demo

**Current status (2026-08-21): device generation remains blocked pending physical verification.** The main target now validates and embeds the full `iphoneos` no-JIT MLX pack plus the separate Direct metallib/manifest, and `GenerationEngine` resolves both before constructing the pipeline. The integration compiles in the UI-only Simulator, where artifact embedding and model execution are intentionally skipped; it has not yet been signed, installed, or exercised on a physical device. Keep the blocker open until the installed app’s artifacts, entitlements, first serialized 512² output, pixel report, and direct PNG inspection pass. The 2026-08-16 results—512² T2I in **11.6 s** and 512² I2I at strength 0.8—remain historical pre-no-JIT evidence. Weights are **not** bundled; resync after every `devicectl install`.

The app links `ImarelloRuntime` (same staged Klein 4B path as the Mac CLI). Agent contract: [`../AGENTS.md`](../AGENTS.md) § iOS contract.

## Simulator vs device

| Surface | What works |
|---------|------------|
| **iOS Simulator** | UI preview only — Create shows an honest "Simulator preview" state. **MLX does not run.** Create/Edit actions never initialize the model. Debug UI-test scenarios are static interface fixtures, not generated output. |
| **Physical iPhone** | Build integration exists, but generation remains unverified until a signed install proves both artifact sets and the serialized 512² bring-up gate. |
| **Mac Catalyst** | Not supported. Do not add a Mac destination. |

XcodeBuildMCP on this host is **Simulator-only** (device workflow not enabled). Use `xcodebuild` + `devicectl` for the phone.

Regenerate the Xcode project after `project.yml` edits:

```bash
./Scripts/generate-ios-project.sh
```

Open `Apps/ImarelloIOS/ImarelloIOS.xcodeproj`. Bundle ID: `app.imarello.demo`. Deployment: iOS 26. Team: `FK2D8X36G2`.

## Device build / install

mlx-swift ships a CUDA package plugin. Xcode refuses it unless you skip plugin validation. Automatic signing must be allowed to create/update the development profile and register the phone.

```bash
# Hardware UDID (0000…), not the CoreDevice UUID (6AE2…)
xcodebuild -project Apps/ImarelloIOS/ImarelloIOS.xcodeproj -scheme ImarelloIOS -showdestinations

xcodebuild -project Apps/ImarelloIOS/ImarelloIOS.xcodeproj -scheme ImarelloIOS \
  -configuration Debug \
  -destination 'platform=iOS,id=<hardware-udid>' \
  -allowProvisioningUpdates \
  -allowProvisioningDeviceRegistration \
  DEVELOPMENT_TEAM=FK2D8X36G2 \
  CODE_SIGN_STYLE=Automatic \
  -skipPackagePluginValidation \
  build

# CoreDevice id or the device name both work:
xcrun devicectl list devices
xcrun devicectl device install app --device "<name-or-id>" \
  /path/to/Debug-iphoneos/Imarello.app
xcrun devicectl device process launch --device "<name-or-id>" app.imarello.demo
```

If SpringBoard will not open the icon: **Settings → General → VPN & Device Management** → trust **Apple Development**.

`xcodebuild` for `iphoneos` compiles mlx-swift Metal. That is a Metal owner — do not run `imarello` generate/bench at the same time ([`HOST_SAFETY.md`](HOST_SAFETY.md)).

## Memory entitlements

`ImarelloIOS.entitlements` (public Booleans — no extra Developer Portal grant):

- `com.apple.developer.kernel.increased-memory-limit` — higher jetsam limit on supported iPhones; still run if extra RAM is refused
- `com.apple.developer.kernel.extended-virtual-addressing` — larger VA for mapping 4-bit weights / DiT

The Simulator ignores both. On device, pin **`SystemCapabilities`** in `project.yml` so XcodeGen keeps the keys, and sign with **`iOS Team Provisioning Profile: app.imarello.demo`**. The wildcard `iOS Team Provisioning Profile: *` **drops** those keys. Hand-`codesign` of extra keys without a matching profile fails install (`0xe8008015`).

Save-to-Photos uses **add-only** photo access. There is no photo picker and no file import.

## Weights (not in the bundle)

Do not put the ~5 GB pack in the app bundle.

On first launch the app looks in its sandbox `Caches/Imarello/models/` for:

- `mlx-community--FLUX.2-Klein-4B-4bit` @ `1cebb9b45c21ece14a42615b16bf5fa4de9b56da`
- `black-forest-labs--FLUX.2-small-decoder` @ `a3efc24f613ef42d9428af62fdbd6f5fd8856c4a`

Download on a Mac (same as the CLI), then copy into the app container after the first install so the container exists. Helper: `Scripts/sync-ios-device-weights.sh`. Pins: [`WEIGHTS.md`](WEIGHTS.md).

Missing Klein 4-bit on the Mac cache blocks the copy. Small Decoder alone is not enough. The sync script **resolves symlinked snapshot dirs itself** (e.g. `Imarello/models` links into `~/Library/Caches/Aestrix/models/…` — devicectl cannot open a symlinked source) and auto-detects any **paired** physical iPhone (an idle CoreDevice tunnel reads `disconnected`; that is fine). Both fixed 2026-08-16; `DEVICE=<coredevice-id>` still overrides.

A new `devicectl device install app` often creates a **new data container**. The previous `models/` tree is gone. Run the sync script again before Generate.

`ImarelloPipeline.snapshot` is set in `init`. If the app created a pipeline **before** the copy, Generate can look ready on the stage and still throw `weightsNotFound` at the Klein path. `GenerationEngine.ensureReady()` rebuilds the pipeline when `hasLocalSnapshot` is false.

## First generate (blocked pending signed-device verification)

Do not submit a device matrix yet. First generate both `iphoneos` artifact sets, build/sign/install the main app, inspect that the full MLX pack plus Direct metallib/manifest are present, confirm both kernel entitlements, and make the readiness gate pass before pipeline construction. Then copy both snapshots, run one serialized 512² bring-up case, export/evaluate/inspect its PNG, and only then attempt the repository-locked 1024² default. Same seed does not mean a 512 image scaled up; anatomy still requires direct visual review.

Edit runs from **any Gallery image** at strength 0.8. Selecting Edit in detail switches to the dedicated Edit workspace; a successful result becomes the next source. No `--identity` in this demo. Create and Edit keep independent prompt/seed drafts, while retry captures the exact immutable request that originally ran.

## Metallib

Xcode’s automatic `mlx-swift_Cmlx.bundle/default.metallib` (~3–5 MB) is still thin and never accepted as a substitute. The mlx-swift 0.32.1 fork requires the complete roughly 155 MB no-JIT pack. The main target’s `embed-ios-metallib.sh` phase now validates/copies that full pack plus `imarello-direct.metallib` and `imarello-direct-manifest.json`; `GenerationEngine` verifies both descriptors before pipeline construction. The phase is `iphoneos`-only and explicitly skips Simulator builds. `Scripts/ensure-metallib.sh` remains macOS-only; `Scripts/ensure-direct-metallib.sh --sdk iphoneos` builds the Direct artifact, while the full iOS MLX recipe remains sourced from `ImarelloSpikes`. Physical bundle selection/execution is still the P7 blocker.

## Device harness (generate + Mac eval)

Agents cannot rely on UI automation for device generation. Drive one Klein job by dropping JSON into the app container; the app runs the same pipeline as the Create/Edit actions, then the Mac pulls the PNG and optionally pixel-evals it. **Not** XCUITest. Simulator jobs are written `skipped` (no fake Klein).

```bash
# Historical bring-up example only; current device submission remains blocked. Does not rebuild the app.
./Scripts/ios-device-harness.sh --eval --fail-on-pixel-gate

# 1024 is opt-in (minutes on device; one Metal owner)
./Scripts/ios-device-harness.sh --width 1024 --allow-1024 --eval
```

The frozen wire schema still carries `--steps`, `--text-tokens`, and `--strength`. The script and app now validate IDs, dimensions, steps, seed, strength, timeout, stale reuse, and possible Metal owners before mutating app state or launching work. Completion is durable: a running recovery marker is retained until the matching done result is atomically persisted, and startup recovery treats an already-committed done result as authoritative.

Device auto-detection accepts any paired physical iPhone (same fix as the sync script); `DEVICE=` overrides.

The app polls `Library/Caches/Imarello/jobs/inbox/` every 2s while active. Results land in `jobs/done/{id}.json`; harness-pull PNGs stay under `Caches/Imarello/outputs/`, while canonical history lives in Application Support. Copied artifacts: `/tmp/imarello-ios-eval/{id}/`.

Copy the job to **`…/jobs/inbox/{id}.json`**. Copying onto `…/jobs/inbox` when that name is missing (or is already a file) **replaces the inbox directory with the JSON** and the job never runs. The default job id is timestamped. Explicit reuse is rejected when a running marker or done result already exists; hidden/path-like IDs are invalid.

Pixel eval is not a vision pass — open the PNG (`EVAL_WORKFLOW.md`). Do not run a generate matrix on the phone in v1.

## Four-tab UI (reengineered 2026-08-19)

The iOS 26 shell is native and typed:

| Surface | What it owns |
|------|--------------|
| **Create tab** | Atmospheric uncropped current image, resolution-only picker, seed options, independent prompt, Create action |
| **Edit tab** | Newest-first Gallery source picker followed by a source-locked edit workspace with its own prompt/seed draft |
| **Gallery tab** | Newest-first persistent 2-point image grid with asynchronous thumbnails |
| **Settings tab** | Fail-closed readiness, Gallery count/size, Photos status, app/build/runtime information |
| **Image detail** | Pushed immersive paging/zoom plus local Edit, Share, Save, Delete feedback |
| **Tab accessory** | App-global run owner, phase/step, elapsed time, completion, failure, Cancel/Retry/Dismiss |
| **Generation Options** | Transactional seed-only draft; Create resolution stays separate and Edit resolution stays source-locked |

Each tab owns an independent `NavigationStack`. The persistent system-owned Liquid Glass tab bar uses Create, Edit, Gallery, and Settings and never minimizes on scroll. Create and Gallery push the same image detail; detail hides the tab bar but keeps the system back gesture. Editing selects the source and switches to the Edit root. The active composer withdraws while global activity is present, and tapping activity returns to the originating Create or Edit workspace.

Layers:

| File | Role |
|------|------|
| `AppNavigation.swift` | `AppTab`, `AppRoute`, `AppSheet`, independent tab paths |
| `GenerationActivity.swift` / `StudioModel.swift` | Typed owner/immutable request/activity plus independent Create/Edit drafts |
| `GenerationEngine.swift` | `GenerationServing` boundary plus the fail-closed device runtime |
| `HarnessMonitor.swift` / `HarnessService.swift` | One scene-aware poll task plus frozen durable harness execution |
| `PhotoExportService.swift` | Serialized Photos authorization/save lane; feedback remains detail-local |
| `PrintImageLoader.swift` | Actor-isolated thumbnail/full-image decode and bounded cache |
| `PrintStore.swift` | Unchanged versioned index contract and transactional canonical PNG persistence |
| `StudioRootView.swift` / `StudioPage.swift` / `EditPage.swift` / `SettingsPage.swift` | Four native tabs and their Create, Edit, Gallery, Settings destinations |

Debug-only deterministic interface states:

```text
--ui-test-scenario empty|library|pending-edit|running|failed
```

They use real repository PNGs as development resources. They do not construct `ImarelloPipeline`, initialize MLX, or imitate generation.

Verification starts with XcodeBuildMCP `session_show_defaults`, then runs filtered unit/UI targets serially. Use Simulator only for navigation, UI, accessibility, keyboard, and persistence. Do not make device-runtime claims from it.
