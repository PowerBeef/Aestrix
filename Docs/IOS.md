# Imarello iOS 26 demo

**Status (2026-08-16):** `Apps/ImarelloIOS` runs staged Klein 4B on a physical iPhone (Debug, Apple Development). Simulator remains UI-only. **512² T2I smoke passed** (fox, seed 42; pixel PASS + vision pass). Device jobs: `Scripts/ios-device-harness.sh`. **1024² T2I** runs but can fail vision anatomy (Klein 4-step / μ=1.15 / 4096 tokens — not a seed-commit bug). Weights are **not** in the bundle; **resync after every `devicectl install`**. Profile `app.imarello.demo` signs both kernel entitlements. Metallib resolution walks the `.app` for `mlx-swift_Cmlx.bundle`.

The app links `ImarelloRuntime` (same staged Klein 4B path as the Mac CLI). Agent map: [`../CLAUDE.md`](../CLAUDE.md) § P7 iOS.

## Simulator vs device

| Surface | What works |
|---------|------------|
| **iOS Simulator** | UI preview only (no gate banner). **MLX does not run.** Generate is a chrome no-op. |
| **Physical iPhone** | App installs and launches. T2I + last-in-app I2I once weights + entitlements are right. |
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

Download on a Mac (same as the CLI), then copy into the app container after the first install so the container exists. Helper notes: `Scripts/sync-ios-device-weights.sh`. Pins: [`WEIGHTS.md`](WEIGHTS.md).

Missing Klein 4-bit on the Mac cache blocks the copy. Small Decoder alone is not enough. If the Mac path is a **symlink** into `~/Library/Caches/Aestrix/models/…`, copy the **real** directories (the phone cannot follow that symlink).

A new `devicectl device install app` often creates a **new data container**. The previous `models/` tree is gone. Run the sync script again before Generate.

`ImarelloPipeline.snapshot` is set in `init`. If the app created a pipeline **before** the copy, Generate can show **no banner** and still throw `weightsNotFound` at the Klein path. `GenerationModel.ensureReady()` now rebuilds the pipeline when `hasLocalSnapshot` is false.

## First generate (after weights + entitlements)

1. Confirm the signed app actually contains both kernel entitlements (`codesign -d --entitlements - Imarello.app`).
2. Copy both snapshots into `Caches/Imarello/models/`.
3. **512² T2I first.** One Metal owner.
4. Export the PNG and run [`EVAL_WORKFLOW.md`](EVAL_WORKFLOW.md) on the Mac.
5. 1024² only after 512 succeeds. Same seed ≠ the 512 image scaled up (different noise shape + scheduler μ). Vision-check anatomy — pixel can PASS a headless chimera.

Edit uses the last in-app PNG at strength 0.8. No `--identity` in this demo. Seed field: number-pad **Done**, commit on Generate; caption `{side} · seed {n}` is what actually ran.

## Metallib

Xcode compiles mlx-swift Cmlx Metal into `Imarello.app/mlx-swift_Cmlx.bundle/default.metallib` (~3–4 MB, Steel + RMSNorm). That is a **resource** bundle (`BNDL`), not a loaded framework — it does not appear in `Bundle.allBundles`. Asking every loaded bundle for `default.metallib` either hits a UIKit/SwiftUI stub (~157 KB) or finds nothing. The resolver walks the `.app` wrapper on disk (same path Cmlx `try_load_bundle` uses). `Scripts/ensure-metallib.sh` is macOS-only (`-sdk macosx`) and must not be copied onto the phone.

4-bit `affine_qmm` is JIT in mlx-swift 0.31.6 and is not required in the file.

## Device harness (generate + Mac eval)

Agents cannot tap **Generate**. Drive one Klein job by dropping JSON into the app container; the app runs the same pipeline as the dock button, then the Mac pulls the PNG and optionally pixel-evals it. **Not** XCUITest. Simulator jobs are written `skipped` (no fake Klein).

```bash
# Default: fox prompt, 512², seed 42. Does not rebuild the app.
./Scripts/ios-device-harness.sh --eval --fail-on-pixel-gate

# 1024 is opt-in (minutes on device; one Metal owner)
./Scripts/ios-device-harness.sh --width 1024 --allow-1024 --eval
```

The app polls `Library/Caches/Imarello/jobs/inbox/` every 2s while active. Results land in `jobs/done/{id}.json`; PNGs stay under `Caches/Imarello/outputs/`. Copied artifacts: `/tmp/imarello-ios-eval/{id}/`.

Copy the job to **`…/jobs/inbox/{id}.json`**. Copying onto `…/jobs/inbox` when that name is missing (or is already a file) **replaces the inbox directory with the JSON** and the job never runs. Use a **new `--id`** on retry so a leftover `done/{id}.json` is not treated as the new result.

Pixel eval is not a vision pass — open the PNG (`EVAL_WORKFLOW.md`). Do not run a generate matrix on the phone in v1.

## Simulator UI

Impeccable `layout` + axiom Liquid Glass (Regular, not Clear). Verify with `xcui` + AXe (`describe-ui`, tap **by label**, `axe screenshot`). Do not fake Klein generate on the Simulator.
