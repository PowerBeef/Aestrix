# Imarello iOS 26 demo

**Status (2026-08-15):** `Apps/ImarelloIOS` is installed and launches on a physical iPhone (Debug, Apple Development). Simulator remains UI-only. **On-device generate is not done yet** — pinned Klein 4-bit weights are not in the app container, and the wildcard team profile strips the memory entitlements from the signed binary.

The app links `ImarelloRuntime` (same staged Klein 4B path as the Mac CLI). Agent map: [`../AGENTS.md`](../AGENTS.md) § P7.

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

The Simulator ignores both. On device, automatic signing currently picks the wildcard `iOS Team Provisioning Profile: *` and **drops those keys** from the merged `.xcent`. Hand-`codesign` of extra keys without a matching profile fails install (`0xe8008015`). Fix before a real 512² generate: explicit App ID `app.imarello.demo` whose development profile includes both kernel entitlements.

Save-to-Photos uses **add-only** photo access. There is no photo picker and no file import.

## Weights (not in the bundle)

Do not put the ~5 GB pack in the app bundle.

On first launch the app looks in its sandbox `Caches/Imarello/models/` for:

- `mlx-community--FLUX.2-Klein-4B-4bit` @ `1cebb9b45c21ece14a42615b16bf5fa4de9b56da`
- `black-forest-labs--FLUX.2-small-decoder` @ `a3efc24f613ef42d9428af62fdbd6f5fd8856c4a`

Download on a Mac (same as the CLI), then copy into the app container after the first install so the container exists. Helper notes: `Scripts/sync-ios-device-weights.sh`. Pins: [`WEIGHTS.md`](WEIGHTS.md).

Missing Klein 4-bit on the Mac cache blocks the copy. Small Decoder alone is not enough.

## First generate (after weights + entitlements)

1. Confirm the signed app actually contains both kernel entitlements (`codesign -d --entitlements - Imarello.app`).
2. Copy both snapshots into `Caches/Imarello/models/`.
3. **512² T2I first.** One Metal owner.
4. Export the PNG and run [`EVAL_WORKFLOW.md`](EVAL_WORKFLOW.md) on the Mac.
5. 1024² only after 512 succeeds.

Edit uses the last in-app PNG at strength 0.8. No `--identity` in this demo.

## Metallib

Xcode + mlx-swift resources provide the Metal library (not `Scripts/ensure-metallib.sh`, which is the SPM CLI). The app refuses generate if the bundled metallib is a stub or missing Steel symbols (`MetallibVerification.resolveFromBundles()`).

## Simulator UI

Impeccable `layout` + axiom Liquid Glass (Regular, not Clear). Verify with `xcui` + AXe (`describe-ui`, tap **by label**, `axe screenshot`). Do not fake Klein generate on the Simulator.
