# Host safety (8 GB Apple Silicon)

**Last updated:** 2026-08-15

## What crashed

On this **Mac mini M2 / 8 GB** (`Mac14,3`), DiagnosticReports show two classes:

| Class | Evidence | Symptom |
|-------|----------|---------|
| **A. System reboot** | `panic-full-2026-08-13-025033` | `userspace watchdog timeout: no successful checkins from WindowServer … 127 seconds`. Top CPU: `Cursor Helper (Renderer)`. Snapshot: many Cursor helpers, `MTLCompilerService`, `swift-package`. Compressor + **24 swapfiles**. Then power-button `btn_rst`. |
| **B. Process abort** | `~/Library/Logs/DiagnosticReports/imarello-*.ips` (Aug 10–11) | `SIGABRT` in `mlx::core::gpu::check_error` on Metal completion; stack in **VAE decode**. `AGXMetalG14G`. |

Class A is **not** “any shell command panics the kernel.” It is **Cursor (Electron GPU) + SwiftPM/Metal compile + 8 GB unified memory** starving the compositor. Class B can contribute if `imarello` is also on the GPU.

## What we did

- Uncommitted Cursor optimization tree was isolated on **`cursor-opt-quarantine`** and later **deleted** after the safe leftovers were ported. Audit: [`CURSOR_QUARANTINE.md`](CURSOR_QUARANTINE.md).
- `HostPreflight`: exclusive lock + sibling `imarello` detect. Swap is **not** a run gate.
- `dit-compile-spike` refused on `DeviceTier.low` unless `--force`.

## Operator checklist before 1024² or bench

```bash
pgrep -x imarello || true     # none (second instance is refused)
# no second IDE compiling this package
.build/release/imarello t2i "…" --width 512 --height 512
```

## Agent defaults

See **CLAUDE.md → Host safety** and [`AGENT_WORKFLOW.md`](AGENT_WORKFLOW.md). Prefer **filtered** `swift test` (unfiltered Metal FA tests have hung after GPU aborts). No parallel Metal agents.

1024² T2I is measured-safe on this host (product path ~**71 s**, watermark **3.00 GiB**, post-Tier-2). Default smokes stay **512²**. Do not start a 4-trial `identity-i2i` at 1024 unless asked. **Never decode 1024² untiled** — measured Metal abort (`imarello-2026-08-16-104942.ips`); the VAE tile threshold (128) is load-bearing there.

**Ambient vs competing:** `WindowServer`, Ghostty, `grok`, and `MTLCompilerService` are expected on this session and no longer mark a trial `CONTAMINATED`. Cursor / `swift-package` still do. Swap is recorded only.

## I2I / identity bench (safe slice)

`imarello bench --mode i2i|identity-i2i --image PATH` records host contention per trial (`CONTAMINATED` if a non-ambient process is ≥15% CPU). Prefer `--width 512 --height 512` first on 8 GB.

`--with-quality` on identity mode adds Vision **face-crop SSIM** (not a biometric ID score).
