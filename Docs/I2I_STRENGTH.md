# I2I strength curves (color vs structure)

**P8 deliverable.** Klein is distilled **4-step**. I2I does **not** slice a T2I schedule. It always runs all N steps from a start-noise set by `--strength` and `--schedule`.

Code: `StrengthScheduleCurve`, `Flux2Scheduler.strengthSchedule`. CLI: `imarello i2i`, `imarello schedule` (T2I sigmas only).

## Mapping

User `--strength` \(s \in (0,1]\) → unshifted start time `startT` (then the same μ time-shift as T2I):

| Curve | `--schedule` | Formula | When |
|-------|----------------|---------|------|
| **Color** (default) | `color` | \(1-(1-s)^{1.5}\) | Objects, recolor, style. Mid \(s\) still re-noises hard. |
| **Identity** | `identity` | \(1-(1-s)^{0.85}\) | With `--identity`. Milder start noise so face/pose survive high \(s\). |
| **Linear** | `linear` | \(s\) | Debug / A/B only. |

`--identity` selects the identity curve (plus ref latents, face mask, clean-pull). Override with `--schedule`.

`startT` at common strengths (before μ-shift):

| \(s\) | color | identity | linear |
|------:|------:|---------:|-------:|
| 0.35 | 0.476 | 0.307 | 0.350 |
| 0.50 | 0.646 | 0.445 | 0.500 |
| 0.65 | 0.793 | 0.590 | 0.650 |
| 0.75 | 0.875 | 0.692 | 0.750 |
| 0.80 | 0.911 | 0.745 | 0.800 |
| 0.85 | 0.942 | 0.801 | 0.850 |
| 0.90 | 0.968 | 0.859 | 0.900 |
| 1.00 | 1.000 | 1.000 | 1.000 |

Color \(s=0.8\) ≈ identity \(s=0.93\) in start-noise. That is why object recolors at 0.8 work without `--identity`, and why people edits need **higher \(s\) plus** the identity stack.

CLI defaults: strength-only **0.8** + color curve; `--identity` **0.9** + identity curve. Face start-σ = global × `--face-strength-scale` (preset **0.5**).

## Recipes

Match canvas to the reference (omit `--width`/`--height` or set native size).

| Intent | Strength | Flags | Prompt |
|--------|----------|-------|--------|
| Mild grade / lighting | 0.35–0.55 | (none) | Say what **stays** |
| Object / mug **recolor** | **≥ 0.8** | default | Hex + color words; same object |
| Style / weather | 0.8–0.9 | default | What changes vs composition |
| Person, **recolor same cut** | 0.85–0.9 | `--identity` | “exact same cut; only fabric color …” |
| Person, **replace outfit** / scene | 0.85–0.9 | `--identity` | “replace the blouse with a different …” + lock face/pose |
| Face lock too weak | keep \(s\) | `--face-strength-scale 0.35` | — |
| Edit not landing | raise \(s\) | stay on the same curve | Stronger color/object verbs |

Do **not** turn on `--identity` for a mug or still life. Strength-only + color curve is the product default.

```bash
# Object recolor
.build/release/imarello i2i \
  "the same ceramic mug but emerald green glaze #0B5F4B, same morning light" \
  --image src.png --strength 0.8 --output edit.png --analyze --vision-brief

# Same person, new wardrobe + light
.build/release/imarello i2i \
  "Same woman, identical face, eyes, freckles, hair, pose. Replace the ivory blouse with a different emerald silk wrap top. Golden hour balcony." \
  --image portrait.png --strength 0.9 --identity --output edit-id.png --analyze --vision-brief
```

Eval prompts: [`eval-prompts.md`](eval-prompts.md) I2I + identity sections.

## Eval read-out

Pass `--strength` to `analyze-image` (automatic after `i2i --analyze`):

| Finding | Meaning | Action |
|---------|---------|--------|
| `color_mismatch` | Prompt color absent | Raise \(s\), hex, fewer competing colors |
| `strength_too_low_for_edit` | \(s \ge 0.75\), color edit, SSIM still high | Raise \(s\) or stronger prompt |
| `expected_structure_change` | \(s \ge 0.75\), low SSIM | Expected for outfit/scene |
| `high_structure_fidelity` | Edit ≈ reference | Strength too low for the ask |
| `unexpected_identity_drift` | Low \(s\), low SSIM | Prompt or curve mismatch |

High SSIM is **not** a failed identity edit. For a recolor, high SSIM + color match is success. For a new outfit, lower SSIM + face lock is success.

## Implementation notes

- Full N-step Euler from `startT` → \(1/N\), then μ-shift (same as T2I). Not `startStep` slicing.
- Identity extras: clean ref tokens at RoPE \(t=10\); Vision face mask; clean-pull α 0.2 decaying over steps (`IdentityPreserveConfig.identityPreset`).
- `--ref-downsample N` shrinks ref tokens (faster, weaker identity). Experimental.

Unit lock for the table: `Flux2MathTests.strengthCurveStartTMatchesDocs`.
