#!/bin/bash
# Human-anatomy reference set (2026-08-18): the fox smoke hides exactly the
# failures Klein 4-step actually has (limb count, proportions, navel placement,
# hands). These subjects expose them. Vision checklist per image (see
# Docs/EVAL_WORKFLOW.md "Anatomy addendum"): limb/finger count, joint bend
# direction, navel on the midline above the hips, arm span ≈ height,
# leg-to-torso ratio, face symmetry.
#
#   IMARELLO=.build/release/imarello ./Scripts/eval-anatomy.sh [--canvas 512|1024]
# Extra t2i flags: T2I_EXTRA='--vae-engine direct'  (same contract as eval-regression.sh).
set -euo pipefail

IMARELLO="${IMARELLO:-.build/release/imarello}"
CANVAS=512
if [ "${1:-}" = "--canvas" ]; then CANVAS="${2:-512}"; fi
OUT="${OUT:-$HOME/Library/Caches/Imarello/outputs/anatomy-$CANVAS}"
mkdir -p "$OUT"

PROMPTS=(
  "A woman in a turquoise bikini standing at the water's edge on a sunny beach, full body visible from head to toe, natural relaxed pose, photorealistic travel photography"
  "A shirtless male swimmer standing on a pool deck after a race, arms at his sides, full body visible, photorealistic sports photography"
  "A ballet dancer in mid-leap across a sunlit rehearsal studio, full body visible from head to pointed toes, arms extended, photorealistic, soft window light reflecting off the wooden floor"
  "Studio portrait of an elderly fisherman with weathered hands clasped under his chin, warm side lighting, photorealistic"
  "A yoga instructor holding warrior pose on a mat in a bright studio, full body side view, photorealistic"
  "Two friends walking along a city street at dusk, full bodies visible, candid photorealistic street photography"
)
NAMES=(bikini-beach swimmer dancer portrait-hands yoga street-pair)

FAILED=0
for i in "${!PROMPTS[@]}"; do
  name="${NAMES[$i]}"
  for seed in 42 0; do
    png="$OUT/${name}-s${seed}.png"
    echo "== ${name} seed ${seed} (${CANVAS}²) =="
    if ! "$IMARELLO" t2i "${PROMPTS[$i]}" \
        --width "$CANVAS" --height "$CANVAS" --steps 4 --seed "$seed" \
        --output "$png" --analyze --fail-on-pixel-gate ${T2I_EXTRA:-}; then
      echo "   -> PIXEL FAIL ${name}-s${seed}"
      FAILED=$((FAILED + 1))
    fi
  done
done

echo ""
echo "outputs: $OUT"
echo "pixel hard-fails: $FAILED"
echo "REMINDER: pixel metrics cannot judge anatomy — vision-review every image"
echo "(limb/finger count, joint direction, navel midline placement, proportions)."
exit "$([ "$FAILED" -eq 0 ] && echo 0 || echo 2)"
