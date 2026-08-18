#!/usr/bin/env bash
# Sequential 512² T2I pixel regression from Docs/eval-prompts.md (seeds 42, 0, 7).
# Does not commit PNGs. Requires a built imarello + local snapshot + metallib.
# Extra t2i flags: T2I_EXTRA='--text-tokens 512'  (see Docs/TEXT_TOKENS.md; CLI default is auto).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IMARELLO="${IMARELLO:-${AESTRIX:-.build/release/imarello}}"
OUT="${OUT_DIR:-/tmp/imarello-eval-regression}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
SEEDS="${SEEDS:-42 0 7}"

if [[ ! -x "$IMARELLO" ]]; then
  echo "error: $IMARELLO not executable (build release first)" >&2
  exit 1
fi

mkdir -p "$OUT"
SUMMARY="$OUT/summary.md"
: > "$SUMMARY"
echo "# Imarello 512² eval regression" >> "$SUMMARY"
echo "" >> "$SUMMARY"
echo "binary: \`$IMARELLO\`  canvas: ${WIDTH}×${HEIGHT}  $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SUMMARY"
echo "" >> "$SUMMARY"
echo "| id | seed | gate | technical | notes |" >> "$SUMMARY"
echo "|----|-----:|------|----------:|-------|" >> "$SUMMARY"

# Prompts 1–5 from Docs/eval-prompts.md (T2I distilled 4-step).
declare -a PROMPTS=(
  'A cozy coffee shop interior bathed in warm afternoon light, steam rising lazily from ceramic cups, worn leather armchairs arranged around small wooden tables, bookshelves lining exposed brick walls, dust motes floating in sunbeams through tall windows.'
  'A weathered fisherman in his 70s with deep wrinkles and a salt-and-pepper beard, wearing a navy cable-knit sweater, standing at the helm of his wooden boat. Golden hour sunlight from the left creates dramatic rim lighting on his profile. Shallow depth of field with harbor lights soft in the background.'
  'A modern product photo of a matte ceramic mug in #C45C26 terracotta orange on a #F5F0E8 linen surface, soft window light from the left, subtle shadow, clean commercial still life.'
  'A minimal poster with the headline "OPEN STUDIO" in bold condensed white sans-serif on a deep indigo #1A1A40 background, soft gradient, centered composition, print-ready graphic design.'
  'A red fox in a snowy forest at sunrise, photorealistic.'
)

fail=0
idx=0
for prompt in "${PROMPTS[@]}"; do
  idx=$((idx + 1))
  for seed in $SEEDS; do
    tag="p${idx}_s${seed}"
    png="$OUT/${tag}.png"
    echo "== $tag =="
    set +e
    # shellcheck disable=SC2086
    "$IMARELLO" t2i "$prompt" \
      --width "$WIDTH" --height "$HEIGHT" --steps 4 --seed "$seed" \
      --output "$png" --analyze --fail-on-pixel-gate \
      ${T2I_EXTRA:-}
    rc=$?
    set -e
    tech="?"
    notes=""
    if [[ -f "${png%.png}.eval.json" ]]; then
      # Path goes in via argv, not interpolated into python source (quote-safe).
      tech=$(python3 -c 'import json, sys
t = json.load(open(sys.argv[1])).get("technical") or {}
print(round(t.get("technical_score", t.get("technicalScore", float("nan"))), 1))' \
        "${png%.png}.eval.json" 2>/dev/null || echo "?")
    fi
    if [[ $rc -eq 0 ]]; then
      gate=PASS
    elif [[ $rc -eq 2 ]]; then
      gate=FAIL
      fail=$((fail + 1))
      notes="pixel gate"
    else
      gate=ERROR
      fail=$((fail + 1))
      notes="exit $rc"
    fi
    echo "| $tag | $seed | $gate | $tech | $notes |" >> "$SUMMARY"
    echo "  -> $gate tech=$tech"
  done
done

echo "" >> "$SUMMARY"
echo "failures: $fail" >> "$SUMMARY"
echo "summary: $SUMMARY"
cat "$SUMMARY"
exit "$fail"
