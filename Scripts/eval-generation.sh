#!/usr/bin/env bash
# Pixel eval + vision brief for an existing generation.
# See Docs/EVAL_WORKFLOW.md
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# A user-supplied IMARELLO must be honored or rejected — never silently swapped.
IMARELLO_EXPLICIT="${IMARELLO:-${AESTRIX:-}}"
if [[ -n "$IMARELLO_EXPLICIT" && ! -x "$IMARELLO_EXPLICIT" ]]; then
  echo "error: IMARELLO is set but not executable: $IMARELLO_EXPLICIT" >&2
  exit 1
fi
# Prefer the release binary when present (matches generate/bench convention);
# analyze-image itself is a CPU/Vision path, so debug is only slower, not wrong.
if [[ -n "$IMARELLO_EXPLICIT" ]]; then
  IMARELLO="$IMARELLO_EXPLICIT"
elif [[ -x "$ROOT/.build/release/imarello" ]]; then
  IMARELLO="$ROOT/.build/release/imarello"
else
  IMARELLO="$ROOT/.build/debug/imarello"
fi
IMAGE=""
PROMPT=""
REFERENCE=""
MODE=""
JSON_OUT=""
FAIL_GATE=0

usage() {
  cat <<'EOF'
Usage:
  Scripts/eval-generation.sh <image.png> --prompt "..." [--reference src.png] [--mode t2i|i2i]
                             [--json report.json] [--fail-on-pixel-gate]

Runs imarello analyze-image (text report + JSON) and writes a vision brief sidecar.

Environment:
  IMARELLO   path to imarello binary (default: .build/release/imarello when
             present, else .build/debug/imarello; AESTRIX is a legacy alias)
EOF
}

if [[ $# -lt 1 ]]; then usage; exit 1; fi
IMAGE="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt) PROMPT="${2:-}"; shift 2 ;;
    --reference) REFERENCE="${2:-}"; shift 2 ;;
    --mode) MODE="${2:-}"; shift 2 ;;
    --json) JSON_OUT="${2:-}"; shift 2 ;;
    --fail-on-pixel-gate) FAIL_GATE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -f "$IMAGE" ]]; then
  echo "error: image not found: $IMAGE" >&2
  exit 1
fi

if [[ ! -x "$IMARELLO" ]]; then
  if pgrep -x xcodebuild >/dev/null 2>&1 || pgrep -x imarello >/dev/null 2>&1; then
    echo "error: imarello binary missing and a Metal owner is active; build later" >&2
    echo "       (swift build -c release && ./Scripts/ensure-metallib.sh)" >&2
    exit 1
  fi
  echo "building imarello…"
  (cd "$ROOT" && swift build --product imarello)
  IMARELLO="$ROOT/.build/debug/imarello"
fi

if [[ -z "$MODE" ]]; then
  if [[ -n "$REFERENCE" ]]; then MODE=i2i; else MODE=t2i; fi
fi

ABS_IMAGE="$(cd "$(dirname "$IMAGE")" && pwd)/$(basename "$IMAGE")"
STEM="${ABS_IMAGE%.*}"
JSON_OUT="${JSON_OUT:-${STEM}.eval.json}"
BRIEF_OUT="${STEM}.vision-brief.md"

ARGS=(analyze-image "$ABS_IMAGE" --json "$JSON_OUT" --vision-mode "$MODE")
if [[ -n "$PROMPT" ]]; then ARGS+=(--prompt "$PROMPT"); fi
if [[ -n "$REFERENCE" ]]; then ARGS+=(--reference "$REFERENCE"); fi

echo "=== Imarello eval-generation ==="
echo "image: $ABS_IMAGE"
echo "mode:  $MODE"
[[ -n "$PROMPT" ]] && echo "prompt: $PROMPT"
[[ -n "$REFERENCE" ]] && echo "reference: $REFERENCE"

set +e
"$IMARELLO" "${ARGS[@]}"
CODE=$?
set -e

# Always write vision brief (analyze-image --vision-brief replaces text; run separately)
BRIEF_ARGS=(analyze-image "$ABS_IMAGE" --vision-brief --vision-mode "$MODE")
if [[ -n "$PROMPT" ]]; then BRIEF_ARGS+=(--prompt "$PROMPT"); fi
if [[ -n "$REFERENCE" ]]; then BRIEF_ARGS+=(--reference "$REFERENCE"); fi
"$IMARELLO" "${BRIEF_ARGS[@]}" > "$BRIEF_OUT" || true
echo "vision_brief: $BRIEF_OUT"
echo "eval_json:    $JSON_OUT"

if [[ $CODE -eq 2 ]]; then
  echo "quality_gate: pixel FAIL (exit 2)" >&2
  if [[ $FAIL_GATE -eq 1 ]]; then exit 2; fi
  exit 0
elif [[ $CODE -ne 0 ]]; then
  echo "error: analyze-image failed (exit $CODE)" >&2
  exit "$CODE"
fi

echo "quality_gate: pixel PASS"
echo "next: open PNG with vision tools, complete checklist in vision-brief, merge Assessment"
echo "docs: Docs/EVAL_WORKFLOW.md"
exit 0
