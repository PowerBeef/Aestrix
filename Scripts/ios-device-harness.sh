#!/bin/zsh
# Drive one Klein generate on a physical iPhone, pull the PNG, optional pixel eval.
# Does not tap the UI. The app polls Caches/Imarello/jobs/inbox/.
#
# Usage:
#   ./Scripts/ios-device-harness.sh
#   ./Scripts/ios-device-harness.sh --width 512 --seed 42 --eval --fail-on-pixel-gate
#   ./Scripts/ios-device-harness.sh --width 1024 --allow-1024 --eval
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE_ID="${BUNDLE_ID:-app.imarello.demo}"
DEVICE="${DEVICE:-}"
APP="${APP:-/tmp/ImarelloIOS-iphoneos-dd/Build/Products/Debug-iphoneos/Imarello.app}"
OUT_ROOT="${OUT_ROOT:-/tmp/imarello-ios-eval}"

MODE="t2i"
PROMPT="A red fox in a snowy forest at sunrise, photorealistic, golden rim light, shallow depth of field."
WIDTH=512
HEIGHT=""
STEPS=4
SEED=42
TEXT_TOKENS="512"
STRENGTH=0.8
JOB_ID=""
DO_EVAL=0
FAIL_GATE=0
ALLOW_1024=0
DO_INSTALL=0
TIMEOUT=180

usage() {
  cat <<'EOF'
Usage:
  Scripts/ios-device-harness.sh [options]

  --mode t2i|i2i          default t2i
  --prompt TEXT           default fox prompt
  --width N               512 (default) or 1024
  --height N              must match width
  --seed N                default 42
  --steps N               default 4
  --text-tokens 512|auto  default 512
  --strength F            i2i only, default 0.8
  --id NAME               job id (default derived)
  --eval                  run Scripts/eval-generation.sh on the PNG
  --fail-on-pixel-gate    eval exit 2 on pixel fail
  --allow-1024            required for --width 1024 (timeout 600s)
  --install               devicectl install APP if present
  --timeout SEC           poll timeout (overrides default)
  --device ID             CoreDevice id (else first connected physical)
  --app PATH              .app for --install

Environment: BUNDLE_ID, DEVICE, APP, OUT_ROOT, IMARELLO
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --prompt) PROMPT="${2:-}"; shift 2 ;;
    --width) WIDTH="${2:-}"; shift 2 ;;
    --height) HEIGHT="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;
    --steps) STEPS="${2:-}"; shift 2 ;;
    --text-tokens) TEXT_TOKENS="${2:-}"; shift 2 ;;
    --strength) STRENGTH="${2:-}"; shift 2 ;;
    --id) JOB_ID="${2:-}"; shift 2 ;;
    --eval) DO_EVAL=1; shift ;;
    --fail-on-pixel-gate) FAIL_GATE=1; shift ;;
    --allow-1024) ALLOW_1024=1; shift ;;
    --install) DO_INSTALL=1; shift ;;
    --timeout) TIMEOUT="${2:-}"; shift 2 ;;
    --device) DEVICE="${2:-}"; shift 2 ;;
    --app) APP="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

case "$MODE" in
  t2i|i2i) ;;
  *) echo "error: --mode must be t2i or i2i (got '$MODE')" >&2; exit 1 ;;
esac
case "$TEXT_TOKENS" in
  512|auto) ;;
  *) echo "error: --text-tokens must be 512 or auto (got '$TEXT_TOKENS')" >&2; exit 1 ;;
esac
if ! [[ "$STEPS" =~ ^[0-9]+$ ]] || (( STEPS < 1 )); then
  echo "error: --steps must be a positive integer (got '$STEPS')" >&2
  exit 1
fi
if ! [[ "$SEED" =~ ^[0-9]+$ ]] || (( SEED > 9999999 )); then
  echo "error: --seed must be an integer from 0 through 9999999 (got '$SEED')" >&2
  exit 1
fi
if ! [[ "$STRENGTH" =~ ^[0-9]*\.?[0-9]+$ ]] || ! awk -v value="$STRENGTH" 'BEGIN { exit !(value > 0 && value <= 1) }'; then
  echo "error: --strength must be a number in (0, 1] (got '$STRENGTH')" >&2
  exit 1
fi
if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]] || (( TIMEOUT < 1 )); then
  echo "error: --timeout must be a positive integer number of seconds (got '$TIMEOUT')" >&2
  exit 1
fi

HEIGHT="${HEIGHT:-$WIDTH}"
if [[ "$WIDTH" != "$HEIGHT" ]]; then
  echo "error: canvas must be square (got ${WIDTH}x${HEIGHT})" >&2
  exit 1
fi
if [[ "$WIDTH" != "512" && "$WIDTH" != "1024" ]]; then
  echo "error: width must be 512 or 1024" >&2
  exit 1
fi
if [[ "$WIDTH" == "1024" && "$ALLOW_1024" -ne 1 ]]; then
  echo "error: 1024 requires --allow-1024 (device time + host safety)" >&2
  exit 1
fi
if [[ "$WIDTH" == "1024" && "$TIMEOUT" -eq 180 ]]; then
  TIMEOUT=600
fi

METAL_OWNERS=(Xcode xcodebuild swift swiftc swift-frontend xctest metal metallib imarello aestrix)
for owner in "${METAL_OWNERS[@]}"; do
  if pgrep -x "$owner" >/dev/null 2>&1; then
    echo "error: possible Metal owner '$owner' is active; stop it before device generation." >&2
    pgrep -lf "$owner" >&2 || true
    exit 1
  fi
done

if [[ -z "$DEVICE" ]]; then
  LIST_JSON="$(mktemp /tmp/imarello-devices.XXXXXX.json)"
  xcrun devicectl list devices --json-output "$LIST_JSON" >/dev/null
  DEVICE="$(python3 - "$LIST_JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for dev in d.get("result", {}).get("devices", []):
    props = dev.get("properties") or {}
    hw = (props.get("hardware") or {})
    conn = (props.get("connection") or {})
    old_h = dev.get("hardwareProperties") or {}
    old_c = dev.get("connectionProperties") or {}
    reality = hw.get("reality") or old_h.get("reality")
    pairing = str(conn.get("pairingState") or old_c.get("pairingState") or "").lower()
    if reality == "physical" and pairing == "paired":
        print(dev.get("identifier") or "")
        break
PY
)"
  rm -f "$LIST_JSON"
fi
if [[ -z "$DEVICE" ]]; then
  echo "error: no connected physical iPhone." >&2
  xcrun devicectl list devices || true
  exit 1
fi

# The default id carries a timestamp: a reused id would match a leftover
# jobs/done/{id}.json from a previous run and return that run's PNG instantly.
SCRIPT_START_EPOCH="$(date +%s)"
if [[ -z "$JOB_ID" ]]; then
  JOB_ID="${MODE}-${WIDTH}-s${SEED}-$(date +%Y%m%d%H%M%S)"
fi
if ! [[ "$JOB_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "error: --id must be 1-128 safe, non-hidden [A-Za-z0-9._-] characters" >&2
  exit 1
fi

if [[ "$DO_INSTALL" -eq 1 ]]; then
  if [[ ! -d "$APP" ]]; then
    echo "error: --install but app missing: $APP" >&2
    echo "build Debug-iphoneos first (one Metal owner). See Docs/IOS.md." >&2
    exit 1
  fi
  echo "installing $APP → $DEVICE"
  xcrun devicectl device install app --device "$DEVICE" "$APP"
fi

echo "launching $BUNDLE_ID on $DEVICE"
xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE_ID" >/dev/null

WORK="$(mktemp -d /tmp/imarello-harness.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
# Reject reused IDs before submission. The app also refuses collisions, but a
# host-side check gives an immediate deterministic error and cannot overwrite
# or accidentally accept an earlier run's result.
for STATE in inbox running done; do
  if xcrun devicectl device copy from \
      --device "$DEVICE" \
      --domain-type appDataContainer \
      --domain-identifier "$BUNDLE_ID" \
      --source "Library/Caches/Imarello/jobs/${STATE}/${JOB_ID}.json" \
      --destination "$WORK/existing-${STATE}.json" >/dev/null 2>&1; then
    echo "error: harness job id '$JOB_ID' already exists in jobs/${STATE}; choose a fresh --id" >&2
    exit 1
  fi
done
JOB_JSON="$WORK/${JOB_ID}.json"
python3 - "$JOB_JSON" "$JOB_ID" "$MODE" "$PROMPT" "$WIDTH" "$HEIGHT" "$STEPS" "$SEED" "$TEXT_TOKENS" "$STRENGTH" <<'PY'
import json, sys
path, job_id, mode, prompt, width, height, steps, seed, tokens, strength = sys.argv[1:]
job = {
    "id": job_id,
    "mode": mode,
    "prompt": prompt,
    "width": int(width),
    "height": int(height),
    "steps": int(steps),
    "seed": int(seed),
    "textTokens": tokens,
    "strength": float(strength),
}
with open(path, "w") as f:
    json.dump(job, f, indent=2, sort_keys=True)
    f.write("\n")
PY

echo "copying job $JOB_ID (${WIDTH}² seed ${SEED} ${MODE})"
# Destination must be the file path. Copying onto .../inbox when that name
# is missing (or is already a file) replaces the directory with the JSON.
COPIED=0
for _ in {1..30}; do
  if xcrun devicectl device copy to \
      --device "$DEVICE" \
      --domain-type appDataContainer \
      --domain-identifier "$BUNDLE_ID" \
      --destination "Library/Caches/Imarello/jobs/inbox/${JOB_ID}.json" \
      --source "$JOB_JSON" >/dev/null 2>&1; then
    COPIED=1
    break
  fi
  xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE_ID" >/dev/null 2>&1 || true
  sleep 1
done
if [[ "$COPIED" -ne 1 ]]; then
  echo "error: could not copy job into jobs/inbox (is the app installed?)" >&2
  exit 1
fi

xcrun devicectl device process launch --device "$DEVICE" "$BUNDLE_ID" >/dev/null

echo "polling jobs/done/${JOB_ID}.json (timeout ${TIMEOUT}s)"
DEADLINE=$((SECONDS + TIMEOUT))
DEST="$OUT_ROOT/$JOB_ID"
mkdir -p "$DEST"
while (( SECONDS < DEADLINE )); do
  if xcrun devicectl device copy from \
      --device "$DEVICE" \
      --domain-type appDataContainer \
      --domain-identifier "$BUNDLE_ID" \
      --source "Library/Caches/Imarello/jobs/done/${JOB_ID}.json" \
      --destination "$DEST/${JOB_ID}.json" >/dev/null 2>&1; then
    # Guard against a leftover done file from a previous run with the same
    # (explicit) id: only accept results whose startedAt is not older than
    # this script's start, with 120 s of clock-skew allowance.
    if python3 - "$DEST/${JOB_ID}.json" "$SCRIPT_START_EPOCH" <<'PY' >/dev/null 2>&1
import json, sys, datetime
r = json.load(open(sys.argv[1]))
started = r.get("startedAt")
if not started:
    sys.exit(1)
t = datetime.datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp()
sys.exit(0 if t >= int(sys.argv[2]) - 120 else 1)
PY
    then
      DONE_JSON=1
      break
    fi
    rm -f "$DEST/${JOB_ID}.json"
  fi
  sleep 2
done

if [[ -z "${DONE_JSON:-}" ]]; then
  echo "error: timed out waiting for jobs/done/${JOB_ID}.json" >&2
  exit 1
fi

STATUS=0
python3 - "$DEST/${JOB_ID}.json" <<'PY' || STATUS=$?
import json, sys
r = json.load(open(sys.argv[1]))
print("status:", r.get("status"))
print("error:", r.get("error"))
print("png:", r.get("pngRelativePath"))
print("elapsedSec:", r.get("elapsedSec"))
print("metallib:", r.get("metallibNote"))
if r.get("status") != "ok":
    sys.exit(2 if r.get("status") == "failed" else 3)
PY
if [[ "$STATUS" -ne 0 ]]; then
  echo "harness job did not succeed (see $DEST/${JOB_ID}.json)" >&2
  exit "$STATUS"
fi

PNG_REL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pngRelativePath") or "")' "$DEST/${JOB_ID}.json")"
if [[ -z "$PNG_REL" ]]; then
  echo "error: result has no pngRelativePath" >&2
  exit 1
fi
PNG_NAME="$(basename "$PNG_REL")"
xcrun devicectl device copy from \
  --device "$DEVICE" \
  --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" \
  --source "$PNG_REL" \
  --destination "$DEST/$PNG_NAME"

echo "png: $DEST/$PNG_NAME"
echo "result: $DEST/${JOB_ID}.json"

if [[ "$DO_EVAL" -eq 1 ]]; then
  # Prefer the release binary for eval so a missing IMARELLO does not silently
  # fall back to (or build) a debug one inside eval-generation.sh.
  if [[ -z "${IMARELLO:-}" && -x "$ROOT/.build/release/imarello" ]]; then
    export IMARELLO="$ROOT/.build/release/imarello"
  fi
  EVAL_FLAGS=(--prompt "$PROMPT" --mode "$MODE")
  if [[ "$MODE" == "i2i" ]]; then
    EVAL_FLAGS+=(--strength "$STRENGTH")
  fi
  if [[ "$FAIL_GATE" -eq 1 ]]; then
    EVAL_FLAGS+=(--fail-on-pixel-gate)
  fi
  "$ROOT/Scripts/eval-generation.sh" "$DEST/$PNG_NAME" "${EVAL_FLAGS[@]}"
  echo "vision: open $DEST/$PNG_NAME (pixel eval is not a vision pass)"
fi

echo "done. id=$JOB_ID out=$DEST"
