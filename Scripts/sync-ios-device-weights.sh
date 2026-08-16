#!/bin/zsh
# Copy the pinned Klein 4-bit + Small Decoder snapshots into a connected
# iPhone's Imarello app container. Does not run generation.
#
# Usage (after the demo app is installed and has been launched once):
#   ./Scripts/sync-ios-device-weights.sh
#
# Requires: Xcode, a paired iPhone, and a local snapshot from `hf download`
# (or a leftover ~/Library/Caches/Aestrix/models copy).
set -euo pipefail

BUNDLE_ID="${BUNDLE_ID:-app.imarello.demo}"
HOST_MODELS="${HOST_MODELS:-$HOME/Library/Caches/Imarello/models}"
LEGACY_MODELS="${LEGACY_MODELS:-$HOME/Library/Caches/Aestrix/models}"
KLEIN="mlx-community--FLUX.2-Klein-4B-4bit"
SMALL="black-forest-labs--FLUX.2-small-decoder"
DEVICE="${DEVICE:-}"

resolve_snapshot() {
  local name="$1"
  local dir=""
  if [[ -d "$HOST_MODELS/$name" ]]; then
    dir="$HOST_MODELS/$name"
  elif [[ -d "$LEGACY_MODELS/$name" ]]; then
    dir="$LEGACY_MODELS/$name"
  else
    return 1
  fi
  # devicectl cannot open a symlinked source dir — hand it the real path.
  print -r -- "${dir:A}"
}

KLEIN_SRC="$(resolve_snapshot "$KLEIN" || true)"
SMALL_SRC="$(resolve_snapshot "$SMALL" || true)"
if [[ -z "$KLEIN_SRC" || -z "$SMALL_SRC" ]]; then
  echo "Missing host snapshots." >&2
  echo "Need $KLEIN and $SMALL under $HOST_MODELS (or $LEGACY_MODELS)." >&2
  echo "See Docs/WEIGHTS.md / Docs/IOS.md for hf download commands." >&2
  exit 1
fi

if [[ -z "$DEVICE" ]]; then
  # devicectl mixes log lines into stdout; write JSON to a real file.
  DEVICE_JSON="$(mktemp)"
  xcrun devicectl list devices --json-output "$DEVICE_JSON" >/dev/null 2>&1 || true
  DEVICE="$(python3 -c 'import json,sys
d=json.load(sys.stdin)
for dev in d.get("result",{}).get("devices",[]):
    p=dev.get("deviceProperties") or {}
    h=dev.get("hardwareProperties") or {}
    c=dev.get("connectionProperties") or {}
    if h.get("reality")=="physical" and c.get("pairingState")=="paired":
        print(dev.get("identifier") or "")
        break' < "$DEVICE_JSON")"
  rm -f "$DEVICE_JSON"
fi
if [[ -z "$DEVICE" ]]; then
  echo "No connected physical iPhone. Pair one, then retry." >&2
  xcrun devicectl list devices || true
  exit 1
fi

DEST="Library/Caches/Imarello/models"
echo "Copying snapshots to $BUNDLE_ID on $DEVICE"
echo "  $KLEIN_SRC"
echo "  $SMALL_SRC"
echo "  -> $DEST/"

xcrun devicectl device copy to \
  --device "$DEVICE" \
  --domain-type appDataContainer \
  --domain-identifier "$BUNDLE_ID" \
  --destination "$DEST" \
  --source "$KLEIN_SRC" \
  --source "$SMALL_SRC"

echo "Done. Relaunch Imarello so refreshGate() sees the files."
