#!/bin/zsh
# Copy the pinned Klein 4-bit + Small Decoder snapshots into a connected
# iPhone's Imarello app container. Does not run generation.
#
# Usage (after the demo app is installed on a device):
#   ./Scripts/sync-ios-device-weights.sh
#
# Requires: Xcode, a paired iPhone, and the local Mac cache from `hf download`.
set -euo pipefail

BUNDLE_ID="${BUNDLE_ID:-app.imarello.demo}"
HOST_MODELS="${HOST_MODELS:-$HOME/Library/Caches/Imarello/models}"
KLEIN="mlx-community--FLUX.2-Klein-4B-4bit"
SMALL="black-forest-labs--FLUX.2-small-decoder"

if [[ ! -d "$HOST_MODELS/$KLEIN" || ! -d "$HOST_MODELS/$SMALL" ]]; then
  echo "Missing host snapshots under $HOST_MODELS" >&2
  echo "See Docs/WEIGHTS.md / Docs/IOS.md for hf download commands." >&2
  exit 1
fi

if ! xcrun devicectl list devices >/dev/null 2>&1; then
  echo "devicectl is not available. Install Xcode." >&2
  exit 1
fi

echo "This script lists devices, then you copy snapshots into the app container:"
echo "  Caches/Imarello/models/$KLEIN"
echo "  Caches/Imarello/models/$SMALL"
echo
echo "Find the app container after first launch:"
echo "  xcrun devicectl device info appContainer --device <udid> --bundle-id $BUNDLE_ID"
echo
echo "Then rsync the two directories from:"
echo "  $HOST_MODELS/$KLEIN"
echo "  $HOST_MODELS/$SMALL"
echo
xcrun devicectl list devices || true
