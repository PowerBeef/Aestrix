#!/usr/bin/env bash
# CI: golden metric floors for AestrixEval (no model weights required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== AestrixEval golden floors (P1) =="
swift test --filter GoldenMetricFloorsTests
swift test --filter ImageAnalyzerTests

echo "OK — floors satisfied (see Docs/eval-floors.json, Docs/IMAGE_ANALYSIS.md)"
