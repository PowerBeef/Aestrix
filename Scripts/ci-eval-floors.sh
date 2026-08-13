#!/usr/bin/env bash
# CI: Hub revision pins + golden metric floors (no model weights, no generate).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== Imarello Hub pins + eval golden floors =="
# Single invocation so SPM compiles once. Do not drop the filter: Metal FA tests
# have hung after GPU aborts on constrained hosts.
swift test --filter 'HubPinTests|GoldenMetricFloorsTests|ImageAnalyzerTests'

echo "OK — pins + floors satisfied (Docs/hub-pins.json, Docs/eval-floors.json)"
