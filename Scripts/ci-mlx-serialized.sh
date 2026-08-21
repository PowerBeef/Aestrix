#!/usr/bin/env bash
# Explicit serialized MLX gate. Run alone under AGENTS.md's one-owner rule.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== Imarello serialized MLX suites =="
IMARELLO_MLX_TESTS=1 swift test --filter 'MemoryStaging|IdentityPreserve|Flux2Math'
echo "OK — serialized MLX suites passed"
