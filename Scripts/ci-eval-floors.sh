#!/usr/bin/env bash
# CI: Hub revision pins + golden metric floors (no model weights, no generate).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== Imarello Hub pins + eval golden floors + pure-math suites =="
# Single invocation so SPM compiles once. Do not drop the filter: Metal FA tests
# have hung after GPU aborts on constrained hosts. Every suite listed here is
# verified MLX-free (no GPU work), so it is safe on a virtualized CI runner;
# VAEAttention and DiTOpProfile stay local-only because they touch MLX arrays.
swift test --filter 'HubPinTests|GoldenMetricFloorsTests|ImageAnalyzerTests|DeviceHarness|TextTokenMode|Metallib|AppCache|VAETileMath|CompVisVAEMapper|EvalCachePolicy|PromptEmbedCacheKey|HostPreflight|Qwen|MetricsAggregator|FaceRegionCompare'

echo "OK — pins + floors + math suites satisfied (Docs/hub-pins.json, Docs/eval-floors.json)"
