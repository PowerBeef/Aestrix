#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHADERS="$ROOT/Sources/ImarelloDirect/Shaders"
SDK="macosx"
OUTPUT=""

while (( $# > 0 )); do
  case "$1" in
    --sdk)
      SDK="${2:?missing value for --sdk}"
      shift 2
      ;;
    --output)
      OUTPUT="${2:?missing value for --output}"
      shift 2
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$SDK" in
  macosx)
    PLATFORM="macosx"
    MIN_VERSION_FLAG="-mmacosx-version-min=26.2"
    DEFAULT_OUTPUT="$ROOT/Tools/Metal/imarello-direct.metallib"
    ;;
  iphoneos)
    PLATFORM="iphoneos"
    MIN_VERSION_FLAG="-mios-version-min=26.2"
    DEFAULT_OUTPUT="$ROOT/Apps/ImarelloIOS/ImarelloSpikes/imarello-direct.metallib"
    ;;
  *)
    echo "error: --sdk must be macosx or iphoneos" >&2
    exit 2
    ;;
esac

OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"
MANIFEST="$(dirname "$OUTPUT")/imarello-direct-manifest.json"
SDK_ROOT="$(xcrun -sdk "$SDK" --show-sdk-path)"
METAL_TOOL="$(xcrun -f metal)"
METALLIB_TOOL="$(dirname "$METAL_TOOL")/metallib"
if [[ ! -x "$METAL_TOOL" || ! -x "$METALLIB_TOOL" ]]; then
  echo "error: the Xcode Metal Toolchain is unavailable; open Xcode Settings > Components and install Metal Toolchain" >&2
  exit 1
fi
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
AIR="$WORK/air"
mkdir -p "$AIR" "$(dirname "$OUTPUT")"

compile() {
  local source="$1"
  local output="$2"
  shift 2
  "$METAL_TOOL" -isysroot "$SDK_ROOT" \
    -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions \
    "$MIN_VERSION_FLAG" "$@" -c "$source" -o "$output"
}

compile "$SHADERS/DirectDiT.metal" "$AIR/DirectDiT.air"
compile "$SHADERS/DirectVAE.metal" "$AIR/DirectVAE.air"
compile "$SHADERS/DirectGlue.metal" "$AIR/DirectGlueHalf.air" \
  -DDQ_DT=half -DDQ_SUFFIX=_half
compile "$SHADERS/DirectGlue.metal" "$AIR/DirectGlueBFloat.air" \
  -DDQ_DT=bfloat -DDQ_SUFFIX=_bfloat
"$METALLIB_TOOL" "$AIR"/*.air -o "$OUTPUT"

REQUIRED_SYMBOLS=(
  dd_ln_mod_prescale dd_rmsnorm_pitched dd_rope_interleaved
  dd_scale_cast_pitched dd_swiglu_pitched dd_scale_inplace dd_silu_f32
  dd_cast_prescale dd_cast_postscale dd_gate_add
  dq_rmsnorm_half dq_rope_half dq_silu_mul_half dq_add_half
  dq_rmsnorm_bfloat dq_rope_bfloat dq_silu_mul_bfloat dq_add_bfloat
  dv_gn_partial dv_gn_finalize dv_gn_apply dv_bias_act dv_upsample2 dv_add
  dv_matmul_nt dv_matmul_nn dv_softmax_rows
)
for symbol in "${REQUIRED_SYMBOLS[@]}"; do
  if ! LC_ALL=C grep -a -q "$symbol" "$OUTPUT"; then
    echo "error: Direct metallib is missing required symbol: $symbol" >&2
    exit 1
  fi
done

SDK_VERSION="$(xcrun -sdk "$SDK" --show-sdk-version)"
SOURCE_REVISION="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git -C "$ROOT" status --porcelain -- Sources/ImarelloDirect/Shaders)" ]]; then
  SOURCE_REVISION="${SOURCE_REVISION}-dirty"
fi
SHADER_SHA="$({
  for source in DirectDiT.metal DirectGlue.metal DirectVAE.metal; do
    printf '%s\n' "$source"
    shasum -a 256 "$SHADERS/$source"
  done
} | shasum -a 256 | awk '{print $1}')"
METALLIB_SHA="$(shasum -a 256 "$OUTPUT" | awk '{print $1}')"

{
  printf '{\n'
  printf '  "schemaVersion": 1,\n'
  printf '  "abiVersion": 1,\n'
  printf '  "platform": "%s",\n' "$PLATFORM"
  printf '  "sdkVersion": "%s",\n' "$SDK_VERSION"
  printf '  "minimumOS": "26.2",\n'
  printf '  "sourceRevision": "%s",\n' "$SOURCE_REVISION"
  printf '  "shaderSHA256": "%s",\n' "$SHADER_SHA"
  printf '  "metallibSHA256": "%s",\n' "$METALLIB_SHA"
  printf '  "requiredSymbols": [\n'
  for index in "${!REQUIRED_SYMBOLS[@]}"; do
    comma=','
    if (( index == ${#REQUIRED_SYMBOLS[@]} - 1 )); then comma=''; fi
    printf '    "%s"%s\n' "${REQUIRED_SYMBOLS[$index]}" "$comma"
  done
  printf '  ],\n'
  printf '  "functionConstants": {\n'
  printf '    "attention.alignedQueries": 200,\n'
  printf '    "attention.alignedKeys": 201,\n'
  printf '    "attention.hasMask": 300,\n'
  printf '    "attention.causal": 301,\n'
  printf '    "attention.doReads": 302\n'
  printf '  }\n'
  printf '}\n'
} > "$MANIFEST"

if [[ "$SDK" == "macosx" ]]; then
  for directory in \
    "$ROOT/.build/arm64-apple-macosx/debug" \
    "$ROOT/.build/arm64-apple-macosx/release" \
    "$ROOT/.build/debug" \
    "$ROOT/.build/release"
  do
    if [[ -d "$directory" ]]; then
      cp "$OUTPUT" "$directory/imarello-direct.metallib"
      cp "$MANIFEST" "$directory/imarello-direct-manifest.json"
    fi
  done
fi

echo "Direct metallib ready → $OUTPUT"
echo "Direct manifest ready → $MANIFEST"
