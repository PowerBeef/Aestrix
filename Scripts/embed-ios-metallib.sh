#!/usr/bin/env bash
set -euo pipefail

if [[ "${PLATFORM_NAME:-iphoneos}" != "iphoneos" ]]; then
  echo "skipping MLX/Direct metallib embedding for ${PLATFORM_NAME}; Simulator is UI-only"
  exit 0
fi

SOURCE="${IMARELLO_IOS_METALLIB_SOURCE:-${SRCROOT}/ImarelloSpikes/mlx-full.metallib}"
DESTINATION="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/mlx.metallib"
DIRECT_SOURCE="${IMARELLO_IOS_DIRECT_METALLIB_SOURCE:-${SRCROOT}/ImarelloSpikes/imarello-direct.metallib}"
DIRECT_MANIFEST_SOURCE="${IMARELLO_IOS_DIRECT_MANIFEST_SOURCE:-${SRCROOT}/ImarelloSpikes/imarello-direct-manifest.json}"
DIRECT_DESTINATION="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/imarello-direct.metallib"
DIRECT_MANIFEST_DESTINATION="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/imarello-direct-manifest.json"
MIN_BYTES=100000000

if [[ ! -f "$SOURCE" ]]; then
  echo "error: full iphoneos no-JIT metallib missing at $SOURCE" >&2
  echo "error: rebuild Apps/ImarelloIOS/ImarelloSpikes/mlx-full.metallib before the device app" >&2
  exit 1
fi

SIZE="$(stat -f%z "$SOURCE")"
if (( SIZE < MIN_BYTES )); then
  echo "error: refusing partial/JIT-era metallib ($SIZE bytes; need at least $MIN_BYTES)" >&2
  exit 1
fi

for SYMBOL in steel_attention rms_norm affine_qmm; do
  if ! LC_ALL=C grep -a -q "$SYMBOL" "$SOURCE"; then
    echo "error: full metallib is missing required symbol family: $SYMBOL" >&2
    exit 1
  fi
done

if [[ ! -f "$DIRECT_SOURCE" || ! -f "$DIRECT_MANIFEST_SOURCE" ]]; then
  echo "error: Direct iphoneos metallib or manifest is missing under ${SRCROOT}/ImarelloSpikes" >&2
  echo "error: rebuild both with Scripts/ensure-direct-metallib.sh --sdk iphoneos" >&2
  exit 1
fi

DIRECT_PLATFORM="$(plutil -extract platform raw -o - "$DIRECT_MANIFEST_SOURCE")"
DIRECT_ABI="$(plutil -extract abiVersion raw -o - "$DIRECT_MANIFEST_SOURCE")"
DIRECT_MINIMUM_OS="$(plutil -extract minimumOS raw -o - "$DIRECT_MANIFEST_SOURCE")"
DIRECT_EXPECTED_SHA="$(plutil -extract metallibSHA256 raw -o - "$DIRECT_MANIFEST_SOURCE")"
DIRECT_ACTUAL_SHA="$(shasum -a 256 "$DIRECT_SOURCE" | awk '{print $1}')"
if [[ "$DIRECT_PLATFORM" != "iphoneos" || "$DIRECT_ABI" != "1" || "$DIRECT_MINIMUM_OS" != "26.2" ]]; then
  echo "error: incompatible Direct manifest (platform=$DIRECT_PLATFORM abi=$DIRECT_ABI minimumOS=$DIRECT_MINIMUM_OS)" >&2
  exit 1
fi
if [[ "$DIRECT_EXPECTED_SHA" != "$DIRECT_ACTUAL_SHA" ]]; then
  echo "error: Direct metallib SHA-256 does not match its manifest" >&2
  exit 1
fi

DIRECT_SYMBOLS=(
  dd_ln_mod_prescale dd_rmsnorm_pitched dd_rope_interleaved
  dd_scale_cast_pitched dd_swiglu_pitched dd_scale_inplace dd_silu_f32
  dd_cast_prescale dd_cast_postscale dd_gate_add
  dq_rmsnorm_half dq_rope_half dq_silu_mul_half dq_add_half
  dq_rmsnorm_bfloat dq_rope_bfloat dq_silu_mul_bfloat dq_add_bfloat
  dv_gn_partial dv_gn_finalize dv_gn_apply dv_bias_act dv_upsample2 dv_add
  dv_matmul_nt dv_matmul_nn dv_softmax_rows
)
for SYMBOL in "${DIRECT_SYMBOLS[@]}"; do
  if ! LC_ALL=C grep -a -q "$SYMBOL" "$DIRECT_SOURCE"; then
    echo "error: Direct metallib is missing required symbol: $SYMBOL" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$DESTINATION")"
if [[ ! -f "$DESTINATION" ]] || ! cmp -s "$SOURCE" "$DESTINATION"; then
  cp "$SOURCE" "$DESTINATION"
fi
if [[ ! -f "$DIRECT_DESTINATION" ]] || ! cmp -s "$DIRECT_SOURCE" "$DIRECT_DESTINATION"; then
  cp "$DIRECT_SOURCE" "$DIRECT_DESTINATION"
fi
if [[ ! -f "$DIRECT_MANIFEST_DESTINATION" ]] || ! cmp -s "$DIRECT_MANIFEST_SOURCE" "$DIRECT_MANIFEST_DESTINATION"; then
  cp "$DIRECT_MANIFEST_SOURCE" "$DIRECT_MANIFEST_DESTINATION"
fi

echo "embedded full iphoneos MLX metallib → $DESTINATION ($SIZE bytes)"
echo "embedded Direct iphoneos metallib + manifest → $DIRECT_DESTINATION"
