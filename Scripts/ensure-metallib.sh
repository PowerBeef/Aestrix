#!/usr/bin/env bash
# Build a real mlx.metallib from mlx-swift Metal kernels and install it next to
# SPM-built aestrix / Cmlx bundle so MLX can load RMSNorm, SDPA, RoPE, etc.
#
# Usage: Scripts/ensure-metallib.sh   (run after `swift build` or when metallib is missing)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/Tools/Metal/mlx.metallib"
CHECKOUT="$ROOT/.build/checkouts/mlx-swift"
MLX_ROOT="$CHECKOUT/Source/Cmlx/mlx"
KERNELS="$MLX_ROOT/mlx/backend/metal/kernels"

need_rebuild() {
  if [[ ! -f "$OUT" ]]; then return 0; fi
  # Stub / noop metallib is a few KB; real one is tens of MB.
  local sz
  sz=$(stat -f%z "$OUT" 2>/dev/null || echo 0)
  if (( sz < 1000000 )); then return 0; fi
  return 1
}

build_metallib() {
  if [[ ! -d "$KERNELS" ]]; then
    echo "error: mlx-swift checkout not found at $KERNELS" >&2
    echo "       run: swift package resolve && swift build  (once) then re-run this script" >&2
    exit 1
  fi

  local WORK AIR_DIR
  WORK="$(mktemp -d)"
  AIR_DIR="$WORK/air"
  mkdir -p "$AIR_DIR"
  local OK=0 FAIL=0
  local f rel base out

  echo "compiling Metal kernels from $KERNELS …"
  while IFS= read -r f; do
    rel="${f#"$KERNELS/"}"
    base="$(echo "$rel" | tr '/' '_')"
    base="${base%.metal}"
    out="$AIR_DIR/${base}.air"
    if xcrun -sdk macosx metal \
      -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions \
      -I"$MLX_ROOT" -c "$f" -o "$out" 2>"$WORK/${base}.err"
    then
      OK=$((OK + 1))
    else
      FAIL=$((FAIL + 1))
      echo "  FAIL $base: $(head -1 "$WORK/${base}.err")" >&2
    fi
  done < <(find "$KERNELS" -name '*.metal' | sort)

  if (( OK == 0 )); then
    echo "error: no metal kernels compiled" >&2
    rm -rf "$WORK"
    exit 1
  fi
  echo "compiled OK=$OK FAIL=$FAIL → linking metallib"
  mkdir -p "$(dirname "$OUT")"
  xcrun -sdk macosx metallib "$AIR_DIR"/*.air -o "$OUT"
  rm -rf "$WORK"
  ls -la "$OUT"
}

install_metallib() {
  local dir
  for dir in \
    "$ROOT/.build/arm64-apple-macosx/debug" \
    "$ROOT/.build/arm64-apple-macosx/release" \
    "$ROOT/.build/debug" \
    "$ROOT/.build/release"
  do
    if [[ -d "$dir" ]]; then
      cp "$OUT" "$dir/mlx.metallib"
      cp "$OUT" "$dir/default.metallib"
      mkdir -p "$dir/Resources"
      cp "$OUT" "$dir/Resources/mlx.metallib"
      # Keep Cmlx SwiftPM bundle in sync (MLX may load default.metallib from there).
      local cmlx
      for cmlx in "$dir"/mlx-swift_Cmlx.bundle "$dir"/mlx_Cmlx.bundle; do
        if [[ -d "$cmlx" ]]; then
          cp "$OUT" "$cmlx/default.metallib"
          mkdir -p "$cmlx/Contents/Resources"
          cp "$OUT" "$cmlx/Contents/Resources/default.metallib"
        fi
      done
      echo "installed metallib → $dir"
    fi
  done
  # Do NOT overwrite Sources/AestrixCLI/Resources/mlx.metallib — that path holds a
  # tiny SPM package stub committed to git. Full kernels go next to the binary only.
}

if need_rebuild; then
  build_metallib
else
  echo "reusing existing $OUT ($(stat -f%z "$OUT") bytes)"
fi
install_metallib
