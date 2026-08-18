#!/usr/bin/env bash
# Build a real mlx.metallib from mlx-swift Metal kernels and install it next to
# SPM-built imarello / Cmlx bundle so MLX can load RMSNorm, SDPA, RoPE, etc.
#
# Usage: Scripts/ensure-metallib.sh   (run after `swift build` or when metallib is missing)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/Tools/Metal/mlx.metallib"
CHECKOUT="$ROOT/.build/checkouts/mlx-swift"
MLX_ROOT="$CHECKOUT/Source/Cmlx/mlx"
KERNELS="$MLX_ROOT/mlx/backend/metal/kernels"

# The mlx-swift revision the lib was built from (git checkout, else Package.resolved).
mlx_revision() {
  git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null && return 0
  python3 - "$ROOT/Package.resolved" 2>/dev/null <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for p in d.get("pins", []):
    if p.get("identity") == "mlx-swift":
        print(p["state"]["revision"])
        break
PY
}

need_rebuild() {
  if [[ ! -f "$OUT" ]]; then return 0; fi
  # Stub / noop metallib is a few KB; real one is tens of MB.
  local sz
  sz=$(stat -f%z "$OUT" 2>/dev/null || echo 0)
  if (( sz < 1000000 )); then return 0; fi
  # A lib built from an older mlx-swift silently mismatches the host code
  # (kernel/runtime ABI): rebuild whenever the recorded revision differs.
  local want have
  want="$(mlx_revision || true)"
  have="$(cat "$OUT.rev" 2>/dev/null || true)"
  if [[ -z "$want" || -z "$have" || "$want" != "$have" ]]; then return 0; fi
  return 1
}

build_metallib() {
  if [[ ! -d "$KERNELS" ]]; then
    echo "error: mlx-swift checkout not found at $KERNELS" >&2
    echo "       run: swift package resolve && swift build  (once) then re-run this script" >&2
    exit 1
  fi

  local AIR_DIR
  WORK="$(mktemp -d)"
  trap 'rm -rf "${WORK:-}"' EXIT
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
    exit 1
  fi
  # A partial kernel set links fine and only fails later as a runtime MLX
  # function-load error mid-generate — refuse to install one.
  if (( FAIL > 0 )); then
    echo "error: $FAIL kernel(s) failed to compile; refusing to link a partial metallib" >&2
    echo "       see the FAIL lines above (toolchain vs mlx-swift kernel mismatch?)" >&2
    exit 1
  fi
  echo "compiled OK=$OK FAIL=$FAIL → linking metallib"
  mkdir -p "$(dirname "$OUT")"
  xcrun -sdk macosx metallib "$AIR_DIR"/*.air -o "$OUT"
  mlx_revision > "$OUT.rev" || true
  ls -la "$OUT"
}

install_metallib() {
  local installed=0
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
      installed=$((installed + 1))
    fi
  done
  # Do NOT overwrite Sources/ImarelloCLI/Resources/mlx.metallib — that path holds a
  # tiny SPM package stub committed to git. Full kernels go next to the binary only.
  if (( installed == 0 )); then
    echo "error: no .build product directory found — nothing received the metallib." >&2
    echo "       run 'swift build' first (the binary would run on the stub otherwise)." >&2
    exit 1
  fi
}

verify_steel() {
  local lib="$1"
  local sz
  sz=$(stat -f%z "$lib" 2>/dev/null || echo 0)
  if (( sz < 1000000 )); then
    echo "error: $lib is a stub ($sz bytes); expected a full MLX metallib" >&2
    exit 1
  fi
  python3 - "$lib" <<'PY'
import sys
from pathlib import Path
data = Path(sys.argv[1]).read_bytes()
need = ("steel_attention", "affine_qmm", "rms_norm")
missing = [s for s in need if s.encode() not in data]
if missing:
    print("error: Steel symbols missing:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)
print("steel symbols ok in", sys.argv[1], f"({len(data)} bytes)")
PY
}

if need_rebuild; then
  build_metallib
else
  echo "reusing existing $OUT ($(stat -f%z "$OUT") bytes)"
fi
verify_steel "$OUT"
install_metallib
