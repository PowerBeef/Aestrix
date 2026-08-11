#!/usr/bin/env bash
# Optional: install Core ML CLIP encoders for real CLIPScore (P1).
#
# Expected layout after install:
#   ~/Library/Caches/Aestrix/models/clip-coreml/
#     image_encoder.mlmodelc   # RGB 224 → L2-normalized embedding multiarray
#     text_encoder.mlmodelc    # string input → L2-normalized embedding multiarray
#
# Export tips:
#   - MobileCLIP / OpenAI CLIP converted with coremltools
#   - Text encoder should accept MLFeatureType.string when possible
#   - Matching embedding dims (e.g. 512)
#
# Without these models, AestrixEval uses VNClassifyImageRequest proxy
# (backend=vision_proxy) — always available, weaker signal.
set -euo pipefail
DEST="${AESTIX_CLIP_DIR:-$HOME/Library/Caches/Aestrix/models/clip-coreml}"
mkdir -p "$DEST"
echo "CLIP Core ML directory: $DEST"
echo ""
echo "Place compiled models here:"
echo "  $DEST/image_encoder.mlmodelc"
echo "  $DEST/text_encoder.mlmodelc"
echo ""
if [[ -d "$DEST/image_encoder.mlmodelc" && -d "$DEST/text_encoder.mlmodelc" ]]; then
  echo "OK — both encoders present. SemanticAlignment will use backend=coreml_clip."
else
  echo "Missing encoders — analyze-image will use vision_proxy until installed."
  exit 0
fi
