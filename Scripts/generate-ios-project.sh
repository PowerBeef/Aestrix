#!/bin/zsh
set -euo pipefail
# Regenerates Apps/ImarelloIOS/ImarelloIOS.xcodeproj from project.yml (XcodeGen).
cd "$(dirname "$0")/../Apps/ImarelloIOS"
xcodegen generate
echo "Wrote $(pwd)/ImarelloIOS.xcodeproj"
