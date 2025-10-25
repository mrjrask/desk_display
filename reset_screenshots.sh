#!/usr/bin/env bash
# reset_screenshots.sh
# Deletes and recreates the local screenshots/ and screenshot_archive/ folders
# relative to this script's directory.

set -Eeuo pipefail

# Resolve the absolute directory of this script (works with symlinks)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"

# Target directories (inside the script's directory)
TARGETS=(
  "$SCRIPT_DIR/screenshots"
  "$SCRIPT_DIR/screenshot_archive"
)

# Safety check to refuse obviously dangerous deletions
refuse_dangerous_path() {
  local path="$1"
  if [[ -z "$path" || "$path" == "/" || "$path" == "$HOME" ]]; then
    echo "❌ Refusing to operate on dangerous path: '$path'"
    exit 1
  fi
  # Ensure the path is within the script directory
  case "$path" in
    "$SCRIPT_DIR"/*) : ;; # ok
    *) echo "❌ Refusing to operate outside script directory: '$path'"; exit 1 ;;
  esac
}

echo "📂 Working in: $SCRIPT_DIR"

for dir in "${TARGETS[@]}"; do
  refuse_dangerous_path "$dir"

  if [[ -e "$dir" ]]; then
    echo "🗑️  Removing: $dir"
    rm -rf -- "$dir"
  else
    echo "ℹ️  Not present (ok): $dir"
  fi

  echo "📁 Creating: $dir"
  mkdir -p -- "$dir"
  chmod 775 -- "$dir" || true
done

echo "✅ Done. Recreated: screenshots/ and screenshot_archive/"
