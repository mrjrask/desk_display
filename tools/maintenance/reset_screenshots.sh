#!/usr/bin/env bash
# reset_screenshots.sh
# Clears all contents of the local screenshots/ and screenshot_archive/ folders
# relative to the project root, without deleting the folders themselves.

set -Eeuo pipefail

# Resolve the absolute directory of this script (works with symlinks)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd -P)"

# Target directories (inside the project root)
TARGETS=(
  "$PROJECT_ROOT/screenshots"
  "$PROJECT_ROOT/screenshot_archive"
)

# Safety check to refuse obviously dangerous deletions
refuse_dangerous_path() {
  local path="$1"
  if [[ -z "$path" || "$path" == "/" || "$path" == "$HOME" ]]; then
    echo "❌ Refusing to operate on dangerous path: '$path'"
    exit 1
  fi
  # Ensure the path is within the project root
  case "$path" in
    "$PROJECT_ROOT"/*) : ;; # ok
    *) echo "❌ Refusing to operate outside project root: '$path'"; exit 1 ;;
  esac
}

echo "📂 Working in: $PROJECT_ROOT"

for dir in "${TARGETS[@]}"; do
  refuse_dangerous_path "$dir"

  if [[ ! -d "$dir" ]]; then
    echo "📁 Creating missing directory: $dir"
    mkdir -p -- "$dir"
    chmod 775 -- "$dir" || true
  else
    echo "🧹 Clearing directory: $dir"
    find "$dir" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
  fi

done

echo "✅ Reset complete."
