#!/usr/bin/env bash
# reset_screenshots.sh
# Clears all contents of the local screenshots/ and screenshot_archive/ folders
# relative to the project root, without deleting the folders themselves.
#
# Files under these folders are often written by the desk_display systemd
# service, which may run as a different user (or root) than whoever runs
# this script interactively. When that happens, plain `rm` fails with
# "Permission denied". This script retries such entries with `sudo` instead
# of aborting on the first failure, and reports anything it still couldn't
# remove at the end.

set -Eeuo pipefail

# Resolve the absolute directory of this script (works with symlinks)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd -P)"

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

failed_entries=()

# Remove a single top-level entry. Falls back to `sudo rm -rf` when a plain
# removal fails with a permissions error, so ownership mismatches (e.g. the
# systemd service writing as a different user) don't abort the whole run.
remove_entry() {
  local entry="$1"
  local err_file
  err_file="$(mktemp)"

  if rm -rf -- "$entry" 2>"$err_file"; then
    rm -f -- "$err_file"
    return 0
  fi

  local err
  err="$(cat -- "$err_file" 2>/dev/null || true)"
  rm -f -- "$err_file"

  if [[ "$err" == *"Permission denied"* ]] && command -v sudo &>/dev/null; then
    echo "  ⚠️  Permission denied removing $(basename -- "$entry"); retrying with sudo..."
    if sudo rm -rf -- "$entry"; then
      return 0
    fi
  fi

  echo "  ❌ Failed to remove: $entry"
  [[ -n "$err" ]] && echo "     $err"
  failed_entries+=("$entry")
  return 1
}

echo "📂 Working in: $PROJECT_ROOT"

for dir in "${TARGETS[@]}"; do
  refuse_dangerous_path "$dir"

  if [[ ! -d "$dir" ]]; then
    echo "📁 Creating missing directory: $dir"
    mkdir -p -- "$dir"
    chmod 775 -- "$dir" || true
    continue
  fi

  echo "🧹 Clearing directory: $dir"
  while IFS= read -r -d '' entry; do
    remove_entry "$entry" || true
  done < <(find "$dir" -mindepth 1 -maxdepth 1 -print0)
done

if ((${#failed_entries[@]} > 0)); then
  echo ""
  echo "⚠️  ${#failed_entries[@]} item(s) could not be removed, even with sudo:"
  for entry in "${failed_entries[@]}"; do
    echo "   - $entry"
  done
  echo ""
  echo "   These are likely owned by another user (e.g. the desk_display"
  echo "   systemd service running as root or a different account)."
  echo "   Fix ownership with:"
  echo "     sudo chown -R \"\$(whoami)\":\"\$(whoami)\" \"$PROJECT_ROOT/screenshots\" \"$PROJECT_ROOT/screenshot_archive\""
  exit 1
fi

echo "✅ Reset complete."
