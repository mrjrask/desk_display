#!/usr/bin/env bash
# reset_screenshots.sh
# Clears all contents of the screenshot and screenshot archive folders without
# deleting the folders themselves. They are screenshots/ and screenshot_archive/
# in the project root unless SCREENSHOT_DIR or SCREENSHOT_ARCHIVE_BASE moves
# them (in the environment or .env, relative paths from the project root, as
# main.py resolves them). Only main.py (the standalone renderer) writes
# screenshots; a server or client install has none of its own, so on those
# this only empties the folders if an earlier standalone install left any.
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

# A KEY=value from .env, the file main.py loads its settings from.
env_file_value() {
  local key="$1"
  local env_path="$PROJECT_ROOT/.env"
  [[ -f "$env_path" ]] || return 0
  awk -v key="$key" '
    /^[[:space:]]*(#|$)/ { next }
    {
      line=$0
      sub(/^[[:space:]]*export[[:space:]]+/, "", line)
      if (line !~ "^[[:space:]]*" key "[[:space:]]*=") next
      sub("^[[:space:]]*" key "[[:space:]]*=[[:space:]]*", "", line)
      sub(/[[:space:]]+#.*$/, "", line)
      sub(/^[[:space:]]+|[[:space:]]+$/, "", line)
      if ((line ~ /^".*"$/) || (line ~ /^\047.*\047$/)) line=substr(line, 2, length(line)-2)
      print line
      exit 0
    }
  ' "$env_path"
}

# Where main.py writes: the environment, then .env, then the default.
configured_dir() {
  local key="$1" default="$2"
  local value="${!key-}"
  if [[ -z "$value" ]]; then
    value="$(env_file_value "$key")"
  fi
  if [[ -z "$value" ]]; then
    value="$default"
  fi
  value="${value/#\~/$HOME}"
  if [[ "$value" != /* ]]; then
    value="$PROJECT_ROOT/$value"
  fi
  # Normalise ./ and ../ so the project-root check below sees the real path.
  realpath -m -- "$value" 2>/dev/null || printf '%s' "$value"
}

TARGETS=(
  "$(configured_dir SCREENSHOT_DIR screenshots)"
  "$(configured_dir SCREENSHOT_ARCHIVE_BASE screenshot_archive)"
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
    *) echo "❌ Refusing to operate outside project root: '$path' (clear it by hand)"; exit 1 ;;
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

# Check every target before clearing any of them.
for dir in "${TARGETS[@]}"; do
  refuse_dangerous_path "$dir"
done

for dir in "${TARGETS[@]}"; do
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
  echo "     sudo chown -R \"\$(whoami)\":\"\$(whoami)\" \"${TARGETS[0]}\" \"${TARGETS[1]}\""
  exit 1
fi

echo "✅ Reset complete."
