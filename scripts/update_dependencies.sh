#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
COMMON_SCRIPT="$PROJECT_DIR/scripts/helpers/common.sh"
PYTHON_BIN="${PYTHON:-python3}"
REQUIREMENTS_FILE_OVERRIDE="${REQUIREMENTS_FILE:-}"
OUTPUT_MODE="${DESK_DISPLAY_OUTPUT:-}"
INCLUDE_VENDOR_REQUIREMENTS="${INCLUDE_VENDOR_REQUIREMENTS:-0}"
INCLUDE_PLATFORM_SPECIFIC_REQUIREMENTS="${INCLUDE_PLATFORM_SPECIFIC_REQUIREMENTS:-0}"

if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "[ERROR] Missing helper script: $COMMON_SCRIPT" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$COMMON_SCRIPT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --requirements)
      REQUIREMENTS_FILE_OVERRIDE="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_MODE="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --include-vendor)
      INCLUDE_VENDOR_REQUIREMENTS=1
      shift
      ;;
    --include-platform-specific)
      INCLUDE_PLATFORM_SPECIFIC_REQUIREMENTS=1
      shift
      ;;
    *)
      echo "Usage: $0 [--requirements <file>] [--output <displayhatmini|minipitft|kernel|framebuffer>] [--python <python-bin>] [--include-vendor] [--include-platform-specific]" >&2
      exit 1
      ;;
  esac
done

pick_requirements_file() {
  if [[ -n "$REQUIREMENTS_FILE_OVERRIDE" ]]; then
    echo "$REQUIREMENTS_FILE_OVERRIDE"
    return 0
  fi

  local normalized_output_mode
  normalized_output_mode=$(printf '%s' "$OUTPUT_MODE" | tr '[:upper:]' '[:lower:]')

  case "$normalized_output_mode" in
    kernel)
      echo "requirements_kernel.txt"
      ;;
    framebuffer)
      echo "requirements_framebuffer.txt"
      ;;
    minipitft)
      echo "requirements_minipitft.txt"
      ;;
    *)
      echo "requirements.txt"
      ;;
  esac
}

REQUIREMENTS_FILE=$(pick_requirements_file)
cleanup_stale_egg_info() {
  local vendor_dir="$PROJECT_DIR/vendor"
  local had_permission_errors=0

  if [[ ! -d "$vendor_dir" ]]; then
    return 0
  fi

  while IFS= read -r -d '' egg_info_dir; do
    warn "Removing stale metadata directory: ${egg_info_dir#$PROJECT_DIR/}"
    if ! rm -rf "$egg_info_dir" 2>/dev/null; then
      had_permission_errors=1
      warn "Unable to remove ${egg_info_dir#$PROJECT_DIR/}; this usually means ownership/permissions are incorrect."
    fi
  done < <(find "$vendor_dir" -mindepth 2 -maxdepth 2 -type d -name '*.egg-info' -print0)

  if [[ "$had_permission_errors" -ne 0 ]]; then
    echo "[ERROR] One or more *.egg-info directories under vendor/ could not be removed." >&2
    echo "[ERROR] Editable installs will fail until permissions are fixed." >&2
    echo "[ERROR] Try: sudo chown -R \"$(id -un):$(id -gn)\" \"$PROJECT_DIR/vendor\" && rerun this script." >&2
    return 1
  fi
}

VENV_DIR="$PROJECT_DIR/venv"
EXISTING_VENV=$(detect_existing_venv "$PROJECT_DIR" || true)
if [[ -n "$EXISTING_VENV" ]]; then
  VENV_DIR="$EXISTING_VENV"
  log "Found existing virtual environment at $VENV_DIR"
fi

if [[ ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
  if [[ -d "$VENV_DIR" ]]; then
    warn "$VENV_DIR exists but does not look like a virtual environment. Recreating."
  fi
  log "Creating virtual environment with $PYTHON_BIN at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  log "Virtual environment already exists at $VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

check_venv_permissions() {
  local current_user current_group
  current_user=$(id -un)
  current_group=$(id -gn)

  # pip upgrades can fail mid-uninstall when these files are not writable
  # (for example after previously running the script with sudo).
  local required_paths=(
    "$VENV_DIR/bin"
    "$VENV_DIR/bin/pip"
    "$VENV_DIR/lib"
  )

  local path
  for path in "${required_paths[@]}"; do
    if [[ -e "$path" && ! -w "$path" ]]; then
      echo "[ERROR] Virtual environment is not writable: $path" >&2
      echo "[ERROR] This usually means the venv is owned by another user (often root)." >&2
      echo "[ERROR] Fix ownership, then rerun:" >&2
      echo "[ERROR]   sudo chown -R \"$current_user:$current_group\" \"$VENV_DIR\"" >&2
      return 1
    fi
  done
}

check_venv_permissions

log "Upgrading pip"
pip install --upgrade pip

build_install_requirements_file() {
  local source_requirements="$1"
  local install_requirements="$2"
  local host_os
  host_os=$(uname -s)

  if [[ "$INCLUDE_VENDOR_REQUIREMENTS" == "1" ]]; then
    cp "$source_requirements" "$install_requirements"
    return 0
  fi

  log "Skipping local vendor requirements (use --include-vendor to include them)"
  awk '
    /^[[:space:]]*#/ { print; next }
    /^[[:space:]]*$/ { print; next }
    /^[[:space:]]*-e[[:space:]]+\.\/vendor\// { next }
    /^[[:space:]]*\.\/vendor\// { next }
    /^[[:space:]]*-e[[:space:]]+file:\/\/.*\/vendor\// { next }
    /^[[:space:]]*file:\/\/.*\/vendor\// { next }
    { print }
  ' "$source_requirements" > "$install_requirements"

  if [[ "$host_os" == "Darwin" && "$INCLUDE_PLATFORM_SPECIFIC_REQUIREMENTS" != "1" ]]; then
    log "macOS detected; skipping Linux-only hardware dependencies (use --include-platform-specific to include them)"
    awk '
      /^[[:space:]]*#/ { print; next }
      /^[[:space:]]*$/ { print; next }
      /^[[:space:]]*(spidev|smbus|lgpio|rpi\.gpio|gpiozero|displayhatmini)[[:space:]]*([<>=!~].*)?$/ { next }
      { print }
    ' "$install_requirements" > "${install_requirements}.macos"
    mv "${install_requirements}.macos" "$install_requirements"
  fi
}

if [[ -f "$PROJECT_DIR/$REQUIREMENTS_FILE" ]]; then
  log "Installing Python dependencies from $REQUIREMENTS_FILE"
  if [[ "$INCLUDE_VENDOR_REQUIREMENTS" == "1" ]]; then
    cleanup_stale_egg_info
  fi
  INSTALL_REQUIREMENTS_FILE=$(mktemp)
  trap 'rm -f "$INSTALL_REQUIREMENTS_FILE"' EXIT
  build_install_requirements_file "$PROJECT_DIR/$REQUIREMENTS_FILE" "$INSTALL_REQUIREMENTS_FILE"
  pushd "$PROJECT_DIR" >/dev/null
  pip install -r "$INSTALL_REQUIREMENTS_FILE"
  popd >/dev/null
else
  warn "$REQUIREMENTS_FILE not found; skipping pip install."
fi

deactivate

log "Dependency update complete."
