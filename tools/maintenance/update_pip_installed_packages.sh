#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
COMMON_SCRIPT="$PROJECT_DIR/scripts/helpers/common.sh"
PYTHON_BIN="${PYTHON:-python3}"
DRY_RUN=0

if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "[ERROR] Missing helper script: $COMMON_SCRIPT" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$COMMON_SCRIPT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Usage: $0 [--python <python-bin>] [--dry-run]" >&2
      exit 1
      ;;
  esac
done

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

log "Upgrading pip"
pip install --upgrade pip

log "Checking for outdated installed packages"
OUTDATED_PACKAGES=$($PYTHON_BIN - <<'PY'
import json
import subprocess

result = subprocess.run(
    ["pip", "list", "--outdated", "--format=json"],
    check=True,
    capture_output=True,
    text=True,
)
packages = json.loads(result.stdout or "[]")
for package in packages:
    name = package.get("name")
    if name:
        print(name)
PY
)

if [[ -z "$OUTDATED_PACKAGES" ]]; then
  log "All installed packages are already up to date."
  deactivate
  exit 0
fi

log "Outdated packages detected:"
printf '%s\n' "$OUTDATED_PACKAGES"

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Dry run enabled; skipping upgrade step."
  deactivate
  exit 0
fi

log "Upgrading all outdated packages"
printf '%s\n' "$OUTDATED_PACKAGES" | xargs -r -n 1 pip install --upgrade

deactivate

log "Package upgrade complete."
