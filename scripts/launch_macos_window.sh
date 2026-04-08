#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
VENV_DIR="$PROJECT_DIR/venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[ERROR] Missing virtual environment at $VENV_DIR" >&2
  echo "Run ./scripts/update_dependencies.sh first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

export DESK_DISPLAY_OUTPUT="${DESK_DISPLAY_OUTPUT:-window}"
export HYPERPIXEL_PANEL="${HYPERPIXEL_PANEL:-hyperpixel4}"
export DISPLAY_WIDTH="${DISPLAY_WIDTH:-800}"
export DISPLAY_HEIGHT="${DISPLAY_HEIGHT:-480}"
export DISPLAY_ROTATION="${DISPLAY_ROTATION:-0}"
export DESK_DISPLAY_WINDOW_SCALE="${DESK_DISPLAY_WINDOW_SCALE:-1}"

cd "$PROJECT_DIR"
exec python main.py
