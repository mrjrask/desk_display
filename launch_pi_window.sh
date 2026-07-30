#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
VENV_DIR="$PROJECT_DIR/venv"

load_env_file() {
  local env_path="$1"
  if [[ -f "$env_path" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$env_path"
    set +a
  fi
}

expand_home_path_var() {
  local name="$1"
  local raw="${!name:-}"
  if [[ "$raw" == "~/"* ]]; then
    export "$name=$HOME/${raw#~/}"
  fi
}

if [[ -z "${DESK_DISPLAY_ENV_PATH:-}" ]]; then
  if [[ -f "$PROJECT_DIR/.env" ]]; then
    DESK_DISPLAY_ENV_PATH="$PROJECT_DIR/.env"
  elif [[ -f "$HOME/desk_display/.env" ]]; then
    DESK_DISPLAY_ENV_PATH="$HOME/desk_display/.env"
  else
    DESK_DISPLAY_ENV_PATH=""
  fi
fi

if [[ -n "${DESK_DISPLAY_ENV_PATH:-}" ]]; then
  load_env_file "$DESK_DISPLAY_ENV_PATH"
fi

expand_home_path_var "WEATHERKIT_KEY_PATH"

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
