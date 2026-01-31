#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_NAME="${SERVICE_NAME:-desk_display.service}"

COMMON_SCRIPT="$SCRIPT_DIR/install_common.sh"
if [[ -f "$COMMON_SCRIPT" ]]; then
  # shellcheck source=/dev/null
  source "$COMMON_SCRIPT"
fi

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

service_was_active="false"
if command -v systemctl >/dev/null 2>&1; then
  if systemctl is-active --quiet "$SERVICE_NAME"; then
    service_was_active="true"
    log "Stopping $SERVICE_NAME before launching in the desktop session."
    $SUDO systemctl stop "$SERVICE_NAME"
  fi
fi

cleanup() {
  if [[ "$service_was_active" == "true" ]] && command -v systemctl >/dev/null 2>&1; then
    log "Restarting $SERVICE_NAME."
    $SUDO systemctl restart "$SERVICE_NAME"
  fi
}
trap cleanup EXIT

if [[ -z "${DESK_DISPLAY_OUTPUT:-}" ]]; then
  export DESK_DISPLAY_OUTPUT="kernel"
fi

if [[ -z "${SDL_VIDEODRIVER:-}" ]]; then
  if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    export SDL_VIDEODRIVER="wayland"
  elif [[ -n "${DISPLAY:-}" ]]; then
    export SDL_VIDEODRIVER="x11"
  fi
fi

VENV_DIR=""
if declare -F detect_existing_venv >/dev/null 2>&1; then
  VENV_DIR=$(detect_existing_venv "$PROJECT_DIR" || true)
fi
if [[ -z "$VENV_DIR" ]]; then
  VENV_DIR="$PROJECT_DIR/venv"
fi

PYTHON_BIN="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  warn "Python virtualenv not found at $VENV_DIR. Run the installer first."
  exit 1
fi

log "Launching Desk Display with $PYTHON_BIN"
exec "$PYTHON_BIN" "$PROJECT_DIR/main.py"
