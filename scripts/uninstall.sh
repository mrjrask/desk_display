#!/usr/bin/env bash
set -euo pipefail

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

SERVICE_NAME="desk_display.service"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
VENV_DIR="$PROJECT_DIR/venv"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"

if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

log "Starting uninstall for $PROJECT_DIR"

if command -v systemctl >/dev/null 2>&1; then
  if systemctl list-unit-files | grep -q "^$SERVICE_NAME"; then
    log "Stopping $SERVICE_NAME"
    $SUDO systemctl stop "$SERVICE_NAME" || warn "Failed to stop $SERVICE_NAME"
    log "Disabling $SERVICE_NAME"
    $SUDO systemctl disable "$SERVICE_NAME" || warn "Failed to disable $SERVICE_NAME"
  else
    warn "$SERVICE_NAME not registered with systemd"
  fi

  if [[ -f "$SERVICE_PATH" ]]; then
    log "Removing systemd unit at $SERVICE_PATH"
    $SUDO rm -f "$SERVICE_PATH"
    log "Reloading systemd daemon"
    $SUDO systemctl daemon-reload || warn "Failed to reload systemd daemon"
  else
    warn "No systemd unit found at $SERVICE_PATH"
  fi
else
  warn "systemctl not found; skipping service removal"
fi

if [[ -d "$VENV_DIR" ]]; then
  if [[ "${KEEP_VENV:-}" == "1" ]]; then
    log "Keeping virtual environment at $VENV_DIR (KEEP_VENV=1)"
  else
    log "Removing virtual environment at $VENV_DIR"
    rm -rf "$VENV_DIR"
  fi
else
  warn "No virtual environment found at $VENV_DIR"
fi

log "Uninstall complete. Project files remain in $PROJECT_DIR"
