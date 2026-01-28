#!/usr/bin/env bash
set -euo pipefail

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

SERVICE_NAME="desk_display.service"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
COMMON_SCRIPT="$PROJECT_DIR/scripts/install_common.sh"

if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

if command -v systemctl >/dev/null 2>&1; then
  if systemctl list-unit-files | grep -q "^$SERVICE_NAME"; then
    log "Stopping $SERVICE_NAME"
    $SUDO systemctl stop "$SERVICE_NAME" || warn "Failed to stop $SERVICE_NAME"
  fi
fi

log "Starting uninstall for $PROJECT_DIR"

if [[ -f "$COMMON_SCRIPT" ]]; then
  # shellcheck source=/dev/null
  source "$COMMON_SCRIPT"
fi

VENV_DIR="$PROJECT_DIR/venv"
EXISTING_VENV=$(detect_existing_venv "$PROJECT_DIR" || true)
if [[ -n "$EXISTING_VENV" ]]; then
  VENV_DIR="$EXISTING_VENV"
fi

if command -v systemctl >/dev/null 2>&1; then
  if systemctl list-unit-files | grep -q "^$SERVICE_NAME"; then
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
  keep_choice="${KEEP_VENV:-}"

  if [[ -z "$keep_choice" && -t 0 ]]; then
    read -r -p "Keep virtual environment at $VENV_DIR? [y/N]: " keep_reply
    case "${keep_reply,,}" in
      y|yes) keep_choice="yes" ;;
      *) keep_choice="no" ;;
    esac
  fi

  if [[ "$keep_choice" == "1" || "$keep_choice" == "yes" ]]; then
    log "Keeping virtual environment at $VENV_DIR"
  else
    log "Removing virtual environment at $VENV_DIR"
    rm -rf "$VENV_DIR"
  fi
else
  warn "No virtual environment found at $VENV_DIR"
fi

log "Uninstall complete. Project files remain in $PROJECT_DIR"
