#!/usr/bin/env bash
set -euo pipefail

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
# The service that drives the panel: desk_display_client.service on a client
# or combined install, desk_display.service when standalone.
if [[ -z "${SERVICE_NAME:-}" ]]; then
  SERVICE_NAME="desk_display.service"
  if command -v python3 >/dev/null 2>&1 && [[ -f "$PROJECT_DIR/install_modes.py" ]]; then
    case "$(python3 "$PROJECT_DIR/install_modes.py" detect --project-dir "$PROJECT_DIR" 2>/dev/null || true)" in
      client|combined) SERVICE_NAME="desk_display_client.service" ;;
    esac
  fi
fi

if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

if [[ -t 0 ]]; then
  read -r -p "This will stop the desktop display manager and switch to framebuffer output. Continue? [y/N]: " reply
  case "${reply,,}" in
    y|yes) ;;
    *) log "Launcher cancelled."; exit 0 ;;
  esac
else
  warn "No interactive terminal detected; proceeding without confirmation."
fi

if command -v systemctl >/dev/null 2>&1; then
  if systemctl is-active --quiet display-manager; then
    log "Stopping display-manager to free the framebuffer."
    $SUDO systemctl stop display-manager
  else
    log "display-manager is not active."
  fi

  log "Restarting $SERVICE_NAME"
  $SUDO systemctl restart "$SERVICE_NAME"
else
  warn "systemctl not found; unable to manage $SERVICE_NAME"
fi

log "Framebuffer launcher complete. Run scripts/restore_desktop.sh to bring back the desktop."
