#!/usr/bin/env bash
set -euo pipefail

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

SERVICE_NAME="${SERVICE_NAME:-desk_display.service}"

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
