#!/usr/bin/env bash
set -euo pipefail

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

action="${1:-}"
output="${DESK_DISPLAY_OUTPUT:-}"

if [[ "$output" != "framebuffer" ]]; then
  log "DESK_DISPLAY_OUTPUT is not framebuffer; skipping display-manager handling."
  exit 0
fi

if ! command -v systemctl >/dev/null 2>&1; then
  warn "systemctl not found; unable to manage display-manager."
  exit 0
fi

if ! systemctl list-unit-files --no-legend display-manager.service 2>/dev/null | grep -q display-manager.service; then
  warn "display-manager service not found; skipping."
  exit 0
fi

case "$action" in
  start)
    if systemctl is-active --quiet display-manager; then
      log "Stopping display-manager to free the framebuffer."
      systemctl stop display-manager
    else
      log "display-manager is not active."
    fi
    ;;
  stop)
    log "Starting display-manager to restore the desktop."
    systemctl start display-manager
    ;;
  *)
    warn "Unknown action '$action'. Use start or stop."
    exit 1
    ;;
esac
