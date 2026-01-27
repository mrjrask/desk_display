#!/usr/bin/env bash
set -euo pipefail

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

if command -v systemctl >/dev/null 2>&1; then
  log "Starting display-manager."
  $SUDO systemctl start display-manager
else
  warn "systemctl not found; unable to start display-manager."
fi
