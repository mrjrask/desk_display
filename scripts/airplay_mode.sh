#!/usr/bin/env bash
set -euo pipefail

AIRPLAY_SERVICE_NAME="${AIRPLAY_SERVICE_NAME:-airplay_desk_display.service}"

if command -v systemctl >/dev/null 2>&1; then
  if [[ $EUID -ne 0 ]]; then
    sudo systemctl enable --now "$AIRPLAY_SERVICE_NAME"
    sudo systemctl restart "$AIRPLAY_SERVICE_NAME"
    sudo systemctl status --no-pager "$AIRPLAY_SERVICE_NAME"
  else
    systemctl enable --now "$AIRPLAY_SERVICE_NAME"
    systemctl restart "$AIRPLAY_SERVICE_NAME"
    systemctl status --no-pager "$AIRPLAY_SERVICE_NAME"
  fi
else
  echo "systemctl is required for AirPlay mode." >&2
  exit 1
fi
