#!/usr/bin/env bash
set -euo pipefail

SUDO="${SUDO:-}"
SERVICES=("desk_display.service")

if systemctl list-unit-files --type=service --no-legend 2>/dev/null | awk '{print $1}' | grep -qx "desk_display_waveshare_oled.service"; then
  SERVICES+=("desk_display_waveshare_oled.service")
fi

log() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

restart_service() {
  local service="$1"

  log "Restarting $service using stop/start fallback..."

  $SUDO systemctl reset-failed "$service" >/dev/null 2>&1 || true

  if ! $SUDO systemctl stop "$service"; then
    warn "Stop failed for $service; continuing with start attempt."
  fi

  for _ in {1..20}; do
    if ! $SUDO systemctl is-active --quiet "$service"; then
      break
    fi
    sleep 0.25
  done

  if ! $SUDO systemctl start "$service"; then
    warn "Start failed for $service. Showing recent status and logs."
    $SUDO systemctl --no-pager --full status "$service" || true
    $SUDO journalctl -u "$service" -n 80 --no-pager || true
    return 1
  fi

  log "$service is running."
}

main() {
  local failed=0

  for service in "${SERVICES[@]}"; do
    if ! restart_service "$service"; then
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    exit 1
  fi

  log "Done."
}

main "$@"
