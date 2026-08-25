#!/usr/bin/env bash
# Restart desk_display's systemd services one at a time, in an order that
# respects how the services depend on each other:
#
#   1. desk_display_adsb_collector.service   - writes the ADS-B cache that
#                                               main.py's screens read.
#   2. feed_server_desk_display.service      - serves/collects screenshot
#                                               data other services push to.
#   3. desk_display.service                  - the main renderer; reads the
#                                               caches above and draws to the
#                                               framebuffer/screenshot dir.
#   4. waveshare-fbcp.service                - mirrors desk_display's
#                                               framebuffer onto the physical
#                                               panel, so it should come back
#                                               after desk_display has fresh
#                                               frames to copy.
#   5. desk_display_waveshare_oled.service   - side status OLED helper for
#                                               the Waveshare HAT; independent
#                                               of the framebuffer but part of
#                                               the same display stack.
#   6. screenshot_uploader_desk_display.service - watches the screenshot dir
#                                               desk_display writes and POSTs
#                                               changes to the feed server.
#   7. config_ui_desk_display.service        - web config UI; independent,
#                                               safe to cycle any time.
#   8. airplay_desk_display.service          - AirPlay receiver add-on;
#                                               independent, least urgent.
#
# Only services actually installed on this machine are restarted; the rest
# are skipped. By default every installed service is restarted in the order
# above. Pass one or more service names to restart just those (still one at
# a time, in the order given below rather than the order typed).
set -euo pipefail

SUDO="${SUDO:-}"
if [[ -z "$SUDO" && "${EUID:-$(id -u)}" -ne 0 ]]; then
  SUDO="sudo"
fi

ORDERED_SERVICES=(
  desk_display_adsb_collector.service
  feed_server_desk_display.service
  desk_display.service
  waveshare-fbcp.service
  desk_display_waveshare_oled.service
  screenshot_uploader_desk_display.service
  config_ui_desk_display.service
  airplay_desk_display.service
)

log() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

print_usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [service ...]
  $(basename "$0") --list
  $(basename "$0") --help

With no arguments, restarts every installed desk_display service, one at a
time, in dependency order (see the comment block at the top of this script).

With one or more service names, restarts only those (still one at a time,
in the order below rather than the order given), skipping any that aren't
installed on this machine.

Known services, in restart order:
$(printf '  %s\n' "${ORDERED_SERVICES[@]}")
USAGE
}

is_known() {
  local svc="$1"
  local candidate
  for candidate in "${ORDERED_SERVICES[@]}"; do
    if [[ "$candidate" == "$svc" ]]; then
      return 0
    fi
  done
  return 1
}

is_installed() {
  local svc="$1"
  systemctl list-unit-files --type=service --no-legend 2>/dev/null | awk '{print $1}' | grep -qx "$svc"
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
  local -a requested=("$@")

  for arg in "${requested[@]}"; do
    case "$arg" in
      -h|--help)
        print_usage
        exit 0
        ;;
      --list)
        printf '%s\n' "${ORDERED_SERVICES[@]}"
        exit 0
        ;;
    esac
  done

  local -a targets=()

  if [[ ${#requested[@]} -eq 0 ]]; then
    for svc in "${ORDERED_SERVICES[@]}"; do
      if is_installed "$svc"; then
        targets+=("$svc")
      fi
    done
  else
    for svc in "${requested[@]}"; do
      if ! is_known "$svc"; then
        warn "Unknown service: $svc (run '$(basename "$0") --list' to see known services); skipping."
        continue
      fi
      if ! is_installed "$svc"; then
        warn "$svc is not installed on this system; skipping."
        continue
      fi
      targets+=("$svc")
    done
    # Restart in dependency order regardless of the order the caller typed.
    local -a ordered_targets=()
    for svc in "${ORDERED_SERVICES[@]}"; do
      for chosen in "${targets[@]}"; do
        if [[ "$chosen" == "$svc" ]]; then
          ordered_targets+=("$svc")
          break
        fi
      done
    done
    targets=("${ordered_targets[@]}")
  fi

  if [[ ${#targets[@]} -eq 0 ]]; then
    warn "No matching installed services to restart."
    exit 0
  fi

  log "Restart order: ${targets[*]}"

  local failed=0
  for service in "${targets[@]}"; do
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
