#!/usr/bin/env bash
set -euo pipefail

log() { printf '[AIRPLAY] %s\n' "$*"; }
warn() { printf '[AIRPLAY][WARN] %s\n' "$*"; }

load_env_file() {
  local env_file="$1"
  [[ -f "$env_file" ]] || return 0
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

if [[ "${CONFIG_LOAD_DOTENV:-1}" != "0" ]]; then
  load_env_file "$PROJECT_DIR/.env"
fi

AIRPLAY_BIN="${AIRPLAY_BIN:-uxplay}"
AIRPLAY_NAME="${AIRPLAY_NAME:-Desk Display AirPlay}"
AIRPLAY_PIN="${AIRPLAY_PAIRING_CODE:-}"
AIRPLAY_RESOLUTION="${AIRPLAY_RESOLUTION:-${AIRPLAY_RESOLUTION_DEFAULT:-1920x1080}}"
AIRPLAY_DISPLAY="${AIRPLAY_DISPLAY:-}"
AIRPLAY_EXTRA_ARGS="${AIRPLAY_EXTRA_ARGS:-}"
DESK_DISPLAY_SERVICE_NAME="${DESK_DISPLAY_SERVICE_NAME:-desk_display.service}"
DESK_DISPLAY_USER_SERVICE_NAME="${DESK_DISPLAY_USER_SERVICE_NAME:-desk_display-kernel.service}"
DESK_DISPLAY_SESSION_USER="${DESK_DISPLAY_SESSION_USER:-${SUDO_USER:-$(whoami)}}"

if ! command -v "$AIRPLAY_BIN" >/dev/null 2>&1; then
  warn "AirPlay binary '$AIRPLAY_BIN' is not installed."
  exit 1
fi

stop_desk_display() {
  if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl stop "$DESK_DISPLAY_SERVICE_NAME" >/dev/null 2>&1 || true

    local uid
    uid=$(id -u "$DESK_DISPLAY_SESSION_USER" 2>/dev/null || true)
    if [[ -n "$uid" ]]; then
      sudo -u "$DESK_DISPLAY_SESSION_USER" XDG_RUNTIME_DIR="/run/user/$uid" \
        systemctl --user stop "$DESK_DISPLAY_USER_SERVICE_NAME" >/dev/null 2>&1 || true
    fi
  fi
}

start_desk_display() {
  if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl start "$DESK_DISPLAY_SERVICE_NAME" >/dev/null 2>&1 || true

    local uid
    uid=$(id -u "$DESK_DISPLAY_SESSION_USER" 2>/dev/null || true)
    if [[ -n "$uid" ]]; then
      sudo -u "$DESK_DISPLAY_SESSION_USER" XDG_RUNTIME_DIR="/run/user/$uid" \
        systemctl --user start "$DESK_DISPLAY_USER_SERVICE_NAME" >/dev/null 2>&1 || true
    fi
  fi
}

build_uxplay_command() {
  local -a args=(-fs -nh -n "$AIRPLAY_NAME" -s "$AIRPLAY_RESOLUTION")

  if [[ -n "$AIRPLAY_PIN" ]]; then
    args+=( -pin "$AIRPLAY_PIN" )
  fi

  if [[ -n "$AIRPLAY_DISPLAY" ]]; then
    args+=( -display "$AIRPLAY_DISPLAY" )
  fi

  if [[ -n "$AIRPLAY_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra=( $AIRPLAY_EXTRA_ARGS )
    args+=("${extra[@]}")
  fi

  printf '%q ' "$AIRPLAY_BIN" "${args[@]}"
}

main() {
  local uxplay_cmd
  uxplay_cmd=$(build_uxplay_command)

  log "Starting AirPlay takeover receiver at ${AIRPLAY_RESOLUTION}"
  log "Command: $uxplay_cmd"

  local connected=0
  # shellcheck disable=SC2086
  stdbuf -oL -eL $uxplay_cmd 2>&1 | while IFS= read -r line; do
    printf '%s\n' "$line"

    local lower
    lower=$(printf '%s' "$line" | tr '[:upper:]' '[:lower:]')

    if [[ "$lower" == *"connected"* ]] && [[ $connected -eq 0 ]]; then
      connected=1
      log "AirPlay client connected. Stopping desk_display services."
      stop_desk_display
    fi

    if [[ "$lower" == *"disconnected"* || "$lower" == *"connection closed"* ]] && [[ $connected -eq 1 ]]; then
      connected=0
      log "AirPlay client disconnected. Starting desk_display services."
      start_desk_display
    fi
  done
}

main "$@"
