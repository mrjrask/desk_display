#!/usr/bin/env bash
set -euo pipefail

log() { printf '[AIRPLAY] %s\n' "$*"; }
warn() { printf '[AIRPLAY][WARN] %s\n' "$*"; }

load_env_file() {
  local env_file="$1"
  [[ -f "$env_file" ]] || return 0
  local raw_line line key value

  while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
    line="${raw_line#"${raw_line%%[![:space:]]*}"}"
    [[ -z "$line" ]] && continue
    [[ "${line:0:1}" == "#" ]] && continue
    [[ "$line" == *"="* ]] || continue

    key="${line%%=*}"
    value="${line#*=}"

    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"

    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

    if [[ ${#value} -ge 2 ]]; then
      if [[ "${value:0:1}" == "\"" && "${value: -1}" == "\"" ]]; then
        value="${value:1:${#value}-2}"
      elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
        value="${value:1:${#value}-2}"
      fi
    fi

    export "$key=$value"
  done < "$env_file"
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
AIRPLAY_VIDEO_SINK="${AIRPLAY_VIDEO_SINK:-}"
AIRPLAY_EXTRA_ARGS="${AIRPLAY_EXTRA_ARGS:-}"
DESK_DISPLAY_SERVICE_NAME="${DESK_DISPLAY_SERVICE_NAME:-desk_display.service}"
DESK_DISPLAY_USER_SERVICE_NAME="${DESK_DISPLAY_USER_SERVICE_NAME:-desk_display-kernel.service}"
DESK_DISPLAY_SESSION_USER="${DESK_DISPLAY_SESSION_USER:-${SUDO_USER:-$(whoami)}}"

ensure_runtime_dir() {
  if [[ -n "${XDG_RUNTIME_DIR:-}" && -d "${XDG_RUNTIME_DIR:-}" ]]; then
    return 0
  fi

  local uid runtime_dir fallback_runtime_dir
  uid=$(id -u 2>/dev/null || true)
  runtime_dir="/run/user/$uid"
  if [[ -d "$runtime_dir" ]]; then
    export XDG_RUNTIME_DIR="$runtime_dir"
    return 0
  fi

  fallback_runtime_dir="/tmp/desk-display-xdg-runtime-$uid"
  mkdir -p "$fallback_runtime_dir"
  chmod 700 "$fallback_runtime_dir" >/dev/null 2>&1 || true
  export XDG_RUNTIME_DIR="$fallback_runtime_dir"
  warn "XDG runtime dir /run/user/$uid is unavailable; using $XDG_RUNTIME_DIR"
}

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

desk_display_paused=0
AIRPLAY_VIDEO_SINK_AUTO_FORCED=0
AIRPLAY_AUTO_KMSSINK_ALLOWED=1

resume_desk_display_on_exit() {
  if [[ "$desk_display_paused" -eq 1 ]]; then
    log "AirPlay session ended unexpectedly. Resuming desk_display services."
    start_desk_display
    desk_display_paused=0
  fi
}

build_uxplay_command() {
  local -n cmd_ref=$1
  local -a args=(-fs -nh -n "$AIRPLAY_NAME" -s "$AIRPLAY_RESOLUTION")

  # On Raspberry Pi systems without a desktop session, autovideosink can pick a
  # sink that renders in a small window. Force kmssink in that case so AirPlay
  # content fills the screen.
  if [[ "$AIRPLAY_AUTO_KMSSINK_ALLOWED" -eq 1 && -z "$AIRPLAY_VIDEO_SINK" ]]; then
    if [[ -r /proc/device-tree/model ]] && tr -d '\0' </proc/device-tree/model | grep -qi 'raspberry pi'; then
      if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
        AIRPLAY_VIDEO_SINK="kmssink"
        AIRPLAY_VIDEO_SINK_AUTO_FORCED=1
      fi
    fi
  fi

  if [[ -n "$AIRPLAY_PIN" ]]; then
    args+=( -pin "$AIRPLAY_PIN" )
  fi

  if [[ -n "$AIRPLAY_DISPLAY" ]]; then
    args+=( -display "$AIRPLAY_DISPLAY" )
  fi

  if [[ -n "$AIRPLAY_VIDEO_SINK" ]]; then
    args+=( -vs "$AIRPLAY_VIDEO_SINK" )
  fi

  if [[ -n "$AIRPLAY_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra=( $AIRPLAY_EXTRA_ARGS )
    args+=("${extra[@]}")
  fi

  cmd_ref=( "$AIRPLAY_BIN" "${args[@]}" )
}

run_uxplay_session() {
  local connected=0
  local kms_renderer_failed=0
  local should_abort_for_fallback=0
  local -a uxplay_cmd=()
  build_uxplay_command uxplay_cmd

  log "Starting AirPlay takeover receiver at ${AIRPLAY_RESOLUTION}"
  log "Command: $(printf '%q ' "${uxplay_cmd[@]}")"

  while IFS= read -r line; do
    printf '%s\n' "$line"

    local lower
    lower=$(printf '%s' "$line" | tr '[:upper:]' '[:lower:]')

    if [[ "$lower" == *"begin streaming to gstreamer video pipeline"* || "$lower" == *"starting mirroring"* ]] && [[ $connected -eq 0 ]]; then
      connected=1
      log "AirPlay client connected. Stopping desk_display services."
      stop_desk_display
      desk_display_paused=1
    fi

    if [[ "$lower" == *"disconnected"* || "$lower" == *"teardown"* || "$lower" == *"stopped"* ]] && [[ $connected -eq 1 ]]; then
      connected=0
      log "AirPlay client disconnected. Starting desk_display services."
      start_desk_display
      desk_display_paused=0
    fi

    if [[ "$lower" == *"failed to initialize gstreamer video renderer"* || "$lower" == *"could not get allowed gstcaps of device"* ]]; then
      kms_renderer_failed=1
      if [[ "$AIRPLAY_VIDEO_SINK_AUTO_FORCED" -eq 1 && "$AIRPLAY_VIDEO_SINK" == "kmssink" ]]; then
        should_abort_for_fallback=1
        warn "Detected kmssink renderer failure. Restarting uxplay with default video sink."
        break
      fi
    fi
  done < <(stdbuf -oL -eL "${uxplay_cmd[@]}" 2>&1)

  if [[ "$should_abort_for_fallback" -eq 1 ]]; then
    return 1
  fi

  return "$kms_renderer_failed"
}

main() {
  ensure_runtime_dir
  trap resume_desk_display_on_exit EXIT INT TERM

  if run_uxplay_session; then
    return 0
  fi

  if [[ "$AIRPLAY_VIDEO_SINK_AUTO_FORCED" -eq 1 && "$AIRPLAY_VIDEO_SINK" == "kmssink" ]]; then
    warn "Auto-selected kmssink failed to initialize. Retrying with uxplay default video sink."
    AIRPLAY_AUTO_KMSSINK_ALLOWED=0
    AIRPLAY_VIDEO_SINK=""
    AIRPLAY_VIDEO_SINK_AUTO_FORCED=0
    run_uxplay_session
    return $?
  fi

  return 1
}

main "$@"
