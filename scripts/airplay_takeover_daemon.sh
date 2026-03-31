#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
HELPER_PATH="$PROJECT_DIR/scripts/helpers/airplay_common.sh"

if [[ ! -f "$HELPER_PATH" ]]; then
  echo "[AIRPLAY][ERROR] Missing helper library at $HELPER_PATH" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$HELPER_PATH"
init_sudo

if [[ "${CONFIG_LOAD_DOTENV:-1}" != "0" ]]; then
  load_env_file "$PROJECT_DIR/.env"
fi

AIRPLAY_BIN="${AIRPLAY_BIN:-uxplay}"
AIRPLAY_NAME="${AIRPLAY_NAME:-Desk Display AirPlay}"
AIRPLAY_PIN="${AIRPLAY_PAIRING_CODE:-}"
AIRPLAY_RESOLUTION="${AIRPLAY_RESOLUTION:-${AIRPLAY_RESOLUTION_DEFAULT:-800x480}}"
AIRPLAY_DISPLAY="${AIRPLAY_DISPLAY:-}"
AIRPLAY_EXTRA_ARGS="${AIRPLAY_EXTRA_ARGS:-}"
AIRPLAY_VIDEO_SINK="${AIRPLAY_VIDEO_SINK:-}"

DESK_DISPLAY_SERVICE_NAME="${DESK_DISPLAY_SERVICE_NAME:-desk_display.service}"
DESK_DISPLAY_USER_SERVICE_NAME="${DESK_DISPLAY_USER_SERVICE_NAME:-desk_display-kernel.service}"
DESK_DISPLAY_SESSION_USER="${DESK_DISPLAY_SESSION_USER:-${SUDO_USER:-$(whoami)}}"

SESSION_ACTIVE=0
CURRENT_SINK=""

ensure_runtime_dir() {
  if [[ -n "${XDG_RUNTIME_DIR:-}" && -d "${XDG_RUNTIME_DIR:-}" ]]; then
    return 0
  fi

  local uid runtime_dir fallback
  uid=$(id -u)
  runtime_dir="/run/user/$uid"
  if [[ -d "$runtime_dir" ]]; then
    export XDG_RUNTIME_DIR="$runtime_dir"
    return 0
  fi

  fallback="/tmp/desk-display-airplay-runtime-$uid"
  mkdir -p "$fallback"
  chmod 700 "$fallback" >/dev/null 2>&1 || true
  export XDG_RUNTIME_DIR="$fallback"
  log_warn "XDG runtime dir unavailable; using $XDG_RUNTIME_DIR"
}

stop_desk_display() {
  systemctl_safe stop "$DESK_DISPLAY_SERVICE_NAME" >/dev/null 2>&1 || true
  user_systemctl_safe "$DESK_DISPLAY_SESSION_USER" stop "$DESK_DISPLAY_USER_SERVICE_NAME" >/dev/null 2>&1 || true
}

start_desk_display() {
  systemctl_safe start "$DESK_DISPLAY_SERVICE_NAME" >/dev/null 2>&1 || true
  user_systemctl_safe "$DESK_DISPLAY_SESSION_USER" start "$DESK_DISPLAY_USER_SERVICE_NAME" >/dev/null 2>&1 || true
}

resume_on_exit() {
  if [[ "$SESSION_ACTIVE" -eq 1 ]]; then
    log_info "Resuming Desk Display services after AirPlay session exit"
    start_desk_display
    SESSION_ACTIVE=0
  fi
}

sink_available() {
  local sink="$1"
  [[ -z "$sink" ]] && return 0
  command -v gst-inspect-1.0 >/dev/null 2>&1 || return 0
  gst-inspect-1.0 "$sink" >/dev/null 2>&1
}

is_pi_headless() {
  if [[ -r /proc/device-tree/model ]] && tr -d '\0' </proc/device-tree/model | grep -qi 'raspberry pi'; then
    [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]
    return
  fi
  return 1
}

build_sink_candidates() {
  local -a sinks=()

  if [[ -n "$AIRPLAY_VIDEO_SINK" ]]; then
    sinks+=("$AIRPLAY_VIDEO_SINK")
  else
    if is_pi_headless && sink_available kmssink; then
      sinks+=("kmssink")
    fi
    sinks+=("autovideosink")
    if sink_available glimagesink; then
      sinks+=("glimagesink")
    fi
    if sink_available waylandsink; then
      sinks+=("waylandsink")
    fi
    if sink_available ximagesink; then
      sinks+=("ximagesink")
    fi
    sinks+=("")
  fi

  # de-dup while preserving order
  local -A seen=()
  local item seen_key
  for item in "${sinks[@]}"; do
    seen_key="${item:-__DEFAULT_SINK__}"
    if [[ -z "${seen[$seen_key]+x}" ]]; then
      seen[$seen_key]=1
      printf '%s\n' "$item"
    fi
  done
}

build_uxplay_cmd() {
  local sink="$1"
  local -n cmd_ref=$2

  cmd_ref=("$AIRPLAY_BIN" -fs -nh -n "$AIRPLAY_NAME" -s "$AIRPLAY_RESOLUTION")

  if [[ -n "$AIRPLAY_PIN" ]]; then
    cmd_ref+=(-pin "$AIRPLAY_PIN")
  fi
  if [[ -n "$AIRPLAY_DISPLAY" ]]; then
    cmd_ref+=(-display "$AIRPLAY_DISPLAY")
  fi
  if [[ -n "$sink" ]]; then
    cmd_ref+=(-vs "$sink")
  fi
  if [[ -n "$AIRPLAY_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra=( $AIRPLAY_EXTRA_ARGS )
    cmd_ref+=("${extra[@]}")
  fi
}

run_once() {
  local sink="$1"
  local -a cmd=()
  local connected=0
  local sink_failed=0

  build_uxplay_cmd "$sink" cmd
  CURRENT_SINK="$sink"

  if [[ -n "$sink" ]]; then
    log_info "Starting uxplay with sink '$sink' at ${AIRPLAY_RESOLUTION}"
  else
    log_info "Starting uxplay with uxplay default sink at ${AIRPLAY_RESOLUTION}"
  fi
  log_info "Command: $(printf '%q ' "${cmd[@]}")"

  coproc UXP { stdbuf -oL -eL "${cmd[@]}" 2>&1; }
  local ux_pid=$UXP_PID

  while IFS= read -r line <&"${UXP[0]}"; do
    printf '%s\n' "$line"
    local lower
    lower=$(printf '%s' "$line" | tr '[:upper:]' '[:lower:]')

    if [[ "$lower" == *"begin streaming to gstreamer video pipeline"* || "$lower" == *"starting mirroring"* ]] && [[ $connected -eq 0 ]]; then
      connected=1
      stop_desk_display
      SESSION_ACTIVE=1
      log_info "AirPlay client connected; Desk Display services paused"
    fi

    if [[ "$lower" == *"disconnected"* || "$lower" == *"teardown"* || "$lower" == *"stopped"* ]] && [[ $connected -eq 1 ]]; then
      connected=0
      start_desk_display
      SESSION_ACTIVE=0
      log_info "AirPlay client disconnected; Desk Display services resumed"
    fi

    if [[ "$lower" == *"failed to initialize gstreamer video renderer"* || "$lower" == *"could not get allowed gstcaps of device"* ]]; then
      sink_failed=1
      log_warn "Detected renderer initialization failure for sink '${sink:-default}'"
      kill "$ux_pid" >/dev/null 2>&1 || true
      break
    fi
  done

  wait "$ux_pid" >/dev/null 2>&1 || true

  if [[ $sink_failed -eq 1 ]]; then
    return 2
  fi

  return 0
}

main() {
  ensure_runtime_dir

  if ! command -v "$AIRPLAY_BIN" >/dev/null 2>&1; then
    log_error "AirPlay binary '$AIRPLAY_BIN' is not installed"
    exit 1
  fi

  trap resume_on_exit EXIT INT TERM

  while true; do
    local used_fallback=0
    while IFS= read -r sink_candidate; do
      if run_once "$sink_candidate"; then
        used_fallback=1
        break
      fi

      log_warn "Retrying uxplay with next sink candidate"
      sleep 1
    done < <(build_sink_candidates)

    if [[ $used_fallback -eq 0 ]]; then
      log_warn "No working sink candidate succeeded; waiting before full retry"
    fi

    sleep 2
  done
}

main "$@"
