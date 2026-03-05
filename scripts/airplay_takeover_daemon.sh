#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ENV_PATH="$PROJECT_DIR/.env"
COMMON_SCRIPT="$SCRIPT_DIR/helpers/common.sh"
DESK_DISPLAY_SERVICE_NAME="${DESK_DISPLAY_SERVICE_NAME:-desk_display.service}"
AIRPLAY_RECEIVER_NAME="${AIRPLAY_RECEIVER_NAME:-Desk Display}"
AIRPLAY_PASSWORD="${AIRPLAY_PASSWORD:-}"
AIRPLAY_PIN="${AIRPLAY_PIN:-}"
AIRPLAY_EXTRA_ARGS="${AIRPLAY_EXTRA_ARGS:-}"
AIRPLAY_IDLE_RESUME_SECONDS="${AIRPLAY_IDLE_RESUME_SECONDS:-8}"
AIRPLAY_LOOP_SLEEP_SECONDS="${AIRPLAY_LOOP_SLEEP_SECONDS:-1}"
AIRPLAY_PORTS="${AIRPLAY_PORTS:-7000 7001 7002 7100}"
AIRPLAY_FULLSCREEN="${AIRPLAY_FULLSCREEN:-1}"
AIRPLAY_NATIVE_RESOLUTION="${AIRPLAY_NATIVE_RESOLUTION:-1}"

if [[ -f "$ENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_PATH"
fi

if [[ -f "$COMMON_SCRIPT" ]]; then
  # shellcheck disable=SC1090
  source "$COMMON_SCRIPT"
fi

AIRPLAY_RECEIVER_NAME="${DESK_DISPLAY_AIRPLAY_NAME:-$AIRPLAY_RECEIVER_NAME}"
AIRPLAY_PASSWORD="${DESK_DISPLAY_AIRPLAY_PASSWORD:-$AIRPLAY_PASSWORD}"
AIRPLAY_PIN="${DESK_DISPLAY_AIRPLAY_PIN:-$AIRPLAY_PIN}"
AIRPLAY_EXTRA_ARGS="${DESK_DISPLAY_AIRPLAY_ARGS:-$AIRPLAY_EXTRA_ARGS}"
AIRPLAY_IDLE_RESUME_SECONDS="${DESK_DISPLAY_AIRPLAY_IDLE_RESUME_SECONDS:-$AIRPLAY_IDLE_RESUME_SECONDS}"
AIRPLAY_LOOP_SLEEP_SECONDS="${DESK_DISPLAY_AIRPLAY_POLL_SECONDS:-$AIRPLAY_LOOP_SLEEP_SECONDS}"
AIRPLAY_FULLSCREEN="${DESK_DISPLAY_AIRPLAY_FULLSCREEN:-$AIRPLAY_FULLSCREEN}"
AIRPLAY_NATIVE_RESOLUTION="${DESK_DISPLAY_AIRPLAY_NATIVE_RESOLUTION:-$AIRPLAY_NATIVE_RESOLUTION}"

resolve_display_mode() {
  local mode=""

  if command -v xrandr >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" ]]; then
    mode=$(xrandr --current 2>/dev/null | awk '
      / connected primary / {
        split($4, parts, "+")
        print parts[1]
        exit
      }
      / connected / {
        split($3, parts, "+")
        if (parts[1] ~ /^[0-9]+x[0-9]+$/) {
          print parts[1]
          exit
        }
      }
    ')
    if [[ -n "$mode" ]]; then
      echo "$mode"
      return 0
    fi
  fi

  if command -v fbset >/dev/null 2>&1; then
    mode=$(fbset -s 2>/dev/null | awk '
      /geometry[[:space:]]+/ {
        if ($2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/) {
          print $2 "x" $3
          exit
        }
      }
    ')
    if [[ -n "$mode" ]]; then
      echo "$mode"
      return 0
    fi
  fi

  mode=$(find /sys/class/drm -maxdepth 2 -type f -name status -print 2>/dev/null \
    | while read -r status_path; do
        [[ "$(cat "$status_path" 2>/dev/null)" == "connected" ]] || continue
        connector_dir=$(dirname "$status_path")
        connector_mode="$connector_dir/mode"
        if [[ -s "$connector_mode" ]]; then
          cat "$connector_mode"
          break
        fi
      done \
    | awk '/^[0-9]+x[0-9]+$/ { print; exit }')
  if [[ -n "$mode" ]]; then
    echo "$mode"
    return 0
  fi

  if [[ -r /sys/class/graphics/fb0/virtual_size ]]; then
    mode=$(awk -F, '
      NF >= 2 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ {
        print $1 "x" $2
        exit
      }
    ' /sys/class/graphics/fb0/virtual_size 2>/dev/null)
  fi
  if [[ -n "$mode" ]]; then
    echo "$mode"
    return 0
  fi

  return 1
}

prepare_display_session_env() {
  local session_user="${DESK_DISPLAY_SESSION_USER:-$(whoami)}"

  if declare -F detect_desktop_session >/dev/null 2>&1; then
    detect_desktop_session "$session_user" || true
  fi

  if [[ -z "${XDG_RUNTIME_DIR:-}" ]]; then
    local uid runtime_dir
    uid=$(id -u "$session_user" 2>/dev/null || id -u)
    runtime_dir="/run/user/$uid"
    if [[ -d "$runtime_dir" ]]; then
      export XDG_RUNTIME_DIR="$runtime_dir"
    fi
  fi

  if [[ -n "${DISPLAY:-}" && -z "${XAUTHORITY:-}" ]]; then
    local home_dir xauth_path
    home_dir=$(getent passwd "$session_user" | cut -d: -f6)
    if [[ -z "$home_dir" ]]; then
      home_dir="$HOME"
    fi
    xauth_path="$home_dir/.Xauthority"
    if [[ -f "$xauth_path" ]]; then
      export XAUTHORITY="$xauth_path"
    fi
  fi
}

prepare_display_session_env

if ! command -v uxplay >/dev/null 2>&1; then
  echo "[ERROR] uxplay is not installed. Run scripts/update_airplay_dependencies.sh first." >&2
  exit 1
fi

if [[ -z "$AIRPLAY_PASSWORD" && -z "$AIRPLAY_PIN" ]]; then
  echo "[ERROR] DESK_DISPLAY_AIRPLAY_PASSWORD or DESK_DISPLAY_AIRPLAY_PIN is required." >&2
  exit 1
fi

run_systemctl_action() {
  local action="$1"
  local service="$2"

  if ! command -v systemctl >/dev/null 2>&1; then
    return 1
  fi

  if systemctl --user show "$service" >/dev/null 2>&1; then
    systemctl --user "$action" "$service"
    return $?
  fi

  if [[ $EUID -eq 0 ]]; then
    systemctl "$action" "$service"
    return $?
  fi

  if sudo -n true >/dev/null 2>&1; then
    sudo -n systemctl "$action" "$service"
    return $?
  fi

  return 1
}

restart_display() {
  if ! run_systemctl_action restart "$DESK_DISPLAY_SERVICE_NAME"; then
    echo "[WARN] Unable to restart $DESK_DISPLAY_SERVICE_NAME automatically." >&2
  fi
}

stop_display() {
  if ! run_systemctl_action stop "$DESK_DISPLAY_SERVICE_NAME"; then
    echo "[WARN] Unable to stop $DESK_DISPLAY_SERVICE_NAME automatically." >&2
  fi
}

count_clients() {
  local uxplay_pid="$1"
  local port_pattern

  port_pattern=$(echo "$AIRPLAY_PORTS" | tr ' ' '|')

  ss -Htanup 2>/dev/null | awk -v pid="$uxplay_pid" -v port_pattern="$port_pattern" '
    BEGIN {
      port_re = ":(" port_pattern ")$"
    }
    index($0, "pid=" pid ",") == 0 { next }
    $4 !~ port_re { next }
    $5 ~ /^\*:/ { next }
    $5 ~ /^\[::\]:/ { next }
    $1 == "LISTEN" { next }
    { c += 1 }
    END { print c + 0 }
  '
}
run_uxplay() {
  local detected_mode=""
  local args=(
    -n "$AIRPLAY_RECEIVER_NAME"
    -vsync no
  )

  if [[ "$AIRPLAY_FULLSCREEN" == "1" ]]; then
    args+=( -fs )
  fi

  if [[ "$AIRPLAY_NATIVE_RESOLUTION" == "1" ]]; then
    detected_mode=$(resolve_display_mode || true)
    if [[ -n "$detected_mode" ]]; then
      args+=( -s "$detected_mode" )
      echo "[INFO] Using native display resolution for AirPlay: $detected_mode"
    else
      echo "[WARN] Unable to detect native display resolution; uxplay default resolution will be used."
    fi
  fi

  if [[ -n "$AIRPLAY_PASSWORD" ]]; then
    args+=( -P "$AIRPLAY_PASSWORD" )
  fi
  if [[ -n "$AIRPLAY_PIN" ]]; then
    args+=( -pin "$AIRPLAY_PIN" )
  fi
  if [[ -n "$AIRPLAY_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra=( $AIRPLAY_EXTRA_ARGS )
    args+=( "${extra[@]}" )
  fi

  uxplay "${args[@]}"
}

active_takeover=0
last_connected_epoch=0

while true; do
  echo "[INFO] Starting background AirPlay receiver '$AIRPLAY_RECEIVER_NAME'."
  run_uxplay &
  uxplay_pid=$!

  while kill -0 "$uxplay_pid" 2>/dev/null; do
    client_count=$(count_clients "$uxplay_pid" || echo 0)
    now=$(date +%s)

    if [[ "$client_count" -gt 0 ]]; then
      last_connected_epoch="$now"
      if [[ "$active_takeover" -eq 0 ]]; then
        echo "[INFO] AirPlay client connected. Pausing Desk Display service."
        stop_display
        active_takeover=1
      fi
    elif [[ "$active_takeover" -eq 1 ]]; then
      if (( now - last_connected_epoch >= AIRPLAY_IDLE_RESUME_SECONDS )); then
        echo "[INFO] AirPlay clients disconnected. Resuming Desk Display service."
        restart_display
        active_takeover=0
      fi
    fi

    sleep "$AIRPLAY_LOOP_SLEEP_SECONDS"
  done

  wait "$uxplay_pid" || true

  if [[ "$active_takeover" -eq 1 ]]; then
    echo "[WARN] uxplay exited while takeover was active. Restoring Desk Display service."
    restart_display
    active_takeover=0
  fi

  echo "[WARN] uxplay exited. Restarting receiver in 2 seconds..."
  sleep 2
done
