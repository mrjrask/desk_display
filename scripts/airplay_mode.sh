#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
ENV_PATH="$PROJECT_DIR/.env"
SERVICE_NAME="${DESK_DISPLAY_SERVICE_NAME:-desk_display.service}"
AIRPLAY_RECEIVER_NAME="${AIRPLAY_RECEIVER_NAME:-Desk Display}"
AIRPLAY_PASSWORD="${AIRPLAY_PASSWORD:-}"
AIRPLAY_PIN="${AIRPLAY_PIN:-}"
AIRPLAY_EXTRA_ARGS="${AIRPLAY_EXTRA_ARGS:-}"
AIRPLAY_FULLSCREEN="${AIRPLAY_FULLSCREEN:-1}"
AIRPLAY_NATIVE_RESOLUTION="${AIRPLAY_NATIVE_RESOLUTION:-1}"

if [[ -f "$ENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_PATH"
fi

if [[ -n "${DESK_DISPLAY_AIRPLAY_NAME:-}" ]]; then
  AIRPLAY_RECEIVER_NAME="$DESK_DISPLAY_AIRPLAY_NAME"
fi
if [[ -n "${DESK_DISPLAY_AIRPLAY_PASSWORD:-}" ]]; then
  AIRPLAY_PASSWORD="$DESK_DISPLAY_AIRPLAY_PASSWORD"
fi
if [[ -n "${DESK_DISPLAY_AIRPLAY_PIN:-}" ]]; then
  AIRPLAY_PIN="$DESK_DISPLAY_AIRPLAY_PIN"
fi
if [[ -n "${DESK_DISPLAY_AIRPLAY_ARGS:-}" ]]; then
  AIRPLAY_EXTRA_ARGS="$DESK_DISPLAY_AIRPLAY_ARGS"
fi
if [[ -n "${DESK_DISPLAY_AIRPLAY_FULLSCREEN:-}" ]]; then
  AIRPLAY_FULLSCREEN="$DESK_DISPLAY_AIRPLAY_FULLSCREEN"
fi
if [[ -n "${DESK_DISPLAY_AIRPLAY_NATIVE_RESOLUTION:-}" ]]; then
  AIRPLAY_NATIVE_RESOLUTION="$DESK_DISPLAY_AIRPLAY_NATIVE_RESOLUTION"
fi

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

if ! command -v uxplay >/dev/null 2>&1; then
  echo "[ERROR] uxplay is not installed. Run scripts/update_airplay_dependencies.sh first." >&2
  exit 1
fi

if [[ -z "$AIRPLAY_PASSWORD" && -z "$AIRPLAY_PIN" ]]; then
  echo "[ERROR] AirPlay protection is required. Set DESK_DISPLAY_AIRPLAY_PASSWORD or DESK_DISPLAY_AIRPLAY_PIN in .env." >&2
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

echo "[INFO] Stopping $SERVICE_NAME while AirPlay mode is active..."
if ! run_systemctl_action stop "$SERVICE_NAME"; then
  echo "[WARN] Unable to stop $SERVICE_NAME automatically. AirPlay may overlap with Desk Display output." >&2
fi

cleanup() {
  local rc=$?
  echo "[INFO] Restarting $SERVICE_NAME so scheduled screens resume..."
  if ! run_systemctl_action restart "$SERVICE_NAME"; then
    echo "[WARN] Unable to restart $SERVICE_NAME automatically. Restart it manually after AirPlay exits." >&2
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

uxplay_args=(
  -n "$AIRPLAY_RECEIVER_NAME"
  -vsync no
)

if [[ "$AIRPLAY_FULLSCREEN" == "1" ]]; then
  uxplay_args+=( -fs )
fi

if [[ "$AIRPLAY_NATIVE_RESOLUTION" == "1" ]]; then
  detected_mode=$(resolve_display_mode || true)
  if [[ -n "$detected_mode" ]]; then
    uxplay_args+=( -s "$detected_mode" )
    echo "[INFO] Using native display resolution for AirPlay: $detected_mode"
  else
    echo "[WARN] Unable to detect native display resolution; uxplay default resolution will be used."
  fi
fi

if [[ -n "$AIRPLAY_PASSWORD" ]]; then
  uxplay_args+=( -P "$AIRPLAY_PASSWORD" )
fi
if [[ -n "$AIRPLAY_PIN" ]]; then
  uxplay_args+=( -pin "$AIRPLAY_PIN" )
fi

if [[ -n "$AIRPLAY_EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra=( $AIRPLAY_EXTRA_ARGS )
  uxplay_args+=( "${extra[@]}" )
fi

echo "[INFO] Starting AirPlay receiver '$AIRPLAY_RECEIVER_NAME'."
echo "[INFO] Disconnect from AirPlay (or press Ctrl+C) to resume Desk Display screens."
exec uxplay "${uxplay_args[@]}"
