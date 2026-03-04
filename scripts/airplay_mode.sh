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

if ! command -v uxplay >/dev/null 2>&1; then
  echo "[ERROR] uxplay is not installed. Run scripts/update_airplay_dependencies.sh first." >&2
  exit 1
fi

if [[ -z "$AIRPLAY_PASSWORD" && -z "$AIRPLAY_PIN" ]]; then
  echo "[ERROR] AirPlay protection is required. Set DESK_DISPLAY_AIRPLAY_PASSWORD or DESK_DISPLAY_AIRPLAY_PIN in .env." >&2
  exit 1
fi

SUDO=""
if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
fi

if command -v systemctl >/dev/null 2>&1; then
  echo "[INFO] Stopping $SERVICE_NAME while AirPlay mode is active..."
  ${SUDO:-} systemctl stop "$SERVICE_NAME" || true
fi

cleanup() {
  local rc=$?
  if command -v systemctl >/dev/null 2>&1; then
    echo "[INFO] Restarting $SERVICE_NAME so scheduled screens resume..."
    ${SUDO:-} systemctl restart "$SERVICE_NAME" || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

uxplay_args=(
  -n "$AIRPLAY_RECEIVER_NAME"
  -vsync no
)

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
