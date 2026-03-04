#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
COMMON_SCRIPT="$PROJECT_DIR/scripts/helpers/common.sh"

if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "[ERROR] Missing helper script: $COMMON_SCRIPT" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$COMMON_SCRIPT"

if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

detect_codename() {
  if command -v lsb_release >/dev/null 2>&1; then
    lsb_release -sc
    return 0
  fi

  if [[ -f /etc/os-release ]]; then
    # shellcheck source=/dev/null
    source /etc/os-release
    if [[ -n "${VERSION_CODENAME:-}" ]]; then
      echo "$VERSION_CODENAME"
      return 0
    fi
  fi

  echo "bookworm"
}

EXPECTED_CODENAME="${EXPECTED_CODENAME:-$(detect_codename)}"
export EXPECTED_CODENAME

log "Installing/updating all Desk Display apt dependencies (existing + AirPlay support)..."
install_apt_packages
ensure_avahi_daemon_running

install_airplay_launcher "$PROJECT_DIR" "$SERVICE_USER"
ensure_executable "$PROJECT_DIR/scripts/airplay_mode.sh"
ensure_executable "$PROJECT_DIR/scripts/airplay_takeover_daemon.sh"

if [[ "${DESK_DISPLAY_AIRPLAY_ALWAYS_ON:-1}" == "1" ]]; then
  install_airplay_takeover_service "$PROJECT_DIR" "$SERVICE_USER"
fi

log "AirPlay dependencies installed."
log "Set DESK_DISPLAY_AIRPLAY_PASSWORD or DESK_DISPLAY_AIRPLAY_PIN in $PROJECT_DIR/.env."
log "The background takeover service is now enabled by default (DESK_DISPLAY_AIRPLAY_ALWAYS_ON=1)."
log "You can still run on-demand mode manually via: $PROJECT_DIR/scripts/airplay_mode.sh"
