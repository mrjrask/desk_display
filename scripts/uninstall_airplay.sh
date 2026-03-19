#!/usr/bin/env bash
set -euo pipefail

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

AIRPLAY_SERVICE_NAME="${AIRPLAY_SERVICE_NAME:-desk_display_airplay.service}"
AIRPLAY_SERVICE_PATH="/etc/systemd/system/$AIRPLAY_SERVICE_NAME"

if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

remove_file_if_exists() {
  local target="$1"
  if [[ -e "$target" || -L "$target" ]]; then
    log "Removing $target"
    if [[ -n "$SUDO" ]]; then
      $SUDO rm -f "$target"
    else
      rm -f "$target"
    fi
  fi
}

stop_disable_service_if_exists() {
  if ! command -v systemctl >/dev/null 2>&1; then
    warn "systemctl not found; skipping AirPlay service disable/remove."
    return 0
  fi

  if systemctl list-unit-files | grep -q "^${AIRPLAY_SERVICE_NAME}"; then
    log "Stopping $AIRPLAY_SERVICE_NAME"
    ${SUDO:-} systemctl stop "$AIRPLAY_SERVICE_NAME" || warn "Failed to stop $AIRPLAY_SERVICE_NAME"
    log "Disabling $AIRPLAY_SERVICE_NAME"
    ${SUDO:-} systemctl disable "$AIRPLAY_SERVICE_NAME" || warn "Failed to disable $AIRPLAY_SERVICE_NAME"
  else
    warn "$AIRPLAY_SERVICE_NAME not registered with systemd"
  fi

  if [[ -f "$AIRPLAY_SERVICE_PATH" ]]; then
    log "Removing systemd unit at $AIRPLAY_SERVICE_PATH"
    ${SUDO:-} rm -f "$AIRPLAY_SERVICE_PATH"
    log "Reloading systemd daemon"
    ${SUDO:-} systemctl daemon-reload || warn "Failed to reload systemd daemon"
  fi
}

remove_airplay_launchers() {
  local users=()
  if [[ -n "${SUDO_USER:-}" ]]; then
    users+=("$SUDO_USER")
  fi
  users+=("$(whoami)")

  local seen=""
  for user_name in "${users[@]}"; do
    [[ -n "$user_name" ]] || continue
    if [[ ",$seen," == *",$user_name,"* ]]; then
      continue
    fi
    seen="${seen},${user_name}"

    local home_dir
    home_dir=$(getent passwd "$user_name" | cut -d: -f6)
    if [[ -z "$home_dir" ]]; then
      home_dir="/home/$user_name"
    fi

    remove_file_if_exists "$home_dir/.local/share/applications/desk-display-airplay.desktop"
    remove_file_if_exists "$home_dir/Desktop/Desk Display AirPlay Mode.desktop"
  done
}

remove_airplay_scripts() {
  remove_file_if_exists "$PROJECT_DIR/scripts/airplay_mode.sh"
  remove_file_if_exists "$PROJECT_DIR/scripts/airplay_takeover_daemon.sh"
  remove_file_if_exists "$PROJECT_DIR/scripts/update_airplay_dependencies.sh"
}

main() {
  log "Uninstalling AirPlay components for Desk Display from $PROJECT_DIR"
  stop_disable_service_if_exists
  remove_airplay_launchers
  remove_airplay_scripts
  log "AirPlay uninstall complete."
}

main "$@"
