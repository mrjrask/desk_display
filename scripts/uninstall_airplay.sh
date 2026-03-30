#!/usr/bin/env bash
set -euo pipefail

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

AIRPLAY_SERVICE_NAME="${AIRPLAY_SERVICE_NAME:-airplay_desk_display.service}"
AIRPLAY_SERVICE_PATH="/etc/systemd/system/$AIRPLAY_SERVICE_NAME"
MANIFEST_PATH="${AIRPLAY_MANIFEST_PATH:-/var/lib/desk-display-airplay/packages.txt}"
REMOVE_SHARED_PACKAGES="${REMOVE_AIRPLAY_SHARED_PACKAGES:-0}"

if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

remove_file_if_exists() {
  local target="$1"
  if [[ -e "$target" || -L "$target" ]]; then
    log "Removing $target"
    ${SUDO:-} rm -f "$target"
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
    seen=",${seen},${user_name}"

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

uninstall_manifest_packages() {
  if [[ ! -f "$MANIFEST_PATH" ]]; then
    warn "No AirPlay package manifest found at $MANIFEST_PATH"
    return 0
  fi

  mapfile -t manifest_packages < <(awk 'NF {print $1}' "$MANIFEST_PATH")
  if [[ ${#manifest_packages[@]} -eq 0 ]]; then
    log "AirPlay manifest exists but contains no package entries"
    remove_file_if_exists "$MANIFEST_PATH"
    return 0
  fi

  local -a installed_manifest_packages=()
  local pkg
  for pkg in "${manifest_packages[@]}"; do
    if dpkg -s "$pkg" >/dev/null 2>&1; then
      installed_manifest_packages+=("$pkg")
    fi
  done

  if [[ ${#installed_manifest_packages[@]} -gt 0 ]]; then
    log "Purging packages installed by AirPlay installer: ${installed_manifest_packages[*]}"
    ${SUDO:-} apt-get purge -y "${installed_manifest_packages[@]}" || warn "Failed to purge one or more AirPlay packages"
    ${SUDO:-} apt-get autoremove -y || warn "apt autoremove failed"
  else
    log "No manifest-tracked AirPlay packages are currently installed"
  fi

  remove_file_if_exists "$MANIFEST_PATH"
}

remove_shared_packages_if_requested() {
  if [[ "$REMOVE_SHARED_PACKAGES" != "1" ]]; then
    log "Skipping shared dependency purge (set REMOVE_AIRPLAY_SHARED_PACKAGES=1 to remove all AirPlay dependencies)."
    return 0
  fi

  local -a shared_packages=(
    uxplay
    avahi-daemon
    gstreamer1.0-tools
    gstreamer1.0-libav
    gstreamer1.0-plugins-base
    gstreamer1.0-plugins-good
    gstreamer1.0-plugins-bad
    gstreamer1.0-plugins-ugly
  )

  local -a installed_shared_packages=()
  local pkg
  for pkg in "${shared_packages[@]}"; do
    if dpkg -s "$pkg" >/dev/null 2>&1; then
      installed_shared_packages+=("$pkg")
    fi
  done

  if [[ ${#installed_shared_packages[@]} -eq 0 ]]; then
    log "No shared AirPlay dependency packages detected"
    return 0
  fi

  log "Purging shared AirPlay dependencies: ${installed_shared_packages[*]}"
  ${SUDO:-} apt-get purge -y "${installed_shared_packages[@]}" || warn "Failed to purge shared AirPlay dependencies"
  ${SUDO:-} apt-get autoremove -y || warn "apt autoremove failed"
}

main() {
  log "Uninstalling AirPlay components for Desk Display from $PROJECT_DIR"
  stop_disable_service_if_exists
  remove_airplay_launchers
  remove_airplay_scripts
  uninstall_manifest_packages
  remove_shared_packages_if_requested
  log "AirPlay uninstall complete."
}

main "$@"
