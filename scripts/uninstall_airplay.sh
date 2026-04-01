#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

log_info() { printf '[AIRPLAY][INFO] %s\n' "$*"; }
log_warn() { printf '[AIRPLAY][WARN] %s\n' "$*"; }

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

AIRPLAY_SERVICE_NAME="${AIRPLAY_SERVICE_NAME:-airplay_desk_display.service}"
AIRPLAY_SERVICE_PATH="/etc/systemd/system/$AIRPLAY_SERVICE_NAME"
MANIFEST_PATH="${AIRPLAY_MANIFEST_PATH:-/var/lib/desk-display-airplay/installed_packages.txt}"
REMOVE_SHARED_PACKAGES="${REMOVE_AIRPLAY_SHARED_PACKAGES:-0}"

shared_packages=(
  uxplay
  avahi-daemon
  libnss-mdns
  gstreamer1.0-tools
  gstreamer1.0-libav
  gstreamer1.0-plugins-base
  gstreamer1.0-plugins-good
  gstreamer1.0-plugins-bad
  gstreamer1.0-plugins-ugly
)

remove_path_if_exists() {
  local target="$1"
  if [[ -e "$target" || -L "$target" ]]; then
    log_info "Removing $target"
    ${SUDO:-} rm -f "$target"
  fi
}

remove_service() {
  if command -v systemctl >/dev/null 2>&1; then
    if systemctl list-unit-files | grep -q "^${AIRPLAY_SERVICE_NAME}"; then
      ${SUDO:-} systemctl stop "$AIRPLAY_SERVICE_NAME" || log_warn "Unable to stop $AIRPLAY_SERVICE_NAME"
      ${SUDO:-} systemctl disable "$AIRPLAY_SERVICE_NAME" || log_warn "Unable to disable $AIRPLAY_SERVICE_NAME"
    fi
  fi

  if [[ -f "$AIRPLAY_SERVICE_PATH" ]]; then
    remove_path_if_exists "$AIRPLAY_SERVICE_PATH"
    if command -v systemctl >/dev/null 2>&1; then
      ${SUDO:-} systemctl daemon-reload || log_warn "systemctl daemon-reload failed"
    fi
  fi
}

service_user_home() {
  local user_name="$1"
  local resolved
  resolved=$(getent passwd "$user_name" | cut -d: -f6)
  if [[ -n "$resolved" ]]; then
    printf '%s\n' "$resolved"
  else
    printf '/home/%s\n' "$user_name"
  fi
}

remove_launchers() {
  local -a users=("${SUDO_USER:-}" "$(whoami)")
  local user_name home_dir
  local seen=","

  for user_name in "${users[@]}"; do
    [[ -n "$user_name" ]] || continue
    if [[ "$seen" == *",$user_name,"* ]]; then
      continue
    fi
    seen+="$user_name,"

    home_dir=$(service_user_home "$user_name")
    remove_path_if_exists "$home_dir/.local/share/applications/desk-display-airplay.desktop"
    remove_path_if_exists "$home_dir/Desktop/Desk Display AirPlay Mode.desktop"
  done
}

purge_manifest_packages() {
  [[ -f "$MANIFEST_PATH" ]] || return 0

  mapfile -t manifest_packages < <(awk 'NF {print $1}' "$MANIFEST_PATH")
  if [[ ${#manifest_packages[@]} -gt 0 ]]; then
    local pkg
    local -a installed=()
    for pkg in "${manifest_packages[@]}"; do
      if dpkg -s "$pkg" >/dev/null 2>&1; then
        installed+=("$pkg")
      fi
    done

    if [[ ${#installed[@]} -gt 0 ]]; then
      log_info "Purging manifest packages: ${installed[*]}"
      ${SUDO:-} apt-get purge -y "${installed[@]}"
      ${SUDO:-} apt-get autoremove -y
    fi
  fi

  remove_path_if_exists "$MANIFEST_PATH"
}

purge_shared_packages_if_requested() {
  if [[ "$REMOVE_SHARED_PACKAGES" != "1" ]]; then
    log_info "Skipping shared package purge. Set REMOVE_AIRPLAY_SHARED_PACKAGES=1 to force removal."
    return 0
  fi

  local pkg
  local -a installed=()
  for pkg in "${shared_packages[@]}"; do
    if dpkg -s "$pkg" >/dev/null 2>&1; then
      installed+=("$pkg")
    fi
  done

  if [[ ${#installed[@]} -gt 0 ]]; then
    log_info "Purging shared AirPlay packages: ${installed[*]}"
    ${SUDO:-} apt-get purge -y "${installed[@]}"
    ${SUDO:-} apt-get autoremove -y
  fi
}

main() {
  log_info "Uninstalling Desk Display AirPlay components"
  remove_service
  remove_launchers
  purge_manifest_packages
  purge_shared_packages_if_requested
  log_info "AirPlay uninstall complete"
}

main "$@"
