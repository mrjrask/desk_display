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

SERVICE_USER="${AIRPLAY_SERVICE_USER:-${SUDO_USER:-$(whoami)}}"
AIRPLAY_SERVICE_NAME="${AIRPLAY_SERVICE_NAME:-airplay_desk_display.service}"
AIRPLAY_SERVICE_PATH="/etc/systemd/system/$AIRPLAY_SERVICE_NAME"
MANIFEST_PATH="${AIRPLAY_MANIFEST_PATH:-/var/lib/desk-display-airplay/installed_packages.txt}"
MODE_SCRIPT="$PROJECT_DIR/scripts/airplay_mode.sh"
DAEMON_SCRIPT="$PROJECT_DIR/scripts/airplay_takeover_daemon.sh"
UNINSTALL_SCRIPT="$PROJECT_DIR/scripts/uninstall_airplay.sh"

required_packages=(
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

validate_files() {
  local f
  for f in "$MODE_SCRIPT" "$DAEMON_SCRIPT" "$UNINSTALL_SCRIPT"; do
    if [[ ! -f "$f" ]]; then
      log_error "Required script is missing: $f"
      exit 1
    fi
    ensure_executable "$f"
  done
}

install_packages() {
  log_info "Refreshing apt package metadata"
  ${SUDO:-} apt-get update

  local pkg
  local -a missing=()
  for pkg in "${required_packages[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
      missing+=("$pkg")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    log_info "Installing packages: ${missing[*]}"
    ${SUDO:-} apt-get install -y "${missing[@]}"
  else
    log_info "All AirPlay dependencies are already installed"
  fi

  ${SUDO:-} mkdir -p "$(dirname -- "$MANIFEST_PATH")"
  if [[ ${#missing[@]} -gt 0 ]]; then
    printf '%s\n' "${missing[@]}" | ${SUDO:-} tee "$MANIFEST_PATH" >/dev/null
  else
    : | ${SUDO:-} tee "$MANIFEST_PATH" >/dev/null
  fi

  systemctl_safe enable --now avahi-daemon || log_warn "Unable to enable avahi-daemon automatically"
}

write_service_unit() {
  local resolution
  resolution=$(detect_display_resolution)

  local unit
  unit=$(cat <<UNIT
[Unit]
Description=Desk Display AirPlay takeover service
Wants=network-online.target avahi-daemon.service
After=network-online.target avahi-daemon.service

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$PROJECT_DIR
Environment=PROJECT_DIR=$PROJECT_DIR
Environment=CONFIG_LOAD_DOTENV=1
Environment=AIRPLAY_RESOLUTION_DEFAULT=$resolution
ExecStart=/bin/bash -lc '$DAEMON_SCRIPT'
Restart=always
RestartSec=3
StartLimitIntervalSec=0

[Install]
WantedBy=multi-user.target
UNIT
)

  log_info "Writing service unit: $AIRPLAY_SERVICE_PATH"
  write_file_as_root "$AIRPLAY_SERVICE_PATH" "$unit"
  systemctl_safe daemon-reload
  systemctl_safe enable --now "$AIRPLAY_SERVICE_NAME"
}

install_launcher() {
  local home_dir app_dir desktop_dir launcher_path desktop_copy
  home_dir=$(service_user_home "$SERVICE_USER")
  app_dir="$home_dir/.local/share/applications"
  desktop_dir="$home_dir/Desktop"
  launcher_path="$app_dir/desk-display-airplay.desktop"
  desktop_copy="$desktop_dir/Desk Display AirPlay Mode.desktop"

  local entry
  entry=$(cat <<DESKTOP
[Desktop Entry]
Type=Application
Name=Desk Display AirPlay Mode
Comment=Start Desk Display AirPlay takeover receiver
Exec=/bin/bash -lc '$MODE_SCRIPT'
Terminal=true
Categories=Utility;
DESKTOP
)

  if [[ -n "$SUDO" ]]; then
    $SUDO -u "$SERVICE_USER" mkdir -p "$app_dir"
    printf '%s\n' "$entry" | $SUDO -u "$SERVICE_USER" tee "$launcher_path" >/dev/null
  else
    mkdir -p "$app_dir"
    printf '%s\n' "$entry" > "$launcher_path"
  fi

  if [[ -d "$desktop_dir" ]]; then
    if [[ -n "$SUDO" ]]; then
      $SUDO cp "$launcher_path" "$desktop_copy"
      $SUDO chown "$SERVICE_USER:$SERVICE_USER" "$desktop_copy"
      $SUDO chmod +x "$desktop_copy"
    else
      cp "$launcher_path" "$desktop_copy"
      chmod +x "$desktop_copy"
    fi
  fi
}

main() {
  log_info "Installing AirPlay service for Desk Display"
  validate_files
  install_packages
  write_service_unit
  install_launcher
  log_info "Install complete"
  log_info "Check status with: sudo systemctl status $AIRPLAY_SERVICE_NAME"
}

main "$@"
