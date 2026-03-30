#!/usr/bin/env bash
set -euo pipefail

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
AIRPLAY_SERVICE_NAME="${AIRPLAY_SERVICE_NAME:-airplay_desk_display.service}"
AIRPLAY_SERVICE_PATH="/etc/systemd/system/$AIRPLAY_SERVICE_NAME"
MANIFEST_PATH="/var/lib/desk-display-airplay/packages.txt"

if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

ensure_executable() {
  local file_path="$1"
  if [[ -f "$file_path" ]]; then
    chmod +x "$file_path" || warn "Could not mark $file_path executable"
  else
    warn "Missing expected file: $file_path"
  fi
}

get_connected_mode() {
  local status_path
  for status_path in /sys/class/drm/card*-*/status; do
    [[ -r "$status_path" ]] || continue
    if grep -q "connected" "$status_path"; then
      local modes_path="${status_path%/status}/modes"
      if [[ -r "$modes_path" ]]; then
        local mode
        read -r mode < "$modes_path" || true
        if [[ "$mode" == *x* ]]; then
          echo "$mode"
          return 0
        fi
      fi
    fi
  done

  if command -v xrandr >/dev/null 2>&1; then
    local x_mode
    x_mode=$(xrandr --current 2>/dev/null | awk '/\*/ {print $1; exit}')
    if [[ -n "$x_mode" ]]; then
      echo "$x_mode"
      return 0
    fi
  fi

  return 1
}

install_airplay_packages() {
  local -a required_packages=(
    uxplay
    avahi-daemon
    gstreamer1.0-tools
    gstreamer1.0-libav
    gstreamer1.0-plugins-base
    gstreamer1.0-plugins-good
    gstreamer1.0-plugins-bad
    gstreamer1.0-plugins-ugly
  )

  local -a missing_packages=()
  local pkg
  for pkg in "${required_packages[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
      missing_packages+=("$pkg")
    fi
  done

  log "Refreshing apt metadata"
  ${SUDO:-} apt-get update

  if [[ ${#missing_packages[@]} -gt 0 ]]; then
    log "Installing AirPlay dependencies: ${missing_packages[*]}"
    ${SUDO:-} apt-get install -y "${missing_packages[@]}"
  else
    log "All AirPlay dependencies are already installed"
  fi

  ${SUDO:-} mkdir -p "$(dirname -- "$MANIFEST_PATH")"
  if [[ ${#missing_packages[@]} -gt 0 ]]; then
    printf '%s\n' "${missing_packages[@]}" | ${SUDO:-} tee "$MANIFEST_PATH" >/dev/null
    log "Recorded newly installed packages in $MANIFEST_PATH"
  else
    : | ${SUDO:-} tee "$MANIFEST_PATH" >/dev/null
    log "No new packages recorded in manifest"
  fi
}

write_airplay_service() {
  local default_resolution
  default_resolution=$(get_connected_mode || true)
  if [[ -z "$default_resolution" ]]; then
    default_resolution="1920x1080"
  fi

  local service_contents
  service_contents=$(cat <<EOF_SERVICE
[Unit]
Description=Desk Display AirPlay takeover service
After=network-online.target avahi-daemon.service
Wants=network-online.target avahi-daemon.service

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$PROJECT_DIR
Environment=CONFIG_LOAD_DOTENV=1
Environment=AIRPLAY_RESOLUTION_DEFAULT=$default_resolution
ExecStart=/bin/bash -lc '$PROJECT_DIR/scripts/airplay_takeover_daemon.sh'
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF_SERVICE
)

  log "Writing AirPlay service unit to $AIRPLAY_SERVICE_PATH"
  echo "$service_contents" | ${SUDO:-} tee "$AIRPLAY_SERVICE_PATH" >/dev/null
  ${SUDO:-} systemctl daemon-reload
  ${SUDO:-} systemctl enable --now "$AIRPLAY_SERVICE_NAME"
}

install_airplay_launcher() {
  local home_dir
  home_dir=$(getent passwd "$SERVICE_USER" | cut -d: -f6)
  if [[ -z "$home_dir" ]]; then
    home_dir="/home/$SERVICE_USER"
  fi

  local app_dir="$home_dir/.local/share/applications"
  local desktop_dir="$home_dir/Desktop"
  local launcher_entry="$app_dir/desk-display-airplay.desktop"
  local launcher_contents

  launcher_contents=$(cat <<EOF_LAUNCHER
[Desktop Entry]
Type=Application
Name=Desk Display AirPlay Mode
Comment=Start AirPlay takeover mode for Desk Display
Exec=/bin/bash -lc '$PROJECT_DIR/scripts/airplay_mode.sh'
Terminal=true
Categories=Utility;
EOF_LAUNCHER
)

  if [[ -n "$SUDO" ]]; then
    $SUDO mkdir -p "$app_dir"
    echo "$launcher_contents" | $SUDO tee "$launcher_entry" >/dev/null
    $SUDO chown "$SERVICE_USER":"$SERVICE_USER" "$launcher_entry"
  else
    mkdir -p "$app_dir"
    echo "$launcher_contents" > "$launcher_entry"
  fi

  if [[ -d "$desktop_dir" ]]; then
    local desktop_launcher="$desktop_dir/Desk Display AirPlay Mode.desktop"
    if [[ -n "$SUDO" ]]; then
      $SUDO cp "$launcher_entry" "$desktop_launcher"
      $SUDO chown "$SERVICE_USER":"$SERVICE_USER" "$desktop_launcher"
      $SUDO chmod +x "$desktop_launcher"
    else
      cp "$launcher_entry" "$desktop_launcher"
      chmod +x "$desktop_launcher"
    fi
  fi

  log "Installed desktop launcher at $launcher_entry"
}

main() {
  log "Installing AirPlay support for Desk Display from $PROJECT_DIR"
  ensure_executable "$PROJECT_DIR/scripts/airplay_mode.sh"
  ensure_executable "$PROJECT_DIR/scripts/airplay_takeover_daemon.sh"

  install_airplay_packages
  write_airplay_service
  install_airplay_launcher

  log "AirPlay install complete."
  log "Set AIRPLAY_PAIRING_CODE in .env to require a pairing code."
  log "Service status: sudo systemctl status $AIRPLAY_SERVICE_NAME"
}

main "$@"
