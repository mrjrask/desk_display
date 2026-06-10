#!/usr/bin/env bash
set -euo pipefail

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

ensure_executable() {
  local file_path="$1"

  if [[ -x "$file_path" ]]; then
    log "$(basename "$file_path") already executable"
  elif [[ -f "$file_path" ]]; then
    chmod +x "$file_path" || warn "Could not mark $(basename "$file_path") as executable"
  else
    warn "Missing script: $file_path"
  fi
}

install_framebuffer_launcher() {
  local project_dir="$1"
  local service_name="$2"
  local service_user="$3"
  local launcher_path="$project_dir/scripts/launch_framebuffer.sh"
  local restore_path="$project_dir/scripts/restore_desktop.sh"

  if [[ ! -f "$launcher_path" ]]; then
    warn "Framebuffer launcher not found at $launcher_path"
    return 1
  fi

  ensure_executable "$launcher_path"
  if [[ -f "$restore_path" ]]; then
    ensure_executable "$restore_path"
  fi

  local home_dir
  home_dir=$(getent passwd "$service_user" | cut -d: -f6)
  if [[ -z "$home_dir" ]]; then
    home_dir="/home/$service_user"
  fi

  local app_dir="$home_dir/.local/share/applications"
  local desktop_dir="$home_dir/Desktop"
  local launcher_entry="$app_dir/desk-display-framebuffer.desktop"

  if [[ -n "${SUDO:-}" ]]; then
    $SUDO mkdir -p "$app_dir"
  else
    mkdir -p "$app_dir"
  fi

  local launcher_contents
  launcher_contents=$(cat <<EOF
[Desktop Entry]
Type=Application
Name=Desk Display (Framebuffer)
Comment=Launch the framebuffer service (stops the desktop display manager)
Exec=/bin/bash -lc '$launcher_path'
Terminal=true
Categories=Utility;
EOF
)

  if [[ -n "${SUDO:-}" ]]; then
    echo "$launcher_contents" | $SUDO tee "$launcher_entry" >/dev/null
    $SUDO chown "$service_user":"$service_user" "$launcher_entry"
  else
    echo "$launcher_contents" > "$launcher_entry"
  fi

  if [[ -d "$desktop_dir" ]]; then
    local desktop_launcher="$desktop_dir/Desk Display Framebuffer.desktop"
    if [[ -n "${SUDO:-}" ]]; then
      $SUDO cp "$launcher_entry" "$desktop_launcher"
      $SUDO chown "$service_user":"$service_user" "$desktop_launcher"
      $SUDO chmod +x "$desktop_launcher"
    else
      cp "$launcher_entry" "$desktop_launcher"
      chmod +x "$desktop_launcher"
    fi
  fi

  log "Installed framebuffer launcher for $service_name at $launcher_entry"
}

install_kernel_launcher() {
  local project_dir="$1"
  local service_name="$2"
  local service_user="$3"
  local launcher_path="$project_dir/scripts/launch_kernel_display.sh"

  if [[ ! -f "$launcher_path" ]]; then
    warn "Kernel display launcher not found at $launcher_path"
    return 1
  fi

  ensure_executable "$launcher_path"

  local home_dir
  home_dir=$(getent passwd "$service_user" | cut -d: -f6)
  if [[ -z "$home_dir" ]]; then
    home_dir="/home/$service_user"
  fi

  local app_dir="$home_dir/.local/share/applications"
  local desktop_dir="$home_dir/Desktop"
  local launcher_entry="$app_dir/desk-display-kernel.desktop"

  if [[ -n "${SUDO:-}" ]]; then
    $SUDO mkdir -p "$app_dir"
  else
    mkdir -p "$app_dir"
  fi

  local launcher_contents
  launcher_contents=$(cat <<EOF
[Desktop Entry]
Type=Application
Name=Desk Display (Kernel Display)
Comment=Launch the fullscreen kernel display in the desktop session
Exec=/bin/bash -lc '$launcher_path'
Terminal=true
Categories=Utility;
EOF
)

  if [[ -n "${SUDO:-}" ]]; then
    echo "$launcher_contents" | $SUDO tee "$launcher_entry" >/dev/null
    $SUDO chown "$service_user":"$service_user" "$launcher_entry"
  else
    echo "$launcher_contents" > "$launcher_entry"
  fi

  if [[ -d "$desktop_dir" ]]; then
    local desktop_launcher="$desktop_dir/Desk Display Kernel.desktop"
    if [[ -n "${SUDO:-}" ]]; then
      $SUDO cp "$launcher_entry" "$desktop_launcher"
      $SUDO chown "$service_user":"$service_user" "$desktop_launcher"
      $SUDO chmod +x "$desktop_launcher"
    else
      cp "$launcher_entry" "$desktop_launcher"
      chmod +x "$desktop_launcher"
    fi
  fi

  log "Installed kernel display launcher for $service_name at $launcher_entry"
}

install_kernel_autostart() {
  local project_dir="$1"
  local service_user="$2"
  local launcher_path="$project_dir/scripts/launch_kernel_display.sh"

  if [[ ! -f "$launcher_path" ]]; then
    warn "Kernel display launcher not found at $launcher_path"
    return 1
  fi

  ensure_executable "$launcher_path"

  local home_dir
  home_dir=$(getent passwd "$service_user" | cut -d: -f6)
  if [[ -z "$home_dir" ]]; then
    home_dir="/home/$service_user"
  fi

  local autostart_dir="$home_dir/.config/autostart"
  local autostart_entry="$autostart_dir/desk-display-kernel.desktop"

  if [[ -n "${SUDO:-}" ]]; then
    $SUDO mkdir -p "$autostart_dir"
  else
    mkdir -p "$autostart_dir"
  fi

  local launcher_contents
  launcher_contents=$(cat <<EOF
[Desktop Entry]
Type=Application
Name=Desk Display (Kernel Display)
Comment=Launch the fullscreen kernel display on login
Exec=/bin/bash -lc '$launcher_path'
Terminal=false
Categories=Utility;
X-GNOME-Autostart-enabled=true
EOF
)

  if [[ -n "${SUDO:-}" ]]; then
    echo "$launcher_contents" | $SUDO tee "$autostart_entry" >/dev/null
    $SUDO chown "$service_user":"$service_user" "$autostart_entry"
  else
    echo "$launcher_contents" > "$autostart_entry"
  fi

  log "Installed kernel display autostart entry at $autostart_entry"
}

install_kernel_user_service() {
  local project_dir="$1"
  local service_user="$2"
  local template_path="$3"
  local service_name="${4:-desk_display-kernel.service}"

  if [[ ! -f "$template_path" ]]; then
    warn "Kernel user service template not found at $template_path"
    return 1
  fi

  local home_dir
  home_dir=$(getent passwd "$service_user" | cut -d: -f6)
  if [[ -z "$home_dir" ]]; then
    home_dir="/home/$service_user"
  fi

  local user_systemd_dir="$home_dir/.config/systemd/user"
  local service_path="$user_systemd_dir/$service_name"

  local venv_dir
  venv_dir=$(detect_existing_venv "$project_dir" || true)
  if [[ -z "$venv_dir" ]]; then
    venv_dir="$project_dir/venv"
  fi

  local maintenance_dir="$project_dir/tools/maintenance"
  local project_dir_safe="$project_dir"
  local venv_dir_safe="$venv_dir"
  local maintenance_dir_safe="$maintenance_dir"
  local output_env_line=""

  # Keep any generated DESK_DISPLAY_OUTPUT default before EnvironmentFile so
  # PROJECT_DIR/.env remains authoritative. This is especially important for
  # HyperPixel installs that may fall back from kernel to framebuffer mode.
  if [[ "${DESK_DISPLAY_OUTPUT:-}" == "kernel" ]]; then
    output_env_line="Environment=DESK_DISPLAY_OUTPUT=kernel"
  fi

  local service_contents
  service_contents=$(sed \
    -e "s|@PROJECT_DIR@|$project_dir_safe|g" \
    -e "s|@VENV_DIR@|$venv_dir_safe|g" \
    -e "s|@MAINTENANCE_DIR@|$maintenance_dir_safe|g" \
    -e "s|@DESK_DISPLAY_OUTPUT_ENV@|$output_env_line|g" \
    "$template_path")

  if [[ -n "${SUDO:-}" ]]; then
    $SUDO -u "$service_user" mkdir -p "$user_systemd_dir"
    echo "$service_contents" | $SUDO -u "$service_user" tee "$service_path" >/dev/null
  else
    mkdir -p "$user_systemd_dir"
    echo "$service_contents" > "$service_path"
  fi

  log "Installed user systemd service to $service_path"

  local enable_target="default.target"
  local wants_dir="$user_systemd_dir/${enable_target}.wants"
  local wants_link="$wants_dir/$service_name"

  create_user_wants_link() {
    if [[ -n "${SUDO:-}" ]]; then
      $SUDO -u "$service_user" mkdir -p "$wants_dir"
      $SUDO -u "$service_user" ln -sf "$service_path" "$wants_link"
    else
      mkdir -p "$wants_dir"
      ln -sf "$service_path" "$wants_link"
    fi
    log "Linked $service_name into $enable_target (fallback)."
  }

  if command -v systemctl >/dev/null 2>&1; then
    local uid runtime_dir
    uid=$(id -u "$service_user" 2>/dev/null || true)
    runtime_dir=""
    if [[ -n "$uid" ]]; then
      runtime_dir="/run/user/$uid"
    fi
    local systemctl_env=()
    if [[ -n "$runtime_dir" && -d "$runtime_dir" ]]; then
      systemctl_env=("XDG_RUNTIME_DIR=$runtime_dir")
      if [[ -S "$runtime_dir/bus" ]]; then
        systemctl_env+=("DBUS_SESSION_BUS_ADDRESS=unix:path=$runtime_dir/bus")
      fi
    fi

    if [[ -n "${SUDO:-}" ]]; then
      if ! $SUDO -u "$service_user" env "${systemctl_env[@]}" systemctl --user daemon-reload; then
        warn "Failed to reload user systemd daemon for $service_user."
        warn "Attempting to enable lingering for SSH/headless setups."
        if $SUDO loginctl enable-linger "$service_user"; then
          if $SUDO -u "$service_user" XDG_RUNTIME_DIR="/run/user/$uid" systemctl --user daemon-reload; then
            $SUDO -u "$service_user" XDG_RUNTIME_DIR="/run/user/$uid" systemctl --user enable --now "$service_name" \
              || {
                warn "Failed to enable/start $service_name after enabling linger."
                create_user_wants_link
              }
            return 0
          fi
        fi
        warn "To enable manually, run:"
        warn "  sudo loginctl enable-linger $service_user"
        warn "  sudo -u $service_user XDG_RUNTIME_DIR=/run/user/$uid systemctl --user daemon-reload"
        warn "  sudo -u $service_user XDG_RUNTIME_DIR=/run/user/$uid systemctl --user enable --now $service_name"
        warn "If enable fails, link the unit into the default target:"
        warn "  sudo -u $service_user mkdir -p $wants_dir"
        warn "  sudo -u $service_user ln -sf $service_path $wants_link"
        return 0
      fi
      if detect_desktop_session "$service_user"; then
        $SUDO -u "$service_user" env "${systemctl_env[@]}" systemctl --user enable --now "$service_name" \
          || {
            warn "Failed to enable/start $service_name (user session may be offline)."
            create_user_wants_link
          }
      else
        $SUDO -u "$service_user" env "${systemctl_env[@]}" systemctl --user enable "$service_name" \
          || {
            warn "Failed to enable $service_name (user session may be offline)."
            create_user_wants_link
          }
      fi
    else
      if ! env "${systemctl_env[@]}" systemctl --user daemon-reload; then
        warn "Failed to reload user systemd daemon for $service_user."
        warn "To enable manually on SSH/headless setups, run:"
        warn "  sudo loginctl enable-linger $service_user"
        warn "  sudo -u $service_user XDG_RUNTIME_DIR=/run/user/$uid systemctl --user daemon-reload"
        warn "  sudo -u $service_user XDG_RUNTIME_DIR=/run/user/$uid systemctl --user enable --now $service_name"
        warn "If enable fails, link the unit into the default target:"
        warn "  sudo -u $service_user mkdir -p $wants_dir"
        warn "  sudo -u $service_user ln -sf $service_path $wants_link"
        return 0
      fi
      if detect_desktop_session "$service_user"; then
        env "${systemctl_env[@]}" systemctl --user enable --now "$service_name" \
          || {
            warn "Failed to enable/start $service_name (user session may be offline)."
            create_user_wants_link
          }
      else
        env "${systemctl_env[@]}" systemctl --user enable "$service_name" \
          || {
            warn "Failed to enable $service_name (user session may be offline)."
            create_user_wants_link
          }
      fi
    fi
  else
    warn "systemctl not available; skipping user service enablement."
  fi
}

detect_existing_venv() {
  local project_dir="$1"
  local candidates=(
    "$project_dir/venv"
    "$project_dir/.venv"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate/pyvenv.cfg" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  local venv_cfg
  venv_cfg=$(find "$project_dir" -maxdepth 2 -mindepth 2 -type f -name pyvenv.cfg -print -quit 2>/dev/null || true)
  if [[ -n "$venv_cfg" ]]; then
    dirname "$venv_cfg"
    return 0
  fi

  return 1
}

detect_desktop_session() {
  local service_user="$1"

  if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
    return 0
  fi

  local runtime_dir=""
  local uid=""
  uid=$(id -u "$service_user" 2>/dev/null || true)
  if [[ -n "$uid" ]]; then
    runtime_dir="/run/user/$uid"
    if [[ -z "${XDG_RUNTIME_DIR:-}" && -d "$runtime_dir" ]]; then
      export XDG_RUNTIME_DIR="$runtime_dir"
    fi
  fi

  if command -v loginctl >/dev/null 2>&1; then
    local sessions session active type display
    sessions=$(loginctl show-user "$service_user" -p Sessions --value 2>/dev/null || true)
    for session in $sessions; do
      active=$(loginctl show-session "$session" -p Active --value 2>/dev/null || true)
      type=$(loginctl show-session "$session" -p Type --value 2>/dev/null || true)
      display=$(loginctl show-session "$session" -p Display --value 2>/dev/null || true)
      if [[ "$active" == "yes" && ( "$type" == "x11" || "$type" == "wayland" ) ]]; then
        if [[ -z "${DISPLAY:-}" && "$type" == "x11" && -n "$display" ]]; then
          export DISPLAY="$display"
        fi
        if [[ -z "${WAYLAND_DISPLAY:-}" && "$type" == "wayland" && -n "$runtime_dir" && -S "$runtime_dir/wayland-0" ]]; then
          export WAYLAND_DISPLAY="wayland-0"
        fi
        return 0
      fi
    done
  fi

  if [[ -z "${WAYLAND_DISPLAY:-}" && -n "$runtime_dir" && -S "$runtime_dir/wayland-0" ]]; then
    export WAYLAND_DISPLAY="wayland-0"
    return 0
  fi

  if [[ -z "${DISPLAY:-}" && -S /tmp/.X11-unix/X0 ]]; then
    export DISPLAY=":0"
    return 0
  fi

  return 1
}

# Return the preferred libtiff development package for the given codename,
# falling back gracefully when a codename-specific package is unavailable.
select_libtiff_pkg() {
  local codename="$1"
  local candidates=()

  case "$codename" in
    bookworm) candidates=(libtiff5-dev libtiff-dev) ;;
    trixie) candidates=(libtiff6-dev libtiff5-dev libtiff-dev) ;;
    *) candidates=(libtiff5-dev libtiff-dev) ;;
  esac

  for pkg in "${candidates[@]}"; do
    if apt-cache show "$pkg" >/dev/null 2>&1; then
      echo "$pkg"
      return 0
    fi
  done

  warn "Could not find a libtiff dev package for codename '$codename'; attempted: ${candidates[*]}. Defaulting to ${candidates[-1]}."
  echo "${candidates[-1]}"
}

# Choose the gdk-pixbuf development package for the given codename, preferring
# codename-specific names while still allowing a fallback if the package cache
# does not know the preferred option.
select_gdk_pixbuf_pkg() {
  local codename="$1"
  local candidates=()

  case "$codename" in
    bookworm) candidates=(libgdk-pixbuf2.0-dev libgdk-pixbuf-2.0-dev) ;;
    trixie) candidates=(libgdk-pixbuf-2.0-dev) ;;
    *) candidates=(libgdk-pixbuf-2.0-dev libgdk-pixbuf2.0-dev) ;;
  esac

  for pkg in "${candidates[@]}"; do
    if apt-cache show "$pkg" >/dev/null 2>&1; then
      echo "$pkg"
      return 0
    fi
  done

  warn "Could not find a gdk-pixbuf dev package for codename '$codename'; attempted: ${candidates[*]}. Defaulting to ${candidates[-1]}."
  echo "${candidates[-1]}"
}

run_initial_apt_maintenance() {
  log "Running initial apt maintenance."
  ${SUDO:-} apt update
  ${SUDO:-} apt full-upgrade -y
  ${SUDO:-} apt autoremove -y
}

install_apt_packages() {
  run_initial_apt_maintenance

  local codename="${EXPECTED_CODENAME:-}"
  if [[ -z "$codename" ]]; then
    warn "EXPECTED_CODENAME is not set; defaulting to detected release name"
    codename=$(lsb_release -sc 2>/dev/null || echo "")
  fi

  local shared_packages=(
    python3-venv python3-pip python3-dev python3-opencv
    build-essential libjpeg-dev libopenblas0 libopenblas-dev
    liblgpio-dev
    libopenjp2-7-dev libcairo2-dev libpango1.0-dev
    libffi-dev network-manager wireless-tools swig iproute2
    i2c-tools fonts-dejavu-core fonts-noto-color-emoji libgl1 libx264-dev ffmpeg git
  )

  local packages=("${shared_packages[@]}")
  packages+=("$(select_libtiff_pkg "$codename")")
  packages+=("$(select_gdk_pixbuf_pkg "$codename")")

  log "Installing apt dependencies: ${packages[*]}"
  ${SUDO:-} apt-get install -y "${packages[@]}"
}

prepend_env_vars() {
  local env_path="$1"
  shift
  local lines=("$@")

  if [[ ${#lines[@]} -eq 0 ]]; then
    return 0
  fi

  local tmp_file
  local filtered_file
  tmp_file=$(mktemp)
  filtered_file=$(mktemp)
  local target_owner=""
  local target_group=""
  local target_mode=""

  local keys=()
  local line
  for line in "${lines[@]}"; do
    keys+=("${line%%=*}")
  done

  local regex="^($(IFS='|'; echo "${keys[*]}"))="

  if [[ -f "$env_path" ]]; then
    grep -v -E "$regex" "$env_path" > "$filtered_file" || true
  else
    : > "$filtered_file"
  fi

  {
    printf '%s\n' "${lines[@]}"
    if [[ -s "$filtered_file" ]]; then
      printf '\n'
      cat "$filtered_file"
    fi
  } > "$tmp_file"

  local existing_owner_name=""
  local existing_owner_id=""
  if [[ -f "$env_path" ]]; then
    existing_owner_name=$(stat -c '%U' "$env_path" 2>/dev/null || true)
    existing_owner_id=$(stat -c '%u' "$env_path" 2>/dev/null || true)
    target_owner="$existing_owner_name"
    target_group=$(stat -c '%G' "$env_path" 2>/dev/null || true)
    target_mode=$(stat -c '%a' "$env_path" 2>/dev/null || true)
  fi

  # If an existing file is owned by root and we're running with sudo, prefer
  # SERVICE_USER/SUDO_USER so .env stays editable/readable by the non-root user.
  if [[ -n "${SUDO:-}" && ( "$existing_owner_name" == "root" || "$existing_owner_id" == "0" ) ]]; then
    target_owner=""
    target_group=""
  fi

  if [[ -z "$target_owner" ]]; then
    if [[ -n "${SERVICE_USER:-}" ]]; then
      target_owner="$SERVICE_USER"
    elif [[ -n "${SUDO_USER:-}" ]]; then
      target_owner="$SUDO_USER"
    fi
  fi

  if [[ -z "$target_group" && -n "$target_owner" ]]; then
    target_group=$(id -gn "$target_owner" 2>/dev/null || true)
  fi

  if [[ -z "$target_mode" ]]; then
    target_mode="644"
  fi

  if [[ -n "${SUDO:-}" ]]; then
    $SUDO mv "$tmp_file" "$env_path"
    if [[ -n "$target_owner" && -n "$target_group" ]]; then
      $SUDO chown "$target_owner":"$target_group" "$env_path" 2>/dev/null || true
    fi
    $SUDO chmod "$target_mode" "$env_path" 2>/dev/null || true
  else
    mv "$tmp_file" "$env_path"
    if [[ -n "$target_owner" && -n "$target_group" ]]; then
      chown "$target_owner":"$target_group" "$env_path" 2>/dev/null || true
    fi
    chmod "$target_mode" "$env_path" 2>/dev/null || true
  fi

  rm -f "$filtered_file"
}
