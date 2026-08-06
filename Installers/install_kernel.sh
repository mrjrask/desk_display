#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
SERVICE_NAME="desk_display.service"
USER_SERVICE_NAME="desk_display-kernel.service"
USER_SERVICE_TEMPLATE="$PROJECT_DIR/scripts/desk_display_kernel_user.service"

COMMON_SCRIPT="$PROJECT_DIR/scripts/helpers/common.sh"
if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "[ERROR] Missing common installer helpers at $COMMON_SCRIPT" >&2
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
  return 1
}

if [[ -z "${EXPECTED_CODENAME:-}" ]]; then
  EXPECTED_CODENAME=$(detect_codename || true)
  if [[ -z "$EXPECTED_CODENAME" ]]; then
    echo "[WARN] Unable to detect OS codename; defaulting to bookworm." >&2
    EXPECTED_CODENAME="bookworm"
  fi
fi

export EXPECTED_CODENAME
export DESK_DISPLAY_OUTPUT="${DESK_DISPLAY_OUTPUT:-kernel}"
export REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements/kernel.txt}"

prompt_spi_i2c() {
  if [[ -n "${DISABLE_SPI_I2C:-}" ]]; then
    return 0
  fi

  local note="Note: Hyperpixel 4 and Hyperpixel 4 Square require SPI and I2C to be disabled."
  if [[ -t 0 ]]; then
    printf '%s\n' "$note"
    read -r -p "Enable SPI and I2C? [Y/n] " spi_choice
    spi_choice=${spi_choice,,}
    case "$spi_choice" in
      ""|y|yes)
        export DISABLE_SPI_I2C="0"
        ;;
      n|no)
        export DISABLE_SPI_I2C="1"
        ;;
      *)
        warn "Unrecognized input; defaulting to enabling SPI and I2C."
        export DISABLE_SPI_I2C="0"
        ;;
    esac
  else
    log "$note"
    log "Defaulting to enabling SPI and I2C (set DISABLE_SPI_I2C=1 to disable)."
  fi
}

prompt_launch_kernel_display() {
  local launcher="$PROJECT_DIR/scripts/launch_kernel_display.sh"
  if [[ ! -x "$launcher" ]]; then
    return 0
  fi

  local auto_launch="${AUTO_LAUNCH_KERNEL_DISPLAY:-1}"
  local has_display="false"
  local launch_env=()

  if detect_desktop_session "$SERVICE_USER"; then
    has_display="true"
  fi
  if [[ -n "${DISPLAY:-}" ]]; then
    launch_env+=("DISPLAY=$DISPLAY")
  fi
  if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    launch_env+=("WAYLAND_DISPLAY=$WAYLAND_DISPLAY")
  fi
  if [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
    launch_env+=("XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR")
  fi

  if [[ "$auto_launch" == "0" ]]; then
    log "Skipping kernel display launch (AUTO_LAUNCH_KERNEL_DISPLAY=0)."
    log "Launch manually with: $launcher"
    return 0
  fi

  if [[ -t 0 ]]; then
    if [[ "$has_display" == "false" ]]; then
      log "No active desktop session detected; skipping auto-launch."
      log "Launch manually with: $launcher"
      return 0
    fi
    read -r -p "Launch the kernel display now? [Y/n] " launch_choice
    launch_choice=${launch_choice,,}
    if [[ "$launch_choice" == "n" || "$launch_choice" == "no" ]]; then
      log "Skipping kernel display launch."
      log "Launch manually with: $launcher"
      return 0
    fi
    log "Launching the kernel display in the current desktop session."
    if [[ -n "${SUDO:-}" ]]; then
      $SUDO -u "$SERVICE_USER" env "${launch_env[@]}" /bin/bash -lc "$launcher"
    else
      env "${launch_env[@]}" /bin/bash -lc "$launcher"
    fi
    return 0
  fi

  if [[ "$has_display" == "true" ]]; then
    log "Launching the kernel display in the current desktop session."
    if [[ -n "${SUDO:-}" ]]; then
      $SUDO -u "$SERVICE_USER" env "${launch_env[@]}" /bin/bash -lc "$launcher"
    else
      env "${launch_env[@]}" /bin/bash -lc "$launcher"
    fi
  else
    log "Launch manually with: $launcher"
  fi
}

declare -A RESOLUTION_MAP=(
  ["640x480"]="640x480"
  ["1080p"]="1920x1080"
  ["1440p"]="2560x1440"
  ["2k"]="2048x1080"
  ["4k"]="3840x2160"
)

print_resolution_menu() {
  cat <<'MENU'
Select a fullscreen resolution for your kernel-driven display:
  1) 640x480
  2) 1080p - 1920x1080
  3) 1440p - 2560x1440
  4) 2K - 2048x1080
  5) 4K - 3840x2160
MENU
}

detect_drm_resolution() {
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
  return 1
}

detect_xrandr_resolution() {
  if command -v xrandr >/dev/null 2>&1; then
    local mode
    mode=$(xrandr --current 2>/dev/null | awk '/\*/ {print $1; exit}')
    if [[ -n "$mode" ]]; then
      echo "$mode"
      return 0
    fi
  fi
  return 1
}

apply_resolution() {
  local token="$1"
  local dims="${RESOLUTION_MAP[$token]:-$token}"
  if [[ -n "$dims" ]]; then
    export DISPLAY_RESOLUTION="$token"
    export DISPLAY_WIDTH="${dims%x*}"
    export DISPLAY_HEIGHT="${dims#*x}"
  fi
}

configure_resolution() {
  local detected_mode=""
  local default_label="320x240"
  local selection=""

  if [[ -n "${DISPLAY_WIDTH:-}" && -n "${DISPLAY_HEIGHT:-}" ]]; then
    export DISPLAY_RESOLUTION="${DISPLAY_RESOLUTION:-${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}}"
    log "Using preconfigured resolution ${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}."
    return 0
  fi

  if [[ -n "${DISPLAY_RESOLUTION:-}" ]]; then
    apply_resolution "${DISPLAY_RESOLUTION,,}"
    log "Using preconfigured resolution ${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}."
    return 0
  fi

  detected_mode=$(detect_drm_resolution || true)
  if [[ -z "$detected_mode" ]]; then
    detected_mode=$(detect_xrandr_resolution || true)
  fi
  if [[ -n "$detected_mode" ]]; then
    default_label="$detected_mode"
  fi

  if [[ -t 0 ]]; then
    print_resolution_menu
    read -r -p "Select resolution [1-5], press Enter for ${default_label}, or type WxH: " selection
    selection=${selection,,}
    case "$selection" in
      "")
        if [[ "$default_label" != "320x240" ]]; then
          apply_resolution "$default_label"
        fi
        ;;
      1) apply_resolution "640x480" ;;
      2) apply_resolution "1080p" ;;
      3) apply_resolution "1440p" ;;
      4) apply_resolution "2k" ;;
      5) apply_resolution "4k" ;;
      *x*) apply_resolution "$selection" ;;
      *)
        warn "Unrecognized resolution selection '$selection'; keeping ${default_label}."
        if [[ "$default_label" != "320x240" ]]; then
          apply_resolution "$default_label"
        fi
        ;;
    esac
  elif [[ -n "$detected_mode" ]]; then
    apply_resolution "$detected_mode"
  else
    print_resolution_menu
  fi

  if [[ -n "${DISPLAY_WIDTH:-}" && -n "${DISPLAY_HEIGHT:-}" ]]; then
    log "Configured kernel resolution ${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}."
  else
    log "No resolution override selected; keeping default 320x240 output."
  fi
}

configure_resolution

ENV_PATH="$PROJECT_DIR/.env"
ENV_LINES=()

add_env_line() {
  local key="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    ENV_LINES+=("${key}=${value}")
  fi
}

add_env_line "DESK_DISPLAY_OUTPUT" "${DESK_DISPLAY_OUTPUT:-}"
add_env_line "DISPLAY_RESOLUTION" "${DISPLAY_RESOLUTION:-}"
add_env_line "DISPLAY_WIDTH" "${DISPLAY_WIDTH:-}"
add_env_line "DISPLAY_HEIGHT" "${DISPLAY_HEIGHT:-}"

prepend_env_vars "$ENV_PATH" "${ENV_LINES[@]}"

prompt_spi_i2c

# Only one display loop may own the panel at a time: the per-user kernel
# service and the system-wide desk_display.service must never both run,
# or they race to draw the same framebuffer/DRM plane (flickering,
# "stuck" screens that still report an active service). This installer's
# purpose is the per-user kernel service, so keep the system service
# disabled unless the caller explicitly opts out.
if [[ "${DISABLE_SYSTEM_KERNEL_SERVICE:-1}" == "0" ]]; then
  export SKIP_SYSTEM_DISPLAY_SERVICE="0"
else
  export SKIP_SYSTEM_DISPLAY_SERVICE="1"
fi

"$PROJECT_DIR/scripts/helpers/base_setup.sh"

install_kernel_user_service "$PROJECT_DIR" "$SERVICE_USER" "$USER_SERVICE_TEMPLATE" "$USER_SERVICE_NAME"
if [[ "$SKIP_SYSTEM_DISPLAY_SERVICE" == "1" ]] && command -v loginctl >/dev/null 2>&1; then
  log "Enabling lingering for $SERVICE_USER so $USER_SERVICE_NAME survives reboots without an active login."
  $SUDO loginctl enable-linger "$SERVICE_USER" || warn "Failed to enable linger for $SERVICE_USER."
fi

install_kernel_launcher "$PROJECT_DIR" "$SERVICE_NAME" "$SERVICE_USER"
if [[ "${AUTO_START_KERNEL_DISPLAY:-}" == "1" ]]; then
  install_kernel_autostart "$PROJECT_DIR" "$SERVICE_USER"
fi

prompt_launch_kernel_display

if command -v systemctl >/dev/null 2>&1; then
  host=$(hostname)
  cat <<EOF
SSH service control commands:
  ssh ${SERVICE_USER}@${host} '${PROJECT_DIR}/scripts/ssh_kernel_display.sh status'
  ssh ${SERVICE_USER}@${host} '${PROJECT_DIR}/scripts/ssh_kernel_display.sh restart'
  ssh ${SERVICE_USER}@${host} '${PROJECT_DIR}/scripts/ssh_kernel_display.sh stop'
EOF
fi
