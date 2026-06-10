#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
SYSTEM_SERVICE_NAME="desk_display.service"
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

run_user_systemctl() {
  local service_user="$1"
  shift

  if ! command -v systemctl >/dev/null 2>&1; then
    return 1
  fi

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

  local current_user
  current_user=$(id -un 2>/dev/null || true)

  if [[ "$current_user" == "$service_user" ]]; then
    env "${systemctl_env[@]}" systemctl --user "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo -u "$service_user" env "${systemctl_env[@]}" systemctl --user "$@"
  elif command -v runuser >/dev/null 2>&1; then
    runuser -u "$service_user" -- env "${systemctl_env[@]}" systemctl --user "$@"
  else
    warn "Unable to run systemctl --user as $service_user; sudo or runuser is required."
    return 1
  fi
}

disable_user_kernel_service() {
  local service_user="$1"
  local service_name="$2"
  local home_dir user_systemd_dir wants_link

  log "Disabling $service_name to avoid conflicts with $SYSTEM_SERVICE_NAME."
  run_user_systemctl "$service_user" disable --now "$service_name" \
    || warn "Failed to disable $service_name via systemctl --user; removing fallback wants link if present."

  home_dir=$(getent passwd "$service_user" | cut -d: -f6)
  if [[ -z "$home_dir" ]]; then
    home_dir="/home/$service_user"
  fi
  user_systemd_dir="$home_dir/.config/systemd/user"
  wants_link="$user_systemd_dir/default.target.wants/$service_name"

  if [[ -e "$wants_link" || -L "$wants_link" ]]; then
    if [[ -n "$SUDO" ]]; then
      $SUDO rm -f "$wants_link"
    else
      rm -f "$wants_link"
    fi
    log "Removed fallback user service link $wants_link."
  fi
}

disable_system_display_service() {
  if command -v systemctl >/dev/null 2>&1; then
    log "Disabling $SYSTEM_SERVICE_NAME to avoid conflicts with $USER_SERVICE_NAME."
    $SUDO systemctl disable --now "$SYSTEM_SERVICE_NAME" || warn "Failed to disable $SYSTEM_SERVICE_NAME."
  fi
}

enable_user_linger() {
  local service_user="$1"
  if command -v loginctl >/dev/null 2>&1; then
    if [[ -n "$SUDO" ]]; then
      $SUDO loginctl enable-linger "$service_user" || warn "Failed to enable linger for $service_user."
    else
      loginctl enable-linger "$service_user" || warn "Failed to enable linger for $service_user."
    fi
  else
    warn "loginctl not available; cannot enable linger."
  fi
}

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
    warn "Unable to detect OS codename; defaulting to bookworm."
    EXPECTED_CODENAME="bookworm"
  fi
fi

case "$EXPECTED_CODENAME" in
  bookworm|trixie)
    ;;
  *)
    warn "============================================================"
    warn "Detected codename '$EXPECTED_CODENAME', which is not explicitly supported."
    warn "Dependency package mapping is best-effort on this release and may fail."
    warn "If installation errors occur, use Debian package equivalents for your OS."
    warn "============================================================"
    ;;
esac

export EXPECTED_CODENAME
DESK_DISPLAY_OUTPUT="${DESK_DISPLAY_OUTPUT:-kernel}"
export DESK_DISPLAY_OUTPUT
export REQUIREMENTS_FILE="requirements_kernel.txt"
export DISABLE_SPI_I2C="1"

HYPERPIXEL_PANEL="${HYPERPIXEL_PANEL:-}"
DISPLAY_WIDTH="${DISPLAY_WIDTH:-}"
DISPLAY_HEIGHT="${DISPLAY_HEIGHT:-}"
RENDER_WIDTH="${RENDER_WIDTH:-}"
RENDER_HEIGHT="${RENDER_HEIGHT:-}"
DISPLAY_RENDER_SCALE="${DISPLAY_RENDER_SCALE:-}"
PI_ZERO_HYPERPIXEL_CONFIG="$PROJECT_DIR/screens_config.pi_zero_hyperpixel.json"

export DISPLAY_ROTATION="${DISPLAY_ROTATION:-0}"

prompt_panel_type() {
  if [[ -t 0 ]]; then
    cat <<'MENU'
Select your HyperPixel panel:
  1) HyperPixel 4 (800x480)
  2) HyperPixel 4 Square (720x720)
MENU
    read -r -p "Enter a number [1-2]: " selection
    case "$selection" in
      1)
        HYPERPIXEL_PANEL="hyperpixel4"
        DISPLAY_WIDTH="800"
        DISPLAY_HEIGHT="480"
        ;;
      2)
        HYPERPIXEL_PANEL="hyperpixel4sq"
        DISPLAY_WIDTH="720"
        DISPLAY_HEIGHT="720"
        ;;
      *)
        warn "Unrecognized selection."
        return 1
        ;;
    esac
    return 0
  fi

  warn "Unable to detect HyperPixel panel in non-interactive mode."
  return 1
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

detect_hyperpixel_panel() {
  local detected_mode
  detected_mode=$(detect_drm_resolution || true)
  if [[ -z "$detected_mode" ]]; then
    detected_mode=$(detect_xrandr_resolution || true)
  fi

  case "$detected_mode" in
    800x480*|480x800*)
      HYPERPIXEL_PANEL="hyperpixel4"
      DISPLAY_WIDTH="800"
      DISPLAY_HEIGHT="480"
      return 0
      ;;
    720x720*)
      HYPERPIXEL_PANEL="hyperpixel4sq"
      DISPLAY_WIDTH="720"
      DISPLAY_HEIGHT="720"
      return 0
      ;;
    "")
      warn "No connected display mode detected."
      ;;
    *)
      warn "Detected display mode $detected_mode does not match HyperPixel presets."
      ;;
  esac

  prompt_panel_type
}

set_render_size_for_choice() {
  local choice="$1"
  case "$HYPERPIXEL_PANEL:$choice" in
    hyperpixel4:full)
      RENDER_WIDTH="$DISPLAY_WIDTH"
      RENDER_HEIGHT="$DISPLAY_HEIGHT"
      ;;
    hyperpixel4:balanced)
      RENDER_WIDTH="480"
      RENDER_HEIGHT="288"
      ;;
    hyperpixel4:low)
      RENDER_WIDTH="400"
      RENDER_HEIGHT="240"
      ;;
    hyperpixel4sq:full)
      RENDER_WIDTH="$DISPLAY_WIDTH"
      RENDER_HEIGHT="$DISPLAY_HEIGHT"
      ;;
    hyperpixel4sq:balanced)
      RENDER_WIDTH="480"
      RENDER_HEIGHT="480"
      ;;
    hyperpixel4sq:low)
      RENDER_WIDTH="360"
      RENDER_HEIGHT="360"
      ;;
  esac
}

render_size_label() {
  local choice="$1"
  case "$HYPERPIXEL_PANEL:$choice" in
    hyperpixel4:low) echo "400x240" ;;
    hyperpixel4:balanced) echo "480x288" ;;
    hyperpixel4sq:low) echo "360x360" ;;
    hyperpixel4sq:balanced) echo "480x480" ;;
    *) echo "${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}" ;;
  esac
}

configure_render_size() {
  if [[ -n "$DISPLAY_RENDER_SCALE" ]]; then
    return 0
  fi

  if [[ -n "$RENDER_WIDTH" && -n "$RENDER_HEIGHT" ]]; then
    return 0
  fi

  local default_choice="full"
  if [[ "${DESK_DISPLAY_LOW_POWER:-}" == "1" || ( -z "${DESK_DISPLAY_LOW_POWER:-}" && "${DESK_DISPLAY_PROFILE:-hyperpixel_pi_zero}" == "hyperpixel_pi_zero" ) ]]; then
    default_choice="low"
  fi

  if [[ -t 0 ]]; then
    cat <<MENU
Select internal render size for ${HYPERPIXEL_PANEL} (${DISPLAY_WIDTH}x${DISPLAY_HEIGHT} output):
  1) Low-power ($(render_size_label low))
  2) Balanced ($(render_size_label balanced))
  3) Full panel (${DISPLAY_WIDTH}x${DISPLAY_HEIGHT})
MENU
    read -r -p "Enter a number [1-3, default ${default_choice}]: " selection
    case "$selection" in
      "") set_render_size_for_choice "$default_choice" ;;
      1) set_render_size_for_choice low ;;
      2) set_render_size_for_choice balanced ;;
      3) set_render_size_for_choice full ;;
      *)
        warn "Unrecognized render size selection; using low-power defaults."
        set_render_size_for_choice low
        ;;
    esac
  else
    set_render_size_for_choice "$default_choice"
  fi
}

detect_pi_zero_class_device() {
  local model_path model
  for model_path in \
    "${PI_MODEL_PATH:-}" \
    /proc/device-tree/model \
    /sys/firmware/devicetree/base/model; do
    [[ -n "$model_path" && -r "$model_path" ]] || continue
    model=$(tr -d '\0\r' < "$model_path" | head -n 1)
    case "$model" in
      *"Raspberry Pi Zero"*)
        return 0
        ;;
    esac
  done

  return 1
}

select_schedule_preset() {
  local default_choice="default"
  local selection

  if [[ ! -f "$PI_ZERO_HYPERPIXEL_CONFIG" ]]; then
    warn "Pi Zero HyperPixel schedule preset not found at $PI_ZERO_HYPERPIXEL_CONFIG."
    return 0
  fi

  if [[ -n "${SCREENS_CONFIG_PATH:-}" ]]; then
    log "SCREENS_CONFIG_PATH is already set; leaving the existing schedule selection unchanged."
    return 0
  fi

  if detect_pi_zero_class_device; then
    default_choice="pi_zero"
  fi

  if [[ "$default_choice" == "pi_zero" && -t 0 ]]; then
    cat <<MENU
Detected a Raspberry Pi Zero-class device.
Select the default screen schedule:
  1) Lightweight Pi Zero + HyperPixel preset (recommended)
  2) Standard schedule
MENU
    read -r -p "Enter a number [1-2, default 1]: " selection
    case "$selection" in
      ""|1)
        default_choice="pi_zero"
        ;;
      2)
        default_choice="default"
        ;;
      *)
        warn "Unrecognized schedule selection; using the lightweight Pi Zero + HyperPixel preset."
        default_choice="pi_zero"
        ;;
    esac
  fi

  if [[ "$default_choice" == "pi_zero" ]]; then
    SCREENS_CONFIG_PATH="$PI_ZERO_HYPERPIXEL_CONFIG"
    SCREENS_CONFIG_LOCAL_PATH="${SCREENS_CONFIG_LOCAL_PATH:-$PROJECT_DIR/screens_config.pi_zero_hyperpixel.local.json}"
    export SCREENS_CONFIG_PATH SCREENS_CONFIG_LOCAL_PATH
    log "Using lightweight Pi Zero + HyperPixel schedule preset."
    log "Re-enable radar, sports, standings, and animated/logo screens one at a time after CPU temperature/load is stable."
  fi
}

detect_framebuffer_device() {
  local requested_size="${DISPLAY_WIDTH}x${DISPLAY_HEIGHT}"
  local fb_path sysfs_base mode_value virtual_value candidate_size
  local fallback_device=""

  for fb_path in /dev/fb*; do
    [[ -c "$fb_path" ]] || continue
    sysfs_base="/sys/class/graphics/${fb_path##*/}"

    if [[ -z "$fallback_device" ]]; then
      fallback_device="$fb_path"
    fi

    mode_value=""
    if [[ -r "$sysfs_base/mode" ]]; then
      mode_value=$(tr -d '\r' < "$sysfs_base/mode" | head -n 1)
      mode_value=${mode_value// /}
      mode_value=${mode_value#U:}
    elif [[ -r "$sysfs_base/modes" ]]; then
      mode_value=$(tr -d '\r' < "$sysfs_base/modes" | head -n 1)
      mode_value=${mode_value// /}
      mode_value=${mode_value#U:}
    fi

    if [[ "$mode_value" == "$requested_size" ]]; then
      echo "$fb_path"
      return 0
    fi

    if [[ -r "$sysfs_base/virtual_size" ]]; then
      virtual_value=$(tr -d '\r' < "$sysfs_base/virtual_size")
      candidate_size="${virtual_value/,/x}"
      if [[ "$candidate_size" == "$requested_size" ]]; then
        echo "$fb_path"
        return 0
      fi
    fi
  done

  if [[ -n "$fallback_device" ]]; then
    echo "$fallback_device"
    return 0
  fi

  return 1
}

validate_hyperpixel_env_overrides() {
  case "$HYPERPIXEL_PANEL" in
    "")
      return 1
      ;;
    hyperpixel4)
      if [[ -z "$DISPLAY_WIDTH" || -z "$DISPLAY_HEIGHT" ]]; then
        DISPLAY_WIDTH="800"
        DISPLAY_HEIGHT="480"
      fi
      return 0
      ;;
    hyperpixel4sq)
      if [[ -z "$DISPLAY_WIDTH" || -z "$DISPLAY_HEIGHT" ]]; then
        DISPLAY_WIDTH="720"
        DISPLAY_HEIGHT="720"
      fi
      return 0
      ;;
    *)
      warn "Invalid HYPERPIXEL_PANEL '$HYPERPIXEL_PANEL'. Falling back to auto-detection/prompt."
      HYPERPIXEL_PANEL=""
      return 1
      ;;
  esac
}

if ! validate_hyperpixel_env_overrides && ! detect_hyperpixel_panel; then
  warn "Failed to detect HyperPixel panel."
  exit 1
fi

configure_render_size
select_schedule_preset

ENV_PATH="$PROJECT_DIR/.env"
ENV_LINES=()
ENV_LINES+=("DESK_DISPLAY_OUTPUT=${DESK_DISPLAY_OUTPUT}")
ENV_LINES+=("DESK_DISPLAY_PROFILE=${DESK_DISPLAY_PROFILE:-hyperpixel_pi_zero}")
ENV_LINES+=("DESK_DISPLAY_LOW_POWER=${DESK_DISPLAY_LOW_POWER:-1}")
ENV_LINES+=("HYPERPIXEL_PANEL=${HYPERPIXEL_PANEL}")
ENV_LINES+=("DISPLAY_WIDTH=${DISPLAY_WIDTH}")
ENV_LINES+=("DISPLAY_HEIGHT=${DISPLAY_HEIGHT}")
if [[ -n "$RENDER_WIDTH" && -n "$RENDER_HEIGHT" ]]; then
  ENV_LINES+=("RENDER_WIDTH=${RENDER_WIDTH}")
  ENV_LINES+=("RENDER_HEIGHT=${RENDER_HEIGHT}")
elif [[ -n "$DISPLAY_RENDER_SCALE" ]]; then
  ENV_LINES+=("DISPLAY_RENDER_SCALE=${DISPLAY_RENDER_SCALE}")
fi
ENV_LINES+=("INSIDE_I2C_BUSES=${INSIDE_I2C_BUSES:-13}")
ENV_LINES+=("ENABLE_SCREENSHOTS=${ENABLE_SCREENSHOTS:-0}")
ENV_LINES+=("ENABLE_VIDEO=${ENABLE_VIDEO:-0}")
ENV_LINES+=("SCREEN_CONFIG_AUTOSTART=0")
ENV_LINES+=("ENABLE_WIFI_MONITOR=${ENABLE_WIFI_MONITOR:-0}")
if [[ -n "${SCREENS_CONFIG_PATH:-}" ]]; then
  ENV_LINES+=("SCREENS_CONFIG_PATH=${SCREENS_CONFIG_PATH}")
fi
if [[ -n "${SCREENS_CONFIG_LOCAL_PATH:-}" ]]; then
  ENV_LINES+=("SCREENS_CONFIG_LOCAL_PATH=${SCREENS_CONFIG_LOCAL_PATH}")
fi
if [[ -n "${DISPLAY_ROTATION:-}" ]]; then
  ENV_LINES+=("DISPLAY_ROTATION=${DISPLAY_ROTATION}")
fi

prepend_env_vars "$ENV_PATH" "${ENV_LINES[@]}"

if [[ -e /dev/dri/card0 ]]; then
  log "DRM device detected at /dev/dri/card0"
else
  warn "DRM device /dev/dri/card0 not found. Ensure your HyperPixel dtoverlay is configured correctly."
fi

if [[ "${DESK_DISPLAY_OUTPUT}" == "kernel" ]]; then
  if detect_desktop_session "$SERVICE_USER"; then
    log "Detected an active Wayland/X11 session; HyperPixel will use $USER_SERVICE_NAME."
  elif [[ "${AUTO_FALLBACK_FRAMEBUFFER:-1}" == "1" ]]; then
    warn "No active Wayland/X11 session detected."
    warn "Switching DESK_DISPLAY_OUTPUT from kernel to framebuffer for Lite/headless startup reliability."
    DESK_DISPLAY_OUTPUT="framebuffer"
    export DESK_DISPLAY_OUTPUT

    prepend_env_vars "$ENV_PATH" "DESK_DISPLAY_OUTPUT=${DESK_DISPLAY_OUTPUT}"

    if framebuffer_device=$(detect_framebuffer_device); then
      prepend_env_vars "$ENV_PATH" "DISPLAY_FB_DEVICE=${framebuffer_device}"
      log "Configured DISPLAY_FB_DEVICE=${framebuffer_device} for framebuffer fallback."
    else
      prepend_env_vars "$ENV_PATH" "DISPLAY_FB_DEVICE=/dev/fb0"
      warn "Unable to detect framebuffer device automatically; set DISPLAY_FB_DEVICE=/dev/fb0."
    fi
  else
    warn "No active Wayland/X11 session detected."
    warn "Keeping DESK_DISPLAY_OUTPUT=${DESK_DISPLAY_OUTPUT}. Set AUTO_FALLBACK_FRAMEBUFFER=1 to auto-switch on Lite/headless systems."
  fi
else
  log "DESK_DISPLAY_OUTPUT=${DESK_DISPLAY_OUTPUT}; HyperPixel will use $SYSTEM_SERVICE_NAME."
fi

if [[ "${DESK_DISPLAY_OUTPUT}" == "kernel" ]]; then
  export SKIP_SYSTEM_DISPLAY_SERVICE="1"
  log "HyperPixel runtime choice: $USER_SERVICE_NAME will run the display loop; $SYSTEM_SERVICE_NAME will stay disabled."
else
  export SKIP_SYSTEM_DISPLAY_SERVICE="0"
  log "HyperPixel runtime choice: $SYSTEM_SERVICE_NAME will run the display loop; $USER_SERVICE_NAME will stay disabled."
fi

"$PROJECT_DIR/scripts/helpers/base_setup.sh"

if [[ "${DESK_DISPLAY_OUTPUT}" == "kernel" ]]; then
  install_kernel_user_service "$PROJECT_DIR" "$SERVICE_USER" "$USER_SERVICE_TEMPLATE" "$USER_SERVICE_NAME"
  enable_user_linger "$SERVICE_USER"
  disable_system_display_service
else
  disable_user_kernel_service "$SERVICE_USER" "$USER_SERVICE_NAME"
fi

if command -v systemctl >/dev/null 2>&1; then
  host=$(hostname)
  if [[ "${DESK_DISPLAY_OUTPUT}" == "kernel" ]]; then
    uid=$(id -u "$SERVICE_USER" 2>/dev/null || true)
    if [[ -n "$uid" ]]; then
      cat <<EOF
SSH service control commands:
  ssh ${SERVICE_USER}@${host} '${PROJECT_DIR}/scripts/ssh_kernel_display.sh status'
  ssh ${SERVICE_USER}@${host} '${PROJECT_DIR}/scripts/ssh_kernel_display.sh restart'
  ssh ${SERVICE_USER}@${host} '${PROJECT_DIR}/scripts/ssh_kernel_display.sh stop'
EOF
    else
      warn "Unable to resolve UID for $SERVICE_USER; skipping SSH command hints."
    fi
  else
    cat <<EOF
SSH service control commands:
  ssh ${SERVICE_USER}@${host} 'sudo systemctl status ${SYSTEM_SERVICE_NAME}'
  ssh ${SERVICE_USER}@${host} 'sudo systemctl restart ${SYSTEM_SERVICE_NAME}'
  ssh ${SERVICE_USER}@${host} 'sudo systemctl stop ${SYSTEM_SERVICE_NAME}'
EOF
  fi
fi
