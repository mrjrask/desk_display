#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
SERVICE_NAME="desk_display.service"

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
export REQUIREMENTS_FILE="requirements/kernel.txt"
export DISABLE_SPI_I2C="1"

HYPERPIXEL_PANEL="${HYPERPIXEL_PANEL:-}"
DISPLAY_WIDTH="${DISPLAY_WIDTH:-}"
DISPLAY_HEIGHT="${DISPLAY_HEIGHT:-}"

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

ENV_PATH="$PROJECT_DIR/.env"
ENV_LINES=()
ENV_LINES+=("DESK_DISPLAY_OUTPUT=${DESK_DISPLAY_OUTPUT}")
ENV_LINES+=("HYPERPIXEL_PANEL=${HYPERPIXEL_PANEL}")
ENV_LINES+=("DISPLAY_WIDTH=${DISPLAY_WIDTH}")
ENV_LINES+=("DISPLAY_HEIGHT=${DISPLAY_HEIGHT}")
ENV_LINES+=("INSIDE_I2C_BUSES=${INSIDE_I2C_BUSES:-13}")
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
    log "Detected an active Wayland/X11 session; HyperPixel will use kernel-mode output via $SERVICE_NAME."
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
fi

disable_legacy_kernel_user_service "$SERVICE_USER" "$SERVICE_NAME"

"$PROJECT_DIR/scripts/helpers/base_setup.sh"

if command -v systemctl >/dev/null 2>&1; then
  cat <<EOF
Service control commands:
  sudo systemctl status ${SERVICE_NAME}
  sudo systemctl restart ${SERVICE_NAME}
  sudo systemctl stop ${SERVICE_NAME}
  sudo journalctl -u ${SERVICE_NAME} -f
EOF
fi
