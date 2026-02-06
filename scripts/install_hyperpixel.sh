#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
USER_SERVICE_NAME="desk_display-kernel.service"
USER_SERVICE_TEMPLATE="$SCRIPT_DIR/desk_display_kernel_user.service"

COMMON_SCRIPT="$SCRIPT_DIR/helpers/common.sh"
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

export EXPECTED_CODENAME
export DESK_DISPLAY_OUTPUT="kernel"
export REQUIREMENTS_FILE="requirements_kernel.txt"
export DISABLE_SPI_I2C="1"

HYPERPIXEL_PANEL=""
DISPLAY_WIDTH=""
DISPLAY_HEIGHT=""

export DISPLAY_ROTATION="${DISPLAY_ROTATION:-180}"

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

if ! detect_hyperpixel_panel; then
  warn "Failed to detect HyperPixel panel."
  exit 1
fi

ENV_PATH="$PROJECT_DIR/.env"
ENV_LINES=()
ENV_LINES+=("DESK_DISPLAY_OUTPUT=${DESK_DISPLAY_OUTPUT}")
ENV_LINES+=("HYPERPIXEL_PANEL=${HYPERPIXEL_PANEL}")
ENV_LINES+=("DISPLAY_WIDTH=${DISPLAY_WIDTH}")
ENV_LINES+=("DISPLAY_HEIGHT=${DISPLAY_HEIGHT}")
if [[ -n "${DISPLAY_ROTATION:-}" ]]; then
  ENV_LINES+=("DISPLAY_ROTATION=${DISPLAY_ROTATION}")
fi

prepend_env_vars "$ENV_PATH" "${ENV_LINES[@]}"

"$SCRIPT_DIR/helpers/base_setup.sh"

install_kernel_user_service "$PROJECT_DIR" "$SERVICE_USER" "$USER_SERVICE_TEMPLATE" "$USER_SERVICE_NAME"

if command -v loginctl >/dev/null 2>&1; then
  if [[ -n "$SUDO" ]]; then
    $SUDO loginctl enable-linger "$SERVICE_USER" || warn "Failed to enable linger for $SERVICE_USER."
  else
    loginctl enable-linger "$SERVICE_USER" || warn "Failed to enable linger for $SERVICE_USER."
  fi
else
  warn "loginctl not available; cannot enable linger."
fi

if [[ -e /dev/dri/card0 ]]; then
  log "DRM device detected at /dev/dri/card0"
else
  warn "DRM device /dev/dri/card0 not found. Ensure your HyperPixel dtoverlay is configured correctly."
fi

if detect_desktop_session "$SERVICE_USER"; then
  log "Detected an active Wayland/X11 session."
else
  warn "No active Wayland/X11 session detected. The kernel display can still run headless."
fi

if command -v systemctl >/dev/null 2>&1; then
  uid=$(id -u "$SERVICE_USER" 2>/dev/null || true)
  host=$(hostname)
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
fi
