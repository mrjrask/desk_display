#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
SERVICE_NAME="desk_display.service"

COMMON_SCRIPT="$SCRIPT_DIR/install_common.sh"
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

export DESK_DISPLAY_OUTPUT="${DESK_DISPLAY_OUTPUT:-framebuffer}"
export DISPLAY_FB_DEVICE="${DISPLAY_FB_DEVICE:-/dev/fb0}"
export REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements_framebuffer.txt}"

declare -A RESOLUTION_MAP=(
  ["hyperpixel4"]="480x800"
  ["hyperpixel4-square"]="720x720"
  ["640x480"]="640x480"
  ["1080p"]="1920x1080"
  ["1440p"]="2560x1440"
  ["2k"]="2048x1080"
  ["4k"]="3840x2160"
)

print_resolution_menu() {
  cat <<'MENU'
Select a framebuffer resolution (content is tuned for 320x240 and scaled for larger panels):
  1) Hyperpixel4 - 480x800
  2) Hyperpixel4 Square - 720x720
  3) 640x480
  4) 1080p - 1920x1080
  5) 1440p - 2560x1440
  6) 2K - 2048x1080
  7) 4K - 3840x2160
MENU
}

apply_resolution() {
  local token="$1"
  local dims="${RESOLUTION_MAP[$token]}"
  if [[ -n "$dims" ]]; then
    export DISPLAY_RESOLUTION="$token"
    export DISPLAY_WIDTH="${dims%x*}"
    export DISPLAY_HEIGHT="${dims#*x}"
    if [[ "$token" == "hyperpixel4" || "$token" == "hyperpixel4-square" ]]; then
      export DISABLE_SPI_I2C=1
    fi
  fi
}

if [[ -z "${DISPLAY_WIDTH:-}" || -z "${DISPLAY_HEIGHT:-}" ]]; then
  if [[ -n "${DISPLAY_RESOLUTION:-}" ]]; then
    apply_resolution "${DISPLAY_RESOLUTION,,}"
  elif [[ -t 0 ]]; then
    print_resolution_menu
    read -r -p "Enter a number [1-7] (or press Enter to keep 320x240): " selection
    case "$selection" in
      1) apply_resolution "hyperpixel4" ;;
      2) apply_resolution "hyperpixel4-square" ;;
      3) apply_resolution "640x480" ;;
      4) apply_resolution "1080p" ;;
      5) apply_resolution "1440p" ;;
      6) apply_resolution "2k" ;;
      7) apply_resolution "4k" ;;
      *) ;;
    esac
  else
    print_resolution_menu
  fi
fi

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
add_env_line "DISPLAY_FB_DEVICE" "${DISPLAY_FB_DEVICE:-}"
add_env_line "DISPLAY_FB_PIXEL_FORMAT" "${DISPLAY_FB_PIXEL_FORMAT:-}"
add_env_line "DISPLAY_FB_PIXEL_ORDER" "${DISPLAY_FB_PIXEL_ORDER:-}"
add_env_line "DISPLAY_ROTATION" "${DISPLAY_ROTATION:-}"
add_env_line "DISABLE_SPI_I2C" "${DISABLE_SPI_I2C:-}"

prepend_env_vars "$ENV_PATH" "${ENV_LINES[@]}"

"$SCRIPT_DIR/install_trixie.sh"

install_framebuffer_launcher "$PROJECT_DIR" "$SERVICE_NAME" "$SERVICE_USER"

if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet display-manager; then
  if [[ -t 0 ]]; then
    read -r -p "Desktop display manager is active and may hide the framebuffer. Run the framebuffer launcher now? [y/N]: " launch_reply
    case "${launch_reply,,}" in
      y|yes)
        SERVICE_NAME="$SERVICE_NAME" /bin/bash -lc "$SCRIPT_DIR/launch_framebuffer.sh"
        ;;
      *) ;;
    esac
  else
    log "Desktop display manager is active. Run scripts/launch_framebuffer.sh to switch to framebuffer output."
  fi
fi
