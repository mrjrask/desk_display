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

export DESK_DISPLAY_OUTPUT="${DESK_DISPLAY_OUTPUT:-kernel}"
export REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements_kernel.txt}"

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

  if [[ -t 0 ]]; then
    read -r -p "Launch the kernel display now? [y/N] " launch_choice
    launch_choice=${launch_choice,,}
    if [[ "$launch_choice" == "y" || "$launch_choice" == "yes" ]]; then
      log "Launching the kernel display in the current desktop session."
      if [[ -n "${SUDO:-}" ]]; then
        $SUDO -u "$SERVICE_USER" /bin/bash -lc "$launcher"
      else
        /bin/bash -lc "$launcher"
      fi
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

if [[ -z "${DISPLAY_WIDTH:-}" || -z "${DISPLAY_HEIGHT:-}" ]]; then
  if [[ -n "${DISPLAY_RESOLUTION:-}" ]]; then
    apply_resolution "${DISPLAY_RESOLUTION,,}"
  else
    detected_mode=$(detect_drm_resolution || true)
    if [[ -z "$detected_mode" ]]; then
      detected_mode=$(detect_xrandr_resolution || true)
    fi
    if [[ -n "$detected_mode" ]]; then
      apply_resolution "$detected_mode"
    elif [[ -t 0 ]]; then
      print_resolution_menu
      read -r -p "Enter a number [1-5] (or press Enter to keep 320x240): " selection
      case "$selection" in
        1) apply_resolution "640x480" ;;
        2) apply_resolution "1080p" ;;
        3) apply_resolution "1440p" ;;
        4) apply_resolution "2k" ;;
        5) apply_resolution "4k" ;;
        *) ;;
      esac
    else
      print_resolution_menu
    fi
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

prepend_env_vars "$ENV_PATH" "${ENV_LINES[@]}"

prompt_spi_i2c

"$SCRIPT_DIR/install_trixie.sh"

install_kernel_launcher "$PROJECT_DIR" "$SERVICE_NAME" "$SERVICE_USER"

prompt_launch_kernel_display
