#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_USER="${SUDO_USER:-$(whoami)}"

WAVESHARE_OLED_SERVICE_NAME="desk_display_waveshare_oled.service"
WAVESHARE_OLED_SERVICE_PATH="/etc/systemd/system/${WAVESHARE_OLED_SERVICE_NAME}"

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

export EXPECTED_CODENAME
export DESK_DISPLAY_OUTPUT="${DESK_DISPLAY_OUTPUT:-framebuffer}"
export REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements_framebuffer.txt}"
export DISPLAY_WIDTH="${DISPLAY_WIDTH:-320}"
export DISPLAY_HEIGHT="${DISPLAY_HEIGHT:-240}"
export DISPLAY_FB_DEVICE="${DISPLAY_FB_DEVICE:-/dev/fb1}"
export DISPLAY_ROTATION="${DISPLAY_ROTATION:-0}"
export BUTTON_A="${BUTTON_A:-24}"
export BUTTON_B="${BUTTON_B:-4}"
export BUTTON_X="${BUTTON_X:-17}"
export BUTTON_Y="${BUTTON_Y:-23}"

ENV_PATH="$PROJECT_DIR/.env"
ENV_LINES=()
ENV_LINES+=("DESK_DISPLAY_OUTPUT=${DESK_DISPLAY_OUTPUT}")
ENV_LINES+=("DISPLAY_WIDTH=${DISPLAY_WIDTH}")
ENV_LINES+=("DISPLAY_HEIGHT=${DISPLAY_HEIGHT}")
ENV_LINES+=("DISPLAY_FB_DEVICE=${DISPLAY_FB_DEVICE}")
ENV_LINES+=("DISPLAY_ROTATION=${DISPLAY_ROTATION}")
ENV_LINES+=("BUTTON_A=${BUTTON_A}")
ENV_LINES+=("BUTTON_B=${BUTTON_B}")
ENV_LINES+=("BUTTON_X=${BUTTON_X}")
ENV_LINES+=("BUTTON_Y=${BUTTON_Y}")
prepend_env_vars "$ENV_PATH" "${ENV_LINES[@]}"

log "Desk Display will render to ${DISPLAY_WIDTH}x${DISPLAY_HEIGHT} using ${DISPLAY_FB_DEVICE}."
log "This installer also enables a helper service for the OLED side displays (temperature + time)."
log "Button mapping: A=GPIO${BUTTON_A}, B=GPIO${BUTTON_B}, X=GPIO${BUTTON_X}, Y=GPIO${BUTTON_Y}."

if [[ ! -e "$DISPLAY_FB_DEVICE" ]]; then
  warn "${DISPLAY_FB_DEVICE} does not exist yet."
  warn "Install/enable the Waveshare OLED/LCD HAT (A) LCD kernel driver per the Waveshare wiki, then reboot and rerun this installer."
fi

"$PROJECT_DIR/scripts/helpers/base_setup.sh"

install_framebuffer_launcher "$PROJECT_DIR" "desk_display.service" "$SERVICE_USER" || true

ensure_executable "$PROJECT_DIR/scripts/waveshare_oled_status.py"

VENV_DIR=$(detect_existing_venv "$PROJECT_DIR" || true)
if [[ -z "$VENV_DIR" ]]; then
  VENV_DIR="$PROJECT_DIR/venv"
fi

log "Writing Waveshare OLED helper service to $WAVESHARE_OLED_SERVICE_PATH"
$SUDO tee "$WAVESHARE_OLED_SERVICE_PATH" >/dev/null <<SERVICE
[Unit]
Description=Desk Display Waveshare OLED status helper
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=-$PROJECT_DIR/.env
ExecStart=$VENV_DIR/bin/python $PROJECT_DIR/scripts/waveshare_oled_status.py
Restart=always
RestartSec=2
User=$SERVICE_USER

[Install]
WantedBy=multi-user.target
SERVICE

$SUDO systemctl daemon-reload
$SUDO systemctl enable "$WAVESHARE_OLED_SERVICE_NAME"
$SUDO systemctl restart "$WAVESHARE_OLED_SERVICE_NAME"


