#!/usr/bin/env bash
set -euo pipefail

EXPECTED_CODENAME="bookworm"
SERVICE_NAME="desk_display.service"
PYTHON_BIN="${PYTHON:-python3}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
MAINTENANCE_DIR="$PROJECT_DIR/tools/maintenance"

COMMON_SCRIPT="$PROJECT_DIR/scripts/install_common.sh"
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

if [[ "${DISABLE_SPI_I2C:-}" == "1" ]]; then
  log "Disabling SPI/I2C when raspi-config is available (Hyperpixel panels require this)."
  if command -v raspi-config >/dev/null 2>&1; then
    $SUDO raspi-config nonint do_spi 1 || warn "Failed to disable SPI via raspi-config."
    $SUDO raspi-config nonint do_i2c 1 || warn "Failed to disable I2C via raspi-config."
  else
    warn "raspi-config not found; skipping SPI/I2C disablement."
  fi
else
  log "Enabling SPI/I2C when raspi-config is available."
  if command -v raspi-config >/dev/null 2>&1; then
    $SUDO raspi-config nonint do_spi 0 || warn "Failed to enable SPI via raspi-config."
    $SUDO raspi-config nonint do_i2c 0 || warn "Failed to enable I2C via raspi-config."
  else
    warn "raspi-config not found; skipping SPI/I2C enablement."
  fi
fi

install_apt_packages

if [[ ! -d "$PROJECT_DIR" ]]; then
  log "Creating project directory: $PROJECT_DIR"
  mkdir -p "$PROJECT_DIR"
fi

if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  warn "No git repository detected in $PROJECT_DIR. Clone the project before running this installer."
fi

VENV_DIR="$PROJECT_DIR/venv"
EXISTING_VENV=$(detect_existing_venv "$PROJECT_DIR" || true)
if [[ -n "$EXISTING_VENV" ]]; then
  VENV_DIR="$EXISTING_VENV"
  log "Found existing virtual environment at $VENV_DIR"
fi

if [[ ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
  if [[ -d "$VENV_DIR" ]]; then
    warn "$VENV_DIR exists but does not look like a virtual environment. Recreating."
  fi
  log "Creating virtual environment with $PYTHON_BIN at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  log "Virtual environment already exists at $VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

if [[ -f "$PROJECT_DIR/$REQUIREMENTS_FILE" ]]; then
  log "Installing Python dependencies from $REQUIREMENTS_FILE"
  pushd "$PROJECT_DIR" >/dev/null
  pip install -r "$REQUIREMENTS_FILE"
  popd >/dev/null
else
  warn "$REQUIREMENTS_FILE not found; skipping pip install."
fi

ensure_executable "$MAINTENANCE_DIR/cleanup.sh"
ensure_executable "$MAINTENANCE_DIR/reset_screenshots.sh"
ensure_executable "$PROJECT_DIR/scripts/framebuffer_service.sh"

deactivate

SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
SERVICE_ENV_LINES=()

add_service_env() {
  local key="$1"
  local value="$2"

  if [[ -n "$value" ]]; then
    SERVICE_ENV_LINES+=("Environment=${key}=${value}")
  fi
}

add_service_env "DESK_DISPLAY_OUTPUT" "${DESK_DISPLAY_OUTPUT:-}"
add_service_env "DISPLAY_FB_DEVICE" "${DISPLAY_FB_DEVICE:-}"
add_service_env "DISPLAY_FB_PIXEL_FORMAT" "${DISPLAY_FB_PIXEL_FORMAT:-}"
add_service_env "DISPLAY_FB_PIXEL_ORDER" "${DISPLAY_FB_PIXEL_ORDER:-}"
add_service_env "DISPLAY_WIDTH" "${DISPLAY_WIDTH:-}"
add_service_env "DISPLAY_HEIGHT" "${DISPLAY_HEIGHT:-}"
add_service_env "DISPLAY_ROTATION" "${DISPLAY_ROTATION:-}"
FRAMEBUFFER_PRESTART_LINES=()
FRAMEBUFFER_POSTSTOP_LINES=()
FRAMEBUFFER_UNIT_LINES=()
if [[ "${DESK_DISPLAY_OUTPUT:-}" == "framebuffer" ]]; then
  FRAMEBUFFER_PRESTART_LINES=(
    "PermissionsStartOnly=true"
    "ExecStartPre=/bin/bash -lc '$PROJECT_DIR/scripts/framebuffer_service.sh start'"
  )
  FRAMEBUFFER_POSTSTOP_LINES=(
    "ExecStopPost=/bin/bash -lc '$PROJECT_DIR/scripts/framebuffer_service.sh stop'"
  )
  FRAMEBUFFER_UNIT_LINES=(
    "After=display-manager.service"
    "Conflicts=display-manager.service"
  )
fi
log "Writing systemd service to $SERVICE_PATH"
$SUDO tee "$SERVICE_PATH" >/dev/null <<SERVICE
[Unit]
Description=Desk Display Service - main
After=network-online.target
$(printf '%s\n' "${FRAMEBUFFER_UNIT_LINES[@]}")

[Service]
WorkingDirectory=$PROJECT_DIR
$(printf '%s\n' "${SERVICE_ENV_LINES[@]}")
$(printf '%s\n' "${FRAMEBUFFER_PRESTART_LINES[@]}")
ExecStart=$VENV_DIR/bin/python $PROJECT_DIR/main.py
ExecStop=/bin/bash -lc '$MAINTENANCE_DIR/cleanup.sh'
$(printf '%s\n' "${FRAMEBUFFER_POSTSTOP_LINES[@]}")
Restart=always
User=$SERVICE_USER

[Install]
WantedBy=multi-user.target
SERVICE

log "Reloading systemd, enabling and starting $SERVICE_NAME"
$SUDO systemctl daemon-reload
$SUDO systemctl enable "$SERVICE_NAME"
$SUDO systemctl restart "$SERVICE_NAME"

log "Installation complete. Service status:"
$SUDO systemctl status --no-pager "$SERVICE_NAME"
