#!/usr/bin/env bash
set -euo pipefail

EXPECTED_CODENAME="bookworm"
SERVICE_NAME="desk_display.service"
PYTHON_BIN="${PYTHON:-python3}"

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

log "Enabling SPI/I2C when raspi-config is available."
if command -v raspi-config >/dev/null 2>&1; then
  $SUDO raspi-config nonint do_spi 0 || warn "Failed to enable SPI via raspi-config."
  $SUDO raspi-config nonint do_i2c 0 || warn "Failed to enable I2C via raspi-config."
else
  warn "raspi-config not found; skipping SPI/I2C enablement."
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

if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
  log "Installing Python dependencies from requirements.txt"
  pushd "$PROJECT_DIR" >/dev/null
  pip install -r "requirements.txt"
  popd >/dev/null
else
  warn "requirements.txt not found; skipping pip install."
fi

ensure_executable "$MAINTENANCE_DIR/cleanup.sh"
ensure_executable "$MAINTENANCE_DIR/reset_screenshots.sh"

deactivate

SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
log "Writing systemd service to $SERVICE_PATH"
$SUDO tee "$SERVICE_PATH" >/dev/null <<SERVICE
[Unit]
Description=Desk Display Service - main
After=network-online.target

[Service]
WorkingDirectory=$PROJECT_DIR
ExecStart=$VENV_DIR/bin/python $PROJECT_DIR/main.py
ExecStop=/bin/bash -lc '$MAINTENANCE_DIR/cleanup.sh'
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
