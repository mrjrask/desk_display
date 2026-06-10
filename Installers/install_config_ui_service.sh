#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
SERVICE_NAME="config_ui_desk_display.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

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

VENV_DIR="$PROJECT_DIR/venv"
EXISTING_VENV=$(detect_existing_venv "$PROJECT_DIR" || true)
if [[ -n "$EXISTING_VENV" ]]; then
  VENV_DIR="$EXISTING_VENV"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[ERROR] Missing Python interpreter at $VENV_DIR/bin/python" >&2
  echo "[INFO] Run one of the full installers first (for example, Installers/install_display_hat_mini.sh)." >&2
  exit 1
fi

if [[ ! -f "$PROJECT_DIR/config_ui.py" ]]; then
  echo "[ERROR] Missing config UI entrypoint at $PROJECT_DIR/config_ui.py" >&2
  exit 1
fi

SERVICE_START_DELAY_PRESTART_LINES=()
mapfile -t SERVICE_START_DELAY_PRESTART_LINES < <(systemd_start_delay_lines)

log "Writing systemd service to $SERVICE_PATH"
$SUDO tee "$SERVICE_PATH" >/dev/null <<SERVICE
[Unit]
Description=Desk Display Service - config UI
After=network-online.target

[Service]
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=-$PROJECT_DIR/.env
$(printf '%s\n' "${SERVICE_START_DELAY_PRESTART_LINES[@]}")
ExecStart=$VENV_DIR/bin/python $PROJECT_DIR/config_ui.py
Restart=always
User=$SERVICE_USER

[Install]
WantedBy=multi-user.target
SERVICE

log "Reloading systemd, enabling, and restarting $SERVICE_NAME"
$SUDO systemctl daemon-reload
$SUDO systemctl enable "$SERVICE_NAME"
$SUDO systemctl restart "$SERVICE_NAME"

log "Service status:"
$SUDO systemctl status --no-pager "$SERVICE_NAME"
