#!/usr/bin/env bash
set -euo pipefail

# Installs *only* the Feed server (feed_server.py): a lightweight Flask app
# that hosts /feed/<source> pages built from screenshots pushed to it by
# other desk_display Pis (see scripts/screenshot_uploader.py and
# Installers/install_screenshot_uploader.sh). It does not install the
# rendering stack, GPIO drivers, or the main desk_display/config UI
# services, so it's suitable for a Pi that only aggregates and displays
# screenshots from other machines.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
SERVICE_NAME="feed_server_desk_display.service"
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

if command -v apt-get >/dev/null 2>&1; then
  log "Installing Python venv/pip packages."
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip || \
    warn "Failed to install python3-venv/python3-pip via apt-get; continuing."
fi

VENV_DIR=$(detect_existing_venv "$PROJECT_DIR" || true)
if [[ -z "$VENV_DIR" ]]; then
  VENV_DIR="$PROJECT_DIR/venv"
  log "Creating virtual environment at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

log "Installing Feed server dependencies (requirements/feed_server.txt)."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements/feed_server.txt"

FEED_SERVER_HOST="${FEED_SERVER_HOST:-0.0.0.0}"
FEED_SERVER_PORT="${FEED_SERVER_PORT:-5003}"
ENV_PATH="$PROJECT_DIR/.env"

FEED_UPLOAD_TOKEN="${FEED_UPLOAD_TOKEN:-}"
if [[ -z "$FEED_UPLOAD_TOKEN" && -f "$ENV_PATH" ]]; then
  FEED_UPLOAD_TOKEN=$(grep -E '^FEED_UPLOAD_TOKEN=' "$ENV_PATH" | tail -n1 | cut -d= -f2- || true)
fi
if [[ -z "$FEED_UPLOAD_TOKEN" ]]; then
  FEED_UPLOAD_TOKEN=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
  log "Generated a new FEED_UPLOAD_TOKEN (share this with every Pi running the screenshot uploader)."
fi

ENV_LINES=(
  "FEED_SERVER_HOST=${FEED_SERVER_HOST}"
  "FEED_SERVER_PORT=${FEED_SERVER_PORT}"
  "FEED_UPLOAD_TOKEN=${FEED_UPLOAD_TOKEN}"
)
prepend_env_vars "$ENV_PATH" "${ENV_LINES[@]}"

log "Writing systemd service to $SERVICE_PATH"
$SUDO tee "$SERVICE_PATH" >/dev/null <<SERVICE
[Unit]
Description=Desk Display Service - Feed server
After=network-online.target

[Service]
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=-$PROJECT_DIR/.env
ExecStart=$VENV_DIR/bin/python $PROJECT_DIR/feed_server.py
Restart=always
RestartSec=2
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

HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo
echo "===================================================================="
echo "Feed server installed. On each Pi that should upload to it, run:"
echo
echo "  FEED_UPLOAD_URL=http://${HOST_IP:-<this-pi-ip>}:${FEED_SERVER_PORT} \\"
echo "  FEED_UPLOAD_TOKEN=${FEED_UPLOAD_TOKEN} \\"
echo "  bash ./Installers/install_screenshot_uploader.sh"
echo
echo "Feed pages will appear at http://${HOST_IP:-<this-pi-ip>}:${FEED_SERVER_PORT}/feed/<source>"
echo "===================================================================="
