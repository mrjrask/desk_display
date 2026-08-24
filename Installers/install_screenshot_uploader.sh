#!/usr/bin/env bash
set -euo pipefail

# Installs the screenshot uploader service (scripts/screenshot_uploader.py):
# a lightweight background loop that pushes this Pi's current screenshots to
# a remote Feed server (see Installers/install_feed_server.sh). Run this on
# any existing desk_display install (e.g. a HyperPixel or HyperPixel Square
# Pi) that should mirror its screenshots onto a centralized Feed host.
#
# Requires FEED_UPLOAD_URL (the Feed server's base URL, e.g.
# http://192.168.1.200:5003) and FEED_UPLOAD_TOKEN (the shared secret printed
# by install_feed_server.sh) to be set in the environment, or answered at the
# interactive prompts below.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
SERVICE_NAME="screenshot_uploader_desk_display.service"
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

VENV_DIR=$(detect_existing_venv "$PROJECT_DIR" || true)
if [[ -z "$VENV_DIR" ]]; then
  echo "[ERROR] No existing virtual environment found under $PROJECT_DIR." >&2
  echo "[INFO] Run a full installer first (for example, Installers/install_hyperpixel.sh)." >&2
  exit 1
fi

if [[ ! -f "$PROJECT_DIR/scripts/screenshot_uploader.py" ]]; then
  echo "[ERROR] Missing $PROJECT_DIR/scripts/screenshot_uploader.py" >&2
  exit 1
fi

ensure_executable "$PROJECT_DIR/scripts/screenshot_uploader.py"

ENV_PATH="$PROJECT_DIR/.env"

FEED_UPLOAD_URL="${FEED_UPLOAD_URL:-}"
if [[ -z "$FEED_UPLOAD_URL" && -t 0 ]]; then
  read -r -p "Feed server URL (e.g. http://192.168.1.200:5003): " FEED_UPLOAD_URL
fi
if [[ -z "$FEED_UPLOAD_URL" ]]; then
  echo "[ERROR] FEED_UPLOAD_URL is required (set it in the environment or answer the prompt)." >&2
  exit 1
fi

FEED_UPLOAD_TOKEN="${FEED_UPLOAD_TOKEN:-}"
if [[ -z "$FEED_UPLOAD_TOKEN" && -t 0 ]]; then
  read -r -p "Feed server upload token: " FEED_UPLOAD_TOKEN
fi
if [[ -z "$FEED_UPLOAD_TOKEN" ]]; then
  echo "[ERROR] FEED_UPLOAD_TOKEN is required (set it in the environment or answer the prompt)." >&2
  exit 1
fi

FEED_SOURCE_NAME="${FEED_SOURCE_NAME:-$(hostname)}"
FEED_UPLOAD_INTERVAL_SECONDS="${FEED_UPLOAD_INTERVAL_SECONDS:-5}"

ENV_LINES=(
  "FEED_UPLOAD_URL=${FEED_UPLOAD_URL}"
  "FEED_UPLOAD_TOKEN=${FEED_UPLOAD_TOKEN}"
  "FEED_SOURCE_NAME=${FEED_SOURCE_NAME}"
  "FEED_UPLOAD_INTERVAL_SECONDS=${FEED_UPLOAD_INTERVAL_SECONDS}"
)
prepend_env_vars "$ENV_PATH" "${ENV_LINES[@]}"

log "Writing systemd service to $SERVICE_PATH"
$SUDO tee "$SERVICE_PATH" >/dev/null <<SERVICE
[Unit]
Description=Desk Display Service - screenshot uploader
After=network-online.target desk_display.service

[Service]
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=-$PROJECT_DIR/.env
ExecStart=$VENV_DIR/bin/python $PROJECT_DIR/scripts/screenshot_uploader.py
Restart=always
RestartSec=5
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

log "Uploading source '$FEED_SOURCE_NAME' screenshots to $FEED_UPLOAD_URL every ${FEED_UPLOAD_INTERVAL_SECONDS}s."
