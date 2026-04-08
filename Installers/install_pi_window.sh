#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
COMMON_SCRIPT="$PROJECT_DIR/scripts/helpers/common.sh"

if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "[ERROR] Missing common installer helpers at $COMMON_SCRIPT" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$COMMON_SCRIPT"

if [[ "$(uname -s)" != "Linux" ]]; then
  warn "This installer is intended for Raspberry Pi Desktop/Linux; continuing anyway."
fi

ENV_PATH="$PROJECT_DIR/.env"
WINDOW_SCALE="${DESK_DISPLAY_WINDOW_SCALE:-1}"

ENV_LINES=(
  "DESK_DISPLAY_OUTPUT=window"
  "HYPERPIXEL_PANEL=hyperpixel4"
  "DISPLAY_WIDTH=800"
  "DISPLAY_HEIGHT=480"
  "DISPLAY_ROTATION=0"
  "DESK_DISPLAY_WINDOW_SCALE=${WINDOW_SCALE}"
)

prepend_env_vars "$ENV_PATH" "${ENV_LINES[@]}"

log "Updated $ENV_PATH with a Raspberry Pi Desktop windowed profile."
log "Install Python dependencies with:"
log "  ./scripts/update_dependencies.sh"
log "Run the app with:"
log "  ./launch_pi_window.sh"
