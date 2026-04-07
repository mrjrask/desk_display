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

if [[ "$(uname -s)" != "Darwin" ]]; then
  warn "This installer is intended for macOS; continuing anyway."
fi

ENV_PATH="$PROJECT_DIR/.env"
WINDOW_SCALE="${DESK_DISPLAY_WINDOW_SCALE:-2}"

ENV_LINES=(
  "DESK_DISPLAY_OUTPUT=kernel"
  "HYPERPIXEL_PANEL=hyperpixel4"
  "DISPLAY_WIDTH=800"
  "DISPLAY_HEIGHT=480"
  "DISPLAY_ROTATION=0"
  "DESK_DISPLAY_SDL_FULLSCREEN=0"
  "DESK_DISPLAY_WINDOW_RESIZABLE=1"
  "DESK_DISPLAY_WINDOW_SCALE=${WINDOW_SCALE}"
)

prepend_env_vars "$ENV_PATH" "${ENV_LINES[@]}"

log "Updated $ENV_PATH with a macOS windowed HyperPixel 4 profile."
log "Install Python dependencies with:"
log "  ./scripts/update_dependencies.sh --output kernel"
log "Run the app with:"
log "  ./scripts/launch_macos_window.sh"
