#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
HELPER_PATH="$PROJECT_DIR/scripts/helpers/airplay_common.sh"

if [[ ! -f "$HELPER_PATH" ]]; then
  echo "[AIRPLAY][ERROR] Missing helper library at $HELPER_PATH" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$HELPER_PATH"
init_sudo

AIRPLAY_SERVICE_NAME="${AIRPLAY_SERVICE_NAME:-airplay_desk_display.service}"

main() {
  log_info "Enabling and restarting $AIRPLAY_SERVICE_NAME"
  if ! systemctl_safe enable --now "$AIRPLAY_SERVICE_NAME"; then
    log_error "Failed to enable/start $AIRPLAY_SERVICE_NAME"
    exit 1
  fi

  if ! systemctl_safe restart "$AIRPLAY_SERVICE_NAME"; then
    log_error "Failed to restart $AIRPLAY_SERVICE_NAME"
    exit 1
  fi

  systemctl_safe status --no-pager "$AIRPLAY_SERVICE_NAME" || true
}

main "$@"
