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

main() {
  if systemctl_safe start display-manager; then
    log_info "display-manager started"
  else
    log_warn "Unable to start display-manager"
  fi
}

main "$@"
