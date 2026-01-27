#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export DESK_DISPLAY_OUTPUT="${DESK_DISPLAY_OUTPUT:-displayhatmini}"

exec "$SCRIPT_DIR/install_bookworm.sh"
