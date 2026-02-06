#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

detect_codename() {
  if command -v lsb_release >/dev/null 2>&1; then
    lsb_release -sc
    return 0
  fi
  if [[ -f /etc/os-release ]]; then
    # shellcheck source=/dev/null
    source /etc/os-release
    if [[ -n "${VERSION_CODENAME:-}" ]]; then
      echo "$VERSION_CODENAME"
      return 0
    fi
  fi
  return 1
}

if [[ -z "${EXPECTED_CODENAME:-}" ]]; then
  EXPECTED_CODENAME=$(detect_codename || true)
  if [[ -z "$EXPECTED_CODENAME" ]]; then
    echo "[WARN] Unable to detect OS codename; defaulting to bookworm." >&2
    EXPECTED_CODENAME="bookworm"
  fi
fi

export EXPECTED_CODENAME
export DESK_DISPLAY_OUTPUT="${DESK_DISPLAY_OUTPUT:-displayhatmini}"

exec "$SCRIPT_DIR/helpers/base_setup.sh"
