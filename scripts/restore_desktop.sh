#!/usr/bin/env bash
set -euo pipefail

log_info() { printf '[INFO] %s\n' "$*"; }
log_warn() { printf '[WARN] %s\n' "$*"; }

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

main() {
  if command -v systemctl >/dev/null 2>&1 && ${SUDO:-} systemctl start display-manager; then
    log_info "display-manager started"
  else
    log_warn "Unable to start display-manager"
  fi
}

main "$@"
