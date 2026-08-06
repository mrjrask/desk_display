#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${SERVICE_NAME:-desk_display.service}"
SERVICE_USER="${DESK_DISPLAY_SESSION_USER:-${SUDO_USER:-$(whoami)}}"

usage() {
  cat <<EOF
Usage: $0 <start|stop|restart|status|enable|disable|logs>

Manages the Desk Display kernel user service over SSH by ensuring
the correct user systemd environment variables are set.

Optional environment variables:
  SERVICE_NAME                Override the user service name (default: $SERVICE_NAME)
  DESK_DISPLAY_SESSION_USER   Override the user account used for systemctl --user
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

ACTION="$1"
shift || true

if ! command -v systemctl >/dev/null 2>&1; then
  echo "[ERROR] systemctl is not available on this system." >&2
  exit 1
fi

UID_VALUE=$(id -u "$SERVICE_USER" 2>/dev/null || true)
if [[ -z "$UID_VALUE" ]]; then
  echo "[ERROR] Unable to resolve UID for user '$SERVICE_USER'." >&2
  exit 1
fi

RUNTIME_DIR="/run/user/$UID_VALUE"
SYSTEMCTL_ENV=()
if [[ -d "$RUNTIME_DIR" ]]; then
  SYSTEMCTL_ENV+=("XDG_RUNTIME_DIR=$RUNTIME_DIR")
  if [[ -S "$RUNTIME_DIR/bus" ]]; then
    SYSTEMCTL_ENV+=("DBUS_SESSION_BUS_ADDRESS=unix:path=$RUNTIME_DIR/bus")
  fi
fi

SYSTEMCTL_CMD=(systemctl --user "$ACTION" "$SERVICE_NAME")
JOURNAL_CMD=(journalctl --user --unit "$SERVICE_NAME" -n 200 --no-pager)

case "$ACTION" in
  start|stop|restart|status|enable|disable)
    if [[ $EUID -ne 0 ]]; then
      env "${SYSTEMCTL_ENV[@]}" "${SYSTEMCTL_CMD[@]}"
    else
      sudo -u "$SERVICE_USER" env "${SYSTEMCTL_ENV[@]}" "${SYSTEMCTL_CMD[@]}"
    fi
    ;;
  logs)
    if [[ $EUID -ne 0 ]]; then
      env "${SYSTEMCTL_ENV[@]}" "${JOURNAL_CMD[@]}"
    else
      sudo -u "$SERVICE_USER" env "${SYSTEMCTL_ENV[@]}" "${JOURNAL_CMD[@]}"
    fi
    ;;
  *)
    usage
    exit 1
    ;;
esac
