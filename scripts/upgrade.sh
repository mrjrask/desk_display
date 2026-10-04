#!/usr/bin/env bash
# Upgrade an installed Desk Display in place, for whichever mode it runs in.
#
#   bash scripts/upgrade.sh             # pull, update dependencies, units, restart
#   bash scripts/upgrade.sh --no-pull   # the same, without git pull
#
# Every mode keeps its documented data exactly as it was (python3
# install_modes.py plan lists it): env files, client identity and cache,
# server playlists and assignments, credentials, artifacts and backups. A
# server or combined install also snapshots its state first, into
# .runtime/server/backups/upgrade-<time>/.
#
# On a client or combined install the upgrade asks at the end whether
# desk_display_client.service should be started, whatever its state before.
# The default answer is the state it was in when the upgrade began; run
# without a terminal (the Clients page Upgrade button, cron), that default is
# used, so a running client is restarted and a stopped one stays stopped.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
COMMON_SCRIPT="$SCRIPT_DIR/helpers/common.sh"
SYSTEMD_UNIT_DIR="${SYSTEMD_UNIT_DIR:-/etc/systemd/system}"
PYTHON_BIN="${PYTHON:-python3}"

# shellcheck source=/dev/null
source "$COMMON_SCRIPT"

if [[ -z "${SUDO+x}" ]]; then
  if [[ $EUID -ne 0 ]]; then SUDO="sudo"; else SUDO=""; fi
fi

pull=1
for arg in "$@"; do
  case "$arg" in
    --no-pull) pull=0 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) warn "Unknown option: $arg"; exit 2 ;;
  esac
done

CLIENT_SERVICE="desk_display_client.service"
# Recorded before anything else runs, so the restart below can respect it.
client_was_active=0
if systemctl is-active --quiet "$CLIENT_SERVICE" 2>/dev/null; then
  client_was_active=1
fi

modes() { "$PYTHON_BIN" "$PROJECT_DIR/install_modes.py" "$@" --project-dir "$PROJECT_DIR" --systemd-dir "$SYSTEMD_UNIT_DIR"; }

mode=$(modes detect)
log "Upgrading the $mode install in $PROJECT_DIR"

snapshot=$(modes snapshot --mode "$mode")
if [[ -n "$snapshot" ]]; then
  log "Snapshot of server state: $snapshot"
fi

if [[ $pull -eq 1 ]]; then
  git -C "$PROJECT_DIR" pull --ff-only
fi

requirements=$(modes requirements --mode "$mode")
if [[ "$mode" == "server" ]]; then
  export DESK_DISPLAY_PANEL_ENV_FILE="none"
elif [[ "$mode" != "standalone" ]]; then
  export DESK_DISPLAY_PANEL_ENV_FILE=".env.client"
fi
"$PROJECT_DIR/scripts/update_dependencies.sh" --python "$PYTHON_BIN" --requirements "$requirements"

# Rewrites the mode's units (patches the standalone one in place); the restart
# below brings every service back in dependency order.
bash "$PROJECT_DIR/scripts/update_services.sh" --mode "$mode" --no-restart

has_client=0
if [[ "$mode" == "client" || "$mode" == "combined" ]]; then
  has_client=1
fi

# The client is started (or not) after the question below, never here.
restart_args=()
if [[ $has_client -eq 1 || $client_was_active -eq 0 ]]; then
  restart_args+=(--skip "$CLIENT_SERVICE")
fi
run_with_heartbeat "restarting services" bash "$PROJECT_DIR/scripts/restart_services.sh" \
  "${restart_args[@]+"${restart_args[@]}"}"

# Asks only with someone at a terminal; DESK_DISPLAY_UPGRADE_INTERACTIVE=1/0
# forces it either way.
can_prompt() {
  case "${DESK_DISPLAY_UPGRADE_INTERACTIVE:-}" in
    1) return 0 ;;
    0) return 1 ;;
  esac
  [[ -t 0 ]]
}

if [[ $has_client -eq 1 ]]; then
  if [[ $client_was_active -eq 1 ]]; then
    before="running" choices="[Y/n]" start_client=1
  else
    before="not running" choices="[y/N]" start_client=0
  fi
  if can_prompt; then
    answer=""
    printf '%s was %s before the upgrade. Start it now? %s ' "$CLIENT_SERVICE" "$before" "$choices"
    read -r answer || answer=""
    case "$answer" in
      [Yy]|[Yy][Ee][Ss]) start_client=1 ;;
      [Nn]|[Nn][Oo]) start_client=0 ;;
    esac
  fi
  if [[ $start_client -eq 1 ]]; then
    log "Starting $CLIENT_SERVICE."
    if ! run_with_heartbeat "restarting the client" bash "$PROJECT_DIR/scripts/restart_services.sh" "$CLIENT_SERVICE"; then
      warn "Start failed for $CLIENT_SERVICE; see: journalctl -u $CLIENT_SERVICE -n 80"
    fi
  else
    if [[ $client_was_active -eq 1 ]]; then
      log "Stopping $CLIENT_SERVICE."
      ${SUDO:+"$SUDO"} systemctl stop "$CLIENT_SERVICE" || warn "Stop failed for $CLIENT_SERVICE."
    fi
    log "Leaving $CLIENT_SERVICE stopped. Start it later with: sudo systemctl start $CLIENT_SERVICE"
  fi
fi
log "Upgrade complete ($mode)."
