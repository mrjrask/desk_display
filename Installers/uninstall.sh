#!/usr/bin/env bash
set -euo pipefail

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

SERVICE_NAME="desk_display.service"
CONFIG_UI_SERVICE_NAME="config_ui_desk_display.service"
WAVESHARE_OLED_SERVICE_NAME="desk_display_waveshare_oled.service"
WAVESHARE_FBCP_SERVICE_NAME="waveshare-fbcp.service"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
CONFIG_UI_SERVICE_PATH="/etc/systemd/system/$CONFIG_UI_SERVICE_NAME"
COMMON_SCRIPT="$PROJECT_DIR/scripts/helpers/common.sh"

# Any live foreground launcher (scripts/launch_kernel_display.sh) restarts
# the system service from an EXIT trap when it stops. This lock tells it
# (and any future cooperating script) that an uninstall is underway so it
# must not fight the uninstaller by bringing the service back.
UNINSTALL_LOCK="${DESK_DISPLAY_UNINSTALL_LOCK:-/tmp/.desk_display_uninstall_in_progress}"
export DESK_DISPLAY_UNINSTALL_LOCK="$UNINSTALL_LOCK"

if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

# When run via `sudo`, $HOME is root's home directory, not the invoking
# user's. Resolve the real user's home so backups land in their home
# folder (e.g. /home/pi) instead of /root.
REAL_USER="${SUDO_USER:-}"
if [[ $EUID -eq 0 && -n "$REAL_USER" && "$REAL_USER" != "root" ]]; then
  REAL_HOME=$(getent passwd "$REAL_USER" | cut -d: -f6)
fi
REAL_HOME="${REAL_HOME:-$HOME}"
REAL_USER="${REAL_USER:-$(whoami)}"

BACKUP_DIR="${UNINSTALL_BACKUP_DIR:-$REAL_HOME/desk_display_uninstalled}"

if [[ -f "$COMMON_SCRIPT" ]]; then
  # shellcheck source=/dev/null
  source "$COMMON_SCRIPT"
fi

VENV_DIR="$PROJECT_DIR/venv"
EXISTING_VENV=$(detect_existing_venv "$PROJECT_DIR" || true)
if [[ -n "$EXISTING_VENV" ]]; then
  VENV_DIR="$EXISTING_VENV"
fi

PROJECT_DIR_SUSPICIOUS=0
if [[ -z "$PROJECT_DIR" || "$PROJECT_DIR" == "/" || "$PROJECT_DIR" == "$HOME" || "$PROJECT_DIR" == "$REAL_HOME" ]]; then
  PROJECT_DIR_SUSPICIOUS=1
fi

cat >&2 <<EOF

################################################################################
 WARNING: this permanently uninstalls Desk Display.

 This script will:
   - stop and disable the desk_display systemd services
   - remove the Python virtual environment
   - copy .env and ~/keys/ (if present) into:
       $BACKUP_DIR
   - DELETE the entire project directory:
       $PROJECT_DIR

 This cannot be undone, other than restoring from the backup folder above.
################################################################################

EOF

# All confirmation prompts happen up front so the uninstall itself, once
# started below, runs to completion without stopping partway through to
# wait on input.

if [[ -t 0 ]]; then
  read -r -p 'Proceed with uninstall? [y/N]: ' confirm_reply
  case "${confirm_reply,,}" in
    y|yes) ;;
    *)
      warn 'Confirmation not received. Aborting without changes.'
      exit 1
      ;;
  esac
elif [[ "${CONFIRM_UNINSTALL:-}" != "yes" ]]; then
  warn 'Non-interactive shell and CONFIRM_UNINSTALL is not set to "yes". Aborting without changes.'
  warn 'Set CONFIRM_UNINSTALL=yes to run this uninstaller non-interactively.'
  exit 1
fi

keep_venv_choice="${KEEP_VENV:-}"
if [[ -d "$VENV_DIR" ]]; then
  if [[ -z "$keep_venv_choice" && -t 0 ]]; then
    read -r -p "Keep virtual environment at $VENV_DIR? [y/N]: " keep_reply
    case "${keep_reply,,}" in
      y|yes) keep_venv_choice="yes" ;;
      *) keep_venv_choice="no" ;;
    esac
  fi
fi

keep_project_dir_choice="${KEEP_PROJECT_DIR:-}"
if [[ $PROJECT_DIR_SUSPICIOUS -eq 0 ]]; then
  if [[ -z "$keep_project_dir_choice" && -t 0 ]]; then
    read -r -p "Delete project directory $PROJECT_DIR? [Y/n]: " keep_reply
    case "${keep_reply,,}" in
      n|no) keep_project_dir_choice="yes" ;;
      *) keep_project_dir_choice="no" ;;
    esac
  fi
fi

# From here on the uninstall is committed. Drop the lock so any launcher
# script (or a future re-run) knows not to restart the service out from
# under us, and make sure it's removed however the script exits.
: > "$UNINSTALL_LOCK" 2>/dev/null || $SUDO tee "$UNINSTALL_LOCK" >/dev/null 2>&1 || true
trap 'rm -f "$UNINSTALL_LOCK" 2>/dev/null || $SUDO rm -f "$UNINSTALL_LOCK" 2>/dev/null || true' EXIT

# Collect every user who might have a per-user "kernel" desk_display
# service or desktop launcher/autostart entry installed, so nothing is
# left behind to relaunch the display on next login. Do not rely solely on
# SUDO_USER/whoami: if the uninstaller is run as literal root (e.g. from a
# root console/cron, no SUDO_USER set), whoami is "root" and the real
# service-owning user would otherwise be missed entirely.
declare -a kernel_service_users=()
declare -A seen_kernel_users=()

add_kernel_service_user() {
  local candidate="$1"
  if [[ -n "$candidate" && "$candidate" != "root" && -z "${seen_kernel_users[$candidate]:-}" ]]; then
    seen_kernel_users["$candidate"]=1
    kernel_service_users+=("$candidate")
  fi
}

if [[ -n "${DESK_DISPLAY_SESSION_USER:-}" ]]; then
  add_kernel_service_user "$DESK_DISPLAY_SESSION_USER"
fi
if [[ -n "${SUDO_USER:-}" ]]; then
  add_kernel_service_user "$SUDO_USER"
fi
add_kernel_service_user "$(whoami)"

if command -v systemctl >/dev/null 2>&1; then
  system_service_owner=$($SUDO systemctl show -p User --value "$SERVICE_NAME" 2>/dev/null || true)
  add_kernel_service_user "$system_service_owner"
fi

# Fall back to scanning every home directory for a legacy per-user unit
# (older installs ran kernel-mode output as `systemctl --user`); this is
# what catches the case above (root with no SUDO_USER).
if [[ -d /home ]]; then
  for candidate_unit in /home/*/.config/systemd/user/"$SERVICE_NAME"; do
    [[ -e "$candidate_unit" ]] || continue
    candidate_home="${candidate_unit%/.config/systemd/user/*}"
    add_kernel_service_user "$(basename -- "$candidate_home")"
  done
fi

if [[ ${#kernel_service_users[@]} -eq 0 ]]; then
  warn "Could not determine which user owns the per-user kernel display service; it may not be stopped."
fi

# Kill any foreground/manual instance (e.g. launched via the "Desk Display
# (Kernel Display)" desktop icon, or a bare `python main.py`) before
# touching systemd at all. These are not managed by `systemctl stop` and,
# left running, will keep drawing to the display (and can restart the
# system service from their own exit trap) throughout the rest of this
# script. SIGKILL is used deliberately so no exit trap in a killed process
# can react and restart anything.
kill_stray_processes() {
  local pattern="$1"
  local user="${2:-}"
  if [[ -n "$user" ]]; then
    pkill -9 -u "$user" -f "$pattern" 2>/dev/null || true
  else
    pkill -9 -f "$pattern" 2>/dev/null || true
  fi
}

if command -v pkill >/dev/null 2>&1; then
  log "Killing any running Desk Display processes"
  kill_stray_processes "$PROJECT_DIR/main.py"
  kill_stray_processes "$PROJECT_DIR/scripts/launch_kernel_display.sh"
  kill_stray_processes "$PROJECT_DIR/scripts/launch_framebuffer.sh"
  for service_user in "${kernel_service_users[@]}"; do
    kill_stray_processes "main.py" "$service_user"
    kill_stray_processes "launch_kernel_display.sh" "$service_user"
    kill_stray_processes "launch_framebuffer.sh" "$service_user"
  done
fi

if command -v systemctl >/dev/null 2>&1; then
  log "Stopping $SERVICE_NAME"
  $SUDO systemctl stop "$SERVICE_NAME" || warn "Failed to stop $SERVICE_NAME"
  log "Stopping $CONFIG_UI_SERVICE_NAME"
  $SUDO systemctl stop "$CONFIG_UI_SERVICE_NAME" || warn "Failed to stop $CONFIG_UI_SERVICE_NAME"
  if systemctl list-unit-files | grep -q "^$WAVESHARE_OLED_SERVICE_NAME"; then
    log "Stopping $WAVESHARE_OLED_SERVICE_NAME"
    $SUDO systemctl stop "$WAVESHARE_OLED_SERVICE_NAME" || warn "Failed to stop $WAVESHARE_OLED_SERVICE_NAME"
  fi
  if systemctl list-unit-files | grep -q "^$WAVESHARE_FBCP_SERVICE_NAME"; then
    log "Stopping $WAVESHARE_FBCP_SERVICE_NAME"
    $SUDO systemctl stop "$WAVESHARE_FBCP_SERVICE_NAME" || warn "Failed to stop $WAVESHARE_FBCP_SERVICE_NAME"
  fi
fi

remove_desktop_entries() {
  local service_user="$1"
  local home_dir
  home_dir=$(getent passwd "$service_user" | cut -d: -f6)
  if [[ -z "$home_dir" ]]; then
    home_dir="/home/$service_user"
  fi

  local -a entries=(
    "$home_dir/.config/autostart/desk-display-kernel.desktop"
    "$home_dir/.local/share/applications/desk-display-kernel.desktop"
    "$home_dir/.local/share/applications/desk-display-framebuffer.desktop"
    "$home_dir/Desktop/Desk Display Kernel.desktop"
    "$home_dir/Desktop/Desk Display Framebuffer.desktop"
  )

  local entry
  for entry in "${entries[@]}"; do
    if [[ -e "$entry" ]]; then
      log "Removing desktop entry at $entry"
      $SUDO rm -f "$entry"
    fi
  done
}

for service_user in "${kernel_service_users[@]}"; do
  disable_legacy_kernel_user_service "$service_user" "$SERVICE_NAME"
done

log "Starting uninstall for $PROJECT_DIR"

if command -v systemctl >/dev/null 2>&1; then
  if systemctl list-unit-files | grep -q "^$SERVICE_NAME"; then
    log "Disabling $SERVICE_NAME"
    $SUDO systemctl disable "$SERVICE_NAME" || warn "Failed to disable $SERVICE_NAME"
  else
    warn "$SERVICE_NAME not registered with systemd"
  fi
  if systemctl list-unit-files | grep -q "^$CONFIG_UI_SERVICE_NAME"; then
    log "Disabling $CONFIG_UI_SERVICE_NAME"
    $SUDO systemctl disable "$CONFIG_UI_SERVICE_NAME" || warn "Failed to disable $CONFIG_UI_SERVICE_NAME"
  else
    warn "$CONFIG_UI_SERVICE_NAME not registered with systemd"
  fi
  if systemctl list-unit-files | grep -q "^$WAVESHARE_OLED_SERVICE_NAME"; then
    log "Disabling $WAVESHARE_OLED_SERVICE_NAME"
    $SUDO systemctl disable "$WAVESHARE_OLED_SERVICE_NAME" || warn "Failed to disable $WAVESHARE_OLED_SERVICE_NAME"
  fi
  if systemctl list-unit-files | grep -q "^$WAVESHARE_FBCP_SERVICE_NAME"; then
    log "Disabling $WAVESHARE_FBCP_SERVICE_NAME"
    $SUDO systemctl disable "$WAVESHARE_FBCP_SERVICE_NAME" || warn "Failed to disable $WAVESHARE_FBCP_SERVICE_NAME"
  fi

  if [[ -f "$SERVICE_PATH" ]]; then
    log "Removing systemd unit at $SERVICE_PATH"
    $SUDO rm -f "$SERVICE_PATH"
  else
    warn "No systemd unit found at $SERVICE_PATH"
  fi

  if [[ -f "$CONFIG_UI_SERVICE_PATH" ]]; then
    log "Removing systemd unit at $CONFIG_UI_SERVICE_PATH"
    $SUDO rm -f "$CONFIG_UI_SERVICE_PATH"
  else
    warn "No systemd unit found at $CONFIG_UI_SERVICE_PATH"
  fi

  for service_user in "${kernel_service_users[@]}"; do
    remove_desktop_entries "$service_user"
  done

  log "Reloading systemd daemon"
  $SUDO systemctl daemon-reload || warn "Failed to reload systemd daemon"
else
  warn "systemctl not found; skipping service removal"
fi

if [[ -d "$VENV_DIR" ]]; then
  if [[ "$keep_venv_choice" == "1" || "$keep_venv_choice" == "yes" ]]; then
    log "Keeping virtual environment at $VENV_DIR"
  else
    log "Removing virtual environment at $VENV_DIR"
    rm -rf "$VENV_DIR"
  fi
else
  warn "No virtual environment found at $VENV_DIR"
fi

copy_to_backup() {
  local src="$1"
  local name
  name=$(basename -- "$src")
  local dest="$BACKUP_DIR/$name"

  if [[ -e "$dest" ]]; then
    dest="$BACKUP_DIR/${name}_$(date +%Y%m%d%H%M%S)"
  fi

  $SUDO mkdir -p "$BACKUP_DIR"
  log "Copying $src to $dest"
  $SUDO cp -a "$src" "$dest"

  if [[ -n "$REAL_USER" && "$REAL_USER" != "root" ]]; then
    $SUDO chown -R "$REAL_USER" "$BACKUP_DIR" 2>/dev/null || true
  fi
}

ENV_FILE="$PROJECT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  copy_to_backup "$ENV_FILE"
else
  warn "No .env file found at $ENV_FILE"
fi

KEYS_DIR="$REAL_HOME/keys"
if [[ -d "$KEYS_DIR" ]]; then
  copy_to_backup "$KEYS_DIR"
else
  warn "No keys folder found at $KEYS_DIR"
fi

log "Sensitive files (.env, keys) copied to $BACKUP_DIR if present"

# Belt-and-suspenders: everything above stopped and disabled every known
# service and killed known process patterns once, near the start of the
# script. If anything raced back to life in between (for example a
# foreground launcher that was mid-exec when it was killed, or a restart
# queued just before its unit was disabled), catch it here before the
# project directory is removed, rather than leaving a live process holding
# deleted files open and still drawing to the display.
log "Verifying nothing has respawned before removing the project directory"
if command -v pkill >/dev/null 2>&1; then
  kill_stray_processes "$PROJECT_DIR/main.py"
  for service_user in "${kernel_service_users[@]}"; do
    kill_stray_processes "main.py" "$service_user"
  done
fi
if command -v systemctl >/dev/null 2>&1; then
  $SUDO systemctl stop "$SERVICE_NAME" >/dev/null 2>&1 || true
  $SUDO systemctl stop "$CONFIG_UI_SERVICE_NAME" >/dev/null 2>&1 || true
  $SUDO systemctl stop "$WAVESHARE_OLED_SERVICE_NAME" >/dev/null 2>&1 || true
  $SUDO systemctl stop "$WAVESHARE_FBCP_SERVICE_NAME" >/dev/null 2>&1 || true
  for service_user in "${kernel_service_users[@]}"; do
    disable_legacy_kernel_user_service "$service_user" "$SERVICE_NAME" >/dev/null 2>&1 || true
  done
fi

if [[ $PROJECT_DIR_SUSPICIOUS -eq 1 ]]; then
  warn "Refusing to delete suspicious project directory: $PROJECT_DIR"
else
  if [[ "$keep_project_dir_choice" == "1" || "$keep_project_dir_choice" == "yes" ]]; then
    log "Keeping project directory at $PROJECT_DIR"
    log "Uninstall complete. Project files remain in $PROJECT_DIR"
  else
    log "Removing project directory $PROJECT_DIR"
    cd "$REAL_HOME" 2>/dev/null || cd /
    rm -rf -- "$PROJECT_DIR"
    log "Uninstall complete. $PROJECT_DIR removed."
  fi
fi
