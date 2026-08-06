#!/usr/bin/env bash
set -euo pipefail

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

SERVICE_NAME="desk_display.service"
CONFIG_UI_SERVICE_NAME="config_ui_desk_display.service"
KERNEL_USER_SERVICE_NAME="desk_display-kernel.service"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
CONFIG_UI_SERVICE_PATH="/etc/systemd/system/$CONFIG_UI_SERVICE_NAME"
COMMON_SCRIPT="$PROJECT_DIR/scripts/helpers/common.sh"

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

cat >&2 <<EOF

################################################################################
 WARNING: this permanently uninstalls Desk Display.

 This script will:
   - stop and disable the desk_display systemd services
   - remove the Python virtual environment
   - move .env and ~/keys/ (if present) into:
       $BACKUP_DIR
   - DELETE the entire project directory:
       $PROJECT_DIR

 This cannot be undone, other than restoring from the backup folder above.
################################################################################

EOF

if [[ -t 0 ]]; then
  read -r -p 'Type UNINSTALL (all caps) to continue: ' confirm_word
  if [[ "$confirm_word" != "UNINSTALL" ]]; then
    warn 'Confirmation not received. Aborting without changes.'
    exit 1
  fi
elif [[ "${CONFIRM_UNINSTALL:-}" != "yes" ]]; then
  warn 'Non-interactive shell and CONFIRM_UNINSTALL is not set to "yes". Aborting without changes.'
  warn 'Set CONFIRM_UNINSTALL=yes to run this uninstaller non-interactively.'
  exit 1
fi

if command -v systemctl >/dev/null 2>&1; then
  log "Stopping $SERVICE_NAME"
  $SUDO systemctl stop "$SERVICE_NAME" || warn "Failed to stop $SERVICE_NAME"
  log "Stopping $CONFIG_UI_SERVICE_NAME"
  $SUDO systemctl stop "$CONFIG_UI_SERVICE_NAME" || warn "Failed to stop $CONFIG_UI_SERVICE_NAME"
fi

stop_kernel_user_service() {
  local service_user="$1"
  local user_uid=""
  local runtime_dir=""
  local -a user_env=()
  local stopped=0

  if [[ -z "$service_user" ]]; then
    return 0
  fi

  user_uid=$(id -u "$service_user" 2>/dev/null || true)
  if [[ -n "$user_uid" ]]; then
    runtime_dir="/run/user/$user_uid"
    if [[ -d "$runtime_dir" ]]; then
      user_env+=("XDG_RUNTIME_DIR=$runtime_dir")
    fi
  fi

  log "Stopping $KERNEL_USER_SERVICE_NAME for user $service_user"
  if [[ -n "$SUDO" ]]; then
    if [[ ${#user_env[@]} -gt 0 ]] && \
      $SUDO -u "$service_user" env "${user_env[@]}" systemctl --user stop "$KERNEL_USER_SERVICE_NAME" >/dev/null 2>&1; then
      stopped=1
    elif [[ ${#user_env[@]} -eq 0 ]] && \
      $SUDO -u "$service_user" systemctl --user stop "$KERNEL_USER_SERVICE_NAME" >/dev/null 2>&1; then
      stopped=1
    fi

    if [[ $stopped -eq 0 ]] && \
      $SUDO systemctl --quiet --machine="${service_user}@.host" --user status "$KERNEL_USER_SERVICE_NAME" >/dev/null 2>&1; then
      if $SUDO systemctl --machine="${service_user}@.host" --user stop "$KERNEL_USER_SERVICE_NAME" >/dev/null 2>&1; then
        stopped=1
      fi
    fi
  else
    if [[ ${#user_env[@]} -gt 0 ]] && env "${user_env[@]}" systemctl --user stop "$KERNEL_USER_SERVICE_NAME" >/dev/null 2>&1; then
      stopped=1
    elif [[ ${#user_env[@]} -eq 0 ]] && systemctl --user stop "$KERNEL_USER_SERVICE_NAME" >/dev/null 2>&1; then
      stopped=1
    fi
  fi

  if [[ $stopped -eq 0 ]]; then
    warn "Failed to stop $KERNEL_USER_SERVICE_NAME for $service_user"
  fi
}

if command -v systemctl >/dev/null 2>&1; then
  declare -a kernel_service_users=()
  if [[ -n "${DESK_DISPLAY_SESSION_USER:-}" ]]; then
    kernel_service_users+=("$DESK_DISPLAY_SESSION_USER")
  fi
  if [[ -n "${SUDO_USER:-}" ]]; then
    kernel_service_users+=("$SUDO_USER")
  fi
  if [[ -z "${SUDO_USER:-}" ]]; then
    kernel_service_users+=("$(whoami)")
  fi

  declare -A seen_kernel_users=()
  for service_user in "${kernel_service_users[@]}"; do
    if [[ -n "$service_user" && -z "${seen_kernel_users[$service_user]:-}" ]]; then
      seen_kernel_users["$service_user"]=1
      if [[ "$service_user" != "root" ]]; then
        stop_kernel_user_service "$service_user"
      fi
    fi
  done
fi

log "Starting uninstall for $PROJECT_DIR"

if [[ -f "$COMMON_SCRIPT" ]]; then
  # shellcheck source=/dev/null
  source "$COMMON_SCRIPT"
fi

VENV_DIR="$PROJECT_DIR/venv"
EXISTING_VENV=$(detect_existing_venv "$PROJECT_DIR" || true)
if [[ -n "$EXISTING_VENV" ]]; then
  VENV_DIR="$EXISTING_VENV"
fi

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

  log "Reloading systemd daemon"
  $SUDO systemctl daemon-reload || warn "Failed to reload systemd daemon"
else
  warn "systemctl not found; skipping service removal"
fi

if [[ -d "$VENV_DIR" ]]; then
  keep_choice="${KEEP_VENV:-}"

  if [[ -z "$keep_choice" && -t 0 ]]; then
    read -r -p "Keep virtual environment at $VENV_DIR? [y/N]: " keep_reply
    case "${keep_reply,,}" in
      y|yes) keep_choice="yes" ;;
      *) keep_choice="no" ;;
    esac
  fi

  if [[ "$keep_choice" == "1" || "$keep_choice" == "yes" ]]; then
    log "Keeping virtual environment at $VENV_DIR"
  else
    log "Removing virtual environment at $VENV_DIR"
    rm -rf "$VENV_DIR"
  fi
else
  warn "No virtual environment found at $VENV_DIR"
fi

move_to_backup() {
  local src="$1"
  local name
  name=$(basename -- "$src")
  local dest="$BACKUP_DIR/$name"

  if [[ -e "$dest" ]]; then
    dest="$BACKUP_DIR/${name}_$(date +%Y%m%d%H%M%S)"
  fi

  mkdir -p "$BACKUP_DIR"
  log "Moving $src to $dest"
  mv "$src" "$dest"

  if [[ $EUID -eq 0 && -n "$REAL_USER" && "$REAL_USER" != "root" ]]; then
    chown -R "$REAL_USER" "$BACKUP_DIR" 2>/dev/null || true
  fi
}

ENV_FILE="$PROJECT_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  move_to_backup "$ENV_FILE"
else
  warn "No .env file found at $ENV_FILE"
fi

KEYS_DIR="$REAL_HOME/keys"
if [[ -d "$KEYS_DIR" ]]; then
  move_to_backup "$KEYS_DIR"
else
  warn "No keys folder found at $KEYS_DIR"
fi

log "Sensitive files (.env, keys) moved to $BACKUP_DIR if present"

if [[ -z "$PROJECT_DIR" || "$PROJECT_DIR" == "/" || "$PROJECT_DIR" == "$HOME" || "$PROJECT_DIR" == "$REAL_HOME" ]]; then
  warn "Refusing to delete suspicious project directory: $PROJECT_DIR"
else
  keep_choice="${KEEP_PROJECT_DIR:-}"

  if [[ -z "$keep_choice" && -t 0 ]]; then
    read -r -p "Delete project directory $PROJECT_DIR? [Y/n]: " keep_reply
    case "${keep_reply,,}" in
      n|no) keep_choice="yes" ;;
      *) keep_choice="no" ;;
    esac
  fi

  if [[ "$keep_choice" == "1" || "$keep_choice" == "yes" ]]; then
    log "Keeping project directory at $PROJECT_DIR"
    log "Uninstall complete. Project files remain in $PROJECT_DIR"
  else
    log "Removing project directory $PROJECT_DIR"
    cd "$REAL_HOME" 2>/dev/null || cd /
    rm -rf -- "$PROJECT_DIR"
    log "Uninstall complete. $PROJECT_DIR removed."
  fi
fi
