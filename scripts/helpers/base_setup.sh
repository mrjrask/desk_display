#!/usr/bin/env bash
set -euo pipefail

EXPECTED_CODENAME="${EXPECTED_CODENAME:-bookworm}"
SERVICE_NAME="desk_display.service"
CONFIG_UI_SERVICE_NAME="config_ui_desk_display.service"
PYTHON_BIN="${PYTHON:-python3}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements/displayhatmini.txt}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
SERVICE_USER="${SUDO_USER:-$(whoami)}"
# standalone (the default), server, client or combined; see install_modes.py.
INSTALL_MODE="${DESK_DISPLAY_INSTALL_MODE:-standalone}"

COMMON_SCRIPT="$SCRIPT_DIR/common.sh"
if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "[ERROR] Missing common installer helpers at $COMMON_SCRIPT" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$COMMON_SCRIPT"

if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

if [[ "$INSTALL_MODE" == "server" ]]; then
  log "Server install: leaving SPI/I2C and panel hardware settings alone."
elif [[ "${DISABLE_SPI_I2C:-}" == "1" ]]; then
  log "Disabling SPI/I2C when raspi-config is available (Hyperpixel panels require this)."
  if command -v raspi-config >/dev/null 2>&1; then
    $SUDO raspi-config nonint do_spi 1 || warn "Failed to disable SPI via raspi-config."
    $SUDO raspi-config nonint do_i2c 1 || warn "Failed to disable I2C via raspi-config."
  else
    warn "raspi-config not found; skipping SPI/I2C disablement."
  fi
else
  log "Enabling SPI/I2C when raspi-config is available."
  if command -v raspi-config >/dev/null 2>&1; then
    $SUDO raspi-config nonint do_spi 0 || warn "Failed to enable SPI via raspi-config."
    $SUDO raspi-config nonint do_i2c 0 || warn "Failed to enable I2C via raspi-config."
  else
    warn "raspi-config not found; skipping SPI/I2C enablement."
  fi
fi

install_apt_packages

if [[ ! -d "$PROJECT_DIR" ]]; then
  log "Creating project directory: $PROJECT_DIR"
  mkdir -p "$PROJECT_DIR"
fi

if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  warn "No git repository detected in $PROJECT_DIR. Clone the project before running this installer."
fi

if [[ "$INSTALL_MODE" != "standalone" ]]; then
  # Server and client installs take their own dependency sets: a client gets
  # no upstream provider libraries, a server no panel driver.
  REQUIREMENTS_FILE=$("$PYTHON_BIN" "$PROJECT_DIR/install_modes.py" requirements \
    --mode "$INSTALL_MODE" --output "${DESK_DISPLAY_OUTPUT:-}")
  export DESK_DISPLAY_PANEL_ENV_FILE="${DESK_DISPLAY_PANEL_ENV_FILE:-$([[ "$INSTALL_MODE" == "server" ]] && echo none || echo .env.client)}"
fi

VENV_DIR="$PROJECT_DIR/venv"
"$PROJECT_DIR/scripts/update_dependencies.sh" \
  --python "$PYTHON_BIN" \
  --requirements "$REQUIREMENTS_FILE" \
  --output "${DESK_DISPLAY_OUTPUT:-}"

EXISTING_VENV=$(detect_existing_venv "$PROJECT_DIR" || true)
if [[ -n "$EXISTING_VENV" ]]; then
  VENV_DIR="$EXISTING_VENV"
fi

# cleanup.sh remains available as an operator-invoked maintenance utility, but
# normal service shutdown is owned by main.py's SIGTERM/finalization path.
ensure_executable "$PROJECT_DIR/scripts/cleanup.sh"
ensure_executable "$PROJECT_DIR/scripts/reset_screenshots.sh"
ensure_executable "$PROJECT_DIR/scripts/framebuffer_service.sh"
ensure_executable "$PROJECT_DIR/scripts/prepare_kernel_session_env.sh"
ensure_executable "$PROJECT_DIR/scripts/wait_for_display_ready.sh"

# Server, client and combined installs: the env files and units come from
# install_modes.py, which also names the services to disable, so a mode switch
# never leaves main.py and display_client.py fighting over the panel.
install_mode_services() {
  local venv_python="$VENV_DIR/bin/python"
  local unit_dir
  unit_dir=$(mktemp -d)
  local -a env_args=()
  local key
  for key in DISPLAY_FB_DEVICE DISPLAY_FB_PIXEL_FORMAT DISPLAY_FB_PIXEL_ORDER DISPLAY_WIDTH DISPLAY_HEIGHT DISPLAY_ROTATION; do
    if [[ -n "${!key:-}" ]]; then
      env_args+=(--env "$key=${!key}")
    fi
  done

  log "Preparing env files for the $INSTALL_MODE install."
  local -a prepare_args=(--mode "$INSTALL_MODE" --project-dir "$PROJECT_DIR")
  [[ -n "${DESK_DISPLAY_INSTALL_PROFILE:-}" ]] && prepare_args+=(--install-profile "$DESK_DISPLAY_INSTALL_PROFILE")
  [[ -n "${DESK_DISPLAY_CLIENT_CREDENTIALS:-}" ]] && prepare_args+=(--credentials "$DESK_DISPLAY_CLIENT_CREDENTIALS")
  (cd "$PROJECT_DIR" && "$venv_python" -m install_modes prepare-env "${prepare_args[@]}")

  log "Writing systemd units for the $INSTALL_MODE install."
  local -a unit_output=()
  if [[ "$INSTALL_MODE" != "server" ]]; then
    unit_output=(--output "${DESK_DISPLAY_OUTPUT:-}")
  fi
  "$PYTHON_BIN" "$PROJECT_DIR/install_modes.py" units --mode "$INSTALL_MODE" --dir "$unit_dir" \
    --project-dir "$PROJECT_DIR" --python "$venv_python" --user "$SERVICE_USER" \
    "${unit_output[@]}" "${env_args[@]}" >/dev/null

  local unit
  while read -r unit; do
    [[ -n "$unit" ]] || continue
    if [[ -e "/etc/systemd/system/$unit" ]]; then
      log "Disabling $unit (not part of the $INSTALL_MODE install)"
      $SUDO systemctl disable --now "$unit" 2>/dev/null || true
    fi
  done < <("$PYTHON_BIN" "$PROJECT_DIR/install_modes.py" disable --mode "$INSTALL_MODE")

  local path
  for path in "$unit_dir"/*.service; do
    log "Installing $(basename "$path") to /etc/systemd/system"
    $SUDO install -m 644 "$path" "/etc/systemd/system/$(basename "$path")"
  done
  rm -rf "$unit_dir"

  # Recorded so scripts/upgrade.sh rewrites these exact units.
  "$PYTHON_BIN" "$PROJECT_DIR/install_modes.py" mark --mode "$INSTALL_MODE" --project-dir "$PROJECT_DIR" \
    --user "$SERVICE_USER" "${unit_output[@]}" "${env_args[@]}" >/dev/null
  $SUDO systemctl daemon-reload
  # Server first, then the panel: the client never waits for the server, so it
  # draws from its cache immediately (see install_modes.start_order).
  while read -r unit; do
    [[ -n "$unit" ]] || continue
    $SUDO systemctl enable "$unit"
    $SUDO systemctl restart "$unit"
  done < <("$PYTHON_BIN" "$PROJECT_DIR/install_modes.py" services --mode "$INSTALL_MODE")

  log "Installation complete ($INSTALL_MODE). Service status:"
  while read -r unit; do
    [[ -n "$unit" ]] || continue
    $SUDO systemctl status --no-pager "$unit" || true
  done < <("$PYTHON_BIN" "$PROJECT_DIR/install_modes.py" services --mode "$INSTALL_MODE")
}

if [[ "$INSTALL_MODE" != "standalone" ]]; then
  install_mode_services
  exit 0
fi

SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
CONFIG_UI_SERVICE_PATH="/etc/systemd/system/$CONFIG_UI_SERVICE_NAME"
SERVICE_ENV_LINES=()
SERVICE_ENV_OVERRIDE_LINES=()

add_service_env() {
  local key="$1"
  local value="$2"

  if [[ -n "$value" ]]; then
    SERVICE_ENV_LINES+=("Environment=${key}=${value}")
  fi
}

add_service_env "DISPLAY_FB_DEVICE" "${DISPLAY_FB_DEVICE:-}"
add_service_env "DISPLAY_FB_PIXEL_FORMAT" "${DISPLAY_FB_PIXEL_FORMAT:-}"
add_service_env "DISPLAY_FB_PIXEL_ORDER" "${DISPLAY_FB_PIXEL_ORDER:-}"
add_service_env "DISPLAY_WIDTH" "${DISPLAY_WIDTH:-}"
add_service_env "DISPLAY_HEIGHT" "${DISPLAY_HEIGHT:-}"
add_service_env "DISPLAY_ROTATION" "${DISPLAY_ROTATION:-}"

if [[ -n "${DESK_DISPLAY_OUTPUT:-}" ]]; then
  SERVICE_ENV_OVERRIDE_LINES+=("Environment=DESK_DISPLAY_OUTPUT=${DESK_DISPLAY_OUTPUT}")
fi
SERVICE_ENV_OVERRIDE_LINES+=("Environment=SCREEN_CONFIG_AUTOSTART=0")
FRAMEBUFFER_PRESTART_LINES=()
FRAMEBUFFER_POSTSTOP_LINES=()
FRAMEBUFFER_UNIT_LINES=()
if [[ "${DESK_DISPLAY_OUTPUT:-}" == "framebuffer" ]]; then
  FRAMEBUFFER_PRESTART_LINES=(
    "PermissionsStartOnly=true"
    "ExecStartPre=/bin/bash -lc 'bash $PROJECT_DIR/scripts/framebuffer_service.sh start'"
  )
  FRAMEBUFFER_POSTSTOP_LINES=(
    "ExecStopPost=/bin/bash -lc 'bash $PROJECT_DIR/scripts/framebuffer_service.sh stop'"
  )
  FRAMEBUFFER_UNIT_LINES=(
    "After=display-manager.service"
  )
fi
# Kernel-mode output draws into the desktop user's active X11/Wayland
# session, so it needs that session's DISPLAY/WAYLAND_DISPLAY/XAUTHORITY
# before ExecStart runs. prepare_kernel_session_env.sh polls for them (the
# session may not exist yet this early in boot) and writes whatever it
# finds to KERNEL_SESSION_ENV_FILE, which EnvironmentFile= below re-reads
# for ExecStart. It exits non-zero on timeout so Restart=always retries the
# whole start until the desktop session comes up.
KERNEL_PRESTART_LINES=()
KERNEL_ENV_OVERRIDE_LINES=()
KERNEL_UNIT_LINES=()
if [[ "${DESK_DISPLAY_OUTPUT:-}" == "kernel" ]]; then
  KERNEL_SESSION_ENV_FILE="$PROJECT_DIR/.runtime/kernel-session.env"
  KERNEL_PRESTART_LINES=(
    "ExecStartPre=/bin/bash -lc '$PROJECT_DIR/scripts/wait_for_display_ready.sh'"
    "ExecStartPre=/bin/bash -lc 'DESK_DISPLAY_SESSION_USER=$SERVICE_USER DESK_DISPLAY_SESSION_ENV_FILE=$KERNEL_SESSION_ENV_FILE $PROJECT_DIR/scripts/prepare_kernel_session_env.sh'"
  )
  KERNEL_ENV_OVERRIDE_LINES=(
    "EnvironmentFile=-$KERNEL_SESSION_ENV_FILE"
  )
  # Also order After=multi-user.target explicitly. This unit is
  # WantedBy=multi-user.target, so without an explicit After= on that same
  # target systemd implicitly adds Before=multi-user.target to it. Combined
  # with After=graphical.target (which itself Requires/After=multi-user.target),
  # that implicit Before= creates an ordering cycle:
  # desk_display.service -> multi-user.target -> graphical.target ->
  # desk_display.service. Systemd breaks such cycles at boot by silently
  # deleting this unit's start job, logged only under PID 1's own messages
  # (not under `journalctl -u desk_display.service`), leaving the unit
  # enabled but permanently inactive with no unit-tagged log output.
  KERNEL_UNIT_LINES=(
    "After=graphical.target"
    "Wants=graphical.target"
    "After=multi-user.target"
  )
fi
log "Writing systemd service to $SERVICE_PATH"
$SUDO tee "$SERVICE_PATH" >/dev/null <<SERVICE
[Unit]
Description=Desk Display Service - main
Wants=network-online.target
After=network-online.target
Conflicts=desk_display_client.service
$(printf '%s\n' "${FRAMEBUFFER_UNIT_LINES[@]}")
$(printf '%s\n' "${KERNEL_UNIT_LINES[@]}")

[Service]
WorkingDirectory=$PROJECT_DIR
$(printf '%s\n' "${SERVICE_ENV_LINES[@]}")
EnvironmentFile=-$PROJECT_DIR/.env
$(printf '%s\n' "${SERVICE_ENV_OVERRIDE_LINES[@]}")
$(printf '%s\n' "${FRAMEBUFFER_PRESTART_LINES[@]}")
$(printf '%s\n' "${KERNEL_PRESTART_LINES[@]}")
$(printf '%s\n' "${KERNEL_ENV_OVERRIDE_LINES[@]}")
ExecStart=$VENV_DIR/bin/python $PROJECT_DIR/main.py
$(printf '%s\n' "${FRAMEBUFFER_POSTSTOP_LINES[@]}")
TimeoutStopSec=10
KillSignal=SIGTERM
Restart=always
RestartSec=5
User=$SERVICE_USER

[Install]
WantedBy=multi-user.target
SERVICE

log "Writing systemd service to $CONFIG_UI_SERVICE_PATH"
$SUDO tee "$CONFIG_UI_SERVICE_PATH" >/dev/null <<SERVICE
[Unit]
Description=Desk Display Service - config UI
Wants=network-online.target
After=network-online.target

[Service]
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=-$PROJECT_DIR/.env
ExecStart=$VENV_DIR/bin/python $PROJECT_DIR/config_ui.py
Restart=always
User=$SERVICE_USER

[Install]
WantedBy=multi-user.target
SERVICE

log "Reloading systemd and applying service state."
$SUDO systemctl daemon-reload
$SUDO systemctl enable "$SERVICE_NAME"
$SUDO systemctl restart "$SERVICE_NAME"
$SUDO systemctl enable "$CONFIG_UI_SERVICE_NAME"
$SUDO systemctl restart "$CONFIG_UI_SERVICE_NAME"

log "Installation complete. Service status:"
$SUDO systemctl status --no-pager "$SERVICE_NAME" || true
$SUDO systemctl status --no-pager "$CONFIG_UI_SERVICE_NAME"
