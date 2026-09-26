#!/usr/bin/env bash
# Bring this device's installed desk_display systemd units up to date for its
# installation mode, without re-running the full hardware installer (no apt
# packages, raspi-config, or Python dependency reinstall).
#
#   ./scripts/update_services.sh                # update, then restart what changed
#   ./scripts/update_services.sh --dry-run      # only report what would change
#   ./scripts/update_services.sh --no-restart   # update units but restart nothing
#   ./scripts/update_services.sh --mode client  # override the detected mode
#
# The mode is the one Installers/install.sh --mode recorded in
# .runtime/install_mode, else whatever the installed units imply
# and DESK_DISPLAY_ROLE in .env and .env.client (python3 install_modes.py
# detect), so a .env converted with scripts/convert_env.py switches the units
# over. Per mode:
#
#   server, client, combined
#     Rewrites the mode's units (python3 install_modes.py services) from
#     service_units.py, with the service user, display output and panel
#     Environment= overrides the installer recorded, installs any that are
#     missing, and enables them. Units installed before the mode was recorded
#     keep their User= and Environment= overrides, and the mode is recorded.
#   standalone
#     The standalone installer writes desk_display.service itself (display
#     profile, framebuffer/kernel hooks), so that unit is patched in place
#     instead: moved script paths, the current shutdown settings, and
#     Conflicts= with the display client. A missing config UI unit is added.
#
# In every mode it also stops and disables project units that belong to a
# different mode (so main.py and display_client.py never fight over the
# panel), rewrites moved script paths in the add-on units (feed server,
# screenshot uploader, ADS-B collector, OLED helper, AirPlay), reloads
# systemd, restarts only the units it changed, and prints the state of every
# project unit installed here. Nothing else in a unit is touched.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
COMMON_SCRIPT="$SCRIPT_DIR/helpers/common.sh"
SYSTEMD_UNIT_DIR="${SYSTEMD_UNIT_DIR:-/etc/systemd/system}"
SYSTEMCTL="${SYSTEMCTL:-systemctl}"
PYTHON_BIN="${PYTHON:-python3}"

if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "[ERROR] Missing helper script: $COMMON_SCRIPT" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$COMMON_SCRIPT"

if [[ -z "${SUDO+x}" ]]; then
  if [[ $EUID -ne 0 ]]; then
    SUDO="sudo"
  else
    SUDO=""
  fi
fi

mode=""
dry_run=0
restart=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) mode="${2:-}"; shift 2 ;;
    --mode=*) mode="${1#*=}"; shift ;;
    --dry-run) dry_run=1; shift ;;
    --no-restart) restart=0; shift ;;
    -h|--help) sed -n '2,34p' "$0"; exit 0 ;;
    *) warn "Unknown option: $1"; exit 2 ;;
  esac
done

# Every project unit, in the order scripts/restart_services.sh restarts them.
PROJECT_UNITS=(
  desk_display_adsb_collector.service
  feed_server_desk_display.service
  desk_display_server.service
  desk_display.service
  desk_display_client.service
  desk_display_waveshare_oled.service
  screenshot_uploader_desk_display.service
  config_ui_desk_display.service
  airplay_desk_display.service
)

# Units written by their own add-on installers rather than by a mode. Only
# their moved script paths are patched.
ADDON_UNITS=(
  desk_display_adsb_collector.service
  feed_server_desk_display.service
  desk_display_waveshare_oled.service
  screenshot_uploader_desk_display.service
  airplay_desk_display.service
)

# Relative paths (from the project root) that moved during past script
# consolidations, oldest first. Add a new "old new" pair here whenever a
# future move/rename could leave stale references in installed unit files.
MOVED_PATHS=(
  "tools/maintenance/cleanup.sh scripts/cleanup.sh"
  "tools/maintenance/render_screens.py scripts/render_screens.py"
  "tools/maintenance/reset_screenshots.sh scripts/reset_screenshots.sh"
  "tools/adjust_image_assets.py scripts/adjust_image_assets.py"
  "tools/check_image_assets.py scripts/check_image_assets.py"
  "tools/convert_incorrectly_sized_images.py scripts/convert_incorrectly_sized_images.py"
  "tools/export_screen_rotation_config.py scripts/export_screen_rotation_config.py"
  "tools/font_audit.py scripts/font_audit.py"
  "tools/import_screen_rotation_config.py scripts/import_screen_rotation_config.py"
  "tools/load_default_screen_config.py scripts/load_default_screen_config.py"
  "tools/render_bears_next_season_png.py scripts/render_bears_next_season_png.py"
  "tools/update_screen_config.py scripts/update_screen_config.py"
  "tools/validate_required_files.py scripts/validate_required_files.py"
)

STANDALONE_SERVICE="desk_display.service"
CLIENT_SERVICE="desk_display_client.service"
CONFIG_UI_SERVICE="config_ui_desk_display.service"

modes() {
  "$PYTHON_BIN" "$PROJECT_DIR/install_modes.py" "$@" --project-dir "$PROJECT_DIR" \
    --systemd-dir "$SYSTEMD_UNIT_DIR"
}

# Run a state-changing command, or only describe it with --dry-run.
act() {
  if [[ $dry_run -eq 1 ]]; then
    log "[dry-run] would run: $*"
  else
    "$@"
  fi
}

contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

installed() { [[ -f "$SYSTEMD_UNIT_DIR/$1" ]]; }

unit_state() { $SYSTEMCTL "$1" "$2" 2>/dev/null | head -n1 || true; }

escape_sed_replacement() {
  printf '%s' "$1" | sed -e 's/[&\]/\\&/g'
}

# Rewrite script paths that moved in the repository, in unit text on stdin.
patch_moved_paths() {
  local text
  text=$(cat)
  local pair old_rel new_rel pattern replacement
  for pair in "${MOVED_PATHS[@]}"; do
    old_rel="${pair%% *}"
    new_rel="${pair##* }"
    if [[ ! -e "$PROJECT_DIR/$new_rel" ]]; then
      continue
    fi
    # Match any absolute-looking path ending in the old relative path
    # (regardless of which project checkout path it was installed under)
    # and rewrite it to the current project checkout's new location.
    pattern="[^[:space:]'\"]*/${old_rel//./\\.}"
    replacement=$(escape_sed_replacement "$PROJECT_DIR/$new_rel")
    text=$(printf '%s\n' "$text" | sed -E "s#${pattern}#${replacement}#g")
  done
  printf '%s\n' "$text"
}

# The standalone unit's current shutdown settings and its Conflicts= with the
# display client, which drives the same panel.
patch_standalone_unit() {
  # Let systemd deliver SIGTERM to main.py and give its own display-safe
  # finalizer a bounded window to finish. Remove only the historical
  # cleanup.sh handler; preserve any unrelated, operator-added ExecStop.
  sed -E '\#^[[:space:]]*ExecStop=.*cleanup\.sh['"'"']?[[:space:]]*$#d' | awk -v client="$CLIENT_SERVICE" '
    /^TimeoutStopSec=/ { next }
    /^KillSignal=/ { next }
    /^Conflicts=/ && index($0, client) { has_conflict=1 }
    { lines[++n]=$0 }
    END {
      for (i = 1; i <= n; i++) {
        line = lines[i]
        if (line == "[Install]" && !inserted) {
          print "TimeoutStopSec=10"
          print "KillSignal=SIGTERM"
          inserted=1
        }
        print line
        if (line == "[Unit]" && !has_conflict) {
          print "Conflicts=" client
          has_conflict=1
        }
      }
    }
  '
}

UPDATED_UNITS=()

# Install *text* as unit *name* when it differs from what is installed.
install_unit_text() {
  local name="$1" text="$2" reason="$3"
  local path="$SYSTEMD_UNIT_DIR/$name"
  if [[ -f "$path" && "$(cat "$path")" == "$text" ]]; then
    return 0
  fi
  log "$reason: $path"
  if [[ $dry_run -eq 0 ]]; then
    local tmp_file
    tmp_file=$(mktemp)
    printf '%s\n' "$text" > "$tmp_file"
    $SUDO install -m 644 "$tmp_file" "$path"
    rm -f "$tmp_file"
  fi
  UPDATED_UNITS+=("$name")
}

if [[ ! -d "$SYSTEMD_UNIT_DIR" ]]; then
  warn "$SYSTEMD_UNIT_DIR does not exist; no systemd units to update on this machine."
  exit 0
fi

# Units and operators run these scripts directly, so a checkout that lost its
# executable bits (a copy, an archive) would fail at ExecStart.
for script in "$PROJECT_DIR"/scripts/*.sh "$PROJECT_DIR"/scripts/*.py "$PROJECT_DIR"/scripts/helpers/*.sh; do
  if [[ -f "$script" && ! -x "$script" ]]; then
    if [[ $dry_run -eq 1 ]]; then
      log "[dry-run] would mark ${script#"$PROJECT_DIR"/} executable"
    else
      chmod +x "$script" && log "Marked ${script#"$PROJECT_DIR"/} executable" \
        || warn "Could not mark ${script#"$PROJECT_DIR"/} executable"
    fi
  fi
done

if [[ -z "$mode" ]]; then
  mode=$(modes detect)
fi
case "$mode" in
  standalone|server|client|combined) ;;
  *) warn "Unknown mode: $mode (expected standalone, server, client or combined)"; exit 2 ;;
esac

mapfile -t MODE_UNITS < <(modes services --mode "$mode")
mapfile -t OTHER_MODE_UNITS < <(modes disable --mode "$mode")
if [[ -f "$PROJECT_DIR/.runtime/install_mode" ]]; then
  log "Installed mode: $mode (recorded in .runtime/install_mode)"
else
  log "Installed mode: $mode (detected from the installed units and DESK_DISPLAY_ROLE; nothing recorded yet)"
fi
log "Units for this mode: ${MODE_UNITS[*]}"

VENV_DIR=$(detect_existing_venv "$PROJECT_DIR" || true)
VENV_PYTHON="${VENV_DIR:-$PROJECT_DIR/venv}/bin/python"

unit_dir=$(mktemp -d)
trap 'rm -rf "$unit_dir"' EXIT

if [[ "$mode" == "standalone" ]]; then
  if installed "$STANDALONE_SERVICE"; then
    original=$(cat "$SYSTEMD_UNIT_DIR/$STANDALONE_SERVICE")
    patched=$(printf '%s\n' "$original" | patch_moved_paths | patch_standalone_unit)
    install_unit_text "$STANDALONE_SERVICE" "$patched" "Updating script paths and shutdown settings in"
  else
    warn "$STANDALONE_SERVICE is not installed; run Installers/install.sh to install the display service."
  fi
  if installed "$CONFIG_UI_SERVICE"; then
    original=$(cat "$SYSTEMD_UNIT_DIR/$CONFIG_UI_SERVICE")
    patched=$(printf '%s\n' "$original" | patch_moved_paths)
    install_unit_text "$CONFIG_UI_SERVICE" "$patched" "Updating script paths in"
  else
    user=$(sed -n 's/^User=//p' "$SYSTEMD_UNIT_DIR/$STANDALONE_SERVICE" 2>/dev/null | head -n1 || true)
    user_args=()
    [[ -n "$user" ]] && user_args=(--user "$user")
    modes units --mode standalone --dir "$unit_dir" --python "$VENV_PYTHON" "${user_args[@]}" >/dev/null
    install_unit_text "$CONFIG_UI_SERVICE" "$(cat "$unit_dir/$CONFIG_UI_SERVICE")" "Installing missing"
  fi
else
  # The installer records the user, display output and panel Environment=
  # overrides; recover them from the installed units when it has not yet.
  units_args=()
  if [[ ! -f "$PROJECT_DIR/.runtime/install_mode" ]]; then
    source_unit=""
    for candidate in "$CLIENT_SERVICE" desk_display_server.service "$STANDALONE_SERVICE"; do
      if installed "$candidate"; then
        source_unit="$SYSTEMD_UNIT_DIR/$candidate"
        break
      fi
    done
    if [[ -n "$source_unit" ]]; then
      user=$(sed -n 's/^User=//p' "$source_unit" | head -n1)
      [[ -n "$user" ]] && units_args+=(--user "$user")
      while IFS= read -r assignment; do
        case "${assignment%%=*}" in
          DESK_DISPLAY_ROLE|"") ;;
          DESK_DISPLAY_OUTPUT) [[ "$mode" != "server" ]] && units_args+=(--output "${assignment#*=}") ;;
          *) units_args+=(--env "$assignment") ;;
        esac
      done < <(sed -n 's/^Environment=//p' "$source_unit")
      log "Keeping the service user and Environment= overrides from $source_unit"
    fi
  fi
  modes units --mode "$mode" --dir "$unit_dir" --python "$VENV_PYTHON" "${units_args[@]}" >/dev/null
  for name in "${MODE_UNITS[@]}"; do
    if installed "$name"; then
      reason="Rewriting from service_units.py"
    else
      reason="Installing missing"
    fi
    install_unit_text "$name" "$(cat "$unit_dir/$name")" "$reason"
  done
  if [[ ! -f "$PROJECT_DIR/.runtime/install_mode" ]]; then
    log "Recording the $mode install in .runtime/install_mode"
    act modes mark --mode "$mode" "${units_args[@]}" >/dev/null
  fi
fi

for name in "${ADDON_UNITS[@]}"; do
  if installed "$name"; then
    original=$(cat "$SYSTEMD_UNIT_DIR/$name")
    install_unit_text "$name" "$(printf '%s\n' "$original" | patch_moved_paths)" "Updating script paths in"
  fi
done

for name in "${OTHER_MODE_UNITS[@]}"; do
  [[ -n "$name" ]] || continue
  installed "$name" || continue
  if [[ "$(unit_state is-enabled "$name")" == "enabled" || "$(unit_state is-active "$name")" == "active" ]]; then
    log "Stopping and disabling $name (not part of the $mode install)"
    act $SUDO "$SYSTEMCTL" disable --now "$name"
  fi
done

if [[ ${#UPDATED_UNITS[@]} -gt 0 ]]; then
  log "Reloading systemd."
  act $SUDO "$SYSTEMCTL" daemon-reload
else
  log "Every installed unit is already current."
fi

for name in "${MODE_UNITS[@]}"; do
  installed "$name" || contains "$name" "${UPDATED_UNITS[@]+"${UPDATED_UNITS[@]}"}" || continue
  if [[ "$(unit_state is-enabled "$name")" != "enabled" ]]; then
    log "Enabling $name"
    act $SUDO "$SYSTEMCTL" enable "$name"
  fi
done

if [[ $restart -eq 1 && ${#UPDATED_UNITS[@]} -gt 0 ]]; then
  for name in "${PROJECT_UNITS[@]}"; do
    contains "$name" "${UPDATED_UNITS[@]}" || continue
    contains "$name" "${OTHER_MODE_UNITS[@]+"${OTHER_MODE_UNITS[@]}"}" && continue
    log "Restarting $name"
    act $SUDO "$SYSTEMCTL" restart "$name" || warn "Restart failed for $name; see: journalctl -u $name -n 80"
  done
elif [[ ${#UPDATED_UNITS[@]} -gt 0 ]]; then
  log "Not restarting (--no-restart): ${UPDATED_UNITS[*]}"
fi

log "Project units on this device:"
for name in "${PROJECT_UNITS[@]}"; do
  if ! installed "$name"; then
    if contains "$name" "${MODE_UNITS[@]}"; then
      printf '  %-42s %s\n' "$name" "not installed (part of the $mode install)"
    fi
    continue
  fi
  role="add-on"
  if contains "$name" "${MODE_UNITS[@]}"; then
    role="$mode"
  elif contains "$name" "${OTHER_MODE_UNITS[@]+"${OTHER_MODE_UNITS[@]}"}"; then
    role="other mode"
  fi
  enabled=$(unit_state is-enabled "$name")
  active=$(unit_state is-active "$name")
  printf '  %-42s %-11s enabled=%-9s active=%s\n' "$name" "$role" "${enabled:-unknown}" "${active:-unknown}"
done
