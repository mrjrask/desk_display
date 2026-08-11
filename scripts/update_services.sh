#!/usr/bin/env bash
# Patch stale script paths in already-installed desk_display systemd units
# and restart them, without re-running the full hardware installer (no apt
# packages, raspi-config, or Python dependency reinstall).
#
# Installers bake absolute script paths (e.g. into ExecStart/ExecStop) into
# unit files at install time. When scripts are moved or renamed in the repo
# (for example tools/maintenance/cleanup.sh -> scripts/cleanup.sh), an
# already-installed unit keeps pointing at the old, now-missing path until
# something rewrites it. This script rewrites known moved paths in place and
# reloads/restarts the affected services. It intentionally leaves everything
# else in each unit (display profile, Environment= overrides, etc.) alone,
# unlike re-running the installer, which regenerates the whole unit from the
# environment it happens to be run with.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
COMMON_SCRIPT="$SCRIPT_DIR/helpers/common.sh"
SYSTEMD_UNIT_DIR="${SYSTEMD_UNIT_DIR:-/etc/systemd/system}"

if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "[ERROR] Missing helper script: $COMMON_SCRIPT" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$COMMON_SCRIPT"

if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

# Known project-managed unit names. Not all of these will be installed on
# any given device; missing ones are skipped.
UNIT_NAMES=(
  desk_display.service
  config_ui_desk_display.service
  desk_display_waveshare_oled.service
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

escape_sed_replacement() {
  printf '%s' "$1" | sed -e 's/[&\]/\\&/g'
}

ensure_executable "$PROJECT_DIR/scripts/cleanup.sh"
ensure_executable "$PROJECT_DIR/scripts/reset_screenshots.sh"

UPDATED_UNITS=()

for unit_name in "${UNIT_NAMES[@]}"; do
  unit_path="$SYSTEMD_UNIT_DIR/$unit_name"
  if [[ ! -f "$unit_path" ]]; then
    continue
  fi

  original_contents=$(cat "$unit_path")
  patched_contents="$original_contents"

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
    patched_contents=$(printf '%s\n' "$patched_contents" | sed -E "s#${pattern}#${replacement}#g")
  done

  if [[ "$patched_contents" == "$original_contents" ]]; then
    log "$unit_name already points at current script locations."
    continue
  fi

  log "Updating stale script paths in $unit_path"
  tmp_file=$(mktemp)
  printf '%s\n' "$patched_contents" > "$tmp_file"
  $SUDO cp "$tmp_file" "$unit_path"
  rm -f "$tmp_file"
  UPDATED_UNITS+=("$unit_name")
done

if [[ ${#UPDATED_UNITS[@]} -eq 0 ]]; then
  log "No installed units needed updates."
  exit 0
fi

log "Reloading systemd and restarting updated units."
$SUDO systemctl daemon-reload

for unit_name in "${UPDATED_UNITS[@]}"; do
  $SUDO systemctl restart "$unit_name"
  log "Restarted $unit_name"
  $SUDO systemctl status --no-pager "$unit_name" || true
done
