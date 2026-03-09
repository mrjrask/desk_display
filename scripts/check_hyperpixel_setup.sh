#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH="${PROJECT_DIR}/.env"

section() {
  printf '\n=== %s ===\n' "$1"
}

show_cmd() {
  local label="$1"
  shift
  section "$label"
  if "$@"; then
    true
  else
    echo "(command exited non-zero)"
  fi
}

section "HyperPixel dtoverlay (cmdline + boot config)"
tr ' ' '\n' </proc/cmdline | rg '^dtoverlay=' || echo "No dtoverlay in /proc/cmdline"
for cfg in /boot/firmware/config.txt /boot/config.txt; do
  if [[ -f "$cfg" ]]; then
    echo "-- $cfg"
    rg -n '^(dtoverlay=|dtparam=)' "$cfg" || echo "(no dtoverlay/dtparam lines found)"
  fi
done

show_cmd "DRM/KMS devices" ls -l /dev/dri
show_cmd "Framebuffer devices" ls -l /dev/fb0 /dev/fb1

section "Display mode and connector status"
if command -v modetest >/dev/null 2>&1; then
  modetest -c 2>/dev/null | sed -n '1,180p' || echo "modetest failed"
elif command -v kmsprint >/dev/null 2>&1; then
  kmsprint || echo "kmsprint failed"
else
  echo "Neither modetest nor kmsprint is available."
fi

section "Desk Display environment"
if [[ -f "$ENV_PATH" ]]; then
  rg -n '^(DESK_DISPLAY_OUTPUT|DISPLAY_WIDTH|DISPLAY_HEIGHT|DISPLAY_ROTATION|DISPLAY_ROTATION_STRICT|DISPLAY_FB_DEVICE|HYPERPIXEL_PANEL)=' "$ENV_PATH" || true
else
  echo "No .env found at $ENV_PATH"
fi

show_cmd "System services" systemctl --no-pager --full status desk_display.service
show_cmd "User kernel service" systemctl --user --no-pager --full status desk_display-kernel.service

section "Session/display environment"
echo "XDG_SESSION_TYPE=${XDG_SESSION_TYPE:-<unset>}"
echo "WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-<unset>}"
echo "DISPLAY=${DISPLAY:-<unset>}"
echo "XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-<unset>}"
echo "XAUTHORITY=${XAUTHORITY:-<unset>}"

echo
section "Quick hints"
echo "- If DESK_DISPLAY_OUTPUT=kernel and no active desktop session exists, use framebuffer mode instead."
echo "- If you use dtoverlay rotate=..., keep DISPLAY_ROTATION=0 unless DISPLAY_ROTATION_STRICT=0 is intentional."
echo "- If desk_display-kernel.service fails over SSH, run scripts/launch_kernel_display.sh from the Pi desktop session."
