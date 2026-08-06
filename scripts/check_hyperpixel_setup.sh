#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH="${PROJECT_DIR}/.env"

find_lines() {
  local pattern="$1"
  local target="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -n "$pattern" "$target"
  else
    grep -nE "$pattern" "$target"
  fi
}

find_stream() {
  local pattern="$1"
  if command -v rg >/dev/null 2>&1; then
    rg "$pattern"
  else
    grep -E "$pattern"
  fi
}

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
tr ' ' '\n' </proc/cmdline | find_stream '^dtoverlay=' || echo "No dtoverlay in /proc/cmdline"
for cfg in /boot/firmware/config.txt /boot/config.txt; do
  if [[ -f "$cfg" ]]; then
    echo "-- $cfg"
    find_lines '^(dtoverlay=|dtparam=)' "$cfg" || echo "(no dtoverlay/dtparam lines found)"
  fi
done

show_cmd "DRM/KMS devices" ls -l /dev/dri
section "Framebuffer devices"
fb_devices=(/dev/fb*)
if [[ -e "${fb_devices[0]}" ]]; then
  ls -l /dev/fb*
else
  echo "No framebuffer devices found."
fi

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
  find_lines '^(DESK_DISPLAY_OUTPUT|DISPLAY_WIDTH|DISPLAY_HEIGHT|DISPLAY_ROTATION|DISPLAY_ROTATION_STRICT|DISPLAY_FB_DEVICE|HYPERPIXEL_PANEL)=' "$ENV_PATH" || true
else
  echo "No .env found at $ENV_PATH"
fi

show_cmd "System service (desk_display.service)" systemctl --no-pager --full status desk_display.service
show_cmd "User-session service (desk_display.service, systemctl --user)" systemctl --user --no-pager --full status desk_display.service
show_cmd "Recent system desk_display journal" journalctl -u desk_display.service -n 80 --no-pager
show_cmd "Recent user-session desk_display journal" journalctl --user-unit desk_display.service -n 80 --no-pager

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
echo "- If the user-session desk_display.service fails over SSH, run scripts/launch_kernel_display.sh from the Pi desktop session."
echo "- Both are named desk_display.service but are separate units: the system-wide one (sudo systemctl status desk_display.service) and the user-session one (systemctl --user status desk_display.service). They must not both be active; two display loops racing the same panel causes flicker/rapid color changes or a frozen/blank screen even though systemctl reports it as running. Disable whichever one you are not using."
echo "- 'sudo journalctl -u desk_display.service -f' only shows the system-wide unit's logs. For the user-session unit, use 'journalctl --user -u desk_display.service -f' as the desktop user, or 'sudo journalctl --user-unit desk_display.service -f' as root, or './scripts/ssh_kernel_display.sh logs -f'."
