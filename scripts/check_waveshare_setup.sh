#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PATH="${PROJECT_DIR}/.env"
WAVESHARE_OLED_SERVICE_NAME="${WAVESHARE_OLED_SERVICE_NAME:-desk_display_waveshare_oled.service}"

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

section "Kernel cmdline + boot config overlays"
tr ' ' '\n' </proc/cmdline | find_stream '^dtoverlay=' || echo "No dtoverlay in /proc/cmdline"
for cfg in /boot/firmware/config.txt /boot/config.txt; do
  if [[ -f "$cfg" ]]; then
    echo "-- $cfg"
    find_lines '^(dtoverlay=|dtparam=|display_rotate=)' "$cfg" || echo "(no display-related lines found)"
  fi
done

section "Framebuffer devices"
fb_devices=(/dev/fb*)
if [[ -e "${fb_devices[0]}" ]]; then
  ls -l /dev/fb*
else
  echo "No framebuffer devices found."
fi

section "Framebuffer geometry"
for fb in /dev/fb0 /dev/fb1 /dev/fb2; do
  [[ -e "$fb" ]] || continue
  echo "-- $fb"
  if command -v fbset >/dev/null 2>&1; then
    fbset -fb "$fb" -s 2>/dev/null | sed -n '1,6p' || echo "(fbset failed)"
  else
    echo "fbset not installed"
  fi
  fb_name="$(basename "$fb")"
  for item in mode virtual_size bits_per_pixel name; do
    path="/sys/class/graphics/${fb_name}/${item}"
    if [[ -r "$path" ]]; then
      printf '  %s: %s\n' "$item" "$(cat "$path")"
    fi
  done
done

section "I2C devices"
if command -v i2cdetect >/dev/null 2>&1; then
  i2cdetect -y 1 || echo "(i2cdetect failed on bus 1)"
else
  echo "i2cdetect not installed (install i2c-tools)."
fi

section "Desk Display environment"
if [[ -f "$ENV_PATH" ]]; then
  find_lines '^(DESK_DISPLAY_OUTPUT|DISPLAY_WIDTH|DISPLAY_HEIGHT|DISPLAY_ROTATION|DISPLAY_ROTATION_STRICT|DISPLAY_FB_DEVICE|WAVESHARE_OLED_)=' "$ENV_PATH" || true
else
  echo "No .env found at $ENV_PATH"
fi

show_cmd "System service: desk_display" systemctl --no-pager --full status desk_display.service
show_cmd "System service: Waveshare OLED helper" systemctl --no-pager --full status "$WAVESHARE_OLED_SERVICE_NAME"
show_cmd "Recent desk_display journal" journalctl -u desk_display.service -n 80 --no-pager
show_cmd "Recent Waveshare OLED journal" journalctl -u "$WAVESHARE_OLED_SERVICE_NAME" -n 80 --no-pager

section "Quick hints"
echo "- White cursor on black screen usually means Linux console is active but app failed to bind framebuffer."
echo "- For this HAT, DESK_DISPLAY_OUTPUT should usually be framebuffer and DISPLAY_FB_DEVICE should match the 320x240 fb node."
echo "- OLED helper needs I2C bus 1 with addresses 0x3c and 0x3d visible."
echo "- If /dev/fb1 no longer exists after kernel updates, set DISPLAY_FB_DEVICE=/dev/fb0 then restart desk_display.service."
