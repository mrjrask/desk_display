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
  find_lines '^(DESK_DISPLAY_OUTPUT|DISPLAY_WIDTH|DISPLAY_HEIGHT|RENDER_WIDTH|RENDER_HEIGHT|DISPLAY_RENDER_SCALE|DISPLAY_ROTATION|DISPLAY_ROTATION_STRICT|DISPLAY_FB_DEVICE|HYPERPIXEL_PANEL)=' "$ENV_PATH" || true
else
  echo "No .env found at $ENV_PATH"
fi


read_env_value() {
  local key="$1"
  local target="$2"
  awk -F= -v key="$key" '$1 == key {print substr($0, length(key) + 2); exit}' "$target"
}

read_unit_output_value() {
  local target="$1"
  awk '
    /^Environment=/ {
      line = $0
      sub(/^Environment=/, "", line)
      count = split(line, parts, /[[:space:]]+/)
      for (i = 1; i <= count; i++) {
        gsub(/^"|"$/, "", parts[i])
        if (parts[i] ~ /^DESK_DISPLAY_OUTPUT=/) {
          sub(/^DESK_DISPLAY_OUTPUT=/, "", parts[i])
          print parts[i]
          exit
        }
      }
    }
  ' "$target"
}

add_unit_path() {
  local target="$1"
  [[ -n "$target" && -f "$target" ]] || return 0

  local existing
  for existing in "${UNIT_PATHS[@]}"; do
    [[ "$existing" == "$target" ]] && return 0
  done
  UNIT_PATHS+=("$target")
}

section "Desk Display output agreement"
env_output=""
if [[ -f "$ENV_PATH" ]]; then
  env_output=$(read_env_value "DESK_DISPLAY_OUTPUT" "$ENV_PATH")
  echo ".env DESK_DISPLAY_OUTPUT=${env_output:-<unset>}"
else
  echo ".env DESK_DISPLAY_OUTPUT=<missing .env>"
fi

UNIT_PATHS=()
if command -v systemctl >/dev/null 2>&1; then
  unit_fragment=$(systemctl --user show -p FragmentPath --value desk_display-kernel.service 2>/dev/null || true)
  add_unit_path "$unit_fragment"
fi
add_unit_path "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/desk_display-kernel.service"
add_unit_path "$PROJECT_DIR/scripts/desk_display_kernel_user.service"

if [[ ${#UNIT_PATHS[@]} -eq 0 ]]; then
  echo "No desk_display-kernel.service unit file found to compare."
else
  for unit_path in "${UNIT_PATHS[@]}"; do
    unit_output=$(read_unit_output_value "$unit_path")
    echo "-- $unit_path"
    if [[ -n "$unit_output" ]]; then
      echo "unit DESK_DISPLAY_OUTPUT=${unit_output}"
      if [[ -n "$env_output" && "$unit_output" != "$env_output" ]]; then
        echo "[WARN] desk_display-kernel.service sets DESK_DISPLAY_OUTPUT=${unit_output}, but .env sets DESK_DISPLAY_OUTPUT=${env_output}."
        echo "[WARN] If the unit assignment appears after EnvironmentFile, it can override .env; keep any default before EnvironmentFile or remove it."
      fi
    else
      echo "unit DESK_DISPLAY_OUTPUT=<not explicitly set; .env controls output mode>"
    fi
  done
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
