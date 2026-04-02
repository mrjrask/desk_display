#!/usr/bin/env bash
set -euo pipefail

# Prints the current screen rotation configuration as JSON.
# Sources checked (in order):
# 1) App-level DISPLAY_ROTATION env var
# 2) Wayland transform (wlr-randr)
# 3) X11 connector rotation (xrandr)
# 4) DRM connector status/mode + panel orientation sysfs hints

json_escape() {
  local value=${1:-}
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  value=${value//$'\n'/\\n}
  printf '%s' "$value"
}

get_wayland_rotation() {
  [[ -n "${WAYLAND_DISPLAY:-}" ]] || return 1
  command -v wlr-randr >/dev/null 2>&1 || return 1

  local output
  output="$({ wlr-randr 2>/dev/null || true; } | awk '
    /^[^[:space:]]/ { connector=$1 }
    /Transform:/ { print connector "=" $2 }
  ' | paste -sd, -)"
  [[ -n "$output" ]] || return 1
  printf '%s' "$output"
}

get_x11_rotation() {
  command -v xrandr >/dev/null 2>&1 || return 1

  local output
  output="$({ DISPLAY="${DISPLAY:-:0}" xrandr --current 2>/dev/null || true; } | awk '
    / connected/ {
      connector=$1
      rotation="unknown"
      for (i = 1; i <= NF; i++) {
        if ($i == "normal" || $i == "left" || $i == "right" || $i == "inverted") {
          rotation=$i
          break
        }
      }
      print connector "=" rotation
    }
  ' | paste -sd, -)"
  [[ -n "$output" ]] || return 1
  printf '%s' "$output"
}

collect_drm() {
  local status_path connector mode orientation orientation_path first=1

  printf '['
  for status_path in /sys/class/drm/card*-*/status; do
    [[ -r "$status_path" ]] || continue

    connector="${status_path%/status}"
    connector="${connector##*/}"

    mode="unknown"
    if [[ -r "${status_path%/status}/modes" ]] && [[ -s "${status_path%/status}/modes" ]]; then
      mode="$(head -n 1 "${status_path%/status}/modes")"
    fi

    orientation="unknown"
    orientation_path="${status_path%/status}/panel_orientation"
    if [[ -r "$orientation_path" ]]; then
      orientation="$(tr -d '\n' < "$orientation_path")"
    fi

    if (( first == 0 )); then
      printf ','
    fi
    first=0

    printf '{"connector":"%s","status":"%s","mode":"%s","panel_orientation":"%s"}' \
      "$(json_escape "$connector")" \
      "$(json_escape "$(tr -d '\n' < "$status_path")")" \
      "$(json_escape "$mode")" \
      "$(json_escape "$orientation")"
  done
  printf ']'
}

app_rotation="${DISPLAY_ROTATION:-unset}"
wayland_rotation=""
x11_rotation=""

if wayland_rotation="$(get_wayland_rotation)"; then
  :
fi

if x11_rotation="$(get_x11_rotation)"; then
  :
fi

printf '{\n'
printf '  "app_display_rotation": "%s",\n' "$(json_escape "$app_rotation")"
printf '  "wayland_rotation": "%s",\n' "$(json_escape "${wayland_rotation:-unknown}")"
printf '  "x11_rotation": "%s",\n' "$(json_escape "${x11_rotation:-unknown}")"
printf '  "drm_connectors": '
collect_drm
printf '\n}\n'
