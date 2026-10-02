#!/usr/bin/env bash
set -euo pipefail

# Wait briefly during boot for HyperPixel/DRM and desktop sockets to settle.
# This avoids a race where SDL initializes before the display stack is ready.

timeout_seconds="${DESK_DISPLAY_STARTUP_TIMEOUT_SECONDS:-30}"
interval_seconds="${DESK_DISPLAY_STARTUP_POLL_INTERVAL_SECONDS:-1}"
stable_polls_required="${DESK_DISPLAY_STARTUP_STABLE_POLLS:-3}"
drm_root="${DESK_DISPLAY_DRM_SYSFS_ROOT:-/sys/class/drm}"

if (( stable_polls_required < 2 )); then
  stable_polls_required=2
fi

get_rotation_state() {
  local output

  if [[ -n "${WAYLAND_DISPLAY:-}" ]] && command -v wlr-randr >/dev/null 2>&1; then
    output="$({ wlr-randr 2>/dev/null || true; } | awk '
      /^[^[:space:]]/ { connector=$1 }
      /Transform:/ { print connector "=" $2 }
    ' | paste -sd, -)"
    if [[ -n "$output" ]]; then
      printf 'wayland:%s' "$output"
      return 0
    fi
  fi

  if command -v xrandr >/dev/null 2>&1; then
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
    if [[ -n "$output" ]]; then
      printf 'x11:%s' "$output"
      return 0
    fi
  fi

  return 1
}

get_display_state() {
  local status_path modes_path connector mode rotation_state
  # Initialised empty: with `set -u`, bash treats a declared-but-unassigned
  # array as unbound, so `${#connectors[@]}` aborted the poll whenever no
  # connector was connected yet.
  local -a connectors=()

  # DRM/KMS path (preferred for HyperPixel kernel output). Pi 4 exposes the
  # display controller as card0/card1; Pi 5 puts HDMI on card1 or card2
  # (card0 is the render-only v3d GPU) and DPI/DSI panels on the RP1 card,
  # so match every card rather than a fixed one.
  for status_path in "$drm_root"/card*-*/status; do
    [[ -r "$status_path" ]] || continue
    if grep -qx "connected" "$status_path"; then
      modes_path="${status_path%/status}/modes"
      if [[ -r "$modes_path" ]] && [[ -s "$modes_path" ]]; then
        connector="${status_path%/status}"
        connector="${connector##*/}"
        mode="$(head -n 1 "$modes_path")"
        connectors+=("${connector}:${mode}")
      fi
    fi
  done

  if (( ${#connectors[@]} == 0 )); then
    return 1
  fi

  if rotation_state="$(get_rotation_state)"; then
    printf 'connectors=%s|rotation=%s\n' "$(printf '%s,' "${connectors[@]}" | sed 's/,$//')" "$rotation_state"
  else
    printf 'connectors=%s|rotation=unknown\n' "$(printf '%s,' "${connectors[@]}" | sed 's/,$//')"
  fi
}

elapsed=0
stable_count=0
last_state=""

while (( elapsed < timeout_seconds )); do
  current_state=""
  if current_state="$(get_display_state)"; then
    if [[ "$current_state" == "$last_state" ]]; then
      stable_count=$((stable_count + 1))
    else
      last_state="$current_state"
      stable_count=1
    fi

    if (( stable_count >= stable_polls_required )); then
      echo "[INFO] Display ready after ${elapsed}s (${stable_count} stable polls): ${current_state}" >&2
      exit 0
    fi
  else
    stable_count=0
    last_state=""
  fi

  sleep "$interval_seconds"
  elapsed=$((elapsed + interval_seconds))
done

# Describe what DRM reported so a timeout in the journal says why.
describe_drm_connectors() {
  local status_path connector status mode_count model=""
  if [[ -r /proc/device-tree/model ]]; then
    model="$(tr -d '\0' </proc/device-tree/model)"
  fi
  echo "[WARN] Board: ${model:-unknown}" >&2
  local found=0
  for status_path in "$drm_root"/card*-*/status; do
    [[ -r "$status_path" ]] || continue
    found=1
    connector="${status_path%/status}"
    status="$(cat "$status_path" 2>/dev/null || echo unreadable)"
    mode_count="$(wc -l <"${connector}/modes" 2>/dev/null || echo 0)"
    echo "[WARN] DRM connector ${connector##*/}: status=${status} modes=${mode_count// /}" >&2
  done
  if (( found == 0 )); then
    echo "[WARN] No DRM connectors under ${drm_root}; no panel driver is bound. Check that /boot/firmware/config.txt loads dtoverlay=vc4-kms-v3d plus the panel overlay (vc4-kms-dpi-hyperpixel4sq for a HyperPixel square) outside any [pi4] section." >&2
  fi
}

# Don't fail the service start permanently; allow normal retries/restarts.
if [[ -n "$last_state" ]]; then
  echo "[WARN] Display readiness wait timed out after ${timeout_seconds}s; last detected state: ${last_state}. Continuing startup." >&2
else
  echo "[WARN] Display readiness wait timed out after ${timeout_seconds}s; no connected mode detected. Continuing startup." >&2
  describe_drm_connectors
fi
exit 0
