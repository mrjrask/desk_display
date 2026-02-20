#!/usr/bin/env bash
set -euo pipefail

# Wait briefly during boot for HyperPixel/DRM and desktop sockets to settle.
# This avoids a race where SDL initializes before the display stack is ready.

timeout_seconds="${DESK_DISPLAY_STARTUP_TIMEOUT_SECONDS:-30}"
interval_seconds="${DESK_DISPLAY_STARTUP_POLL_INTERVAL_SECONDS:-1}"

is_display_ready() {
  local status_path modes_path

  # DRM/KMS path (preferred for HyperPixel kernel output)
  for status_path in /sys/class/drm/card*-*/status; do
    [[ -r "$status_path" ]] || continue
    if grep -q "connected" "$status_path"; then
      modes_path="${status_path%/status}/modes"
      if [[ -r "$modes_path" ]] && [[ -s "$modes_path" ]]; then
        return 0
      fi
    fi
  done

  # Wayland/X11 sockets if a desktop session is the chosen backend.
  if [[ -n "${XDG_RUNTIME_DIR:-}" && -S "${XDG_RUNTIME_DIR}/wayland-0" ]]; then
    return 0
  fi

  if [[ -S /tmp/.X11-unix/X0 ]]; then
    return 0
  fi

  return 1
}

if is_display_ready; then
  exit 0
fi

elapsed=0
while (( elapsed < timeout_seconds )); do
  sleep "$interval_seconds"
  elapsed=$((elapsed + interval_seconds))
  if is_display_ready; then
    exit 0
  fi
done

# Don't fail the service start permanently; allow normal retries/restarts.
echo "[WARN] Display readiness wait timed out after ${timeout_seconds}s; continuing startup." >&2
exit 0
