#!/usr/bin/env bash
set -euo pipefail

# Runs as an ExecStartPre step of the system-wide desk_display.service when
# DESK_DISPLAY_OUTPUT=kernel. Kernel-mode output draws into the desktop
# user's active X11/Wayland session rather than raw DRM/KMS, so unlike the
# framebuffer/waveshare output modes it needs that session's DISPLAY,
# WAYLAND_DISPLAY, XDG_RUNTIME_DIR, and XAUTHORITY before main.py starts.
#
# Those variables don't exist yet at boot until the desktop session comes
# up, so this script polls for them and writes whatever it finds to an
# env file that the unit's EnvironmentFile= re-reads for ExecStart. If no
# session shows up before the timeout, it exits non-zero; combined with the
# unit's Restart=always/RestartSec, systemd just retries this step until
# the desktop session is available (e.g. after autologin finishes).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

if [[ "${DESK_DISPLAY_OUTPUT:-}" != "kernel" ]]; then
  exit 0
fi

COMMON_SCRIPT="$SCRIPT_DIR/helpers/common.sh"
if [[ -f "$COMMON_SCRIPT" ]]; then
  # shellcheck source=/dev/null
  source "$COMMON_SCRIPT"
fi

if ! declare -F detect_desktop_session >/dev/null 2>&1; then
  warn "detect_desktop_session helper unavailable; cannot prepare kernel session environment."
  exit 1
fi

SESSION_USER="${DESK_DISPLAY_SESSION_USER:-$(whoami)}"
ENV_OUT="${DESK_DISPLAY_SESSION_ENV_FILE:-$PROJECT_DIR/.runtime/kernel-session.env}"
timeout_seconds="${DESK_DISPLAY_SESSION_TIMEOUT_SECONDS:-25}"
interval_seconds="${DESK_DISPLAY_SESSION_POLL_INTERVAL_SECONDS:-1}"

mkdir -p "$(dirname "$ENV_OUT")"

elapsed=0
while (( elapsed < timeout_seconds )); do
  if detect_desktop_session "$SESSION_USER" && [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
    break
  fi
  sleep "$interval_seconds"
  elapsed=$((elapsed + interval_seconds))
done

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  warn "No active X11/Wayland desktop session detected for $SESSION_USER after ${timeout_seconds}s."
  exit 1
fi

if [[ -n "${DISPLAY:-}" && -z "${XAUTHORITY:-}" ]]; then
  home_dir=$(getent passwd "$SESSION_USER" | cut -d: -f6)
  default_xauth="${home_dir:-$HOME}/.Xauthority"
  if [[ -f "$default_xauth" ]]; then
    XAUTHORITY="$default_xauth"
  fi
fi

sdl_drivers=""
if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
  sdl_drivers="wayland,x11,kmsdrm,fbcon,directfb"
elif [[ -n "${DISPLAY:-}" ]]; then
  sdl_drivers="x11,wayland,kmsdrm,fbcon,directfb"
fi

tmp_out=$(mktemp "${ENV_OUT}.XXXXXX")
{
  [[ -n "${DISPLAY:-}" ]] && printf 'DISPLAY=%s\n' "$DISPLAY"
  [[ -n "${WAYLAND_DISPLAY:-}" ]] && printf 'WAYLAND_DISPLAY=%s\n' "$WAYLAND_DISPLAY"
  [[ -n "${XDG_RUNTIME_DIR:-}" ]] && printf 'XDG_RUNTIME_DIR=%s\n' "$XDG_RUNTIME_DIR"
  [[ -n "${XAUTHORITY:-}" ]] && printf 'XAUTHORITY=%s\n' "$XAUTHORITY"
  [[ -n "$sdl_drivers" ]] && printf 'DESK_DISPLAY_SDL_DRIVERS=%s\n' "$sdl_drivers"
} > "$tmp_out"
mv "$tmp_out" "$ENV_OUT"

log "Wrote kernel display session environment to $ENV_OUT"
