#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Ensure Unix line endings and executable bit:
#   sed -i 's/\r$//' cleanup.sh && chmod +x cleanup.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd -P)"

echo "⏱  Running cleanup at $(date +%Y%m%d_%H%M%S)…"
cd "$PROJECT_ROOT"

stop_kernel_user_service() {
  local system_service="desk_display.service"
  local kernel_service="desk_display-kernel.service"
  local service_user=""
  local uid=""
  local runtime_dir=""
  local -a systemctl_env=()

  command -v systemctl >/dev/null 2>&1 || return 0

  service_user="$(systemctl show -p User --value "$system_service" 2>/dev/null || true)"
  if [[ -z "$service_user" || "$service_user" == "root" ]]; then
    service_user="${SUDO_USER:-${USER:-}}"
  fi
  if [[ -z "$service_user" ]]; then
    return 0
  fi

  uid="$(id -u "$service_user" 2>/dev/null || true)"
  if [[ -n "$uid" ]]; then
    runtime_dir="/run/user/$uid"
  fi
  if [[ -n "$runtime_dir" && -d "$runtime_dir" ]]; then
    systemctl_env=("XDG_RUNTIME_DIR=$runtime_dir")
    if [[ -S "$runtime_dir/bus" ]]; then
      systemctl_env+=("DBUS_SESSION_BUS_ADDRESS=unix:path=$runtime_dir/bus")
    fi
  fi

  local current_user=""
  current_user="$(id -un 2>/dev/null || true)"

  if [[ -n "$current_user" && "$current_user" == "$service_user" ]]; then
    env "${systemctl_env[@]}" systemctl --user --no-block stop "$kernel_service" >/dev/null 2>&1 || true
  elif command -v sudo >/dev/null 2>&1; then
    sudo -u "$service_user" env "${systemctl_env[@]}" systemctl --user --no-block stop "$kernel_service" >/dev/null 2>&1 || true
  fi
}

stop_kernel_user_service

# If present, stop the Waveshare OLED helper so it doesn't redraw while cleanup
# is blanking the side OLED panels.
if command -v systemctl >/dev/null 2>&1; then
  systemctl --no-block stop desk_display_waveshare_oled.service >/dev/null 2>&1 || true
fi

# Prefer the repo's virtualenv interpreter when available so optional
# dependencies such as Pillow are on the path even during shutdown.
python_bin="python3"
if [[ -x "${PROJECT_ROOT}/venv/bin/python" ]]; then
  python_bin="${PROJECT_ROOT}/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
fi

# Ask the running service to stop scheduling new screens immediately.
if command -v systemctl >/dev/null 2>&1; then
  SERVICE_NAME="desk_display.service"
  main_pid="$(systemctl show -p MainPID --value "$SERVICE_NAME" 2>/dev/null || true)"
  if [[ -n "${main_pid}" && "${main_pid}" != "0" ]]; then
    echo "    → Requesting ${SERVICE_NAME} shutdown (SIGTERM to PID ${main_pid})…"
    kill -TERM "${main_pid}" 2>/dev/null || true
    # Give the process a brief moment to halt screen rotation before we touch
    # the display directly.
    sleep 1
  fi
fi

# 1) Clear the display before touching the filesystem
echo "    → Clearing display…"
# Intentionally avoid forcing headless mode here: cleanup should blank the
# physical Display HAT Mini panel when hardware output is available.
"${python_bin}" - <<'PY'
import logging
import time
import sys
from pathlib import Path

from PIL import Image

# `python -` runs as <stdin>, so derive the repository root from cwd.
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils import Display, clear_display, clear_update_indicator
except Exception as exc:  # pragma: no cover - best effort during shutdown
    logging.warning("Display cleanup skipped: %s", exc)
else:
    try:
        display = Display()
        clear_update_indicator(display)

        # Run a deliberate multi-pass blackout sequence; some panels can show
        # stale frame fragments if only one clear command is issued during
        # service shutdown while transitions are still winding down.
        black = Image.new(
            "RGB",
            (getattr(display, "width", 320), getattr(display, "height", 240)),
            "black",
        )
        for _ in range(3):
            display.image(black)
            display.set_led(0.0, 0.0, 0.0)
            display.show()
            time.sleep(0.06)

        # Keep the legacy clear path as a final fallback redraw.
        clear_display(display)
        display.clear()

        # Blank the panel backlight at shutdown so any residual LCD ghosting is
        # not visible once cleanup finishes.
        if hasattr(display, "set_backlight"):
            display.set_backlight(0.0)
    except Exception as exc:  # pragma: no cover - best effort during shutdown
        logging.warning("Display cleanup failed: %s", exc)

try:
    from smbus import SMBus
except Exception as exc:  # pragma: no cover - optional Waveshare dependency
    logging.info("Waveshare OLED cleanup skipped (smbus unavailable): %s", exc)
else:
    try:
        scripts_dir = PROJECT_ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        import waveshare_oled_status as waveshare_oled

        with SMBus(waveshare_oled.I2C_BUS) as bus:
            for addr in (waveshare_oled.TEMP_ADDR, waveshare_oled.TIME_ADDR):
                oled = waveshare_oled.SSD1306Display(
                    bus,
                    addr,
                    waveshare_oled.OLED_WIDTH,
                    waveshare_oled.OLED_HEIGHT,
                )
                oled.initialize()
                oled.clear()
    except Exception as exc:  # pragma: no cover - best effort during shutdown
        logging.warning("Waveshare OLED cleanup failed: %s", exc)
PY

# 2) Remove __pycache__ directories
echo "    → Removing __pycache__ directories (excluding virtualenv)…"
find "$PROJECT_ROOT" \
  -path "$PROJECT_ROOT/venv" -prune -o \
  -type d -name "__pycache__" -prune -exec rm -rf {} +

# 3) Archive any straggler screenshots/videos left behind
SCREENSHOTS_DIR="$PROJECT_ROOT/screenshots"
ARCHIVE_BASE="$PROJECT_ROOT/screenshot_archive"   # singular, to match main.py
ARCHIVE_DEFAULT_FOLDER="Screens"
timestamp="$(date +%Y%m%d_%H%M%S)"
batch="${timestamp#*_}"

declare -a leftover_files=()
if [[ -d "${SCREENSHOTS_DIR}" ]]; then
  while IFS= read -r -d $'\0' file; do
    leftover_files+=("$file")
  done < <(
    find "${SCREENSHOTS_DIR}" \
      -path "${SCREENSHOTS_DIR}/current" -prune -o \
      -type f \
      \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \
         -o -iname '*.mp4' -o -iname '*.avi' \) -print0 | sort -z
  )
fi

if (( ${#leftover_files[@]} > 0 )); then
  echo "    → Archiving leftover screenshots/videos to screenshot_archive/<screen>/"
  for src in "${leftover_files[@]}"; do
    rel_path="${src#${SCREENSHOTS_DIR}/}"
    screen_folder="${ARCHIVE_DEFAULT_FOLDER}"
    remainder="${rel_path}"

    if [[ "${rel_path}" != "${src}" ]]; then
      IFS='/' read -r first rest <<< "${rel_path}"
      if [[ -n "${rest}" ]]; then
        screen_folder="${first}"
        remainder="${rest}"
      else
        remainder="${first}"
      fi
    else
      remainder="$(basename "${src}")"
    fi

    dest_dir="${ARCHIVE_BASE}/${screen_folder}"
    dest="${dest_dir}/${remainder}"
    mkdir -p "$(dirname "${dest}")"

    if [[ -e "${dest}" ]]; then
      ext="${dest##*.}"
      base="${dest%.*}"
      dest="${base}_cleanup_${batch}.${ext}"
    fi

    mv -f "${src}" "${dest}"
  done
  if [[ -d "${SCREENSHOTS_DIR}" ]]; then
    find "${SCREENSHOTS_DIR}" -type d -empty -delete
  fi
else
  echo "    → No leftover screenshots/videos to archive."
fi

echo "🏁  Cleanup complete."
