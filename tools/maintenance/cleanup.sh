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

FALSEY_ENV_VALUES_REGEX='^(|0|false|no|off)$'

lookup_config_value() {
  local name="$1"
  local value="${!name-}"

  if [[ -n "$value" || -v "$name" ]]; then
    printf '%s' "$value"
    return 0
  fi

  if [[ -f "$PROJECT_ROOT/.env" ]]; then
    value="$(awk -F= -v key="$name" '
      $0 ~ /^[[:space:]]*(#|$)/ { next }
      {
        candidate = $1
        sub(/^[[:space:]]*/, "", candidate)
        sub(/[[:space:]]*$/, "", candidate)
        if (candidate == key) {
          $1 = ""
          sub(/^=/, "")
          sub(/^[[:space:]]*/, "")
          sub(/[[:space:]]*(#.*)?$/, "")
          print
          exit
        }
      }
    ' "$PROJECT_ROOT/.env")"
    printf '%s' "$value"
  fi
}

is_truthy_config() {
  local value
  value="$(lookup_config_value "$1" | tr '[:upper:]' '[:lower:]')"
  [[ ! "$value" =~ $FALSEY_ENV_VALUES_REGEX ]]
}

is_waveshare_oled_profile() {
  local marker output hyperpixel_panel
  marker="$(lookup_config_value WAVESHARE_OLED_LCD_HAT_A_INSTALLED | tr '[:upper:]' '[:lower:]')"
  if [[ -n "$marker" ]]; then
    [[ ! "$marker" =~ $FALSEY_ENV_VALUES_REGEX ]]
    return
  fi

  output="$(lookup_config_value DESK_DISPLAY_OUTPUT | tr '[:upper:]' '[:lower:]')"
  hyperpixel_panel="$(lookup_config_value HYPERPIXEL_PANEL | tr '[:upper:]' '[:lower:]')"
  if [[ "$output" == "displayhatmini" || "$output" == "display-hat-mini" || "$output" == "pimoroni" || -n "$hyperpixel_panel" ]]; then
    return 1
  fi

  [[ -n "$(lookup_config_value WAVESHARE_OLED_MAX_VALUE_FONT_SIZE)" ]] \
    || [[ -n "$(lookup_config_value WAVESHARE_OLED_MAX_TIME_FONT_SIZE)" ]] \
    || is_truthy_config WAVESHARE_OLED_CLEANUP
}

# If present and configured for this display profile, stop the Waveshare OLED
# helper so it doesn't redraw while cleanup is blanking the side OLED panels.
# Gate this before calling systemctl: on non-Waveshare displays a stale installed
# helper can otherwise trigger a desktop authentication prompt during service
# restarts.
if command -v systemctl >/dev/null 2>&1 && is_waveshare_oled_profile; then
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
import os
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

FALSEY_ENV_VALUES = {"", "0", "false", "no", "off"}


def _truthy_env(name):
    return os.environ.get(name, "").strip().lower() not in FALSEY_ENV_VALUES


def _waveshare_oled_cleanup_enabled():
    marker = os.environ.get("WAVESHARE_OLED_LCD_HAT_A_INSTALLED")
    if marker is not None:
        return marker.strip().lower() not in FALSEY_ENV_VALUES

    display_output = os.environ.get("DESK_DISPLAY_OUTPUT", "").strip().lower()
    hyperpixel_panel = os.environ.get("HYPERPIXEL_PANEL", "").strip().lower()
    if display_output in {"displayhatmini", "display-hat-mini", "pimoroni"} or hyperpixel_panel:
        return False

    # The Waveshare installer writes these OLED-specific settings to the
    # service environment.  Avoid touching I2C on unrelated displays (for
    # example HyperPixel) where smbus may exist but the Waveshare OLEDs do not.
    return (
        "WAVESHARE_OLED_MAX_VALUE_FONT_SIZE" in os.environ
        or "WAVESHARE_OLED_MAX_TIME_FONT_SIZE" in os.environ
        or _truthy_env("WAVESHARE_OLED_CLEANUP")
    )


if not _waveshare_oled_cleanup_enabled():
    logging.info("Waveshare OLED cleanup skipped (not configured for this display profile)")
else:
    try:
        from smbus import SMBus
    except Exception:
        try:
            from smbus2 import SMBus
        except Exception as exc:  # pragma: no cover - optional Waveshare dependency
            logging.info("Waveshare OLED cleanup skipped (smbus/smbus2 unavailable): %s", exc)
            SMBus = None

    if SMBus is not None:
        try:
            scripts_dir = PROJECT_ROOT / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))

            import waveshare_oled_status as waveshare_oled

            i2c_device = Path(f"/dev/i2c-{waveshare_oled.I2C_BUS}")
            if not i2c_device.exists():
                logging.info(
                    "Waveshare OLED cleanup skipped (%s not present)",
                    i2c_device,
                )
            else:
                bus = SMBus(waveshare_oled.I2C_BUS)
                try:
                    for addr in (waveshare_oled.TEMP_ADDR, waveshare_oled.TIME_ADDR):
                        oled = waveshare_oled.SSD1306Display(
                            bus,
                            addr,
                            waveshare_oled.OLED_WIDTH,
                            waveshare_oled.OLED_HEIGHT,
                        )
                        oled.initialize()
                        oled.clear()
                finally:
                    close_bus = getattr(bus, "close", None)
                    if callable(close_bus):
                        close_bus()
        except FileNotFoundError as exc:  # pragma: no cover - hardware specific
            logging.info("Waveshare OLED cleanup skipped (I2C device missing): %s", exc)
        except OSError as exc:  # pragma: no cover - hardware specific
            logging.info("Waveshare OLED cleanup skipped (I2C unavailable): %s", exc)
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
