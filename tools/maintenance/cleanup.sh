#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Ensure Unix line endings and executable bit:
#   sed -i 's/\r$//' cleanup.sh && chmod +x cleanup.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd -P)"

echo "⏱  Running cleanup at $(date +%Y%m%d_%H%M%S)…"
cd "$PROJECT_ROOT"

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
# Force headless mode so cleanup doesn't fight a still-shutting-down service
# for the Display HAT Mini hardware.
DESK_DISPLAY_FORCE_HEADLESS=1 "${python_bin}" - <<'PY'
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils import Display, clear_display
except Exception as exc:  # pragma: no cover - best effort during shutdown
    logging.warning("Display cleanup skipped: %s", exc)
else:
    try:
        display = Display()
        clear_display(display)
        display.set_led(0.0, 0.0, 0.0)
    except Exception as exc:  # pragma: no cover - best effort during shutdown
        logging.warning("Display cleanup failed: %s", exc)
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
