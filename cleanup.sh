#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# Ensure Unix line endings and executable bit:
#   sed -i 's/\r$//' cleanup.sh && chmod +x cleanup.sh

echo "⏱  Running cleanup at $(date +%Y%m%d_%H%M%S)…"

dir="$(dirname "$0")"
cd "$dir"

# 1) Remove __pycache__ directories
echo "    → Removing __pycache__ directories…"
find . -type d -name "__pycache__" -prune -exec rm -rf {} +

# 2) Archive any straggler screenshots/videos left behind
SCREENSHOTS_DIR="screenshots"
ARCHIVE_BASE="screenshot_archive"   # singular, to match main.py
ARCHIVE_DATED_DIR="${ARCHIVE_BASE}/dated_folders"
ARCHIVE_DEFAULT_FOLDER="Screens"
timestamp="$(date +%Y%m%d_%H%M%S)"
day="${timestamp%_*}"
batch="${timestamp#*_}"
target_dir="${ARCHIVE_DATED_DIR}/${ARCHIVE_DEFAULT_FOLDER}/${day}/cleanup_${batch}"

shopt -s nullglob
left_png=( "${SCREENSHOTS_DIR}"/*.png "${SCREENSHOTS_DIR}"/*.jpg "${SCREENSHOTS_DIR}"/*.jpeg )
left_vid=( "${SCREENSHOTS_DIR}"/*.mp4 "${SCREENSHOTS_DIR}"/*.avi )
shopt -u nullglob

if (( ${#left_png[@]} + ${#left_vid[@]} > 0 )); then
  echo "    → Archiving leftover screenshots/videos to ${target_dir}…"
  mkdir -p "${target_dir}"
  # Move images first (primary “screenshots”), then any videos if present
  for f in "${left_png[@]}"; do
    mv -f "$f" "${target_dir}/"
  done
  for f in "${left_vid[@]}"; do
    mv -f "$f" "${target_dir}/"
  done
else
  echo "    → No leftover screenshots/videos to archive."
fi

echo "🏁  Cleanup complete."
