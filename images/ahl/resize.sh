#!/usr/bin/env bash
# resize_pngs_inplace_300.sh
# Resize all PNG files in the current directory to 300x300,
# overwriting the original files.

set -Eeuo pipefail

# Ensure ImageMagick is available
if ! command -v convert >/dev/null 2>&1; then
  echo "❌ ImageMagick 'convert' command not found."
  echo "   Install it with: sudo apt install imagemagick"
  exit 1
fi

shopt -s nullglob

for img in *.png; do
  tmp="${img}.tmp.png"

  echo "🔧 Resizing: $img"

  # Write to a temporary file first for safety
  convert "$img" -resize 300x300 "$tmp"

  # Replace original
  mv -f "$tmp" "$img"
done

echo "✅ Done! All PNGs are now 300×300 and original files overwritten."
