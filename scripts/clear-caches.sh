#!/usr/bin/env bash
#
# clear-caches.sh — clear the pip and apt download caches on a Raspberry Pi,
# and report how much space each freed.
#
# Usage:
#   ./clear-caches.sh
#
# The apt portion runs `apt-get clean` via sudo (you will be prompted
# if you are not already root).

set -uo pipefail

# sudo prefix unless we are already root
if [[ $EUID -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

# Convert a byte count to a human-readable string
hr() {
  local b=${1:-0}
  if (( b >= 1073741824 )); then
    awk -v b="$b" 'BEGIN { printf "%.2f GiB", b / 1073741824 }'
  elif (( b >= 1048576 )); then
    awk -v b="$b" 'BEGIN { printf "%.2f MiB", b / 1048576 }'
  elif (( b >= 1024 )); then
    awk -v b="$b" 'BEGIN { printf "%.2f KiB", b / 1024 }'
  else
    printf '%s B' "$b"
  fi
}

# Size of a path in bytes (0 if it does not exist or cannot be measured)
dir_bytes() {
  local p=$1
  if [[ -e $p ]]; then
    du -sb "$p" 2>/dev/null | cut -f1 | grep -Eq '^[0-9]+$' \
      && du -sb "$p" 2>/dev/null | cut -f1 || echo 0
  else
    echo 0
  fi
}

freed=0

# ------------------------------------------------------------ pip cache -----
echo "pip cache"
pip_cmd=""
for c in pip3 pip; do
  if command -v "$c" >/dev/null 2>&1; then
    pip_cmd=$c
    break
  fi
done

pip_dir=""
if [[ -n $pip_cmd ]]; then
  pip_dir=$("$pip_cmd" cache dir 2>/dev/null || true)
fi
pip_dir=${pip_dir:-$HOME/.cache/pip}

pip_before=$(dir_bytes "$pip_dir")
if [[ -d $pip_dir ]] && (( pip_before > 0 )); then
  if rm -rf "$pip_dir"; then
    echo "  cleared $pip_dir ($(hr "$pip_before"))"
    freed=$(( freed + pip_before ))
  else
    echo "  FAILED to clear $pip_dir" >&2
  fi
else
  echo "  no cache at $pip_dir — nothing to do"
fi

# ------------------------------------------------------------- apt cache -----
echo "apt cache"
apt_dir="/var/cache/apt/archives"
apt_before=$(dir_bytes "$apt_dir")
if $SUDO apt-get clean; then
  apt_after=$(dir_bytes "$apt_dir")
  apt_freed=$(( apt_before - apt_after ))
  (( apt_freed < 0 )) && apt_freed=0
  echo "  cleared $apt_dir ($(hr "$apt_freed"))"
  freed=$(( freed + apt_freed ))
else
  echo "  FAILED to clean $apt_dir (needs sudo?)" >&2
fi

echo
echo "Total freed: $(hr "$freed")"
