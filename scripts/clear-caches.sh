#!/usr/bin/env bash
#
# clear-caches.sh — clear the display client's local project cache and the
# pip and apt download caches on a Raspberry Pi, and report how much space
# each freed.
#
# Usage:
#   ./clear-caches.sh
#
# The local project cache is the display client's offline cache
# (DESK_DISPLAY_CLIENT_CACHE_DIR, from the environment or .env.client;
# cache/client in the project by default). Its playlists, manifests and
# artifacts are removed and the client downloads them again on its next sync.
# Its lease credential (client_credential) and any local override.json are
# kept. desk_display_client.service is stopped while the cache is cleared
# and started again afterwards if it was running.
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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd -P)"
CLIENT_SERVICE="desk_display_client.service"

# A KEY=value from .env.client, the file the display client loads.
env_client_value() {
  local key="$1"
  local env_path="$PROJECT_ROOT/.env.client"
  [[ -f "$env_path" ]] || return 0
  awk -v key="$key" '
    /^[[:space:]]*(#|$)/ { next }
    {
      line=$0
      sub(/^[[:space:]]*export[[:space:]]+/, "", line)
      if (line !~ "^[[:space:]]*" key "[[:space:]]*=") next
      sub("^[[:space:]]*" key "[[:space:]]*=[[:space:]]*", "", line)
      sub(/[[:space:]]+#.*$/, "", line)
      sub(/^[[:space:]]+|[[:space:]]+$/, "", line)
      if ((line ~ /^".*"$/) || (line ~ /^\047.*\047$/)) line=substr(line, 2, length(line)-2)
      print line
      exit 0
    }
  ' "$env_path"
}

freed=0

# ------------------------------------------------ local project cache -----
echo "local project cache"
cache_dir=${DESK_DISPLAY_CLIENT_CACHE_DIR:-$(env_client_value DESK_DISPLAY_CLIENT_CACHE_DIR)}
cache_dir=${cache_dir:-cache/client}
cache_dir=${cache_dir/#\~/$HOME}
[[ $cache_dir == /* ]] || cache_dir="$PROJECT_ROOT/$cache_dir"

cache_real=$(realpath -m -- "$cache_dir" 2>/dev/null || echo "$cache_dir")
if [[ $cache_real == / || $cache_real == "$PROJECT_ROOT" || $cache_real == "$(realpath -m -- "$HOME" 2>/dev/null)" ]]; then
  echo "  REFUSING to clear $cache_dir (not a cache folder)" >&2
elif [[ ! -d $cache_dir ]]; then
  echo "  no cache at $cache_dir — nothing to do"
else
  cache_before=$(dir_bytes "$cache_dir")
  restart_client=0
  if command -v systemctl >/dev/null 2>&1 && systemctl is-active --quiet "$CLIENT_SERVICE" 2>/dev/null; then
    echo "  stopping $CLIENT_SERVICE"
    if $SUDO systemctl stop "$CLIENT_SERVICE"; then
      restart_client=1
    else
      echo "  could not stop $CLIENT_SERVICE; clearing anyway" >&2
    fi
  fi
  # Keep the lease credential (re-registering needs a fresh enrollment) and
  # any local override; everything else is re-downloaded on the next sync.
  if ! find "$cache_dir" -mindepth 1 -maxdepth 1 ! -name client_credential ! -name override.json \
      -exec rm -rf -- {} + 2>/dev/null && [[ -n $SUDO ]]; then
    $SUDO find "$cache_dir" -mindepth 1 -maxdepth 1 ! -name client_credential ! -name override.json \
      -exec rm -rf -- {} + || true
  fi
  cache_after=$(dir_bytes "$cache_dir")
  cache_freed=$(( cache_before - cache_after ))
  (( cache_freed < 0 )) && cache_freed=0
  if find "$cache_dir" -mindepth 1 -maxdepth 1 ! -name client_credential ! -name override.json \
      -print -quit 2>/dev/null | grep -q .; then
    echo "  FAILED to clear everything in $cache_dir ($(hr "$cache_freed") freed)" >&2
  else
    echo "  cleared $cache_dir ($(hr "$cache_freed"))"
  fi
  freed=$(( freed + cache_freed ))
  if (( restart_client )); then
    echo "  starting $CLIENT_SERVICE"
    $SUDO systemctl start "$CLIENT_SERVICE" || echo "  FAILED to start $CLIENT_SERVICE" >&2
  fi
fi
echo

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
