#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

run_as_root() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "⚠️  Skipping root-required command (sudo not found): $*"
    return 0
  fi
}

echo "======================================"
echo "🧹 Raspberry Pi Cache Cleanup Starting"
echo "======================================"

echo ""
echo "📊 Disk usage BEFORE cleanup:"
df -h /

echo ""
echo "--------------------------------------"
echo "🧼 Cleaning APT cache..."
echo "--------------------------------------"

if command -v apt-get >/dev/null 2>&1; then
  run_as_root apt-get clean
  run_as_root apt-get autoclean -y
  run_as_root apt-get autoremove -y
  echo "✅ APT cache cleaned"
else
  echo "⚠️  apt-get not found; skipping APT cache cleanup"
fi

echo ""
echo "--------------------------------------"
echo "🐍 Cleaning pip cache..."
echo "--------------------------------------"

if command -v pip3 >/dev/null 2>&1; then
  pip3 cache purge || true
else
  echo "⚠️  pip3 not found for current user; skipping user pip cache purge"
fi

# Root pip cache may not exist; best effort only.
if command -v pip3 >/dev/null 2>&1; then
  run_as_root pip3 cache purge || true
fi

echo "🗑 Removing leftover pip cache directories..."
rm -rf "$HOME/.cache/pip"
run_as_root rm -rf /root/.cache/pip

echo "✅ pip cache cleaned"

echo ""
echo "--------------------------------------"
echo "🧾 Final disk usage AFTER cleanup:"
echo "--------------------------------------"
df -h /

echo ""
echo "🎉 Cleanup complete!"
