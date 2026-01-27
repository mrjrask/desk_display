#!/usr/bin/env bash
set -euo pipefail

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }

ensure_executable() {
  local file_path="$1"

  if [[ -x "$file_path" ]]; then
    log "$(basename "$file_path") already executable"
  elif [[ -f "$file_path" ]]; then
    chmod +x "$file_path" || warn "Could not mark $(basename "$file_path") as executable"
  else
    warn "Missing script: $file_path"
  fi
}

detect_existing_venv() {
  local project_dir="$1"
  local candidates=(
    "$project_dir/venv"
    "$project_dir/.venv"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate/pyvenv.cfg" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  local venv_cfg
  venv_cfg=$(find "$project_dir" -maxdepth 2 -mindepth 2 -type f -name pyvenv.cfg -print -quit 2>/dev/null || true)
  if [[ -n "$venv_cfg" ]]; then
    dirname "$venv_cfg"
    return 0
  fi

  return 1
}

# Return the preferred libtiff development package for the given codename,
# falling back gracefully when a codename-specific package is unavailable.
select_libtiff_pkg() {
  local codename="$1"
  local candidates=()

  case "$codename" in
    bookworm) candidates=(libtiff5-dev libtiff-dev) ;;
    trixie) candidates=(libtiff6-dev libtiff5-dev libtiff-dev) ;;
    *) candidates=(libtiff5-dev libtiff-dev) ;;
  esac

  for pkg in "${candidates[@]}"; do
    if apt-cache show "$pkg" >/dev/null 2>&1; then
      echo "$pkg"
      return 0
    fi
  done

  warn "Could not find a libtiff dev package; defaulting to ${candidates[-1]}"
  echo "${candidates[-1]}"
}

# Choose the gdk-pixbuf development package for the given codename, preferring
# codename-specific names while still allowing a fallback if the package cache
# does not know the preferred option.
select_gdk_pixbuf_pkg() {
  local codename="$1"
  local candidates=()

  case "$codename" in
    bookworm) candidates=(libgdk-pixbuf2.0-dev libgdk-pixbuf-2.0-dev) ;;
    trixie) candidates=(libgdk-pixbuf-2.0-dev) ;;
    *) candidates=(libgdk-pixbuf-2.0-dev libgdk-pixbuf2.0-dev) ;;
  esac

  for pkg in "${candidates[@]}"; do
    if apt-cache show "$pkg" >/dev/null 2>&1; then
      echo "$pkg"
      return 0
    fi
  done

  warn "Could not find a gdk-pixbuf dev package; defaulting to ${candidates[-1]}"
  echo "${candidates[-1]}"
}

install_apt_packages() {
  local codename="${EXPECTED_CODENAME:-}"
  if [[ -z "$codename" ]]; then
    warn "EXPECTED_CODENAME is not set; defaulting to detected release name"
    codename=$(lsb_release -sc 2>/dev/null || echo "")
  fi

  log "Updating apt package index."
  ${SUDO:-} apt-get update

  local shared_packages=(
    python3-venv python3-pip python3-dev python3-opencv
    build-essential libjpeg-dev libopenblas0 libopenblas-dev
    liblgpio-dev
    libopenjp2-7-dev libcairo2-dev libpango1.0-dev
    libffi-dev network-manager wireless-tools swig
    i2c-tools fonts-dejavu-core fonts-noto-color-emoji libgl1 libx264-dev ffmpeg git
  )

  local packages=("${shared_packages[@]}")
  packages+=("$(select_libtiff_pkg "$codename")")
  packages+=("$(select_gdk_pixbuf_pkg "$codename")")

  log "Installing apt dependencies: ${packages[*]}"
  ${SUDO:-} apt-get install -y "${packages[@]}"
}
