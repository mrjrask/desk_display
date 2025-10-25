#!/usr/bin/env bash
set -euo pipefail

APT_PACKAGES=(
  python3-venv python3-pip python3-dev python3-opencv
  build-essential libjpeg-dev libopenblas0 libopenblas-dev
  libopenjp2-7-dev libtiff5-dev libcairo2-dev libpango1.0-dev
  libgdk-pixbuf2.0-xlib-dev libffi-dev network-manager wireless-tools
  i2c-tools fonts-dejavu-core libgl1 libx264-dev ffmpeg git
)

DEFAULT_REPO_URL="https://github.com/mrjrask/desk_display.git"
DEFAULT_PROJECT_DIR="$HOME/desk_display"

usage() {
  cat <<USAGE
Usage: ${0##*/} [options]

Options:
  -r <repo_url>      Git repository URL (default: ${DEFAULT_REPO_URL})
  -d <project_dir>   Target directory for the project clone (default: ${DEFAULT_PROJECT_DIR})
  -p <python_exec>   Python interpreter to use for the virtual environment (default: python3)
  -h                 Show this help message and exit

The script will enable SPI & I2C (if raspi-config is available), install apt dependencies,
clone the repository, create a Python virtual environment in the project folder, and install
pip dependencies listed in requirements.txt.
USAGE
}

REPO_URL="$DEFAULT_REPO_URL"
PROJECT_DIR="$DEFAULT_PROJECT_DIR"
PYTHON="python3"

while getopts "hr:d:p:" opt; do
  case "$opt" in
    h)
      usage
      exit 0
      ;;
    r)
      REPO_URL="$OPTARG"
      ;;
    d)
      PROJECT_DIR="$OPTARG"
      ;;
    p)
      PYTHON="$OPTARG"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "[INFO] Running without sudo. Commands requiring elevated privileges will use sudo."
  SUDO="sudo"
else
  SUDO=""
fi

if command -v raspi-config >/dev/null 2>&1; then
  echo "[INFO] Enabling SPI interface via raspi-config."
  $SUDO raspi-config nonint do_spi 0 || echo "[WARN] Failed to enable SPI via raspi-config."
  echo "[INFO] Enabling I2C interface via raspi-config."
  $SUDO raspi-config nonint do_i2c 0 || echo "[WARN] Failed to enable I2C via raspi-config."
else
  echo "[WARN] raspi-config not found; skipping SPI/I2C enablement."
fi

echo "[INFO] Updating apt package index."
$SUDO apt-get update

echo "[INFO] Installing apt dependencies: ${APT_PACKAGES[*]}"
$SUDO apt-get install -y "${APT_PACKAGES[@]}"

PROJECT_PARENT=$(dirname "$PROJECT_DIR")
if [[ ! -d "$PROJECT_PARENT" ]]; then
  echo "[INFO] Creating parent directory: $PROJECT_PARENT"
  mkdir -p "$PROJECT_PARENT"
fi

if [[ -d "$PROJECT_DIR/.git" ]]; then
  echo "[INFO] Existing git repository found at $PROJECT_DIR. Pulling latest changes."
  git -C "$PROJECT_DIR" pull --ff-only
else
  echo "[INFO] Cloning repository from $REPO_URL into $PROJECT_DIR"
  git clone "$REPO_URL" "$PROJECT_DIR"
fi

VENV_DIR="$PROJECT_DIR/venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[INFO] Creating virtual environment with $PYTHON at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
else
  echo "[INFO] Virtual environment already exists at $VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
if [[ -f "$REQUIREMENTS_FILE" ]]; then
  echo "[INFO] Installing Python dependencies from $REQUIREMENTS_FILE"
  pip install -r "$REQUIREMENTS_FILE"
else
  echo "[WARN] requirements.txt not found at $REQUIREMENTS_FILE; skipping pip install."
fi

echo "[INFO] Installation complete. Virtual environment activated."

deactivate

echo "[INFO] Virtual environment deactivated. To start using it, run:"
echo "  source $VENV_DIR/bin/activate"
