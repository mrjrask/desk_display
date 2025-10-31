#!/usr/bin/env bash
set -euo pipefail

APT_PACKAGES=(
  build-essential
  ffmpeg
  fonts-dejavu-core
  git
  i2c-tools
  libcairo2-dev
  libffi-dev
  libgdk-pixbuf2.0-dev
  libgl1
  libjpeg-dev
  libopenblas-dev
  libopenblas0
  libopenjp2-7-dev
  libpango1.0-dev
  libtiff5-dev
  libx264-dev
  network-manager
  python3-dev
  python3-opencv
  python3-pip
  python3-pygame
  python3-venv
  wireless-tools
)

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

echo "[INFO] Apt dependency installation complete."
