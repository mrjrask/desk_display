#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SERVICE_USER="${SUDO_USER:-$(whoami)}"

WAVESHARE_OLED_SERVICE_NAME="desk_display_waveshare_oled.service"
WAVESHARE_OLED_SERVICE_PATH="/etc/systemd/system/${WAVESHARE_OLED_SERVICE_NAME}"

WIKI_PAGE_URL="https://www.waveshare.com/wiki/OLED/LCD_HAT_(A)"
WIRINGPI_ZIP_URL="https://files.waveshare.com/wiki/OLED-LCD-HAT-A/WiringPi.zip"
DEMO_ZIP_URL="https://files.waveshare.com/wiki/OLED-LCD-HAT-A/OLED_LCD_HAT_A_Demo.zip"
OVERLAY_ZIP_URL="https://files.waveshare.com/wiki/OLED-LCD-HAT-A/OLED_LCD_HAT_A.zip"

LOGFILE="/var/log/waveshare_oled_lcd_hat_a_trixie64_install.log"
WORKDIR="/usr/local/src/waveshare_oled_lcd_hat_a"
INSTALL_ROOT="/opt/waveshare/OLED_LCD_HAT_A"

COMMON_SCRIPT="$PROJECT_DIR/scripts/helpers/common.sh"
if [[ ! -f "$COMMON_SCRIPT" ]]; then
  echo "[ERROR] Missing common installer helpers at $COMMON_SCRIPT" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$COMMON_SCRIPT"

if [[ $EUID -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

if [[ -z "${TARGET_USER:-}" || "${TARGET_USER}" == "root" ]]; then
  TARGET_USER="${SUDO_USER:-}"
fi
if [[ -z "${TARGET_USER}" || "${TARGET_USER}" == "root" ]]; then
  TARGET_USER="$(awk -F: '$3 >= 1000 && $3 < 65534 {print $1; exit}' /etc/passwd || true)"
fi
if [[ -z "$TARGET_USER" ]]; then
  echo "ERROR: Could not determine the target non-root user." >&2
  echo "Run this script with sudo from your normal account." >&2
  exit 1
fi
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
if [[ -z "$TARGET_HOME" || ! -d "$TARGET_HOME" ]]; then
  echo "ERROR: Could not determine the home directory for $TARGET_USER." >&2
  exit 1
fi

detect_codename() {
  if command -v lsb_release >/dev/null 2>&1; then
    lsb_release -sc
    return 0
  fi
  if [[ -f /etc/os-release ]]; then
    # shellcheck source=/dev/null
    source /etc/os-release
    if [[ -n "${VERSION_CODENAME:-}" ]]; then
      echo "$VERSION_CODENAME"
      return 0
    fi
  fi
  return 1
}

backup_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    cp -a "$file" "${file}.bak.${ts}"
    log "Backup created: ${file}.bak.${ts}"
  fi
}

append_if_missing() {
  local file="$1"
  local line="$2"
  grep -Fqx "$line" "$file" 2>/dev/null || echo "$line" >>"$file"
}

install_waveshare_stack() {
  mkdir -p "$(dirname "$LOGFILE")"
  touch "$LOGFILE"
  exec > >(tee -a "$LOGFILE") 2>&1

  echo "===================================================================="
  echo "Starting $(basename "$0")"
  echo "Date: $(date)"
  echo "Target user: $TARGET_USER"
  echo "Target home: $TARGET_HOME"
  echo "Log file: $LOGFILE"
  echo "===================================================================="

  command -v apt-get >/dev/null 2>&1 || {
    echo "ERROR: apt-get not found."
    exit 1
  }
  command -v raspi-config >/dev/null 2>&1 || {
    echo "ERROR: raspi-config not found."
    exit 1
  }
  command -v wget >/dev/null 2>&1 || {
    echo "ERROR: wget not found."
    exit 1
  }

  local arch kernel_arch model config_file overlays_dir wiringpi_dir deb_candidate dtbo_file demo_root
  arch="$(dpkg --print-architecture)"
  kernel_arch="$(uname -m)"
  model="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo "Unknown Raspberry Pi")"

  echo "Detected dpkg architecture: $arch"
  echo "Detected kernel architecture: $kernel_arch"
  echo "Detected model: $model"

  if [[ "$arch" != "arm64" ]]; then
    echo "WARNING: This script is intended for 64-bit Raspberry Pi OS (arm64)."
    echo "Continuing anyway, but Waveshare's WiringPi package step on the wiki is arm64-specific."
  fi

  config_file=""
  overlays_dir=""
  if [[ -f /boot/firmware/config.txt ]]; then
    config_file="/boot/firmware/config.txt"
    overlays_dir="/boot/firmware/overlays"
  elif [[ -f /boot/config.txt ]]; then
    config_file="/boot/config.txt"
    overlays_dir="/boot/overlays"
  else
    echo "ERROR: Could not find config.txt."
    exit 1
  fi

  echo "Using config file: $config_file"
  echo "Using overlays directory: $overlays_dir"

  mkdir -p "$WORKDIR" "$INSTALL_ROOT" "$overlays_dir"

  echo
  echo "==> Updating package index"
  apt-get update

  echo
  echo "==> Upgrading system"
  DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
  DEBIAN_FRONTEND=noninteractive apt-get full-upgrade -y

  echo
  echo "==> Installing required tools"
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    unzip \
    wget \
    curl \
    git \
    make \
    gcc \
    g++ \
    cmake \
    pkg-config \
    python3-pip \
    python3-pil \
    python3-numpy \
    python3-luma.oled \
    python3-luma.lcd \
    python3-rpi.gpio \
    python3-spidev \
    raspi-config

  echo
  echo "==> Attempting optional font package from Waveshare wiki"
  if ! DEBIAN_FRONTEND=noninteractive apt-get install -y ttf-mscorefonts-installer; then
    echo "Optional package ttf-mscorefonts-installer was not installed. Continuing."
  fi

  echo
  echo "==> Enabling SPI and I2C"
  raspi-config nonint do_spi 0 || true
  raspi-config nonint do_i2c 0 || true

  echo
  echo "==> Preparing working directories"
  cd "$WORKDIR"
  rm -rf wiringpi_extract overlay_extract demo_extract
  mkdir -p wiringpi_extract overlay_extract demo_extract

  echo
  echo "==> Downloading Waveshare WiringPi package"
  wget -O "$WORKDIR/WiringPi.zip" "$WIRINGPI_ZIP_URL"

  echo "==> Extracting Waveshare WiringPi package"
  unzip -o "$WORKDIR/WiringPi.zip" -d "$WORKDIR/wiringpi_extract"

  wiringpi_dir="$(find "$WORKDIR/wiringpi_extract" -maxdepth 3 -type d -name 'WiringPi' | head -n 1 || true)"
  if [[ -n "$wiringpi_dir" ]]; then
    echo "Found WiringPi source at: $wiringpi_dir"
    cd "$wiringpi_dir"

    echo "==> Building WiringPi Debian package"
    if ./build debian; then
      deb_candidate="$(find . -type f -name 'wiringpi_*_arm64.deb' | head -n 1 || true)"
      if [[ -z "$deb_candidate" ]]; then
        deb_candidate="$(find . -type f -name 'wiringpi_*.deb' | head -n 1 || true)"
      fi

      if [[ -n "$deb_candidate" ]]; then
        echo "==> Installing WiringPi package: $deb_candidate"
        apt-get install -y "$deb_candidate" || dpkg -i "$deb_candidate" || true
      else
        echo "WARNING: WiringPi .deb not found after build. Continuing."
      fi
    else
      echo "WARNING: WiringPi build step failed. Continuing."
    fi
  else
    echo "WARNING: Could not find WiringPi source directory. Continuing."
  fi

  if command -v gpio >/dev/null 2>&1; then
    echo "==> gpio -v output"
    gpio -v || true
  else
    echo "WARNING: gpio command not found after WiringPi step."
  fi

  cd "$WORKDIR"

  echo
  echo "==> Downloading Waveshare overlay archive"
  wget -O "$WORKDIR/OLED_LCD_HAT_A.zip" "$OVERLAY_ZIP_URL"

  echo "==> Extracting overlay archive"
  unzip -o "$WORKDIR/OLED_LCD_HAT_A.zip" -d "$WORKDIR/overlay_extract"

  dtbo_file="$(find "$WORKDIR/overlay_extract" -type f -name 'OLED_LCD_HAT_A.dtbo' | head -n 1 || true)"
  if [[ -z "$dtbo_file" ]]; then
    echo "ERROR: Could not find OLED_LCD_HAT_A.dtbo in the downloaded archive."
    exit 1
  fi

  echo "==> Installing device tree overlay"
  cp -f "$dtbo_file" "$overlays_dir/OLED_LCD_HAT_A.dtbo"
  chmod 644 "$overlays_dir/OLED_LCD_HAT_A.dtbo"
  echo "Installed overlay to $overlays_dir/OLED_LCD_HAT_A.dtbo"

  echo
  echo "==> Downloading Waveshare demo package"
  wget -O "$WORKDIR/OLED_LCD_HAT_A_Demo.zip" "$DEMO_ZIP_URL"

  echo "==> Extracting demo package"
  unzip -o "$WORKDIR/OLED_LCD_HAT_A_Demo.zip" -d "$WORKDIR/demo_extract"

  demo_root="$(find "$WORKDIR/demo_extract" -maxdepth 2 -type d -name 'OLED_LCD_HAT_A_Demo' | head -n 1 || true)"
  if [[ -z "$demo_root" ]]; then
    demo_root="$WORKDIR/demo_extract"
  fi

  echo "==> Installing demo package into $INSTALL_ROOT"
  rm -rf "$INSTALL_ROOT"
  mkdir -p "$INSTALL_ROOT"
  cp -a "$demo_root/." "$INSTALL_ROOT/"
  chown -R "$TARGET_USER:$TARGET_USER" "$INSTALL_ROOT"

  echo
  echo "==> Updating boot config"
  backup_file "$config_file"
  sed -i '/^dtparam=spi=on$/d' "$config_file"
  sed -i '/^dtoverlay=OLED_LCD_HAT_A$/d' "$config_file"
  sed -i '/^dtoverlay=OLED_LCD_HAT_A:rotate=90$/d' "$config_file"
  sed -i '/^display_rotate=0$/d' "$config_file"

  cat >>"$config_file" <<'CFGEOF'

# Waveshare OLED/LCD HAT (A)
dtparam=spi=on
dtoverlay=OLED_LCD_HAT_A:rotate=90
display_rotate=0
CFGEOF

  echo
  echo "==> Leaving existing desktop graphics stack alone"
  echo "This installer does not force Lite-mode Xorg/fbcp/startx boot changes."

  echo
  echo "==> Marking user profile for install state"
  append_if_missing "$TARGET_HOME/.profile" 'export WAVESHARE_OLED_LCD_HAT_A_INSTALLED=1'
  chown "$TARGET_USER:$TARGET_USER" "$TARGET_HOME/.profile"

  echo
  echo "==> Creating helper commands"
  cat >/usr/local/bin/waveshare-oled-lcd-hat-a-demo-lcd <<'BINEOF'
#!/usr/bin/env bash
set -euo pipefail
cd /opt/waveshare/OLED_LCD_HAT_A/python/example
exec sudo -E python3 2inch.py
BINEOF
  chmod 755 /usr/local/bin/waveshare-oled-lcd-hat-a-demo-lcd

  cat >/usr/local/bin/waveshare-oled-lcd-hat-a-demo-oled <<'BINEOF'
#!/usr/bin/env bash
set -euo pipefail
cd /opt/waveshare/OLED_LCD_HAT_A/python/example
exec sudo -E python3 0inch96.py
BINEOF
  chmod 755 /usr/local/bin/waveshare-oled-lcd-hat-a-demo-oled

  cat >/usr/local/bin/waveshare-oled-lcd-hat-a-demo-all <<'BINEOF'
#!/usr/bin/env bash
set -euo pipefail
cd /opt/waveshare/OLED_LCD_HAT_A/python/example
exec sudo -E python3 all.py
BINEOF
  chmod 755 /usr/local/bin/waveshare-oled-lcd-hat-a-demo-all

  echo
  echo "==> Build C demo if present"
  if [[ -d "$INSTALL_ROOT/c" ]]; then
    cd "$INSTALL_ROOT/c"
    make clean || true
    make -j"$(nproc)" || true
    chown -R "$TARGET_USER:$TARGET_USER" "$INSTALL_ROOT/c"
  fi

  echo
  echo "===================================================================="
  echo "Waveshare base installation complete"
  echo "===================================================================="
  echo "Wiki reference: $WIKI_PAGE_URL"
  echo "Overlay: $overlays_dir/OLED_LCD_HAT_A.dtbo"
  echo "Demo:    $INSTALL_ROOT"
}

if [[ -z "${EXPECTED_CODENAME:-}" ]]; then
  EXPECTED_CODENAME=$(detect_codename || true)
  if [[ -z "$EXPECTED_CODENAME" ]]; then
    warn "Unable to detect OS codename; defaulting to bookworm."
    EXPECTED_CODENAME="bookworm"
  fi
fi

export EXPECTED_CODENAME
export DESK_DISPLAY_OUTPUT="${DESK_DISPLAY_OUTPUT:-framebuffer}"
export REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements_framebuffer.txt}"
export DISPLAY_WIDTH="${DISPLAY_WIDTH:-320}"
export DISPLAY_HEIGHT="${DISPLAY_HEIGHT:-240}"
export DISPLAY_FB_DEVICE="${DISPLAY_FB_DEVICE:-/dev/fb1}"
export DISPLAY_ROTATION="${DISPLAY_ROTATION:-0}"
export BUTTON_A="${BUTTON_A:-24}"
export BUTTON_B="${BUTTON_B:-4}"
export BUTTON_X="${BUTTON_X:-17}"
export BUTTON_Y="${BUTTON_Y:-23}"

ENV_PATH="$PROJECT_DIR/.env"
ENV_LINES=()
ENV_LINES+=("DESK_DISPLAY_OUTPUT=${DESK_DISPLAY_OUTPUT}")
ENV_LINES+=("DISPLAY_WIDTH=${DISPLAY_WIDTH}")
ENV_LINES+=("DISPLAY_HEIGHT=${DISPLAY_HEIGHT}")
ENV_LINES+=("DISPLAY_FB_DEVICE=${DISPLAY_FB_DEVICE}")
ENV_LINES+=("DISPLAY_ROTATION=${DISPLAY_ROTATION}")
ENV_LINES+=("BUTTON_A=${BUTTON_A}")
ENV_LINES+=("BUTTON_B=${BUTTON_B}")
ENV_LINES+=("BUTTON_X=${BUTTON_X}")
ENV_LINES+=("BUTTON_Y=${BUTTON_Y}")
prepend_env_vars "$ENV_PATH" "${ENV_LINES[@]}"

log "Desk Display will render to ${DISPLAY_WIDTH}x${DISPLAY_HEIGHT} using ${DISPLAY_FB_DEVICE}."
log "This installer also enables a helper service for the OLED side displays (temperature + time)."
log "Button mapping: A=GPIO${BUTTON_A}, B=GPIO${BUTTON_B}, X=GPIO${BUTTON_X}, Y=GPIO${BUTTON_Y}."

install_waveshare_stack

"$PROJECT_DIR/scripts/helpers/base_setup.sh"

if [[ ! -e "$DISPLAY_FB_DEVICE" ]]; then
  warn "${DISPLAY_FB_DEVICE} does not exist yet."
  warn "Reboot may be required after overlay install, then rerun if needed."
fi

install_framebuffer_launcher "$PROJECT_DIR" "desk_display.service" "$SERVICE_USER" || true

ensure_executable "$PROJECT_DIR/scripts/waveshare_oled_status.py"

VENV_DIR=$(detect_existing_venv "$PROJECT_DIR" || true)
if [[ -z "$VENV_DIR" ]]; then
  VENV_DIR="$PROJECT_DIR/venv"
fi

log "Writing Waveshare OLED helper service to $WAVESHARE_OLED_SERVICE_PATH"
$SUDO tee "$WAVESHARE_OLED_SERVICE_PATH" >/dev/null <<SERVICE
[Unit]
Description=Desk Display Waveshare OLED status helper
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=-$PROJECT_DIR/.env
ExecStart=$VENV_DIR/bin/python $PROJECT_DIR/scripts/waveshare_oled_status.py
Restart=always
RestartSec=2
User=$SERVICE_USER

[Install]
WantedBy=multi-user.target
SERVICE

$SUDO systemctl daemon-reload
$SUDO systemctl enable "$WAVESHARE_OLED_SERVICE_NAME"
$SUDO systemctl restart "$WAVESHARE_OLED_SERVICE_NAME"

log "Installation complete. Reboot is recommended before running Waveshare demos."
