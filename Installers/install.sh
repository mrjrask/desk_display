#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

print_usage() {
  cat <<'USAGE'
Usage:
  bash ./Installers/install.sh [profile]

Profiles:
  display_hat_mini   (default)
  adafruit_minipitft
  hyperpixel
  kernel
  waveshare_oled_lcd_hat_a
USAGE
}

resolve_installer() {
  case "${1:-}" in
    ""|display_hat_mini)
      echo "$PROJECT_DIR/Installers/install_display_hat_mini.sh"
      ;;
    adafruit_minipitft|minipitft|adafruit_minipitft_114)
      echo "$PROJECT_DIR/Installers/install_adafruit_minipitft_114.sh"
      ;;
    hyperpixel)
      echo "$PROJECT_DIR/Installers/install_hyperpixel.sh"
      ;;
    kernel)
      echo "$PROJECT_DIR/Installers/install_kernel.sh"
      ;;
    waveshare|waveshare_oled_lcd_hat_a)
      echo "$PROJECT_DIR/Installers/install_waveshare_oled_lcd_hat_a.sh"
      ;;
    *)
      return 1
      ;;
  esac
}

prompt_profile() {
  cat <<'MENU'
Select an install profile:
  1) display_hat_mini (default)
  2) adafruit_minipitft
  3) hyperpixel
  4) kernel
  5) waveshare_oled_lcd_hat_a
MENU
  read -r -p "Enter choice [1-5]: " choice
  case "$choice" in
    ""|1) echo "display_hat_mini" ;;
    2) echo "adafruit_minipitft" ;;
    3) echo "hyperpixel" ;;
    4) echo "kernel" ;;
    5) echo "waveshare_oled_lcd_hat_a" ;;
    *) return 1 ;;
  esac
}

profile="${1:-}"

if [[ -z "$profile" && -t 0 ]]; then
  profile=$(prompt_profile) || {
    echo "[ERROR] Invalid selection." >&2
    exit 1
  }
fi

installer=$(resolve_installer "$profile") || {
  echo "[ERROR] Unknown profile: ${profile:-<empty>}" >&2
  print_usage >&2
  exit 1
}

if [[ ! -x "$installer" ]]; then
  chmod +x "$installer"
fi

echo "[INFO] Running installer: $installer"
exec "$installer"
