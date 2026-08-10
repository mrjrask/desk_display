#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

print_usage() {
  cat <<'USAGE'
Usage:
  bash ./Installers/install.sh [profile] [screen_defaults]

Profiles:
  display_hat_mini   (default)
  adafruit_minipitft
  hyperpixel
  kernel
  macos_window
  pi_window
  win_window
  waveshare_oled_lcd_hat_a

Screen defaults:
  small
  large               (default)
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
    macos_window|macos|mac)
      echo "$PROJECT_DIR/Installers/install_macos_window.sh"
      ;;
    pi_window|pi_desktop|raspberry_pi_window)
      echo "$PROJECT_DIR/Installers/install_pi_window.sh"
      ;;
    win_window|windows_window|windows11_window)
      echo "$PROJECT_DIR/Installers/install_win_window.sh"
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
  5) macos_window
  6) pi_window
  7) win_window
  8) waveshare_oled_lcd_hat_a
MENU
  read -r -p "Enter choice [1-8]: " choice
  case "$choice" in
    ""|1) echo "display_hat_mini" ;;
    2) echo "adafruit_minipitft" ;;
    3) echo "hyperpixel" ;;
    4) echo "kernel" ;;
    5) echo "macos_window" ;;
    6) echo "pi_window" ;;
    7) echo "win_window" ;;
    8) echo "waveshare_oled_lcd_hat_a" ;;
    *) return 1 ;;
  esac
}

prompt_screen_defaults() {
  cat <<'MENU'
Which default screen rotation should be loaded?
  1) small
  2) large (default)
MENU
  read -r -p "Enter choice [1-2]: " choice
  case "$choice" in
    1) echo "small" ;;
    ""|2) echo "large" ;;
    *) return 1 ;;
  esac
}

profile="${1:-}"
screen_defaults="${2:-}"

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
"$installer"

if [[ -z "$screen_defaults" && -t 0 ]]; then
  screen_defaults=$(prompt_screen_defaults) || {
    echo "[WARN] Invalid selection; skipping screen rotation defaults." >&2
    screen_defaults=""
  }
fi

if [[ -n "$screen_defaults" ]]; then
  echo "[INFO] Loading $screen_defaults screen rotation defaults."
  if ! python3 "$PROJECT_DIR/tools/load_default_screen_config.py" "$screen_defaults"; then
    echo "[WARN] Failed to load $screen_defaults screen rotation defaults." >&2
  fi
else
  echo "[INFO] Skipping screen rotation defaults (no selection made)."
fi
