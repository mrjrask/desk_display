#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

print_usage() {
  cat <<'USAGE'
Usage:
  bash ./Installers/install.sh [--mode MODE] [--credentials FILE] [profile] [screen_defaults] [install_adsb]

Modes (see OPERATIONS.md, "Installation modes"):
  standalone          (default) main.py draws to the panel, as before
  server              render server and config UI; no panel, no profile needed
  client              display client for a remote server; needs a panel profile
  combined            server plus a local panel client; needs a panel profile

--credentials FILE    client: the .env.client the server's "Add a display"
                      form produced (identity and credential)

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

Install ADS-B collector service:
  y / yes
  n / no              (default)
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

prompt_install_adsb() {
  read -r -p "Install the ADS-B collector service (requires ADSB_DEVICE_1_HOST in .env)? [y/N]: " choice
  case "$choice" in
    y|Y|yes|Yes|YES) echo "yes" ;;
    ""|n|N|no|No|NO) echo "no" ;;
    *) return 1 ;;
  esac
}

prompt_mode() {
  cat <<'MENU'
Select an installation mode:
  1) standalone (default)
  2) server
  3) client
  4) combined (server plus this panel)
MENU
  read -r -p "Enter choice [1-4]: " choice
  case "$choice" in
    ""|1) echo "standalone" ;;
    2) echo "server" ;;
    3) echo "client" ;;
    4) echo "combined" ;;
    *) return 1 ;;
  esac
}

mode=""
credentials=""
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) mode="${2:-}"; shift 2 ;;
    --mode=*) mode="${1#*=}"; shift ;;
    --credentials) credentials="${2:-}"; shift 2 ;;
    --credentials=*) credentials="${1#*=}"; shift ;;
    -h|--help) print_usage; exit 0 ;;
    *) positional+=("$1"); shift ;;
  esac
done
set -- "${positional[@]+"${positional[@]}"}"

profile="${1:-}"
screen_defaults="${2:-}"
install_adsb="${3:-}"

if [[ -z "$mode" && -t 0 ]]; then
  mode=$(prompt_mode) || {
    echo "[ERROR] Invalid selection." >&2
    exit 1
  }
fi
mode="${mode:-standalone}"
case "$mode" in
  standalone|server|client|combined) ;;
  *)
    echo "[ERROR] Unknown mode: $mode" >&2
    print_usage >&2
    exit 1
    ;;
esac
if [[ -n "$credentials" ]]; then
  if [[ "$mode" != "client" && "$mode" != "combined" ]]; then
    echo "[ERROR] --credentials applies only to client and combined installs." >&2
    exit 1
  fi
  credentials=$(cd -- "$(dirname -- "$credentials")" && pwd)/$(basename -- "$credentials")
fi
export DESK_DISPLAY_INSTALL_MODE="$mode"
if [[ -n "$credentials" ]]; then
  export DESK_DISPLAY_CLIENT_CREDENTIALS="$credentials"
fi

if [[ "$mode" == "server" ]]; then
  # A server drives no panel: no hardware installer, no display profile.
  echo "[INFO] Running the server installer."
  bash "$PROJECT_DIR/scripts/helpers/base_setup.sh"
  echo "[INFO] Move an existing rotation onto the server with scripts/migrate_standalone_config.py."
  profile="server"
fi

if [[ "$mode" != "server" && -z "$profile" && -t 0 ]]; then
  profile=$(prompt_profile) || {
    echo "[ERROR] Invalid selection." >&2
    exit 1
  }
fi

if [[ "$mode" != "server" ]]; then
  installer=$(resolve_installer "$profile") || {
    echo "[ERROR] Unknown profile: ${profile:-<empty>}" >&2
    print_usage >&2
    exit 1
  }
  if [[ "$mode" != "standalone" ]]; then
    case "$(basename -- "$installer")" in
      install_macos_window.sh|install_pi_window.sh|install_win_window.sh)
        echo "[ERROR] The $profile profile runs by hand and has no $mode service; pick a panel profile." >&2
        exit 1
        ;;
    esac
    export DESK_DISPLAY_INSTALL_PROFILE="$(basename -- "$installer" .sh)"
    DESK_DISPLAY_INSTALL_PROFILE="${DESK_DISPLAY_INSTALL_PROFILE#install_}"
    DESK_DISPLAY_INSTALL_PROFILE="${DESK_DISPLAY_INSTALL_PROFILE%_114}"
    if [[ -f "$PROJECT_DIR/.env.client" ]]; then
      # Keep panel changes with the client's existing identity.
      export DESK_DISPLAY_PANEL_ENV_FILE=".env.client"
    fi
  fi

  if [[ ! -x "$installer" ]]; then
    chmod +x "$installer"
  fi

  echo "[INFO] Running installer ($mode): $installer"
  "$installer"
fi

if [[ "$mode" != "standalone" ]]; then
  # The rotation lives on the server (migrate it or edit playlists in the
  # config UI), and the ADS-B collector feeds the server's screens.
  screen_defaults="${screen_defaults:-skip}"
fi
if [[ "$mode" == "client" ]]; then
  install_adsb="no"
fi

if [[ -z "$screen_defaults" && -t 0 ]]; then
  screen_defaults=$(prompt_screen_defaults) || {
    echo "[WARN] Invalid selection; skipping screen rotation defaults." >&2
    screen_defaults=""
  }
fi

if [[ "$screen_defaults" == "skip" ]]; then
  echo "[INFO] Skipping standalone screen rotation defaults for the $mode install."
elif [[ -n "$screen_defaults" ]]; then
  echo "[INFO] Loading $screen_defaults screen rotation defaults."
  if ! python3 "$PROJECT_DIR/scripts/load_default_screen_config.py" "$screen_defaults"; then
    echo "[WARN] Failed to load $screen_defaults screen rotation defaults." >&2
  fi
else
  echo "[INFO] Skipping screen rotation defaults (no selection made)."
fi

if [[ -z "$install_adsb" && -t 0 ]]; then
  install_adsb=$(prompt_install_adsb) || {
    echo "[WARN] Invalid selection; skipping ADS-B collector service install." >&2
    install_adsb="no"
  }
fi

case "$install_adsb" in
  y|Y|yes|Yes|YES)
    adsb_installer="$PROJECT_DIR/Installers/install_adsb_collector_service.sh"
    if [[ ! -x "$adsb_installer" ]]; then
      chmod +x "$adsb_installer"
    fi
    echo "[INFO] Running installer: $adsb_installer"
    "$adsb_installer"
    ;;
  *)
    echo "[INFO] Skipping ADS-B collector service install."
    ;;
esac
