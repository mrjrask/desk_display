# Desk Display

Desk Display is a Raspberry Pi dashboard app that rotates through weather, date, inside, and sports screens on small displays (or fullscreen desktop/kernel displays).

It supports:

- **Pimoroni Display HAT Mini** (`320x240`) over SPI.
- **Adafruit miniPiTFT 1.14"** (`240x135`) via Python script-driven SPI output.
- **Kernel/desktop fullscreen output** (for HDMI/DSI/HyperPixel-style panels).
- **Framebuffer output** (headless-style rendering direct to `/dev/fb*`).
- **Headless mode** for rendering/testing without writing to a display.

---

## Table of Contents

- [What this project includes](#what-this-project-includes)
- [Requirements](#requirements)
- [Install](#install)
- [Run](#run)
- [Installers (recommended on Raspberry Pi OS)](#installers-recommended-on-raspberry-pi-os)
- [Configuration](#configuration)
- [Screen scheduling](#screen-scheduling)
- [Screen configuration web UI](#screen-configuration-web-ui)
- [Screenshots and video capture](#screenshots-and-video-capture)
- [Services](#services)
- [Developer workflow](#developer-workflow)
- [Troubleshooting](#troubleshooting)
- [API references](#api-references)

---

## What this project includes

- A **main display loop** (`main.py`) with screen scheduling, transitions, screenshot capture, and optional video output.
- A **screen registry/catalog** with weather, sports, inside, and utility screens.
- A **Flask + Waitress configuration UI** (`config_ui.py`) for managing screen order/frequency and importing/exporting settings.
- **Installer scripts** for Display HAT Mini, Waveshare OLED/LCD HAT (A), and kernel-display workflows.
- Optional **Wi-Fi health monitoring/recovery**.

---

## Requirements

- Raspberry Pi OS / Debian-based Linux (tested paths are oriented around Raspberry Pi).
- Python **3.9+**.
- A display target:
  - Display HAT Mini,
  - kernel/desktop panel,
  - framebuffer device,
  - or headless mode.

### Base system packages (Raspberry Pi OS)

```bash
sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-pip python3-dev python3-opencv \
  build-essential libjpeg-dev libopenblas0 libopenblas-dev swig liblgpio-dev \
  libopenjp2-7-dev libtiff5-dev libcairo2-dev libpango1.0-dev \
  libgdk-pixbuf-2.0-dev libffi-dev network-manager wireless-tools iproute2 \
  i2c-tools fonts-dejavu-core fonts-noto-color-emoji libgl1 libx264-dev ffmpeg git
```

> Debian Trixie uses `libgdk-pixbuf-2.0-dev` (not the older `libgdk-pixbuf2.0-dev`).

---

## Install

From the project root:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Alternate requirements files

- `requirements.txt`: Display HAT Mini + full stack.
- `requirements_kernel.txt`: kernel/desktop display stack (no Display HAT Mini dependency).
- `requirements_framebuffer.txt`: framebuffer stack (no Display HAT Mini dependency).
- `requirements_minipitft.txt`: miniPiTFT SPI Python-driver stack.

Example:

```bash
pip install -r requirements_kernel.txt
```

### Upgrade dependencies later

```bash
./scripts/update_dependencies.sh
```

Optional explicit requirements file:

```bash
./scripts/update_dependencies.sh --requirements requirements_kernel.txt
```

Upgrade *all installed* pip packages in the venv:

```bash
./tools/maintenance/update_pip_installed_packages.sh
```

Dry run:

```bash
./tools/maintenance/update_pip_installed_packages.sh --dry-run
```

---

## Run

Basic run:

```bash
python main.py
```

The app defaults `CONFIG_LOAD_DOTENV=1`, so `.env` is loaded automatically during normal app startup.

---

## Installers (recommended on Raspberry Pi OS)

### Display HAT Mini (SPI)

```bash
bash ./Installers/install_display_hat_mini.sh
```

### Adafruit miniPiTFT 1.14" (240×135)

```bash
bash ./Installers/install_adafruit_minipitft_114.sh
```

Notes:

- Defaults to `DESK_DISPLAY_OUTPUT=minipitft` with `DISPLAY_WIDTH=240` and `DISPLAY_HEIGHT=135`.
- Uses `requirements_minipitft.txt` (includes the Adafruit RGB display driver).
- Uses SPI script output (ST7789), not kernel/DRM output.

### Waveshare OLED/LCD HAT (A)

```bash
bash ./Installers/install_waveshare_oled_lcd_hat_a.sh
```

Notes:

- Targets a 320×240 LCD framebuffer workflow.
- Installs an OLED helper for the two side OLEDs (time/temp with auto-fit + anti burn-in swapping/fade behavior).
- Default button mapping on this board:
  - `A -> K4 (D24)`
  - `B -> K1 (D4)`
  - `X -> K2 (D17)`
  - `Y -> K3 (D23)`

### Kernel/fullscreen displays

```bash
# HyperPixel 4 / HyperPixel 4 Square flow
bash ./Installers/install_hyperpixel.sh

# Non-interactive HyperPixel example
HYPERPIXEL_PANEL=hyperpixel4 DISPLAY_WIDTH=800 DISPLAY_HEIGHT=480 bash ./Installers/install_hyperpixel.sh

# Generic kernel-display flow
bash ./Installers/install_kernel.sh
```

Kernel installer behavior includes:

- Sets `DESK_DISPLAY_OUTPUT=kernel`.
- Configures sizing env vars and a default `DISPLAY_ROTATION=0`.
- Installs helpers/services for desktop-session launching.
- Supports fallback to framebuffer on Lite/headless setups.

> `dtoverlay=` lines in `/boot/config.txt` or `/boot/firmware/config.txt` are still your responsibility.

---

## Configuration

Configuration is primarily environment-driven (and may be loaded from `.env`).

### Core display/output variables

| Variable | Meaning |
| --- | --- |
| `DESK_DISPLAY_OUTPUT` | `auto` (default), `displayhatmini`, `minipitft`, `kernel`, `framebuffer`, `headless`. |
| `DESK_DISPLAY_FORCE_HEADLESS` | Force render without writing to hardware. |
| `DISPLAY_WIDTH` / `DISPLAY_HEIGHT` | Override render resolution. |
| `DISPLAY_FB_DEVICE` | Framebuffer path (default `/dev/fb0`). |
| `DISPLAY_FB_PIXEL_FORMAT` | Framebuffer format override (`rgb565`, `rgb888`, `xrgb8888`, etc.). |
| `DISPLAY_FB_PIXEL_ORDER` | Force color order (`rgb` / `bgr`). |
| `DISPLAY_ROTATION` | App rotation (`0`,`90`,`180`,`270` or shorthand `0-3`). |
| `DISPLAY_ROTATION_STRICT` | Rotation conflict guard for kernel overlays. |
| `DISPLAY_HAT_MINI_REINIT_SECONDS` | Reinit interval for long-run panel recovery. |
| `DISPLAY_HAT_MINI_LED_LEVEL` | Display HAT Mini LED brightness (`0.0`-`1.0`). |

### Weather variables

| Variable | Meaning |
| --- | --- |
| `WEATHERKIT_TEAM_ID`, `WEATHERKIT_KEY_ID`, `WEATHERKIT_SERVICE_ID` | Apple WeatherKit credentials. |
| `WEATHERKIT_KEY_PATH` / `WEATHERKIT_PRIVATE_KEY` | WeatherKit private key source. |
| `WEATHERKIT_LANGUAGE`, `WEATHERKIT_TIMEZONE` | Weather localization settings. |
| `OWM_API_KEY*` | OpenWeatherMap fallback API key(s). |
| `OWM_UNITS`, `OWM_LANGUAGE` | OpenWeatherMap localization options. |
| `WEATHER_REFRESH_SECONDS` | Refresh interval (minimum clamped in code). |
| `WEATHER_USE_EMOJI_ICONS` | Enable emoji icons for weather glyphs. |

### Travel variables

| Variable | Meaning |
| --- | --- |
| `TRAVEL_TO_HOME_ORIGIN`, `TRAVEL_TO_HOME_DESTINATION` | Home commute route endpoints. |
| `TRAVEL_TO_WORK_ORIGIN`, `TRAVEL_TO_WORK_DESTINATION` | Work commute route endpoints. |
| `APPLE_MAPS_TEAM_ID`, `APPLE_MAPS_KEY_ID` | Apple Maps JWT identifiers. |
| `APPLE_MAPS_KEY_PATH` / `APPLE_MAPS_PRIVATE_KEY` | Apple Maps private key source. |

### Indoor sensor variables

| Variable | Meaning |
| --- | --- |
| `INSIDE_SENSOR` | Optional indoor sensor override (`pim_sensor_stick`, `pimoroni_bme280`, `adafruit_bme280`, `pimoroni_bme680`, `pimoroni_bme68x`, `adafruit_bme680`, `adafruit_sht41`). |
| `PIM_SENSOR_STICK_I2C_BUS` | Optional Linux I2C bus for Pimoroni Multi-Sensor Stick (`LTR559` + `LSM6DS3`) reads. |
| `INSIDE_I2C_BUSES` | Comma-separated fallback I2C buses used for inside/sensor probing. |

### Wi-Fi monitor/recovery variables

| Variable | Meaning |
| --- | --- |
| `ENABLE_WIFI_MONITOR` | Enable background Wi-Fi monitor thread. |
| `ENABLE_WIFI_RECOVERY` | Allow automated recovery actions. |
| `WIFI_INTERFACE` | Force interface name. |
| `WIFI_TCP_PROBE_URLS`, `WIFI_TCP_PROBE_HOSTS`, `WIFI_TCP_PROBE_PORT` | Connectivity probes. |
| `WIFI_RECOVERY_LOG` | Recovery log path override. |

### Screenshots/video variables

| Variable | Meaning |
| --- | --- |
| `ENABLE_SCREENSHOTS` | Enable per-screen image capture. |
| `ENABLE_VIDEO` | Enable rolling H.264 MP4 capture. |
| `SCREENSHOT_DIR` | Screenshot root path override. |
| `SCREENSHOT_ARCHIVE_BASE` | Screenshot archive path override. |

### Waveshare OLED helper variables

| Variable | Meaning |
| --- | --- |
| `WAVESHARE_OLED_I2C_BUS` | I2C bus index for helper. |
| `WAVESHARE_OLED_TEMP_ADDR`, `WAVESHARE_OLED_TIME_ADDR` | OLED I2C addresses. |
| `WAVESHARE_OLED_WIDTH`, `WAVESHARE_OLED_HEIGHT` | OLED dimensions. |
| `WAVESHARE_OLED_TEMP_SOURCE` | `weather1` (default), `weather`, `cpu`, or `command`. |
| `WAVESHARE_OLED_TEMP_COMMAND` | Command source when using `command`. |
| `WAVESHARE_OLED_TEMP_UNIT` | `C` or `F`. |
| `WAVESHARE_OLED_REFRESH_SECONDS` | OLED refresh cadence. |
| `WAVESHARE_OLED_FADE_STEPS` | Fade-step count. |
| `WAVESHARE_OLED_FADE_STEP_MS` | Delay per fade step. |
| `WAVESHARE_OLED_FONT_PATH` | Optional custom TTF for OLED values. |
| `BUTTON_A`, `BUTTON_B`, `BUTTON_X`, `BUTTON_Y` | GPIO BCM pin mapping overrides. |

---

## Screen scheduling

Screen sequencing is configured in `screens_config.json` with a `screens` mapping.

Simple example:

```json
{
  "screens": {
    "date": 1,
    "weather1": 1,
    "inside": 2,
    "NFL Scoreboard": 4
  }
}
```

Rules:

- `1` = show every pass.
- `2` = show every other pass.
- Higher numbers reduce cadence.
- `0` disables the screen while keeping its entry.

Advanced entries can use an object with `frequency` and optional `alt` schedule metadata.

---

## Screen configuration web UI

Run:

```bash
python config_ui.py
```

Defaults:

- URL: `http://localhost:5002`
- Host env var: `SCREEN_CONFIG_HOST` (default `0.0.0.0`)
- Port env var: `SCREEN_CONFIG_PORT` (default `5002`)

Authentication:

- Set `SCREEN_UI_PASSWORD` to require login.
- Optionally set `SCREEN_UI_USERNAME`.
- `SCREEN_AUTH_ENABLED=1` can force auth mode behavior.

---

## Screenshots and video capture

With screenshots enabled:

- Per-screen captures are written under `screenshots/<Screen Name>/`.
- Current/latest mirrors are kept under `screenshots/current/`.
- Archive rollover occurs when screenshot volume reaches the internal threshold (current default: `500`).
- Archived batches land under `screenshot_archive/<Screen Name>/`.

---

## Services

Installers can provision service units.

### Main service

```bash
sudo systemctl status desk_display.service
sudo systemctl restart desk_display.service
./scripts/restart_services.sh
```

If `systemctl restart ...` returns `Job for <service>.service canceled`, use
`./scripts/restart_services.sh` to perform a stop/start fallback for
`desk_display.service` and (when installed) `desk_display_waveshare_oled.service`.

### Kernel desktop-session user service

```bash
systemctl --user status desk_display-kernel.service
systemctl --user restart desk_display-kernel.service
```

SSH helper:

```bash
./scripts/ssh_kernel_display.sh status
./scripts/ssh_kernel_display.sh restart
./scripts/ssh_kernel_display.sh stop
```

Uninstall helper:

```bash
./scripts/uninstall.sh
```

AirPlay cleanup helper (removes legacy AirPlay service/launchers/remnants from existing installs):

```bash
./scripts/uninstall_airplay.sh
```

---

## Developer workflow

### Useful commands

```bash
# Run test suite
pytest

# Validate required files
python tools/validate_required_files.py

# Render sample screens to files (maintenance)
python tools/maintenance/render_all_screens.py
```

### Project layout (high-level)

- `main.py`: display loop runtime.
- `config.py`: env parsing + config constants.
- `config_ui.py`: web UI and config APIs.
- `screens/`: screen renderers.
- `services/`: network/data providers and utilities.
- `tools/`: maintenance/render/testing helper scripts.
- `tests/`: pytest test suite.

---

## Troubleshooting

- **No weather data:** verify WeatherKit vars or provide an OpenWeatherMap key.
- **Radar map missing:** verify `GOOGLE_MAPS_API_KEY`.
- **Travel shows N/A:** verify `TRAVEL_*` endpoints and Apple Maps credentials.
- **Blank kernel display:** ensure valid desktop session/env or use framebuffer mode.
- **Rotation looks wrong on HyperPixel/kernel:** avoid double rotation (`dtoverlay rotate` + app rotation).
- **Wi-Fi monitor loops:** review Wi-Fi recovery logs and consider `ENABLE_WIFI_RECOVERY=0`.
- **Waveshare OLED/LCD HAT (A) shows black screen + cursor / OLEDs off:** run `scripts/check_waveshare_setup.sh` and verify `DESK_DISPLAY_OUTPUT=framebuffer`, `DISPLAY_FB_DEVICE`, and I2C addresses `0x3c` + `0x3d`.

---

## API references

For third-party endpoint and field details, see [`README_APIS.md`](README_APIS.md).
