# Desk Display

Desk Display is a Python dashboard application for Raspberry Pi and Linux displays. It rotates through configurable screens (weather, date/time, indoor sensors, sports, standings, scoreboards, and more) and supports multiple output targets including SPI TFTs, framebuffer devices, kernel/fullscreen displays, and headless rendering.

---

## Table of Contents

- [Features](#features)
- [Supported output modes](#supported-output-modes)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running](#running)
- [Installers (recommended on Raspberry Pi)](#installers-recommended-on-raspberry-pi)
- [Configuration](#configuration)
- [Screens and scheduling](#screens-and-scheduling)
- [Web UI (screen configuration)](#web-ui-screen-configuration)
- [Services](#services)
- [Developer workflow](#developer-workflow)
- [Troubleshooting](#troubleshooting)
- [API notes](#api-notes)

---

## Features

- Rotating screen engine with per-screen frequency control.
- Sports screens for NFL/NHL/NBA/MLB/NCAAM (including scoreboards and standings).
- Team-specific screens for Chicago teams (Bears, Blackhawks, Wolves, Bulls, Cubs, Sox).
- Cubs/Sox **series screens**:
  - `cubs current series`
  - `cubs next series`
  - `cubs next home series`
  - `sox current series`
  - `sox next series`
  - `sox next home series`
- Optional screenshots and rolling video capture.
- Flask-based screen configuration UI with optional auth.
- Installer scripts for common Raspberry Pi display setups.
- Optional Wi-Fi monitor/recovery automation.

---

## Supported output modes

Set with `DESK_DISPLAY_OUTPUT`:

- `auto` (default)
- `displayhatmini`
- `minipitft`
- `kernel`
- `framebuffer`
- `headless`

Hardware/workflows currently supported in this repo:

- Pimoroni Display HAT Mini (320×240)
- Adafruit miniPiTFT 1.14" (240×135)
- HyperPixel/kernel fullscreen displays
- Framebuffer rendering (`/dev/fb*`)
- Headless render/test mode

---

## Project structure

- `main.py` – runtime loop, refresh orchestration, transitions, display writes.
- `config.py` – environment/config parsing and runtime constants.
- `config_ui.py` – Flask/Waitress screen configuration app.
- `screens/` – all screen renderer modules and the screen registry.
- `services/` – API/data-provider clients.
- `data_fetch.py` – sports/team schedule and standings fetching.
- `screens_config.json` – default screen frequencies and playlists.
- `screens_catalog.py` – canonical list of valid screen IDs.
- `tests/` – pytest suite.
- `Installers/` and `scripts/` – deployment/service helpers.

---

## Requirements

- Linux environment (Raspberry Pi OS recommended for hardware use).
- Python 3.9+.
- Display target or headless mode.

### Typical OS packages (Raspberry Pi OS/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-pip python3-dev python3-opencv \
  build-essential libjpeg-dev libopenblas0 libopenblas-dev swig liblgpio-dev \
  libopenjp2-7-dev libtiff5-dev libcairo2-dev libpango1.0-dev \
  libgdk-pixbuf-2.0-dev libffi-dev network-manager wireless-tools iproute2 \
  i2c-tools fonts-dejavu-core fonts-noto-color-emoji libgl1 libx264-dev ffmpeg git
```

> On newer Debian releases, `libgdk-pixbuf-2.0-dev` is the correct package name.

---

## Installation

From project root:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Alternative requirements sets:

- `requirements.txt` (full/default)
- `requirements_kernel.txt`
- `requirements_framebuffer.txt`
- `requirements_minipitft.txt`

Example:

```bash
pip install -r requirements_kernel.txt
```

Update dependencies later:

```bash
./scripts/update_dependencies.sh
```

---

## Running

```bash
python main.py
```

Notes:

- By default, startup loads `.env` (`CONFIG_LOAD_DOTENV=1`).
- Use `DESK_DISPLAY_FORCE_HEADLESS=1` for render-only mode.

---

## Installers (recommended on Raspberry Pi)

### Display HAT Mini

```bash
bash ./Installers/install_display_hat_mini.sh
```

### Adafruit miniPiTFT 1.14"

```bash
bash ./Installers/install_adafruit_minipitft_114.sh
```

### Waveshare OLED/LCD HAT (A)

```bash
bash ./Installers/install_waveshare_oled_lcd_hat_a.sh
```

### HyperPixel / kernel display

```bash
bash ./Installers/install_hyperpixel.sh
bash ./Installers/install_kernel.sh
```

### AirPlay takeover mode

```bash
bash ./Installers/install_airplay.sh
```

---

## Configuration

Configuration is environment-driven. Most values can be placed in `.env`.

### Core display variables

| Variable | Description |
| --- | --- |
| `DESK_DISPLAY_OUTPUT` | Output mode (`auto`, `displayhatmini`, `minipitft`, `kernel`, `framebuffer`, `headless`) |
| `DESK_DISPLAY_FORCE_HEADLESS` | Force headless behavior |
| `DISPLAY_WIDTH` / `DISPLAY_HEIGHT` | Render size override |
| `DISPLAY_ROTATION` | App rotation (`0`,`90`,`180`,`270` or `0-3`) |
| `DISPLAY_FB_DEVICE` | Framebuffer device (default `/dev/fb0`) |
| `DISPLAY_FB_PIXEL_FORMAT` | FB pixel format override |
| `DISPLAY_FB_PIXEL_ORDER` | FB channel order (`rgb` / `bgr`) |

### Weather/travel/sensor variables (common)

| Variable group | Purpose |
| --- | --- |
| `WEATHERKIT_*`, `OWM_*` | Weather provider credentials/settings |
| `TRAVEL_TO_*`, `APPLE_MAPS_*` | Travel route + Apple Maps config |
| `INSIDE_SENSOR`, `INSIDE_I2C_BUSES` | Indoor sensor selection and I2C probing |
| `ENABLE_WIFI_MONITOR`, `ENABLE_WIFI_RECOVERY` | Wi-Fi monitor/recovery controls |

### UI/auth variables

| Variable | Description |
| --- | --- |
| `SCREEN_CONFIG_HOST` | Config UI bind host (default `0.0.0.0`) |
| `SCREEN_CONFIG_PORT` | Config UI port (default `5002`) |
| `SCREEN_UI_PASSWORD` | Enable password-protected UI |
| `SCREEN_UI_USERNAME` | Optional username |
| `SCREEN_AUTH_ENABLED` | Force auth mode behavior |

### Capture variables

| Variable | Description |
| --- | --- |
| `ENABLE_SCREENSHOTS` | Enable screenshot capture |
| `ENABLE_VIDEO` | Enable rolling MP4 capture |
| `SCREENSHOT_DIR` | Screenshot output location |
| `SCREENSHOT_ARCHIVE_BASE` | Archive location |

For API-specific keys/fields, see [README_APIS.md](README_APIS.md).

---

## Screens and scheduling

Screen IDs are defined in:

- `screens_catalog.py` (canonical IDs)
- `screens_config.json` (`screens` frequencies + `playlists`)

Frequency behavior:

- `1` = every pass
- `2` = every other pass
- `0` = disabled
- object form supports `frequency` + optional `alt` schedule

### MLB series screens (Cubs/Sox)

The series screens display all games in an upcoming or in-progress series:

- Scheduled games show relative day labels (`Today` / `Tomorrow` / `Tonight`) and start time.
- Final games show score lines.
- "Next Home Series" intentionally skips duplicating the same game set as "Next Series" when the next series is already at home.

Default playlist order is:

- Cubs: `cubs next` → `cubs next home` → `cubs current series` → `cubs next series` → `cubs next home series`
- Sox: `sox next` → `sox next home` → `sox current series` → `sox next series` → `sox next home series`

---

## Web UI (screen configuration)

Start the UI:

```bash
python config_ui.py
```

Default URL:

- `http://localhost:5002`

The UI allows:

- enabling/disabling screens,
- editing frequencies,
- managing playlists,
- import/export of config payloads.

---

## Services

Common service operations:

```bash
sudo systemctl status desk_display.service
sudo systemctl restart desk_display.service
./scripts/restart_services.sh
```

Kernel user service (if installed):

```bash
systemctl --user status desk_display-kernel.service
systemctl --user restart desk_display-kernel.service
```

Uninstall helpers:

```bash
./scripts/uninstall.sh
./scripts/uninstall_airplay.sh
```

---

## Developer workflow

### Typical commands

```bash
# Run tests
pytest

# Validate required files
python tools/validate_required_files.py

# Render screens for verification
python tools/maintenance/render_all_screens.py
```

### Focused checks

```bash
pytest -q tests/test_screens_catalog.py tests/test_screen_registry.py
python -m py_compile main.py data_fetch.py screens/registry.py screens/mlb_schedule.py
```

---

## Troubleshooting

- Weather empty: verify `WEATHERKIT_*` or `OWM_API_KEY` values.
- Travel blank: verify travel route variables and Apple Maps credentials.
- Wrong rotation: avoid double-rotation between overlay and app rotation.
- Blank framebuffer/kernel output: verify `DESK_DISPLAY_OUTPUT`, dimensions, and target device path.
- Waveshare OLED/LCD issues: run `scripts/check_waveshare_setup.sh` and verify I2C addresses and framebuffer settings.
- AirPlay not visible: verify `avahi-daemon` and local-network mDNS support.

---

## API notes

External API endpoint and payload references: [README_APIS.md](README_APIS.md).
