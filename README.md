# Desk Display

Desk Display is a Python dashboard application for Raspberry Pi and Linux displays. It rotates through configurable screens (weather, date/time, indoor sensors, sports, standings, scoreboards, and more) and supports multiple output targets including SPI TFTs, framebuffer devices, kernel/fullscreen displays, scalable SDL windows, and headless rendering.

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
- [Screen catalog (canonical IDs)](#screen-catalog-canonical-ids)
- [Web UI (screen configuration)](#web-ui-screen-configuration)
- [Services](#services)
- [Developer workflow](#developer-workflow)
- [Troubleshooting](#troubleshooting)
- [API notes](#api-notes)

---

## Features

- Rotating screen engine with per-screen frequency control.
- Per-screen playback extension via optional `extra_seconds` in schedule config.
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
- `window` (SDL windowed mode; useful on macOS/desktop)
- `framebuffer`
- `headless`

Hardware/workflows currently supported in this repo:

- Pimoroni Display HAT Mini (320×240)
- Adafruit miniPiTFT 1.14" (240×135)
- HyperPixel/kernel fullscreen displays
- Windowed SDL rendering on desktop/macOS
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

- Raspberry Pi OS / Linux for hardware deployments.
- macOS and Windows are supported for SDL window mode via the provided launch/install scripts.
- Python 3.9+ (project tooling targets Python 3.11).
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

### Single entrypoint (fresh installs)

```bash
bash ./Installers/install.sh
```

Pass an optional profile (`display_hat_mini`, `adafruit_minipitft`, `hyperpixel`, `kernel`, `macos_window`, `pi_window`, `win_window`, `waveshare_oled_lcd_hat_a`) to skip prompts.

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

The HyperPixel installer intentionally runs exactly one display loop:

- On desktop sessions, HyperPixel uses the per-user kernel service, `desk_display-kernel.service`, and keeps the system display service, `desk_display.service`, disabled. The display loop sets `SCREEN_CONFIG_AUTOSTART=0`, so it does not spawn `config_ui.py` itself.
- On Lite/headless systems where the installer falls back to framebuffer output, HyperPixel uses the system display service, `desk_display.service`, and disables the per-user `desk_display-kernel.service`.
- The separate `config_ui_desk_display.service` is the only config UI process to start when you want the web UI. Start it temporarily with `sudo systemctl start config_ui_desk_display.service`, stop it with `sudo systemctl stop config_ui_desk_display.service`, or enable it persistently only if you want it at every boot.

After installing, verify that only the selected runtime service is active, and that the config UI service is active only when you intentionally started it:

```bash
systemctl --user status desk_display-kernel.service
sudo systemctl status desk_display.service
sudo systemctl status config_ui_desk_display.service
```

### macOS scalable HyperPixel 4 window (800×480 render)

```bash
bash ./Installers/install_macos_window.sh
./scripts/update_dependencies.sh
# Normal desktop window launcher
./launch_macos_window.sh

# Lower CPU / smoother UI profile (opt-in)
./scripts/launch_macos_window_perf.sh
```

Use `launch_macos_window.sh` for the standard desktop profile. Use `launch_macos_window_perf.sh` when you want conservative defaults that reduce CPU load on macOS (window scale `1`, screenshots/video disabled, and Wi-Fi monitor/recovery disabled by default).

### Raspberry Pi Desktop window mode (800×480 render)

```bash
bash ./Installers/install_pi_window.sh
./scripts/update_dependencies.sh
./launch_pi_window.sh
```

### Windows 11 window mode (800×480 render)

```bash
bash ./Installers/install_win_window.sh
# Activate/install dependencies in your preferred shell first, then:
./launch_win_window.sh
```

### Config UI service only (for existing installs)

```bash
bash ./Installers/install_config_ui_service.sh
```

## Configuration

Configuration is environment-driven. Most values can be placed in `.env`.

### Core display variables

| Variable | Description |
| --- | --- |
| `DESK_DISPLAY_OUTPUT` | Output mode (`auto`, `displayhatmini`, `minipitft`, `kernel`, `window`, `framebuffer`, `headless`) |
| `DESK_DISPLAY_PROFILE` | Optional deployment profile (for example `hyperpixel_pi_zero`) |
| `DESK_DISPLAY_LOW_POWER` | Low-power defaults flag; defaults to `1` when `DESK_DISPLAY_PROFILE=hyperpixel_pi_zero` |
| `DESK_DISPLAY_SERVICE_START_DELAY` | Optional systemd launch delay in seconds; defaults to `15` for low-power or detected Raspberry Pi Zero-style deployments and `0` otherwise |
| `DISPLAY_TARGET_FPS` | Overall display output write cap; all normal display writes are paced centrally to this target and default to 60 FPS normally or 12 FPS in low-power mode. Explicit clears/shutdown blanking may bypass the cap so the screen blanks immediately. |
| `DISPLAY_SCROLL_TARGET_FPS` | Scroll/scoreboard frame pacing target; defaults to the active display profile normally and 12 FPS in low-power mode |
| `DISPLAY_ANIMATION_TARGET_FPS` | General animation/fade frame pacing target; defaults to 60 FPS normally and 12 FPS in low-power mode |
| `DESK_DISPLAY_FORCE_HEADLESS` | Force headless behavior |
| `DISPLAY_WIDTH` / `DISPLAY_HEIGHT` | Physical output size override (for example `800`×`480` for HyperPixel 4) |
| `RENDER_WIDTH` / `RENDER_HEIGHT` | Optional internal render canvas size; frames are scaled once to `DISPLAY_WIDTH` / `DISPLAY_HEIGHT` at output |
| `DISPLAY_RENDER_SCALE` | Optional internal render scale used when `RENDER_WIDTH` / `RENDER_HEIGHT` are unset (for example `0.5` renders an 800×480 panel at 400×240) |
| `DISPLAY_ROTATION` | App rotation (`0`,`90`,`180`,`270` or `0-3`) |
| `DISPLAY_FB_DEVICE` | Framebuffer device (default `/dev/fb0`) |
| `DISPLAY_FB_PIXEL_FORMAT` | FB pixel format override |
| `DISPLAY_FB_PIXEL_ORDER` | FB channel order (`rgb` / `bgr`) |

### Weather/travel/sensor variables (common)

| Variable group | Purpose |
| --- | --- |
| `WEATHERKIT_*`, `OWM_*` | Weather provider credentials/settings |
| `TRAVEL_MODE`, `TRAVEL_TO_*`, `GOOGLE_MAPS_*`, `APPLE_MAPS_*`, `MAPKIT_TOKEN` | Travel route + provider selection/auth |
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
| `ENABLE_SCREENSHOTS` | Enable screenshot capture (default: `0` on macOS when `DESK_DISPLAY_OUTPUT=window` or when low-power mode is active; otherwise `1`) |
| `ENABLE_VIDEO` | Enable rolling MP4 capture |
| `SCREENSHOT_DIR` | Screenshot output location |
| `SCREENSHOT_ARCHIVE_BASE` | Archive location |

On macOS desktop/window setups (`DESK_DISPLAY_OUTPUT=window`), screenshot capture defaults to disabled to avoid extra capture load unless explicitly enabled. HyperPixel/Pi Zero low-power deployments (`DESK_DISPLAY_LOW_POWER=1` or `DESK_DISPLAY_PROFILE=hyperpixel_pi_zero`) also default screenshots off, and the HyperPixel installer writes `ENABLE_SCREENSHOTS=0`, `ENABLE_VIDEO=0`, `SCREEN_CONFIG_AUTOSTART=0`, and `ENABLE_WIFI_MONITOR=0` unless overridden in the installer environment. Screenshots and video can be re-enabled manually for debugging with `ENABLE_SCREENSHOTS=1` and `ENABLE_VIDEO=1`, but should stay off for normal Pi Zero 2 W operation. Set `ENABLE_SCREENSHOTS=0` to force-disable regardless of platform/output mode.

For Pi Zero 2 W or HyperPixel deployments, use the low-power frame pacing profile unless you are actively debugging animation smoothness:

```bash
DESK_DISPLAY_PROFILE=hyperpixel_pi_zero
DESK_DISPLAY_LOW_POWER=1
# Keep the HyperPixel panel output at full resolution while composing fewer pixels.
DISPLAY_WIDTH=800
DISPLAY_HEIGHT=480
RENDER_WIDTH=400
RENDER_HEIGHT=240
# Optional explicit override; the low-power defaults already use 12 FPS.
DISPLAY_TARGET_FPS=12
DISPLAY_SCROLL_TARGET_FPS=12
DISPLAY_ANIMATION_TARGET_FPS=12
```

The low-power profile caps all normal display output writes at `DISPLAY_TARGET_FPS` and keeps scrolling and animations around 10-15 FPS. Scroll and animation loops also use their own `DISPLAY_SCROLL_TARGET_FPS` and `DISPLAY_ANIMATION_TARGET_FPS` pacing values, but the central `DISPLAY_TARGET_FPS` cap still applies to their physical output writes. The HyperPixel installer can write a reduced internal render canvas such as `RENDER_WIDTH=400` and `RENDER_HEIGHT=240` while preserving the full `DISPLAY_WIDTH=800` and `DISPLAY_HEIGHT=480` panel output. This reduces CPU load and heat on Pi Zero-class hardware while preserving normal 60 FPS/full-render defaults on faster desktop or Raspberry Pi hardware. Low-power or detected Raspberry Pi Zero-style service installs also add a 15-second systemd pre-start delay so the Pi has more time to settle before the display, config UI, and helper services launch; set `DESK_DISPLAY_SERVICE_START_DELAY=0` before installing to opt out or another whole-second value to tune it.

Pi Zero-class HyperPixel installs also select `screens_config.pi_zero_hyperpixel.json` by default when the installer detects a Raspberry Pi Zero model and no custom `SCREENS_CONFIG_PATH` is already set. That lightweight preset enables only `date`, `weather1`, `weather2`, and `inside` by default, while keeping radar, scoreboards, standings, live sports, stock/league/team logos, and animated/quad-style screens disabled. The installer pairs it with `screens_config.pi_zero_hyperpixel.local.json` for UI edits so the standard schedule remains untouched.

After the display has run long enough to confirm stable CPU temperature and load, re-enable heavier screens one at a time from the config UI or by editing the Pi Zero local config. Start with one candidate, run for a while, and keep it only if temperature/load remain acceptable before enabling the next expensive screen.

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
- object form supports:
  - `frequency` (required)
  - optional `extra_seconds` (adds hold time after normal screen duration)
  - optional `alt` schedule

Example object form:

```json
{
  "screens": {
    "weather1": { "frequency": 1, "extra_seconds": 5 },
    "date": {
      "frequency": 1,
      "alt": { "screen": "inside", "frequency": 2 }
    }
  }
}
```

### MLB series screens (Cubs/Sox)

The series screens display all games in an upcoming or in-progress series:

- Scheduled games show relative day labels (`Today` / `Tomorrow` / `Tonight`) and start time.
- Final games show score lines.
- "Next Home Series" intentionally skips duplicating the same game set as "Next Series" when the next series is already at home.

Default playlist order is:

- Cubs: `cubs next` → `cubs next home` → `cubs current series` → `cubs next series` → `cubs next home series`
- Sox: `sox next` → `sox next home` → `sox current series` → `sox next series` → `sox next home series`

---

## Screen catalog (canonical IDs)

The authoritative list of valid screen IDs lives in `screens_catalog.py` as `RAW_SCREEN_IDS`.

Current canonical IDs:

- Core/weather/inside: `date`, `nixie`, `quad`, `weather logo`, `weather1`, `weather2`, `weather hourly`, `weather daily`, `weather quad`, `weather radar`, `inside`, `verano logo`, `vrnof`
- NFL/Bears: `bears logo`, `bears stand1`, `bears stand2`, `bears next`, `bears next season`, `nfl logo`, `NFL Scoreboard`, `NFL Overview NFC`, `NFL Overview AFC`, `NFL Standings NFC`, `NFL Standings AFC`
- NBA/Bulls/NCAAM: `nba logo`, `NBA Scoreboard`, `NBA Playoffs`, `NCAAM Scoreboard`, `bulls logo`, `bulls stand1`, `bulls last`, `bulls live`, `bulls next`, `bulls next home`, `bulls schedule quad`
- NHL/Blackhawks/Wolves: `hawks logo`, `hawks stand1`, `hawks last`, `hawks live`, `hawks next`, `hawks next home`, `hawks schedule quad`, `nhl logo`, `NHL Scoreboard`, `NHL Playoffs`, `NHL Standings Overview West`, `NHL Standings Overview East`, `NHL Standings West`, `NHL Standings West v2`, `NHL Standings East`, `NHL Standings East v2`, `wolves logo`, `wolves last`, `wolves next`, `wolves next home`
- MLB/Cubs/Sox/league: `cubs logo`, `cubs stand1`, `cubs stand2`, `cubs stand3`, `cubs last`, `cubs result`, `cubs live`, `cubs next`, `cubs next home`, `cubs current series`, `cubs next series`, `cubs next home series`, `cubs schedule quad`, `sox logo`, `sox stand1`, `sox stand2`, `sox stand3`, `sox last`, `sox live`, `sox next`, `sox next home`, `sox current series`, `sox next series`, `sox next home series`, `sox schedule quad`, `mlb logo`, `MLB Scoreboard`, `NL Overview`, `AL Overview`, `MLB AL Standings`, `MLB NL Standings`

Legacy IDs are canonicalized automatically (`time` → `nixie`, `sensors` → `inside`, and legacy `* v2` scoreboard aliases map to current names), so older saved configs still load.

---

## Web UI (screen configuration)

Start the UI manually for a local development session:

```bash
python config_ui.py
```

On systemd installs, start the dedicated config UI service only while you need it:

```bash
sudo systemctl start config_ui_desk_display.service
# make your changes at http://localhost:5002
sudo systemctl stop config_ui_desk_display.service
```

HyperPixel installs write `SCREEN_CONFIG_AUTOSTART=0` and the kernel user service also forces that value, so the display loop will not spawn a second config UI process.

Default URL:

- `http://localhost:5002`

The UI allows:

- enabling/disabling screens,
- editing frequencies,
- editing per-screen additional playback seconds (`extra_seconds`),
- managing playlists,
- import/export of config payloads.

### Quad touch behavior (HyperPixel 4 / HyperPixel 4 Square)

When `quad` or `weather quad` is currently displayed on a touch-capable HyperPixel setup:

- Tapping one of the 4 tiles opens that tile as fullscreen.
- The fullscreen tile uses the normal screen display duration (ignores `extra_seconds` for that touch-initiated play).
- After fullscreen playback ends, the app returns to the quad screen.
- The return-to-quad interval also uses the normal duration (again ignoring `extra_seconds` once), then normal rotation resumes.

---

## Services

Common service operations:

```bash
sudo systemctl status desk_display.service
sudo systemctl status config_ui_desk_display.service
sudo systemctl restart desk_display.service
sudo systemctl restart config_ui_desk_display.service
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

# Validate external API connectivity/credentials
python scripts/test_api_connections.py

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
- High CPU on Mac: use `./scripts/launch_macos_window_perf.sh` or set `DESK_DISPLAY_WINDOW_SCALE=1`, `ENABLE_SCREENSHOTS=0`, and `ENABLE_VIDEO=0` (and optionally `ENABLE_WIFI_MONITOR=0` / `ENABLE_WIFI_RECOVERY=0` for desktop setups).

---

## API notes

External API endpoint and payload references: [README_APIS.md](README_APIS.md).
