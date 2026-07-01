# Desk Display

Desk Display is a Python dashboard for always-on Raspberry Pi, Linux, macOS, and Windows displays. It renders a configurable rotation of weather, date/time, indoor sensor, finance, travel, sports schedule, scoreboard, standings, and playoff screens to small SPI TFTs, kernel/framebuffer displays, SDL windows, or a headless renderer.

The project is optimized for desk-sized devices but also includes larger 800×480 and 1080p display profiles, a browser-based screen configuration UI, install scripts for common Raspberry Pi hardware, diagnostics for external feeds, and maintenance tools for screenshot/render validation.

---

## Table of contents

- [Highlights](#highlights)
- [Supported displays and output modes](#supported-displays-and-output-modes)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Installer workflows](#installer-workflows)
- [Running the app](#running-the-app)
- [Configuration](#configuration)
- [Screens, playlists, and scheduling](#screens-playlists-and-scheduling)
- [Canonical screen IDs](#canonical-screen-ids)
- [Configuration UI](#configuration-ui)
- [Services and operations](#services-and-operations)
- [Developer workflow](#developer-workflow)
- [Troubleshooting](#troubleshooting)
- [External APIs](#external-apis)

---

## Highlights

- Rotating screen engine with per-screen frequency, alternate-screen, and extra-hold-time controls.
- JSON-backed playlists and sequence ordering in `screens_config.json`.
- Optional quad layouts in `screens_layouts.json`, including touch-to-fullscreen behavior on supported HyperPixel setups.
- Weather screens for current conditions, forecast details, hourly forecast, daily forecast, astronomical/sun events, and radar imagery.
- Indoor sensor screen with BME280/BME680/BME688/SHT4x-style sensor support and configurable I2C probing.
- Sports coverage for NFL, NHL, NBA, MLB, NCAAM, and AHL/Wolves helpers.
- Chicago-focused team screens for Bears, Blackhawks, Wolves, Bulls, Cubs, and White Sox.
- MLB series screens for current, next, and next-home Cubs/Sox series.
- NHL and NBA playoff bracket screens.
- Optional screenshot capture and rolling video capture.
- Flask/Waitress configuration UI with optional password protection.
- Installer scripts for Display HAT Mini, Adafruit miniPiTFT, Waveshare OLED/LCD HAT (A), HyperPixel/kernel display, Pi desktop window mode, macOS window mode, and Windows window mode.
- Optional Wi-Fi monitor/recovery utilities for Raspberry Pi deployments.

---

## Supported displays and output modes

Select the renderer with `DESK_DISPLAY_OUTPUT`.

| Mode | Purpose |
| --- | --- |
| `auto` | Default mode. Runtime attempts to pick an available display path. |
| `displayhatmini` | Pimoroni Display HAT Mini, normally 320×240. |
| `minipitft` | Adafruit miniPiTFT 1.14", normally 240×135. |
| `kernel` | Kernel/fullscreen display path, commonly used for HyperPixel and KMS/DRM setups. |
| `window` | SDL desktop window mode for macOS, Windows, Raspberry Pi desktop, and Linux desktops. |
| `framebuffer` | Direct framebuffer output to `/dev/fb*`. |
| `headless` | Render/test mode without writing to hardware. |

Supported workflow profiles include:

- Pimoroni Display HAT Mini.
- Adafruit miniPiTFT 1.14".
- Waveshare OLED/LCD HAT (A) helper/status workflow.
- HyperPixel 4 / HyperPixel 4 Square via kernel display paths.
- HDMI/fallback HD profiles.
- SDL desktop windows on macOS, Windows, Raspberry Pi desktop, and Linux.
- Direct framebuffer rendering.
- Headless rendering for tests and maintenance tools.

---

## Repository layout

| Path | Purpose |
| --- | --- |
| `main.py` | Runtime loop, refresh orchestration, transitions, capture, touch/button handling, and display writes. |
| `config.py` | Environment parsing, defaults, display profile detection, style config, API credentials, and runtime constants. |
| `config_ui.py` | Flask/Waitress screen configuration web app. |
| `screens/` | Screen renderer modules and registry integration. |
| `screens/registry.py` | Screen registration, playlist config loading, layout config loading, aliases, and special display helpers. |
| `screens_catalog.py` | Canonical screen IDs and legacy ID canonicalization. |
| `screens_config.json` | Default screen frequencies, playlists, and sequence. Runtime/local overrides may be stored separately by config helpers. |
| `screens_layouts.json` | Quad/layout configuration. |
| `data_fetch.py` | Weather, sports, finance, AHL, and team data fetch/normalization helpers. |
| `services/` | Shared API/provider clients, HTTP session helpers, sports service modules, and Wi-Fi utilities. |
| `templates/` | Config UI and screenshot UI templates. |
| `Installers/` | Platform/display install scripts. |
| `scripts/` | Service, launch, diagnostics, setup, and operations scripts. |
| `tools/` | Import/export, validation, font audit, and maintenance rendering/cleanup tools. |
| `tests/` | Pytest suite. |
| `images/` | Team, league, and static image assets. |
| `fonts/` | Bundled fonts used by renderers. |
| `vendor/` | Vendored sensor libraries used by Raspberry Pi installs. |

---

## Requirements

### Runtime

- Python 3.9+; project lint/tooling targets Python 3.11.
- A display target, SDL desktop session, framebuffer device, or headless mode.
- Network access for live weather, map, finance, and sports feeds.
- Optional API credentials for WeatherKit, OpenWeatherMap, Google Maps, and Apple Maps.

### Typical Debian/Raspberry Pi OS packages

```bash
sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-pip python3-dev python3-opencv \
  build-essential libjpeg-dev libopenblas0 libopenblas-dev swig liblgpio-dev \
  libopenjp2-7-dev libtiff5-dev libcairo2-dev libpango1.0-dev \
  libgdk-pixbuf-2.0-dev libffi-dev network-manager wireless-tools iproute2 \
  i2c-tools fonts-dejavu-core fonts-noto-color-emoji libgl1 libx264-dev ffmpeg git
```

> On newer Debian releases, `libgdk-pixbuf-2.0-dev` is the expected package name.

### Python requirement files

| File | Intended use |
| --- | --- |
| `requirements.txt` | Full/default install, including hardware and desktop dependencies. |
| `requirements_kernel.txt` | Kernel/HyperPixel-focused install set. |
| `requirements_framebuffer.txt` | Framebuffer-focused install set. |
| `requirements_minipitft.txt` | Adafruit miniPiTFT-focused install set. |
| `requirements_sensors_pimoroni.txt` | Optional editable Pimoroni BME280/BME680/BME68x sensor drivers. |

Pimoroni sensor drivers are optional because they use editable installs from
`vendor/`. The installer adds them automatically only when `INSIDE_SENSOR` (or
legacy `INDOOR_SENSOR`) is configured as `pimoroni_bme280`, `pimoroni_bme680`,
or `pimoroni_bme68x` in the environment or `.env` before running the installer:

```bash
INSIDE_SENSOR=pimoroni_bme680 bash ./Installers/install.sh display_hat_mini
```

For manual setup, install the regular requirements first, then add the optional
sensor requirements from the repository root:

```bash
pip install -r requirements.txt
pip install -r requirements_sensors_pimoroni.txt
```

---

## Quick start

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

For a desktop-safe render path:

```bash
DESK_DISPLAY_OUTPUT=window python main.py
```

For a render-only smoke test:

```bash
DESK_DISPLAY_FORCE_HEADLESS=1 DESK_DISPLAY_OUTPUT=headless python main.py
```

Update installed Python dependencies later with:

```bash
./scripts/update_dependencies.sh
```

---

## Installer workflows

### Single entry point

```bash
bash ./Installers/install.sh
```

You can pass a profile to skip prompts:

```bash
bash ./Installers/install.sh display_hat_mini
bash ./Installers/install.sh adafruit_minipitft
bash ./Installers/install.sh hyperpixel
bash ./Installers/install.sh kernel
bash ./Installers/install.sh macos_window
bash ./Installers/install.sh pi_window
bash ./Installers/install.sh win_window
bash ./Installers/install.sh waveshare_oled_lcd_hat_a
```

### Hardware/display-specific installers

```bash
# Pimoroni Display HAT Mini
bash ./Installers/install_display_hat_mini.sh

# Adafruit miniPiTFT 1.14"
bash ./Installers/install_adafruit_minipitft_114.sh

# Waveshare OLED/LCD HAT (A)
bash ./Installers/install_waveshare_oled_lcd_hat_a.sh

# HyperPixel / kernel display
bash ./Installers/install_hyperpixel.sh
bash ./Installers/install_kernel.sh

# Existing install: add/update config UI service only
bash ./Installers/install_config_ui_service.sh
```

### Desktop/window installers

```bash
# macOS 800×480-style desktop window
bash ./Installers/install_macos_window.sh
./scripts/update_dependencies.sh
./launch_macos_window.sh

# macOS lower-CPU profile
./scripts/launch_macos_window_perf.sh

# Raspberry Pi desktop window
bash ./Installers/install_pi_window.sh
./scripts/update_dependencies.sh
./launch_pi_window.sh

# Windows 11 window mode
bash ./Installers/install_win_window.sh
./launch_win_window.sh
```

`launch_macos_window_perf.sh` uses conservative defaults intended to reduce desktop CPU load, including a window scale of `1`, disabled screenshots/video by default, and disabled Wi-Fi monitor/recovery by default.

---

## Running the app

```bash
python main.py
```

Useful launch variants:

```bash
# Load .env explicitly (default is enabled unless CONFIG_LOAD_DOTENV=0)
CONFIG_LOAD_DOTENV=1 python main.py

# Run an SDL window
DESK_DISPLAY_OUTPUT=window python main.py

# Force headless behavior
DESK_DISPLAY_FORCE_HEADLESS=1 DESK_DISPLAY_OUTPUT=headless python main.py

# Run the configuration UI
python config_ui.py
```

At startup the app loads `.env` when `CONFIG_LOAD_DOTENV` is enabled, initializes display/profile settings, fetches startup data, and begins rotating enabled screens from the configured sequence.

---

## Configuration

Configuration is environment-driven. Put local values in `.env` for development and most installs, or configure them in systemd service environment files for long-running deployments.

### Core display variables

| Variable | Description |
| --- | --- |
| `CONFIG_LOAD_DOTENV` | Load `.env` at startup. Defaults to enabled. |
| `DESK_DISPLAY_OUTPUT` | Output mode: `auto`, `displayhatmini`, `minipitft`, `kernel`, `window`, `framebuffer`, or `headless`. |
| `DESK_DISPLAY_FORCE_HEADLESS` | Force headless behavior even if another output is configured. |
| `DISPLAY_WIDTH` / `DISPLAY_HEIGHT` | Render dimensions override. |
| `DISPLAY_ROTATION` | App rotation. Accepts degrees (`0`, `90`, `180`, `270`) or quarter-turn values (`0`-`3`). |
| `DISPLAY_FB_DEVICE` | Framebuffer device path. Defaults to `/dev/fb0` where applicable. |
| `DISPLAY_FB_PIXEL_FORMAT` | Framebuffer pixel format override. |
| `DISPLAY_FB_PIXEL_ORDER` | Framebuffer channel order, usually `rgb` or `bgr`. |
| `DISPLAY_FB_HIDE_CONSOLE_CURSOR` | Hide the Linux console cursor in framebuffer mode. Defaults to enabled; set to `0` to disable. |
| `DISPLAY_FB_CONSOLE_GRAPHICS` | Switch the active console to graphics mode while framebuffer output is open, preventing fbcon cursor/text redraws. Defaults to enabled; set to `0` to only send cursor-hide controls. |
| `HYPERPIXEL_PANEL` | HyperPixel panel hint, such as `hyperpixel4`. |
| `DESK_DISPLAY_WINDOW_SCALE` | SDL window scaling factor. |
| `DESK_DISPLAY_WINDOW_RESIZABLE` | Allow SDL window resizing. |
| `DESK_DISPLAY_SDL_FULLSCREEN` | Start SDL in fullscreen mode. |
| `DESK_DISPLAY_SDL_DRIVERS` / `SDL_VIDEODRIVER` | SDL driver selection/override. |

### Runtime and interaction variables

| Variable | Description |
| --- | --- |
| `SCREEN_CONFIG_AUTOSTART` | Controls automatic config UI startup from `main.py` where supported. |
| `STARTUP_CRITICAL_FEED_TIMEOUT_SECONDS` | Startup feed timeout guard. |
| `DESK_DISPLAY_GC_INTERVAL_SECONDS` | Periodic garbage collection interval. |
| `TOUCH_DOUBLE_TAP_MAX_INTERVAL_SECONDS` | Double-tap timing for touch interactions. |
| `ESC_DOUBLE_PRESS_ACTION` | Action for double-pressing Escape in SDL/window contexts. |
| `ESC_DOUBLE_PRESS_MAX_INTERVAL_SECONDS` | Double-Escape timing window. |
| `DARK_HOURS` | Time windows used to suppress/alter display behavior during dark hours. |
| `DISPLAY_FADE_IN_ENABLED` | Enables fade-in behavior where supported. |
| `DISPLAY_HAT_MINI_REINIT_SECONDS` | Periodic Display HAT Mini reinitialization interval; `0` disables. |
| `DISPLAY_HAT_MINI_LED_ENABLED` | Enables Display HAT Mini LED behavior. |
| `DISPLAY_HAT_MINI_LED_LEVEL` | Display HAT Mini LED level. |
| `DISPLAY_HAT_MINI_LED_INDICATOR_BORDER_ENABLED` | Adds the LED indicator border on Display HAT Mini. |
| `HYPERPIXEL_LED_INDICATOR_BORDER_ENABLED` | Adds an indicator border on HyperPixel profiles. |
| `HYPERPIXEL_LED_INDICATOR_BORDER_WIDTH` | Indicator border width. |

### Weather, maps, travel, and sensor variables

| Variable group | Purpose |
| --- | --- |
| `WEATHER_LATITUDE`, `WEATHER_LONGITUDE` | Weather and map center location. |
| `WEATHER_REFRESH_SECONDS` | Weather refresh interval. |
| `HOURLY_FORECAST_HOURS` | Number of hourly forecast entries displayed. |
| `WEATHERKIT_*` | Apple WeatherKit team/key/service/private-key settings. |
| `OWM_API_KEY`, `OWM_UNITS`, `OWM_LANGUAGE` | OpenWeatherMap fallback settings. |
| `TRAVEL_MODE`, `TRAVEL_TO_HOME_*`, `TRAVEL_TO_WORK_*` | Travel origin/destination selection. |
| `GOOGLE_MAPS_API_KEY` | Google Directions and Static Maps credentials. |
| `APPLE_MAPS_*`, `MAPKIT_TOKEN` | Apple Maps Directions/Snapshot credentials and endpoint overrides. |
| `INSIDE_SENSOR`, `INSIDE_I2C_BUSES` | Indoor sensor selection and I2C bus probing. |
| `PRESSURE_HISTORY_PATH` | Pressure history cache path for trend display. |

Set `INSIDE_SENSOR` to `pimoroni_bme280`, `pimoroni_bme680`, or
`pimoroni_bme68x` before running an installer when you need the optional
vendored Pimoroni sensor drivers. If those optional drivers are absent,
the inside screen keeps its normal fallback behavior and is skipped when no
supported sensor can be probed.

### Sports and data variables

| Variable | Description |
| --- | --- |
| `TEAM_STANDINGS_DISPLAY_SECONDS` | Hold time for team standings pages. |
| `NCAAM_SCOREBOARD_MODE` | NCAAM scoreboard mode; defaults to `top25`. |
| `NHL_BREAK_WINDOWS_JSON` | Optional NHL break-window override JSON used by registry helpers. |
| `SMALL_RESULT_FLAG_H` | Small flag height for MLB result/schedule renderers. |
| `AHL_API_BASE_URL`, `AHL_API_KEY`, `AHL_CLIENT_CODE`, `AHL_LEAGUE_ID`, `AHL_SITE_ID`, `AHL_SEASON_ID`, `AHL_TEAM_ID`, `AHL_TEAM_TRICODE`, `AHL_TEAM_NAME`, `AHL_SCHEDULE_ICS_URL` | AHL/Wolves feed configuration. |

### Config UI and auth variables

| Variable | Description |
| --- | --- |
| `SCREEN_CONFIG_HOST` | Config UI bind host; default is `0.0.0.0`. |
| `SCREEN_CONFIG_PORT` | Config UI port; default is `5002`. |
| `SCREEN_UI_PASSWORD` | Enables password-protected UI when set. |
| `SCREEN_UI_USERNAME` | Optional username. |
| `SCREEN_AUTH_ENABLED` | Force auth behavior. |

### Capture and artifacts

| Variable | Description |
| --- | --- |
| `ENABLE_SCREENSHOTS` | Enables screenshot capture. Defaults to disabled on macOS window mode and enabled elsewhere unless overridden. |
| `ENABLE_VIDEO` | Enables rolling MP4 capture. |
| `SCREENSHOT_DIR` | Current screenshot output location. |
| `SCREENSHOT_ARCHIVE_BASE` | Screenshot archive base location. |

### Wi-Fi monitor/recovery variables

| Variable | Description |
| --- | --- |
| `ENABLE_WIFI_MONITOR` | Enables Wi-Fi monitoring where wired into launch/service scripts. |
| `ENABLE_WIFI_RECOVERY` | Enables Wi-Fi recovery actions where supported. |
| `WIFI_INTERFACE` | Interface to monitor/recover. |
| `WIFI_TCP_PROBE_HOST`, `WIFI_TCP_PROBE_HOSTS`, `WIFI_TCP_PROBE_PORT`, `WIFI_TCP_PROBE_URL`, `WIFI_TCP_PROBE_URLS`, `WIFI_HTTPS_PROBE_URL` | Connectivity probes. |
| `WIFI_RECOVERY_LOG` | Wi-Fi recovery log path. |
| `RPI_CONNECT_CONTROL_HOST` | Raspberry Pi Connect host used by recovery helpers. |

### Style and layout config

| File/variable | Purpose |
| --- | --- |
| `SCREENS_STYLE_PATH` | Optional path override for screen style configuration. |
| `screens_layouts.json` | Quad pages, tiles, and layout-specific settings. |
| `screens_config.json` | Default screen frequencies, playlists, and sequence. |

For detailed third-party credentials and endpoint behavior, see [README_APIS.md](README_APIS.md).

---

## Screens, playlists, and scheduling

Screen rotation is driven by `screens_config.json`.

Top-level fields:

| Field | Purpose |
| --- | --- |
| `screens` | Per-screen frequency config. |
| `playlists` | Named groups of screen steps. |
| `sequence` | Ordered list of playlist references to rotate through. |

Frequency values:

- `1` means show every pass.
- `2` means show every other pass.
- Larger integers show less often.
- `0` disables a screen.
- Object form enables extended metadata:
  - `frequency` is required.
  - `extra_seconds` adds hold time after the normal display duration.
  - `alt` can define an alternate screen and alternate frequency.

Example:

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

The default playlists include starter, weather, sensors, stocks, Hawks, NHL, Wolves, Cubs, Sox, MLB, Bears, NFL, Bulls, NBA, and quad groups.

### MLB series screens

Cubs and Sox series screens display all games in an upcoming or in-progress series:

- `cubs current series`
- `cubs next series`
- `cubs next home series`
- `sox current series`
- `sox next series`
- `sox next home series`

Scheduled games show relative day labels such as `Today`, `Tonight`, and `Tomorrow` plus start time. Final games show score lines. The next-home-series helpers avoid duplicating the same game set as next-series when the next series is already at home.

### Quad touch behavior

When `quad` or `weather quad` is shown on a touch-capable HyperPixel setup:

- Tapping a tile opens that tile as a fullscreen screen.
- Touch-initiated fullscreen playback uses the normal screen display duration.
- `extra_seconds` is ignored for that touch-initiated fullscreen play and for the immediate return-to-quad interval.
- Normal rotation resumes afterward.

---

## Canonical screen IDs

The authoritative list is `RAW_SCREEN_IDS` in `screens_catalog.py`. Legacy IDs are canonicalized automatically, including `time` → `nixie`, `sensors` → `inside`, and old `* v2` scoreboard aliases → current scoreboard IDs.

### Core, weather, sensor, and finance

- `date`
- `nixie`
- `quad`
- `weather logo`
- `weather1`
- `weather2`
- `weather alert`
- `weather hourly`
- `weather daily`
- `astronomical`
- `weather quad`
- `weather radar`
- `inside`
- `verano logo`
- `vrnof`

### NFL and Bears

- `bears logo`
- `bears stand1`
- `bears stand2`
- `bears next`
- `bears next season`
- `bears next season sched`
- `nfl logo`
- `NFL Scoreboard`
- `NFL Overview NFC`
- `NFL Overview AFC`
- `NFL Standings NFC`
- `NFL Standings AFC`

### NBA, Bulls, and NCAAM

- `nba logo`
- `NBA Scoreboard`
- `NBA Playoffs`
- `NCAAM Scoreboard`
- `bulls logo`
- `bulls stand1`
- `bulls last`
- `bulls live`
- `bulls next`
- `bulls next home`
- `bulls schedule quad`

### NHL, Blackhawks, Wolves, and playoffs

- `hawks logo`
- `hawks stand1`
- `hawks last`
- `hawks live`
- `hawks next`
- `hawks next home`
- `hawks schedule quad`
- `nhl logo`
- `NHL Scoreboard`
- `NHL Playoffs`
- `NHL Standings Overview West`
- `NHL Standings Overview East`
- `NHL Standings West`
- `NHL Standings West v2`
- `NHL Standings East`
- `NHL Standings East v2`
- `wolves logo`
- `wolves last`
- `wolves next`
- `wolves next home`

### MLB, Cubs, White Sox, and league

- `cubs logo`
- `cubs stand1`
- `cubs stand2`
- `cubs stand3`
- `cubs last`
- `cubs result`
- `cubs live`
- `cubs no game`
- `cubs next`
- `cubs next home`
- `cubs current series`
- `cubs next series`
- `cubs next home series`
- `cubs schedule quad`
- `sox logo`
- `sox stand1`
- `sox stand2`
- `sox stand3`
- `sox last`
- `sox live`
- `sox no game`
- `sox next`
- `sox next home`
- `sox current series`
- `sox next series`
- `sox next home series`
- `sox schedule quad`
- `mlb logo`
- `MLB Scoreboard`
- `NL Overview`
- `AL Overview`
- `MLB AL Standings`
- `MLB NL Standings`

---

## Configuration UI

Start the UI locally:

```bash
python config_ui.py
```

Default URL:

```text
http://localhost:5002
```

The UI supports:

- enabling and disabling screens,
- editing screen frequencies,
- editing per-screen `extra_seconds`,
- managing playlists and sequence order,
- importing/exporting screen rotation payloads,
- optional login protection with `SCREEN_UI_PASSWORD` and `SCREEN_UI_USERNAME`,
- screenshot browsing via the included screenshots template where capture is enabled.

Install only the config UI service for an existing deployment with:

```bash
bash ./Installers/install_config_ui_service.sh
```

---

## Services and operations

Common system service commands:

```bash
sudo systemctl status desk_display.service
sudo systemctl status config_ui_desk_display.service
sudo systemctl restart desk_display.service
sudo systemctl restart config_ui_desk_display.service
./scripts/restart_services.sh
```

Kernel user service commands, when installed:

```bash
systemctl --user status desk_display-kernel.service
systemctl --user restart desk_display-kernel.service
```

Useful operations helpers:

```bash
./scripts/show_screen_rotation_config.sh
./scripts/check_hyperpixel_setup.sh
./scripts/check_waveshare_setup.sh
./scripts/restore_desktop.sh
./scripts/uninstall.sh
./scripts/uninstall_airplay.sh
```

---

## Developer workflow

### Common checks

```bash
# Full test suite
pytest

# Focused registry/config checks
pytest -q tests/test_screens_catalog.py tests/test_screen_registry.py tests/test_config_flags.py

# Validate required files
python tools/validate_required_files.py

# Validate external API connectivity and configured credentials
python scripts/test_api_connections.py
python scripts/test_api_connections.py --json

# Render all screens for visual/regression review
python tools/maintenance/render_all_screens.py

# Render selected screens with the older maintenance helper
python tools/maintenance/render_screens.py
```

### Import/export helpers

```bash
python tools/export_screen_rotation_config.py
python tools/import_screen_rotation_config.py path/to/export.json
```

### Style/lint context

`pyproject.toml` configures Ruff for Python 3.11 with `E`, `F`, and `I` rules and a line length of 100.

---

## Troubleshooting

| Symptom | Things to check |
| --- | --- |
| Weather screens are empty | Verify `WEATHERKIT_*` signing values or `OWM_API_KEY`; run `python scripts/test_api_connections.py`. |
| Radar/map is blank | Verify network access, RainViewer reachability, and `GOOGLE_MAPS_API_KEY` if Google Static Maps is expected. |
| Travel route is blank | Verify `TRAVEL_MODE`, route origin/destination variables, and Google/Apple Maps credentials. |
| Indoor sensor is blank | Verify I2C is enabled, sensor wiring, `INSIDE_SENSOR`, `INSIDE_I2C_BUSES`, optional Pimoroni requirements (`pip install -r requirements_sensors_pimoroni.txt`) when using Pimoroni drivers, and run `i2cdetect`. |
| Wrong rotation/orientation | Avoid double rotation between kernel overlays and `DISPLAY_ROTATION`; check `HYPERPIXEL_PANEL` and display dimensions. |
| Blank framebuffer/kernel output | Verify `DESK_DISPLAY_OUTPUT`, `DISPLAY_FB_DEVICE`, display dimensions, pixel format/order, and device permissions. |
| Blinking cursor on framebuffer output | Keep `DISPLAY_FB_HIDE_CONSOLE_CURSOR=1` and `DISPLAY_FB_CONSOLE_GRAPHICS=1` so Linux fbcon does not redraw a cursor over direct framebuffer animation. |
| macOS/window mode uses too much CPU | Use `./scripts/launch_macos_window_perf.sh` or set `DESK_DISPLAY_WINDOW_SCALE=1`, `ENABLE_SCREENSHOTS=0`, `ENABLE_VIDEO=0`, `ENABLE_WIFI_MONITOR=0`, and `ENABLE_WIFI_RECOVERY=0`. |
| Waveshare OLED/LCD HAT issues | Run `scripts/check_waveshare_setup.sh`, verify I2C addresses, framebuffer config, and `WAVESHARE_OLED_LCD_HAT_A_INSTALLED`. |
| API/feed problem | Run `python scripts/test_api_connections.py`; use `--json` for machine-readable details. |

---

## External APIs

See [README_APIS.md](README_APIS.md) for a detailed catalog of third-party endpoints, credentials, payload fields, and diagnostics.
