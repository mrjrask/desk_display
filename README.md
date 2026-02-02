# Desk Display (Display HAT Mini + Kernel Displays)

Desk Display is a Raspberry Pi dashboard for the Pimoroni Display HAT Mini (320×240) or any kernel-driven display. It cycles through weather, commute, sensor, and sports screens with smooth animations, configurable scheduling, and built-in screenshot archiving.

## Highlights

- **Always-on dashboards** for date/time, weather (current/hourly/daily/radar), travel time/map, indoor sensors, and sports scoreboards/standings.
- **Screen scheduling** via `screens_config.json` (frequency + alternates) and a drag-and-drop web UI.
- **Display output choices**: Display HAT Mini, kernel-driven fullscreen displays, or headless rendering.
- **Smart capture pipeline**: per-screen screenshots, batch archiving, and optional H.264 video capture.
- **Wi-Fi monitoring** with automatic recovery and on-screen outage status.
- **GitHub update indicator** on date/time screens when upstream commits are available.

---

## Requirements

- Raspberry Pi (tested on Zero/Zero 2 W) or other Linux SBC
- Pimoroni Display HAT Mini **or** a kernel-driven display (HDMI, DSI, etc.)
- Python 3.9+

### System packages (Raspberry Pi OS)

Install the base dependencies before the Python packages:

```bash
sudo apt-get update
sudo apt-get install -y \
    python3-venv python3-pip python3-dev python3-opencv \
    build-essential libjpeg-dev libopenblas0 libopenblas-dev swig liblgpio-dev \
    libopenjp2-7-dev libtiff5-dev libcairo2-dev libpango1.0-dev \
    libgdk-pixbuf-2.0-dev libffi-dev network-manager wireless-tools \
    i2c-tools fonts-dejavu-core fonts-noto-color-emoji libgl1 libx264-dev ffmpeg git
```

> **Note:** Debian Trixie uses `libgdk-pixbuf-2.0-dev` instead of the legacy `libgdk-pixbuf2.0-dev` package name.

---

## Quick start

```bash
cd ~/desk_display
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Use `requirements_kernel.txt` instead when running on a kernel-driven display without the Display HAT Mini driver.

To start the display loop directly:

```bash
python main.py
```

---

## Installer scripts (recommended on Raspberry Pi OS)

### Display HAT Mini (SPI)

```bash
# Bookworm
bash ./scripts/install_display_hat_mini_bookworm.sh

# Trixie
bash ./scripts/install_display_hat_mini_trixie.sh
```

### Kernel-driven displays (fullscreen)

```bash
# Bookworm
bash ./scripts/install_kernel_bookworm.sh

# Trixie
bash ./scripts/install_kernel_trixie.sh
```

The kernel display installers will:

- Configure `DESK_DISPLAY_OUTPUT=kernel`.
- Detect a connected DRM display for fullscreen output (or prompt for a resolution override).
- Install a desktop launcher that can run the display loop inside the desktop session.
- Attempt to launch the fullscreen display at the end of the installer when a desktop session is available (set `AUTO_LAUNCH_KERNEL_DISPLAY=0` to skip).
- Install an autostart entry when `AUTO_START_KERNEL_DISPLAY=1` to launch the kernel display automatically on desktop login.

To uninstall the systemd service and optionally remove the virtualenv:

```bash
bash ./scripts/uninstall.sh
```

---

## Configuration

Desk Display reads configuration from environment variables and (optionally) a `.env` file. The main loop sets `CONFIG_LOAD_DOTENV=1` by default so `.env` is picked up automatically; set it yourself when running other tools directly (like `config_ui.py`).

### Display output

| Variable | Purpose |
| --- | --- |
| `DESK_DISPLAY_OUTPUT` | `auto` (default), `displayhatmini`, `kernel`, `framebuffer`, or `headless`. |
| `DESK_DISPLAY_FORCE_HEADLESS` | Force headless rendering (no display writes). |
| `DISPLAY_WIDTH` / `DISPLAY_HEIGHT` | Override target render size (required for larger fullscreen panels). |
| `DISPLAY_FB_DEVICE` | Framebuffer device path (default `/dev/fb0`). |
| `DISPLAY_FB_PIXEL_FORMAT` | Override framebuffer format (`rgb565`, `rgb888`, `xrgb8888`, etc.). |
| `DISPLAY_FB_PIXEL_ORDER` | Force pixel order (`rgb` or `bgr`). |

### Weather + radar

| Variable | Purpose |
| --- | --- |
| `WEATHERKIT_TEAM_ID`, `WEATHERKIT_KEY_ID`, `WEATHERKIT_SERVICE_ID` | WeatherKit credentials for JWT auth. |
| `WEATHERKIT_KEY_PATH` / `WEATHERKIT_PRIVATE_KEY` | Location or inline PEM for WeatherKit signing key. |
| `WEATHERKIT_LANGUAGE` / `WEATHERKIT_TIMEZONE` | Localization settings. |
| `OWM_API_KEY*` | OpenWeatherMap fallback key(s). Supports `OWM_API_KEY`, `OWM_API_KEY_DEFAULT`, `OWM_API_KEY_WIFFY`, `OWM_API_KEY_VERANO`. |
| `OWM_UNITS`, `OWM_LANGUAGE` | OpenWeatherMap units + locale. |
| `WEATHER_REFRESH_SECONDS` | Minimum refresh interval (clamped to ≥600s). |
| `WEATHER_USE_EMOJI_ICONS` | Use emoji icon set. |
| `GOOGLE_MAPS_API_KEY` | Google Static Maps key for the radar basemap and travel map screen. |

### Travel screens

| Variable | Purpose |
| --- | --- |
| `TRAVEL_TO_HOME_ORIGIN`, `TRAVEL_TO_HOME_DESTINATION` | Commute endpoints for the `to_home` profile. |
| `TRAVEL_TO_WORK_ORIGIN`, `TRAVEL_TO_WORK_DESTINATION` | Commute endpoints for the `to_work` profile. |
| `APPLE_MAPS_API_KEY` / `MAPKIT_TOKEN` | Apple Maps API key or MapKit token (used by the v2 travel screens). |
| `APPLE_MAPS_TEAM_ID`, `APPLE_MAPS_KEY_ID` | Apple Maps JWT credentials (can reuse WeatherKit IDs). |
| `APPLE_MAPS_KEY_PATH` / `APPLE_MAPS_PRIVATE_KEY` | PEM key for Apple Maps JWT signing. |
| `APPLE_MAPS_DIRECTIONS_URL` / `APPLE_MAPS_SNAPSHOT_URL` | Override Apple Maps service endpoints. |

The active travel profile is selected in `config.py` based on the connected SSID (defaults to `to_home`). Update `TRAVEL_MODE` in `config.py` if you want a different default profile or active window.

### Wi-Fi monitoring / recovery

| Variable | Purpose |
| --- | --- |
| `ENABLE_WIFI_MONITOR` | Enable the Wi-Fi background monitor. |
| `ENABLE_WIFI_RECOVERY` | Allow automatic recovery attempts. |
| `WIFI_INTERFACE` | Force a specific wireless interface. |
| `WIFI_TCP_PROBE_URLS` / `WIFI_TCP_PROBE_HOSTS` / `WIFI_TCP_PROBE_PORT` | Customize connectivity probes. |
| `WIFI_RECOVERY_LOG` | Override the per-user recovery log path. |

### Screenshots + video capture

| Variable | Purpose |
| --- | --- |
| `ENABLE_SCREENSHOTS` | Enable per-screen capture (default on). |
| `ENABLE_VIDEO` | Record a rolling H.264 MP4 in the screenshots folder. |
| `SCREENSHOT_DIR` | Override the screenshots root folder. |
| `SCREENSHOT_ARCHIVE_BASE` | Override the archive base folder. |

Screenshots are saved under `screenshots/<Screen Name>/` with the latest frame mirrored in `screenshots/current/`. Once 500+ screenshots are present, the current batch is archived into `screenshot_archive/<Screen Name>/` and trimmed to the most recent 50 per screen.

---

## Screen scheduling

The screen scheduler lives in `screens_config.json`. Each entry is either a frequency number or an object with a `frequency` and optional `alt` schedule.

```json
{
  "screens": {
    "date": 1,
    "weather1": 1,
    "travel": { "frequency": 2, "alt": { "screen": "travel map", "frequency": 2 } }
  }
}
```

- `frequency` is the number of times a screen appears in each rotation.
- `frequency: 0` disables a screen without removing it from the list.
- `alt` can target a single screen or a list of screens to rotate through.

---

## Screen configuration UI

Run the web UI to reorder screens, adjust frequencies, and preview screenshots:

```bash
python config_ui.py
```

Defaults:

- URL: `http://localhost:5002`
- Host: `SCREEN_CONFIG_HOST` (default `0.0.0.0`)
- Port: `SCREEN_CONFIG_PORT` (default `5002`)
- Config path: `SCREENS_CONFIG_PATH`
- Disable autostart: `SCREEN_CONFIG_AUTOSTART=0`

---

## Fonts + assets

Drop custom fonts in `fonts/` (e.g., `TimesSquare-m105.ttf`, `DejaVuSans.ttf`, `DejaVuSans-Bold.ttf`). Team logos and other assets live in `images/`.

---

## Systemd service

The installer scripts create a `desk_display.service` that runs `main.py` in the project virtual environment. Use standard systemd commands to manage it:

```bash
sudo systemctl status desk_display.service
sudo systemctl restart desk_display.service
```

---

## Troubleshooting

- **Weather data missing**: verify WeatherKit credentials or set an OpenWeatherMap API key.
- **Radar background missing**: ensure `GOOGLE_MAPS_API_KEY` is set.
- **Travel screens show N/A**: confirm `TRAVEL_*` addresses and the relevant Maps API key.
- **Kernel displays not showing**: set `DESK_DISPLAY_OUTPUT=kernel` and provide `DISPLAY_WIDTH`/`DISPLAY_HEIGHT`.
- **Kernel displays still blank in desktop mode**: launch with `scripts/launch_kernel_display.sh` from a logged-in desktop session; ensure `WAYLAND_DISPLAY` or `DISPLAY` is set (and `XDG_RUNTIME_DIR`/`XAUTHORITY` are available for SDL).
- **Wi-Fi recovery loops**: check `/var/log/wifi_auto_recover.log` and disable recovery with `ENABLE_WIFI_RECOVERY=0`.

---

## API references

See [`README_APIS.md`](README_APIS.md) for the full list of third-party endpoints and fields used.
