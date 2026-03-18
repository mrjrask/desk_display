# Desk Display (Display HAT Mini + Kernel Displays)

Desk Display is a Raspberry Pi dashboard for the Pimoroni Display HAT Mini (320×240) or any kernel-driven display. It cycles through weather, commute, sensor, and sports screens with smooth animations, configurable scheduling, and built-in screenshot archiving.

## Highlights

- **Always-on dashboards** for date/time, weather (current/hourly/daily/radar), indoor sensors, and sports scoreboards/standings.
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
    libgdk-pixbuf-2.0-dev libffi-dev network-manager wireless-tools iproute2 \
    i2c-tools fonts-dejavu-core fonts-noto-color-emoji libgl1 libx264-dev ffmpeg git \
    avahi-daemon avahi-utils uxplay
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

For existing installs, refresh Python dependencies with:

```bash
./scripts/update_dependencies.sh
```

Pass `--requirements requirements_kernel.txt` (or `requirements_framebuffer.txt`) when needed.

To upgrade every already-installed pip package in the project virtualenv:

```bash
./tools/maintenance/update_pip_installed_packages.sh
```

Use `--dry-run` to preview which packages would be upgraded.

If upgrades fail with permission errors, fix ownership of the virtualenv first (for example: `sudo chown -R "$USER":"$USER" ./venv`).

To start the display loop directly:

```bash
python main.py
```

---

## Installer scripts (recommended on Raspberry Pi OS)

All installers below are combined for Raspberry Pi OS Bookworm and Trixie.

Supported/tested codenames: `bookworm`, `trixie`. On other Debian/Raspberry Pi OS codenames, the installer falls back to best-effort dependency package mapping and may require manual package-name adjustments if apt cannot resolve a dependency.

### Display HAT Mini (SPI)

```bash
bash ./Installers/install_display_hat_mini.sh
```

### Waveshare OLED/LCD HAT (A) (LCD panel)

```bash
bash ./Installers/install_waveshare_oled_lcd_hat_a.sh
```

This installer targets the 320×240 LCD panel using framebuffer output.

It also enables a small helper service for the two OLED side displays:
- Temperature and time are auto-fitted to the largest possible font size under each title.
- The two OLEDs cross-fade and swap temperature/time positions each refresh to reduce burn-in.

Button mapping for Waveshare OLED/LCD HAT (A):
- `A` → `K4 (D24)`
- `B` → `K1 (D4)`
- `X` → `K2 (D17)`
- `Y` → `K3 (D23)`

### Kernel-driven displays (fullscreen)

```bash
# HyperPixel 4 / HyperPixel 4 Square
bash ./Installers/install_hyperpixel.sh

# HyperPixel installer (non-interactive)
HYPERPIXEL_PANEL=hyperpixel4 DISPLAY_WIDTH=800 DISPLAY_HEIGHT=480 bash ./Installers/install_hyperpixel.sh

# Other kernel-driven displays
bash ./Installers/install_kernel.sh
```

The kernel display installers will:

- Configure `DESK_DISPLAY_OUTPUT=kernel`.
- Configure `.env` display sizing for your detected HyperPixel panel and default `DISPLAY_ROTATION=0` (set `DISPLAY_ROTATION=180` for upside-down mounting).
- **Not** modify `/boot/firmware/config.txt` or `/boot/config.txt`; set your HyperPixel `dtoverlay=` line manually (including any `rotate=` value).
- The HyperPixel installer disables the Pi's primary SPI/I2C interfaces and sets `INSIDE_I2C_BUSES=13` so the Inside screen probes the HyperPixel/HyperPixel 4 Square accessory I2C header bus first.
- Prompt for the target fullscreen resolution at install time (defaulting to the detected DRM/X11 mode when available).
- Install a desktop launcher that can run the display loop inside the desktop session.
- Attempt to launch the fullscreen display at the end of the installer when a desktop session is available (set `AUTO_LAUNCH_KERNEL_DISPLAY=0` to skip).
- Install an autostart entry when `AUTO_START_KERNEL_DISPLAY=1` to launch the kernel display automatically on desktop login.
- Provide an SSH-friendly helper (`scripts/ssh_kernel_display.sh`) to manage the user service without manual environment setup.
- On Lite/headless installs with no active desktop session, automatically fall back to `DESK_DISPLAY_OUTPUT=framebuffer` (set `AUTO_FALLBACK_FRAMEBUFFER=0` to keep kernel output).


### Password-protected AirPlay takeover (always ready)

Desk Display can run an always-on background AirPlay receiver using `uxplay`.

When an AirPlay client connects, `desk_display.service` is stopped so AirPlay takes over the display. When the AirPlay client disconnects, the dashboard service is restarted automatically and normal screen playback resumes.

1. Set one of these in `.env` (required for protection):

```bash
DESK_DISPLAY_AIRPLAY_PASSWORD=your-password
# or
DESK_DISPLAY_AIRPLAY_PIN=1234
```

2. Install/update dependencies and background service (existing installs):

```bash
./scripts/update_airplay_dependencies.sh
```

The installer enables the AirPlay takeover service by default. To disable it, set:

```bash
DESK_DISPLAY_AIRPLAY_ALWAYS_ON=0
```

Optional AirPlay settings:

- `DESK_DISPLAY_AIRPLAY_NAME` (default: `Desk Display`)
- `DESK_DISPLAY_AIRPLAY_ARGS` (extra raw `uxplay` arguments)
- `DESK_DISPLAY_AIRPLAY_FULLSCREEN` (default: `1`; pass `-fs` so AirPlay runs fullscreen)
- `DESK_DISPLAY_AIRPLAY_NATIVE_RESOLUTION` (default: `1`; auto-detect native mode and pass it via `uxplay -s`)
- `DESK_DISPLAY_AIRPLAY_IDLE_RESUME_SECONDS` (default: `8`)
- `DESK_DISPLAY_AIRPLAY_POLL_SECONDS` (default: `1`)

Manual on-demand mode is still available:

```bash
./scripts/airplay_mode.sh
```

To uninstall the systemd service and optionally remove the virtualenv:

```bash
bash ./scripts/uninstall.sh
```

---

## Configuration

Desk Display reads configuration from environment variables and (optionally) a `.env` file. The main loop sets `CONFIG_LOAD_DOTENV=1` by default so `.env` is picked up automatically; set it yourself when running other tools directly (like `config_ui.py`).

The config UI is a Flask app served by Waitress when you run `python config_ui.py`.

### Display output

| Variable | Purpose |
| --- | --- |
| `DESK_DISPLAY_OUTPUT` | `auto` (default), `displayhatmini`, `kernel`, `framebuffer`, or `headless`. |
| `DESK_DISPLAY_FORCE_HEADLESS` | Force headless rendering (no display writes). |
| `DESK_DISPLAY_SESSION_USER` | Override the desktop session user for `loginctl` lookups and `/run/user/<uid>` resolution. |
| `DISPLAY_WIDTH` / `DISPLAY_HEIGHT` | Override target render size (required for larger fullscreen panels). |
| `DISPLAY_FB_DEVICE` | Framebuffer device path (default `/dev/fb0`). |
| `DISPLAY_FB_PIXEL_FORMAT` | Override framebuffer format (`rgb565`, `rgb888`, `xrgb8888`, etc.). |
| `DISPLAY_FB_PIXEL_ORDER` | Force pixel order (`rgb` or `bgr`). |
| `DISPLAY_ROTATION` | Additional app-side rotation in degrees (`0`, `90`, `180`, `270`) or shorthand (`0-3`). When kernel `dtoverlay=...,rotate=` is active in strict mode, non-zero app rotation is reset to `0` to avoid double-rotation. |
| `DISPLAY_ROTATION_STRICT` | Rotation guardrail toggle. Defaults to `1` for HyperPixel/kernel outputs (`DESK_DISPLAY_OUTPUT=kernel|kms|drm|sdl` or `HYPERPIXEL_PANEL=hyperpixel*`) and `0` otherwise. Set `0` to preserve legacy stacked kernel+app rotation behavior. |
| `DISPLAY_HAT_MINI_REINIT_SECONDS` | Recreate the Display HAT Mini driver on this interval (default `1800`) to recover from long-run panel stalls. Set `0` to disable. |
| `DISPLAY_HAT_MINI_LED_LEVEL` | Normalized indicator LED brightness (`0.0`-`1.0`) for weather alerts, schedule win/loss markers, and update animations. Defaults to `0.08`; recommended starting range is `0.05`-`0.20`. |
| `WAVESHARE_OLED_I2C_BUS` | I2C bus for Waveshare OLED helper service (default `1`). |
| `WAVESHARE_OLED_TEMP_ADDR` | I2C address for the temperature OLED (default `0x3C`). |
| `WAVESHARE_OLED_TIME_ADDR` | I2C address for the time OLED (default `0x3D`). |
| `WAVESHARE_OLED_WIDTH` / `WAVESHARE_OLED_HEIGHT` | OLED dimensions for the helper renderer (defaults `128` / `64`). |
| `WAVESHARE_OLED_TEMP_SOURCE` | Temperature source: `weather1` (default), `weather`, `cpu`, or `command`. |
| `WAVESHARE_OLED_TEMP_COMMAND` | Shell command used when `WAVESHARE_OLED_TEMP_SOURCE=command`; first numeric value is shown. |
| `WAVESHARE_OLED_TEMP_UNIT` | Temperature unit for display (`C` default, or `F`). |
| `WAVESHARE_OLED_REFRESH_SECONDS` | Refresh interval between OLED fade/swap cycles (default `5`). |
| `WAVESHARE_OLED_FADE_STEPS` | Number of fade steps for OLED transitions (default `8`). |
| `WAVESHARE_OLED_FADE_STEP_MS` | Delay per fade step in milliseconds (default `35`). |
| `WAVESHARE_OLED_FONT_PATH` | Optional TrueType font path for OLED value text auto-sizing. |
| `BUTTON_A` / `BUTTON_B` / `BUTTON_X` / `BUTTON_Y` | GPIO BCM pins for control buttons. Waveshare defaults are `24`, `4`, `17`, and `23`. |

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

### Travel screens

| Variable | Purpose |
| --- | --- |
| `TRAVEL_TO_HOME_ORIGIN`, `TRAVEL_TO_HOME_DESTINATION` | Commute endpoints for the `to_home` profile. |
| `TRAVEL_TO_WORK_ORIGIN`, `TRAVEL_TO_WORK_DESTINATION` | Commute endpoints for the `to_work` profile. |
| `APPLE_MAPS_TEAM_ID`, `APPLE_MAPS_KEY_ID` | Apple Maps JWT credentials (can reuse WeatherKit IDs). |
| `APPLE_MAPS_KEY_PATH` / `APPLE_MAPS_PRIVATE_KEY` | PEM key for Apple Maps JWT signing. |
| `APPLE_MAPS_DIRECTIONS_URL` / `APPLE_MAPS_SNAPSHOT_URL` | Override Apple Maps service endpoints. |



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
  }
}
```

- `frequency` is an interval: `1` shows every pass, `2` shows every other pass, `8` shows once every eight passes.
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
- Set `SCREEN_UI_PASSWORD` to require login for UI pages and API endpoints

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

### Kernel display via user service

For kernel-driven displays inside a desktop session, the kernel installers also install a user service named `desk_display-kernel.service`. Use `systemctl --user` from the logged-in session (or over SSH with the user session available):

```bash
systemctl --user start desk_display-kernel.service
systemctl --user status desk_display-kernel.service
```

To manage the kernel display service over SSH without manually exporting user session environment variables, use:

```bash
~/desk_display/scripts/ssh_kernel_display.sh status
~/desk_display/scripts/ssh_kernel_display.sh restart
~/desk_display/scripts/ssh_kernel_display.sh stop
```

---

## Troubleshooting

- **Weather data missing**: verify WeatherKit credentials or set an OpenWeatherMap API key.
- **Radar background missing**: ensure `GOOGLE_MAPS_API_KEY` is set.
- **Travel screens show N/A**: confirm `TRAVEL_*` addresses and the relevant Maps API key.
- **Kernel displays not showing**: set `DESK_DISPLAY_OUTPUT=kernel` and provide `DISPLAY_WIDTH`/`DISPLAY_HEIGHT`.
- **Kernel displays still blank in desktop mode**: launch with `scripts/launch_kernel_display.sh` from a logged-in desktop session; ensure `WAYLAND_DISPLAY` or `DISPLAY` is set (and `XDG_RUNTIME_DIR`/`XAUTHORITY` are available for SDL).
- **HyperPixel shows boot logs but Desk Display/desktop stays black**: run `scripts/check_hyperpixel_setup.sh` and verify `dtoverlay=` is present, `/dev/dri/card0` exists, and your output mode matches a real desktop session (`desk_display-kernel.service`) or framebuffer fallback (`DESK_DISPLAY_OUTPUT=framebuffer`).
- **Display appears over-rotated on HyperPixel/kernel output**: if `dtoverlay=...,rotate=` is configured, strict mode now forces non-zero `DISPLAY_ROTATION` back to `0`. Keep a single rotation source, or set `DISPLAY_ROTATION_STRICT=0` temporarily for backward-compatible stacked rotation.
- **Wi-Fi recovery loops**: check `/var/log/wifi_auto_recover.log` and disable recovery with `ENABLE_WIFI_RECOVERY=0`.

---


## API references

See [`README_APIS.md`](README_APIS.md) for the full list of third-party endpoints and fields used.
