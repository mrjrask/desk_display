# Desk Scoreboard & Info Display (Pimoroni Display HAT Mini)

A tiny, always‑on scoreboard and info display that runs on a Raspberry Pi and a Pimoroni Display HAT Mini (320×240 ST7789 LCD). It cycles through date/time, weather, travel time, indoor sensors, stocks, Blackhawks, Bulls & Bears screens, MLB standings, and Cubs/White Sox game views (last/live/next).

> **Highlights**
> - Smooth animations: scroll and fade‑in
> - Rich MLB views: last/live/next game, standings (divisions, overview, wild card)
> - **Cubs W/L result** full‑screen flag (animated WebP supported; PNG fallback)
> - **Chicago Wolves** screens that mirror the Hawks layouts and reuse cached HockeyTech data for fast redraws
> - **Smart screenshots** auto‑archived in batches when the live folder reaches 500 images
> - **GitHub update dot** on date/time screens when new commits are available
> - Screen sequencing via `screens_config.json`

---

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [Project layout](#project-layout)
- [Configuration](#configuration)
- [Images & Fonts](#images--fonts)
- [Screens](#screens)
- [Running](#running)
- [Systemd unit](#systemd-unit)
- [Screenshots & archiving](#screenshots--archiving)
- [GitHub update indicator](#github-update-indicator)
- [Troubleshooting](#troubleshooting)

---

## Requirements

- Raspberry Pi (tested on Pi Zero/Zero 2 W)
- Pimoroni **Display HAT Mini (320×240 ST7789 LCD)** wired to SPI0
- Python 3.9+
- Packages (install via apt / pip):
  ```bash
  sudo apt-get update
  sudo apt-get install -y \
      python3-venv python3-pip python3-dev python3-opencv \
      build-essential libjpeg-dev libopenblas0 libopenblas-dev swig liblgpio-dev \
      libopenjp2-7-dev libtiff5-dev libcairo2-dev libpango1.0-dev \
      libgdk-pixbuf-2.0-dev libffi-dev network-manager wireless-tools \
      i2c-tools fonts-dejavu-core fonts-noto-color-emoji libgl1 libx264-dev ffmpeg git
  ```

  > **Note:** Debian Trixie removed the legacy Xlib-flavoured package that used the
  > older `libgdk-pixbuf2.0-dev` name. Users on Bookworm or earlier may still see the
  > transitional package in documentation or when resolving dependencies.

  > **SWIG required:** The `lgpio` dependency is built from source on Raspberry Pi OS
  > images. Ensure `swig` is installed (included in the apt list above) to avoid
  > build failures when installing `requirements.txt`.

  > **Install liblgpio:** The `lgpio` wheel links against the `llgpio` system
  > library. Install `liblgpio-dev` (included in the apt list above) so pip can
  > link the extension successfully.

  Create and activate a virtual environment before installing the Python dependencies:

  ```bash
  python -m venv venv && source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
  Pillow on current Raspberry Pi OS builds usually includes **WebP** support. If animated WebP is not rendering, upgrade Pillow:
  ```bash
  pip install --upgrade pillow
  ```
  The `bme68x` package is required when using the bundled BME688 air quality sensor helper.
  Install `adafruit-circuitpython-sht4x` when wiring an Adafruit SHT41 (STEMMA QT).
  Install `pimoroni-bme280`, `pimoroni-ltr559`, and `pimoroni-lsm6ds3` when using the Pimoroni Multi-Sensor Stick (PIM745). These
  drivers power the BME280, LTR559, and LSM6DS3 integrations rendered by `screens/draw_inside.py` and `screens/draw_sensors.py`.

---

## Install

If you've already cloned this repository (for example into `~/desk_display`), switch into that directory to install dependencies and configure the app.

```bash
cd ~/desk_display
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The `venv` directory is ignored by Git. Re-run `source venv/bin/activate` whenever you start a new shell session to ensure the project uses the isolated Python environment.

### Automated installers (Raspberry Pi OS Bookworm or Trixie)

Two turnkey installers are provided for Raspberry Pi OS. Run the script that matches your OS release **from the project root**. Each script will enable SPI/I2C (when `raspi-config` is present), install the apt and pip dependencies, create a virtual environment in the project folder, and install+start the `desk_display.service` systemd unit that runs `main.py` under the current user.

```bash
# Bookworm (keeps the transitional libgdk-pixbuf2.0-dev package name)
bash ./scripts/install_bookworm.sh

# Trixie (uses libgdk-pixbuf-2.0-dev)
bash ./scripts/install_trixie.sh
```

Override `PROJECT_DIR` when you want the installer to target a different checkout:

```bash
PROJECT_DIR=/home/pi/desk_display bash ./scripts/install_bookworm.sh
```

To uninstall the service and remove the virtual environment, run:

```bash
bash ./scripts/uninstall.sh
```

Set `KEEP_VENV=1` if you want to keep the virtual environment directory in place.

### Screen configuration UI

The screen configuration UI launches automatically with the main service. To edit the screen order, frequencies, and alternates via a drag-and-drop UI, you can also run the configuration app directly:

```bash
python config_ui.py
```

By default it listens on `http://localhost:5002`. Override the binding with:

- `SCREEN_CONFIG_HOST` (default `0.0.0.0`)
- `SCREEN_CONFIG_PORT` (default `5002`)
- `SCREENS_CONFIG_PATH` (path to the `screens_config.json` file)
- `SCREEN_CONFIG_AUTOSTART` (set to `0`, `false`, or `no` to disable autostart)

---

## Project layout

```
desk_display/
├─ main.py / config_ui.py / utils.py / config.py
├─ config_store.py / data_fetch.py / paths.py / schedule.py / schedule_migrations.py / screens_catalog.py
├─ screens_config.json / screens_style.json
├─ services/
│  ├─ __init__.py
│  ├─ http_client.py              # shared requests.Session + NHL headers
│  ├─ network.py                  # background Wi-Fi / internet monitor
│  └─ wifi_utils.py               # Wi-Fi triage exposed to the main loop
├─ screens/
│  ├─ __init__.py
│  ├─ color_palettes.py
│  ├─ draw_bears_schedule.py
│  ├─ draw_bulls_schedule.py
│  ├─ draw_date_time.py
│  ├─ draw_hawks_schedule.py
│  ├─ draw_inside.py
│  ├─ draw_travel_map.py
│  ├─ draw_travel_time.py
│  ├─ draw_vrnof.py
│  ├─ draw_weather.py
│  ├─ mlb_schedule.py
│  ├─ mlb_scoreboard.py
│  ├─ mlb_standings.py
│  ├─ mlb_team_standings.py
│  ├─ nba_scoreboard.py
│  ├─ nhl_scoreboard.py
│  ├─ nhl_standings.py
│  └─ nfl_scoreboard.py / nfl_standings.py
├─ scripts/
│  ├─ check_api_statuses.py
│  ├─ install_common.sh
│  └─ install_bookworm.sh / install_trixie.sh
├─ tools/
│  ├─ test_screens.py             # interactive screen renderer harness
│  └─ maintenance/
│     ├─ cleanup.sh               # clears display + archives leftover assets on shutdown
│     ├─ render_all_screens.py    # renders every screen to PNG and zips them
│     └─ reset_screenshots.sh     # purges screenshots/ and screenshot_archive/
├─ templates/                     # Flask configuration templates
├─ tests/
├─ images/
│  ├─ mlb/<ABBR>.png              # MLB team logos (e.g., CHC.png)
│  ├─ nfl/<ABBR>.png              # NFL logos used by Bears screen
│  ├─ W_flag.webp / L_flag.webp   # animated WebP flags (preferred)
│  ├─ mlb/W.png / mlb/L.png       # fallback PNG flags
│  ├─ mlb/CHC.png, mlb/CWS.png, nhl/CHI.png, mlb/MLB.png, weather.jpg, verano.jpg, nfl/chi.png
└─ fonts/
   ├─ TimesSquare-m105.ttf
   ├─ DejaVuSans.ttf
   └─ DejaVuSans-Bold.ttf
```

Run `python tools/validate_required_files.py` to confirm that every tracked
non-image Python module stays reachable from the main entry points and helper
scripts.

### Render all screens

Use `tools/maintenance/render_all_screens.py` to render every screen to PNG
and optionally archive the results:

```bash
python tools/maintenance/render_all_screens.py [flags]
```

Available flags:

- `-a`, `--all`: ignore `screens_config.json` and render every available screen.
- `--sync-screenshots`: write PNG files for each rendered screen to the
  screenshots directory (default mirrors `ENABLE_SCREENSHOTS` from `config.py`).
- `--no-sync-screenshots`: disable screenshot syncing even if
  `ENABLE_SCREENSHOTS` is true.
- `--no-archive`: skip creating the ZIP archive of rendered screens.
- `--no-images`: render with placeholder frames instead of loading images.

---

## Configuration

Most runtime behavior is controlled in `config.py`:

- **Display:** `WIDTH=320`, `HEIGHT=240`
- **Intervals:** `SCREEN_DELAY`, `TEAM_STANDINGS_DISPLAY_SECONDS`, `SCHEDULE_UPDATE_INTERVAL`
- **Feature flags:** `ENABLE_SCREENSHOTS`, `ENABLE_VIDEO`, `ENABLE_WIFI_MONITOR`
- **Weather:** `ENABLE_WEATHER`, `LATITUDE/LONGITUDE`
- **Travel:** `TRAVEL_MODE` (`to_home` or `to_work`)
- **MLB:** constants and timezone `CENTRAL_TIME`
- **AHL:** `AHL_TEAM_ID`, `AHL_TEAM_TRICODE`, HockeyTech feed overrides (API base/key/site), and `AHL_SCHEDULE_ICS_URL`. Wolves last-game cards keep using HockeyTech scores while the next/next-home cards now read from the published Stanza ICS schedule URL (defaults to the Chicago Wolves feed).
- **Fonts:** make sure `fonts/` contains the TTFs above
- **Wi-Fi monitor:** ICMP pings stay enabled by default. You can add a TCP/HTTPS reachability probe by setting `WIFI_TCP_PROBE_URLS` (comma-separated URLs/hosts), `WIFI_TCP_PROBE_HOSTS`, or `RPI_CONNECT_CONTROL_HOST`, and optionally override `WIFI_TCP_PROBE_PORT` (defaults to `443`). A successful TCP connection is treated as “internet is OK,” which avoids ICMP-only failures on networks that block ping (e.g., rpi-connect).

Set `ENABLE_WIFI_RECOVERY=false` when you want the Wi-Fi monitor to run in **monitor-only** mode (no interface resets). This keeps connectivity status flowing to the UI/logs without toggling the interface, which is useful when rpi-connect or other tooling manages the connection lifecycle.

### Screen sequencing

The scheduler now uses a **playlist-centric schema (v2)** that supports reusable playlists, nested playlists, rule descriptors, and optional conditions. A minimal configuration looks like this:

```json
{
  "version": 2,
  "catalog": {"presets": {}},
  "playlists": {
    "weather": {
      "label": "Weather",
      "steps": [
        {"screen": "date"},
        {"screen": "weather1"},
        {"rule": {"type": "variants", "options": ["travel", "inside"]}}
      ]
    },
    "main": {
      "label": "Primary loop",
      "steps": [
        {"playlist": "weather"},
        {"rule": {"type": "every", "frequency": 3, "item": {"screen": "inside"}}},
        {"rule": {"type": "cycle", "items": [{"screen": "time"}, {"screen": "date"}]}}
      ]
    }
  },
  "sequence": [
    {"playlist": "main"}
  ]
}
```

Key points:

- **`catalog`** holds reusable building blocks (for example, preset playlists).
- **`playlists`** is a dictionary of playlist IDs → definitions. Each playlist contains an ordered `steps` list. Steps may be screen descriptors, nested playlist references, or rule descriptors (`variants`, `cycle`, `every`).
- **`sequence`** is the top-level playlist order for the display loop. Entries can reference playlists or inline descriptors.
- Optional **conditions** may be attached to playlists or individual steps:

  ```json
  {
    "conditions": {
      "days_of_week": ["mon", "wed", "fri"],
      "time_of_day": [{"start": "08:00", "end": "12:00"}]
    },
    "playlist": "weather"
  }
  ```

  The scheduler automatically skips a step when its conditions are not met.

### Screen-specific style overrides

The default fonts, sizes, and logo heights baked into each renderer can be
overridden without touching Python code. Settings live in
`screens_style.json`, which is managed through the same ConfigStore wrapper as
`screens_config.json`. Each screen can opt into overrides via `fonts` and
`images` dictionaries:

```json
{
  "screens": {
    "nba_scoreboard": {
      "fonts": {
        "score": {"family": "fonts/TimesSquare-m105.ttf", "size": 32},
        "ticker": {"size": 20}
      },
      "images": {
        "team_logo": {"scale": 1.15}
      }
    }
  }
}
```

Available font slots correspond to the keys requested inside each renderer
(`"score"`, `"ticker"`, etc.). When a screen requests a slot, the helper in
`config.get_screen_font()` loads the requested family/size, falling back to the
existing font object if no override is present. Logo sizing works similarly via
`config.get_screen_image_scale()`—renderers read the scale during each draw so
edits to `screens_style.json` take effect immediately.

#### Migrating existing configs

Legacy `sequence` arrays are migrated to v2 automatically on startup. For manual conversions or batch jobs run:

```bash
python schedule_migrations.py migrate --input screens_config.json --output screens_config.v2.json
```

This writes a playlist-aware config and validates it using the scheduler parser. The original file is left untouched when `--output` is provided.

### Default playlist reference

The repository ships with a ready-to-run `screens_config.json` that exposes the **Default loop** playlist. The playlist executes the following steps in order (rules are evaluated on each pass through the loop):

1. `date`
2. `weather1`
3. Every third pass, show `weather2`.
4. Every third pass, show `inside` (indoor sensors).
5. `travel`
6. `travel map`
7. Every fourth pass, show `vrnof` (Verano VRNO stock panel).
8. Every other pass, cycle through the Blackhawks cards: `hawks logo`, `hawks last`, `hawks live`, `hawks next`, `hawks next home`.
9. Every fifth pass, show `NHL Scoreboard`.
10. Every sixth pass, cycle through `NHL Standings Overview`, `NHL Standings Overview`, `NHL Standings West`.
11. Every eighteenth pass (starting at phase 12), show `NHL Standings East`.
12. Every fourth pass, show `bears logo`.
13. Every fourth pass, show `bears next`.
14. Every fifth pass, show `NFL Scoreboard`.
15. Every sixth pass, cycle through `NFL Overview NFC`, `NFL Overview NFC`, `NFL Standings NFC`.
16. Every sixth pass, cycle through `NFL Overview AFC`, `NFL Overview AFC`, `NFL Standings AFC`.
17. Every seventh pass, show `NBA Scoreboard`.
18. Every third pass, show `MLB Scoreboard`.

Each step above maps directly to the JSON structure under `playlists.default.steps`.

---

### Secrets & environment variables

API keys are no longer stored directly in `config.py`. Set them as environment variables before running any of the
scripts:

- **Apple WeatherKit (required for weather):**
  - `WEATHERKIT_TEAM_ID` — your 10-character Apple Developer Team ID.
  - `WEATHERKIT_KEY_ID` — the WeatherKit key identifier from the Keys tab in Certificates, IDs & Profiles.
  - `WEATHERKIT_SERVICE_ID` — the Service ID (or bundle ID) you enabled for WeatherKit.
  - Provide the private key via **one** of:
    - `WEATHERKIT_KEY_PATH` pointing to the `.p8` private key you downloaded from Apple, or
    - `WEATHERKIT_PRIVATE_KEY` containing the full PEM contents.
    - If you use environment variables, ensure the PEM retains its real newlines. Literal `\n` sequences, pasted file paths,
      or Windows-style line endings can all prevent the token from being signed.
  - Optional: `WEATHERKIT_LANGUAGE` (default `en`) and `WEATHERKIT_TIMEZONE` (default `America/Chicago`).
  - Optional: `WEATHER_REFRESH_SECONDS` controls how long WeatherKit responses are cached before another API call is made. The
    default is **1800 seconds (30 minutes)** to stay well below the 500k calls/month allowance, even across multiple devices.
- `GOOGLE_MAPS_API_KEY` for travel-time requests (leave unset to disable that screen).
- `TRAVEL_TO_HOME_ORIGIN`, `TRAVEL_TO_HOME_DESTINATION`, `TRAVEL_TO_WORK_ORIGIN`,
  and `TRAVEL_TO_WORK_DESTINATION` to override the default travel addresses.
- `DARK_HOURS` (optional) to blank the display during quiet hours. For example, place the
  following line in your `.env` file to keep the screen dark overnight Monday–Thursday
  and all day Friday through Sunday:

  ```env
  DARK_HOURS="Mon-Thu 19:00-07:00; Fri-Sun 00:00-24:00"
  ```

You can export the variables in your shell session:

```bash
export WEATHERKIT_TEAM_ID="ABCDE12345"
export WEATHERKIT_KEY_ID="1A2BC3D4E5"
export WEATHERKIT_SERVICE_ID="com.example.weather"
export WEATHERKIT_KEY_PATH="/home/pi/desk_display/AuthKey_1A2BC3D4E5.p8"
export GOOGLE_MAPS_API_KEY="your-google-maps-key"
```

Or copy `.env.example` to `.env` and load it with your preferred process manager or a tool such as
[`python-dotenv`](https://github.com/theskumar/python-dotenv).

### Indoor sensor override

The `inside` screen auto-detects supported I²C sensors (Pimoroni BME280/BME680/BME68x/BME688, Adafruit BME280/BME680/SHT41). If your wiring tends to confuse detection—or you want to hide VOC when using a BME280—you can force a specific driver via `.env`:

```bash
# Choose one of: pimoroni_bme280, adafruit_bme280,
# pimoroni_bme680, pimoroni_bme68x, adafruit_bme680,
# adafruit_sht41
INSIDE_SENSOR=pimoroni_bme280
```

`INDOOR_SENSOR` is also accepted. Unrecognized values fall back to auto-detect.

---

## Images & Fonts

- **MLB logos:** put team PNGs into `images/mlb/` named with the team tricode (e.g., `CUBS.png`, `MIL.png`).
- **NFL logos:** for the Bears screen, `images/nfl/<abbr>.png` (e.g., `gb.png`, `min.png`).
- **Cubs W/L flag:** use `images/W_flag.webp` and `images/L_flag.webp` (animated). If missing, the code falls back to `images/mlb/W.png` / `images/mlb/L.png`.
- **Fonts:** copy `TimesSquare-m105.ttf`, `DejaVuSans.ttf`, and `DejaVuSans-Bold.ttf` into `fonts/`.
- **Travel font:** the Google Maps travel screen loads `HWYGNRRW.TTF` (Highway Gothic) directly from `fonts/`. Without this
  file the app will exit on startup, so copy your licensed copy into that folder alongside the other fonts.
- **Emoji font:** the app prefers the system `fonts-noto-color-emoji` package for emoji glyphs. If unavailable, install the Symbola font (`ttf-ancient-fonts` on Debian/Ubuntu) or place `Symbola.ttf` in your system font directory so precipitation/cloud icons render correctly.

---

## Screens

- **Date/Time:** legible, high‑contrast text with the GitHub update dot when upstream commits are available.
- **Weather (1/2):** Apple WeatherKit current conditions + daily/hourly forecasts with built-in caching to minimize API calls.
- **Inside:** BME688/BME280/LTR559/SHT41 summaries when sensors are wired.
- **VRNO:** stock mini‑panel.
- **Travel:** Maps ETA using your configured mode.
- **Bears:** opponent card, logo splash, standings, and NFL scoreboard tie‑ins.
- **Blackhawks:** last/live/next based on the NHL schedule feed, logos included.
- **Wolves:** last/live/next/next‑home cards that inherit the Hawks art direction and now use cached HockeyTech data so a single API pull feeds multiple screens without repeated network calls.
- **Bulls:** last/live/next/home powered by the NBA live scoreboard feed with team logos and ESPN fallback.
- **MLB (Cubs/Sox):**
  - **Last Game:** box score with **bold W/L** in the title.
  - **Live Game:** box score with inning/state as the bottom label.
  - **Next Game:** AWAY @ HOME logos row with day/date/time label using **Today / Tonight / Tomorrow / Yesterday** logic.
  - **Cubs Result:** full‑screen **W/L flag** (animated WebP 100×100 centered; PNG fallback).
- **MLB Standings:**
  - **Overview (AL/NL):** 3 columns of division logos (East/Central/West) with **drop‑in** animation (last place drops first).
  - **Divisions (AL/NL East/Central/West):** scrolling list with W‑L, GB.
  - **Wild Card (AL/NL):** bottom→top scroll with WCGB formatting and separator line.
- **NHL Scoreboard/Standings:** resilient DNS‑aware scoreboard that automatically swaps to the `api-web` endpoint when `statsapi` lookup fails (warnings are suppressed in favor of the informational fallback log).
- **NBA Scoreboard:** rotating marquee of live/final games with throttled 403 logging and ESPN fallback when the CDN blocks scripted traffic.

---

## Running

Run directly:

```bash
python3 main.py
```

Or install the included systemd service (see below).

---

## Systemd unit

Create `/etc/systemd/system/desk_display.service`:

```ini
[Unit]
Description=Desk Display Service - main
After=network-online.target

[Service]
WorkingDirectory=/home/pi/desk_display
ExecStart=/home/pi/desk_display/venv/bin/python /home/pi/desk_display/main.py
ExecStop=/bin/bash -lc '/home/pi/desk_display/tools/maintenance/cleanup.sh'
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

Enable & start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable desk_display.service
sudo systemctl start desk_display.service
journalctl -u desk_display.service -f
```

The service definition above assumes the project’s virtual environment lives at `/home/pi/desk_display/venv` and that the
cleanup helper is executable. Make sure to create the venv first and grant execute permissions to the script:

```bash
python -m venv /home/pi/desk_display/venv
/home/pi/desk_display/venv/bin/pip install -r /home/pi/desk_display/requirements.txt
chmod +x /home/pi/desk_display/tools/maintenance/cleanup.sh
```

`ExecStop` runs `tools/maintenance/cleanup.sh` on every shutdown so the LCD blanks immediately and any lingering screenshots or videos are swept
into the archive folders. The service is marked `Restart=always`, so crashes or manual restarts via `systemctl restart` will
trigger a fresh boot after cleanup completes.

### Display HAT Mini controls

- **X button:** returns to the previously displayed screen.
- **Y button:** requests a `systemctl restart desk_display.service`, which stops the service, runs `tools/maintenance/cleanup.sh`, and starts a
  fresh process.
- **A button:** skips the remainder of the current screen and moves on immediately.
- **B button:** toggles the display on/off without stopping the scheduler.

---

## Screenshots & archiving

- Screenshots land in `./screenshots/` when `ENABLE_SCREENSHOTS=True`.
- `./screenshots/current/` always mirrors the latest capture per screen (flat files, no subfolders) so other tools can
  serve a stable, up-to-date view.
- **Batch archiving:** once the live folder reaches **500** images, the program moves the **entire batch** into `./screenshot_archive/<screen>/` (images only) so the archive mirrors the folder layout under `./screenshots/`.
- You will **not** see per‑image pruning logs; instead you’ll see a single archive log like: `🗃️ Archived 500 screenshot(s) → …`

> Tip: videos (if enabled) are written to `screenshots/display_output.mp4` and aren’t moved by the archiver.

---

## Update indicators

`utils.check_github_updates()` compares local HEAD with `origin/HEAD`. If they differ, a **red dot** appears at the lower‑left of date/time screens.

`utils.check_apt_updates()` runs a simulated `apt-get upgrade` (cached for four hours) and toggles the LED indicator when packages are upgradeable.

The update checker also logs **which files have diverged** when GitHub updates exist, for easier review (uses `git diff --name-only HEAD..origin/HEAD`).

---

## Troubleshooting

- **Too‑dark colors on date/time:** this project forces high‑brightness random RGB values to ensure legibility on the LCD.
- **Missing logos:** you’ll see a warning like `Logo file missing: CHC.png`. Add the correct file into `images/mlb/`.
- **No WebP animation:** ensure your Pillow build supports WebP (`pip3 show pillow`). PNG fallback will still work.
- **Network/API errors:** WeatherKit/MLB requests are time‑bounded; transient timeouts are logged and screens are skipped gracefully.
- **NHL statsapi diagnostics:** run `python3 nhl_scoreboard.py --diagnose-dns` to print resolver details, `/etc/resolv.conf`, and
  quick HTTP checks for both the statsapi and api-web fallbacks. DNS hiccups are now logged at `DEBUG`, so look for the INFO log
  that announces the api-web fallback instead of a warning storm.
- **NBA CDN blocks:** the scoreboard automatically throttles further NBA CDN calls for 30 minutes after a 403 and logs the ESPN
  fallback at INFO level.
- **Font not found:** the code falls back to `ImageFont.load_default()` so the app keeps running; install the missing TTFs to restore look.

---

## License

Personal / hobby project. Use at your own risk. Team names and logos belong to their respective owners.
