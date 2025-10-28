# Desk Scoreboard & Info Display (Pimoroni Display HAT Mini)

A tiny, always‑on scoreboard and info display that runs on a Raspberry Pi and a Pimoroni Display HAT Mini (320×240 ST7789 LCD). It cycles through date/time, weather, travel time, indoor sensors, stocks, Blackhawks, Bulls & Bears screens, MLB standings, and Cubs/White Sox game views (last/live/next).

> **Highlights**
> - Smooth animations: scroll and fade‑in
> - Rich MLB views: last/live/next game, standings (divisions, overview, wild card)
> - **Cubs W/L result** full‑screen flag (animated WebP supported; PNG fallback)
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
      build-essential libjpeg-dev libopenblas0 libopenblas-dev \
      libopenjp2-7-dev libtiff5-dev libcairo2-dev libpango1.0-dev \
      libgdk-pixbuf2.0-dev libffi-dev network-manager wireless-tools \
      i2c-tools fonts-dejavu-core libgl1 libx264-dev ffmpeg git
  ```

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
  Install `pimoroni-bme280` for the Pimoroni Multi-Sensor Stick's BME280 breakout (shares a board with the LTR559 and LSM6DS3).

---

## Install

If you've already cloned this repository (for example into `~/desk_display`), switch into that directory to install dependencies and configure the app.

```bash
cd ~/desk_display
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

The `venv` directory is ignored by Git. Re-run `source venv/bin/activate` whenever you start a new shell session to ensure the project uses the isolated Python environment.

---

## Project layout

```
desk_display/
├─ main.py
├─ config.py
├─ data_fetch.py
├─ screens_catalog.py
├─ screens_config.json
├─ utils.py
├─ scripts_2_text.py
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
├─ images/
│  ├─ mlb/<ABBR>.png              # MLB team logos (e.g., CUBS.png)
│  ├─ nfl/<ABBR>.png              # NFL logos used by Bears screen
│  ├─ W_flag.webp / L_flag.webp   # animated WebP flags (preferred)
│  ├─ W.png / L.png               # fallback PNG flags
│  ├─ cubs.jpg, sox.jpg, hawks.jpg, mlb.jpg, weather.jpg, verano.jpg, bears.png
└─ fonts/
   ├─ TimesSquare-m105.ttf
   ├─ DejaVuSans.ttf
   ├─ DejaVuSans-Bold.ttf
   └─ NotoColorEmoji.ttf
```

---

## Configuration

Most runtime behavior is controlled in `config.py`:

- **Display:** `WIDTH=320`, `HEIGHT=240`
- **Intervals:** `SCREEN_DELAY`, `TEAM_STANDINGS_DISPLAY_SECONDS`, `SCHEDULE_UPDATE_INTERVAL`
- **Feature flags:** `ENABLE_SCREENSHOTS`, `ENABLE_VIDEO`, `ENABLE_WIFI_MONITOR`
- **Weather:** `ENABLE_WEATHER`, `LATITUDE/LONGITUDE`
- **Travel:** `TRAVEL_MODE` (`to_home` or `to_work`)
- **MLB:** constants and timezone `CENTRAL_TIME`
- **Fonts:** make sure `fonts/` contains the TTFs above

### Screen sequencing

The scheduler now uses a **playlist-centric schema (v2)** that supports reusable playlists, nested playlists, rule descriptors, and optional conditions. A minimal configuration looks like this:

```json
{
  "version": 2,
  "catalog": {"presets": {}},
  "metadata": {
    "ui": {"playlist_admin_enabled": true}
  },
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

- **`catalog`** holds reusable building blocks (e.g., preset playlists exposed in the admin UI sidebar).
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

#### Migrating existing configs

Legacy `sequence` arrays are migrated to v2 automatically on startup. For manual conversions or batch jobs run:

```bash
python schedule_migrations.py migrate --input screens_config.json --output screens_config.v2.json
```

This writes a playlist-aware config and validates it using the scheduler parser. The original file is left untouched when `--output` is provided.

#### Admin workflow

- The refreshed admin UI (enabled when `metadata.ui.playlist_admin_enabled` is `true`) provides:
  - Drag-and-drop sequence editing with playlist cards.
  - Rule wizards for **frequency**, **cycle**, and **variants** patterns.
  - Condition editors for days-of-week and time-of-day windows.
  - A preview drawer that simulates the next N screens via the live scheduler.
  - Version history with rollback, backed by `config_versions/` plus an SQLite ledger.
- Set `metadata.ui.playlist_admin_enabled` to `false` (or append `?legacy=1` to the URL) to fall back to the JSON editor.
- Every save records an audit entry (actor, summary, diff summary) and prunes historical versions beyond the configured retention window.

### Default playlist reference

The repository ships with a ready-to-run `screens_config.json` that exposes the **Default loop** playlist shown in the admin UI. The playlist executes the following steps in order (rules are evaluated on each pass through the loop):

1. `date`
2. `weather1`
3. Every third pass, show `weather2`.
4. Every third pass, show `inside` (indoor sensors).
5. `travel`
6. Every fourth pass, show `vrnof` (Verano office status).
7. Every other pass, cycle through the Blackhawks cards: `hawks logo`, `hawks last`, `hawks live`, `hawks next`, `hawks next home`.
8. Every fifth pass, show `NHL Scoreboard`.
9. Every sixth pass, cycle through `NHL Standings Overview`, `NHL Standings Overview`, `NHL Standings West`.
10. Every eighteenth pass (starting at phase 12), show `NHL Standings East`.
11. Every fourth pass, show `bears logo`.
12. Every fourth pass, show `bears next`.
13. Every fifth pass, show `NFL Scoreboard`.
14. Every sixth pass, cycle through `NFL Overview NFC`, `NFL Overview NFC`, `NFL Standings NFC`.
15. Every sixth pass, cycle through `NFL Overview AFC`, `NFL Overview AFC`, `NFL Standings AFC`.
16. Every seventh pass, show `NBA Scoreboard`.
17. Every third pass, show `MLB Scoreboard`.

Each step above maps directly to the JSON structure under `playlists.default.steps`, so any edits made through the admin UI will keep the document and the on-device rotation in sync.

---

### Secrets & environment variables

API keys are no longer stored directly in `config.py`. Set them as environment variables before running any of the
scripts:

- `OWM_API_KEY_VERANO`, `OWM_API_KEY_WIFFY`, or `OWM_API_KEY_DEFAULT` (fallback); the code also accepts a generic
  `OWM_API_KEY` value if you only have a single OpenWeatherMap key.
- `GOOGLE_MAPS_API_KEY` for travel-time requests (leave unset to disable that screen).
- `TRAVEL_TO_HOME_ORIGIN`, `TRAVEL_TO_HOME_DESTINATION`, `TRAVEL_TO_WORK_ORIGIN`,
  and `TRAVEL_TO_WORK_DESTINATION` to override the default travel addresses.

You can export the variables in your shell session:

```bash
export OWM_API_KEY="your-open-weather-map-key"
export GOOGLE_MAPS_API_KEY="your-google-maps-key"
```

Or copy `.env.example` to `.env` and load it with your preferred process manager or a tool such as
[`python-dotenv`](https://github.com/theskumar/python-dotenv).

---

## Images & Fonts

- **MLB logos:** put team PNGs into `images/mlb/` named with your abbreviations (e.g., `CUBS.png`, `MIL.png`).
- **NFL logos:** for the Bears screen, `images/nfl/<abbr>.png` (e.g., `gb.png`, `min.png`).
- **Cubs W/L flag:** use `images/W_flag.webp` and `images/L_flag.webp` (animated). If missing, the code falls back to `images/W.png` / `images/L.png`.
- **Fonts:** copy `TimesSquare-m105.ttf`, `DejaVuSans.ttf`, `DejaVuSans-Bold.ttf`, and `NotoColorEmoji.ttf` into `fonts/`.
- **Travel font:** the Google Maps travel screen loads `HWYGNRRW.TTF` (Highway Gothic) directly from `fonts/`. Without this
  file the app will exit on startup, so copy your licensed copy into that folder alongside the other fonts.
- **Emoji font:** `NotoColorEmoji.ttf` is used by default; if unavailable, install the Symbola font (package `ttf-ancient-fonts` on Debian/Ubuntu) or place `Symbola.ttf` in your system font directory so precipitation/cloud icons render correctly.

---

## Screens

- **Date/Time:** both screens display date & time in bright/legible colors with a red dot when updates are available.
- **Weather (1/2):** Open‑Meteo + OWM configuration.
- **Inside:** BME sensor summary (labels/values) if wired.
- **VRNOF:** stock mini‑panel.
- **Travel:** Maps ETA using your configured mode.
- **Bears Next:** opponent and logos row, formatted bottom line.
- **Blackhawks:** last/live/next based on schedule feed, logos included.
- **Bulls:** last/live/next/home powered by the NBA live scoreboard feed with team logos.
- **MLB (Cubs/Sox):**
  - **Last Game:** box score with **bold W/L** in the title.
  - **Live Game:** box score with inning/state as the bottom label.
  - **Next Game:** AWAY @ HOME logos row with day/date/time label using **Today / Tonight / Tomorrow / Yesterday** logic.
  - **Cubs Result:** full‑screen **W/L flag** (animated WebP 100×100 centered; PNG fallback).

- **MLB Standings:**
  - **Overview (AL/NL):** 3 columns of division logos (East/Central/West) with **drop‑in** animation (last place drops first).
  - **Divisions (AL/NL East/Central/West):** scrolling list with W‑L, GB.
  - **Wild Card (AL/NL):** bottom→top scroll with WCGB formatting and separator line.

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
ExecStop=/bin/bash -lc '/home/pi/desk_display/cleanup.sh'
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
chmod +x /home/pi/desk_display/cleanup.sh
```

`ExecStop` runs `cleanup.sh` on every shutdown so the LCD blanks immediately and any lingering screenshots or videos are swept
into the archive folders. The service is marked `Restart=always`, so crashes or manual restarts via `systemctl restart` will
trigger a fresh boot after cleanup completes.

### Display HAT Mini controls

- **X button:** skips the remainder of the current screen and moves on immediately.
- **Y button:** requests a `systemctl restart desk_display.service`, which stops the service, runs `cleanup.sh`, and starts a
  fresh process.
- **A/B buttons:** currently unused but logged when pressed so you can build new shortcuts.

---

## Screenshots & archiving

- Screenshots land in `./screenshots/` when `ENABLE_SCREENSHOTS=True`.
- **Batch archiving:** once the live folder reaches **500** images, the program moves the **entire batch** into `./screenshot_archive/dated_folders/<screen>/YYYYMMDD/HHMMSS/` (images only) so the archive mirrors the folder layout under `./screenshots/`.
- You will **not** see per‑image pruning logs; instead you’ll see a single archive log like: `🗃️ Archived 500 screenshot(s) → …`

> Tip: videos (if enabled) are written to `screenshots/display_output.mp4` and aren’t moved by the archiver.

---

## GitHub update indicator

`utils.check_github_updates()` compares local HEAD with `origin/HEAD`. If they differ, a **red dot** appears at the lower‑left of date/time screens.

The checker now logs **which files have diverged** when updates exist, for easier review (uses `git diff --name-only HEAD..origin/HEAD`).

---

## Troubleshooting

- **Too‑dark colors on date/time:** this project forces high‑brightness random RGB values to ensure legibility on the LCD.
- **Missing logos:** you’ll see a warning like `Logo file missing: CUBS.png`. Add the correct file into `images/mlb/`.
- **No WebP animation:** ensure your Pillow build supports WebP (`pip3 show pillow`). PNG fallback will still work.
- **Network/API errors:** MLB/OWM requests are time‑bounded; transient timeouts are logged and screens are skipped gracefully.
- **NHL statsapi DNS warning:** run `python3 nhl_scoreboard.py --diagnose-dns` to print resolver details, `/etc/resolv.conf`, and
  quick HTTP checks for both the statsapi and api-web fallbacks. Attach the JSON output when filing an issue.
- **Font not found:** the code falls back to `ImageFont.load_default()` so the app keeps running; install the missing TTFs to restore look.

---

## License

Personal / hobby project. Use at your own risk. Team names and logos belong to their respective owners.
