# Desk Display Operator Guide

A short guide to running the display on a Raspberry Pi. The display is `main.py`
running as `desk_display.service` under systemd, installed by
`Installers/install.sh` and reconfigured through the web UI on port 5002.

`README.md` is the full reference; this file is the operator's path through it.

---

## How it runs on the Pi

One systemd unit does the work: `desk_display.service` runs
`<project>/venv/bin/python <project>/main.py` with the checkout as its working
directory. The installer writes that unit to `/etc/systemd/system/` and bakes in
the display settings it was run with.

What the unit carries:

- `EnvironmentFile=-<project>/.env` — every API key and tuning variable comes
  from there, and the leading `-` means a missing file is not an error.
- `Environment=DESK_DISPLAY_OUTPUT=<mode>` — the renderer chosen at install time.
- `Environment=SCREEN_CONFIG_AUTOSTART=0` — stops `main.py` from spawning its own
  copy of the config UI, because `config_ui_desk_display.service` runs it instead.
- `Restart=always`, `RestartSec=5`, `KillSignal=SIGTERM`, `TimeoutStopSec=10`.

On stop or reboot systemd sends `SIGTERM` straight to `main.py`, which stops
scheduling screens, blanks the panel, finalizes video and exits through its own
shutdown path inside that 10-second window.

`DESK_DISPLAY_OUTPUT` selects the renderer:

| Mode | Used for |
| --- | --- |
| `auto` | Default; picks whatever display path is available. |
| `displayhatmini` | Pimoroni Display HAT Mini, 320×240. |
| `minipitft` | Adafruit miniPiTFT 1.14", 240×135. |
| `kernel` | HyperPixel and other KMS/DRM panels. |
| `framebuffer` | Direct writes to `/dev/fb*`. |
| `window` | SDL desktop window (macOS, Windows, Pi desktop). |
| `headless` | Render without hardware, for tests and tooling. |

Kernel mode draws into the desktop user's X11 or Wayland session, so the unit
gets two `ExecStartPre` steps (`scripts/wait_for_display_ready.sh`, then
`scripts/prepare_kernel_session_env.sh`) that wait for that session and write its
`DISPLAY`/`WAYLAND_DISPLAY`/`XAUTHORITY` into `.runtime/kernel-session.env` for
`ExecStart` to read. If the desktop never comes up, `Restart=always` keeps
retrying and the reason shows in the unit's journal.

Up to eight project units can be installed. `scripts/restart_services.sh`
documents and enforces the dependency order between them:

| Unit | Role |
| --- | --- |
| `desk_display_adsb_collector.service` | Writes the ADS-B cache the aircraft screens read. |
| `feed_server_desk_display.service` | Hosts feed pages built from screenshots other Pis push. |
| `desk_display.service` | The renderer itself. |
| `waveshare-fbcp.service` | Mirrors the framebuffer onto a Waveshare panel. |
| `desk_display_waveshare_oled.service` | Side status OLED helper. |
| `screenshot_uploader_desk_display.service` | POSTs new screenshots to a feed server. |
| `config_ui_desk_display.service` | The web config UI on port 5002. |
| `airplay_desk_display.service` | Optional AirPlay receiver add-on. |

---

## Installing on a fresh Pi

Clone the repository, create `.env` from `.env.example` and fill in the keys you
need (`.env` is gitignored, so it never leaves the Pi), then run the single entry
point from the repository root:

```bash
bash ./Installers/install.sh
```

It asks three questions, each of which can be answered on the command line
instead:

1. **Which hardware profile** — `display_hat_mini` (default),
   `adafruit_minipitft`, `hyperpixel`, `kernel`, `macos_window`, `pi_window`,
   `win_window`, or `waveshare_oled_lcd_hat_a`.
2. **Which default rotation** — `small` or `large` (default), loaded by
   `scripts/load_default_screen_config.py`.
3. **Whether to install the ADS-B collector** — defaults to no; needs
   `ADSB_DEVICE_1_HOST` in `.env`.

So a fully unattended install is:

```bash
bash ./Installers/install.sh display_hat_mini large n
```

Set `INSIDE_SENSOR` in the environment or `.env` before running, and the
installer adds the matching optional sensor requirements automatically.

Every hardware profile is a thin wrapper: it sets `DESK_DISPLAY_OUTPUT` (and
panel-specific variables) and hands off to `scripts/helpers/base_setup.sh`, which
is where the real work happens. That script, in order:

1. Enables SPI and I2C through `raspi-config` — or disables them when
   `DISABLE_SPI_I2C=1`, which HyperPixel panels require.
2. Installs the apt packages (Python build tooling, imaging and font libraries,
   `network-manager`, `i2c-tools`, `ffmpeg`, and so on).
3. Runs `scripts/update_dependencies.sh`, which creates `venv/` and installs the
   requirements file for the chosen output mode.
4. Marks the maintenance scripts executable.
5. Writes `/etc/systemd/system/desk_display.service` and
   `/etc/systemd/system/config_ui_desk_display.service`, baking in the display
   environment described above.
6. Runs `systemctl daemon-reload`, then enables and restarts both units, and
   prints their status.

The unit runs as `$SUDO_USER` (or whoever invoked the script), not as root, so
run the installer with your normal login and let it call `sudo` itself.

---

## What each script in Installers/ does

| Script | What it does |
| --- | --- |
| `install.sh` | The menu. Resolves a profile to one of the scripts below, runs it, then loads the chosen default rotation and optionally the ADS-B collector. |
| `install_display_hat_mini.sh` | Sets `DESK_DISPLAY_OUTPUT=displayhatmini` and hands off to `base_setup.sh`. |
| `install_adafruit_minipitft_114.sh` | Sets `minipitft` output, `requirements/minipitft.txt`, and 240×135 dimensions, then hands off. |
| `install_kernel.sh` | Sets `kernel` output and `requirements/kernel.txt` for generic KMS/DRM panels; also clears out stale per-user units from older installs. |
| `install_hyperpixel.sh` | Same as above plus HyperPixel panel setup (overlays, panel selection), and falls back to `framebuffer` output when kernel mode is not usable. |
| `install_waveshare_oled_lcd_hat_a.sh` | Sets `framebuffer` output and additionally installs `waveshare-fbcp.service` and `desk_display_waveshare_oled.service` for the side status OLED. |
| `install_macos_window.sh`, `install_pi_window.sh`, `install_win_window.sh` | Desktop profiles. These write windowed settings into `.env` and print the launch command — they install no systemd unit and no dependencies. |
| `install_config_ui_service.sh` | Writes and starts `config_ui_desk_display.service` alone, for a machine that already has the venv. |
| `install_adsb_collector_service.sh` | Writes and starts `desk_display_adsb_collector.service`. Needs `ADSB_DEVICE_1_HOST` in `.env`. |
| `install_feed_server.sh` | Installs only `feed_server.py` and its unit, with no rendering or GPIO stack — for a Pi that just aggregates screenshots from other Pis. |
| `install_screenshot_uploader.sh` | Installs the uploader that pushes this Pi's screenshots to that feed server. Needs `FEED_UPLOAD_URL` and `FEED_UPLOAD_TOKEN`. |
| `uninstall.sh` | Destructive. Stops and disables the units, removes the venv, moves `.env` and `~/keys/` into `~/desk_display_uninstalled`, then deletes the project directory. Requires typing `UNINSTALL`, or `CONFIRM_UNINSTALL=yes` non-interactively. |

---

## Running, restarting, and logs

The ordinary systemd commands work:

```bash
sudo systemctl status desk_display.service
sudo systemctl restart desk_display.service
sudo journalctl -u desk_display.service -f
```

To cycle everything this project installed, in the order the services depend on
each other, use the wrapper instead:

```bash
./scripts/restart_services.sh            # all installed project units
./scripts/restart_services.sh --list     # show the known units and their order
./scripts/restart_services.sh desk_display.service config_ui_desk_display.service
```

It only ever touches the eight units listed above, skips the ones that are not
installed here, and restarts in its own dependency order regardless of the order
you type.

After pulling new code:

```bash
git pull
./scripts/update_dependencies.sh     # refresh venv/ from the profile's requirements
./scripts/update_services.sh         # patch stale script paths baked into the units
./scripts/restart_services.sh
```

`update_services.sh` is the safe way to repair an installed unit: it rewrites
script paths that moved in the repository and applies the current shutdown
settings, but leaves the display profile and every `Environment=` override alone.
Re-running a full hardware installer instead regenerates the unit from whatever
environment the installer happens to run with, which is how a working profile
gets lost.

To run the renderer by hand, stop the service first so two processes do not fight
over the panel:

```bash
sudo systemctl stop desk_display.service
source venv/bin/activate
python main.py                        # full rotation
python main.py --list-screens         # print every screen ID
python main.py --screen               # pick one screen from a searchable menu
python main.py --screen "news headlines"
DESK_DISPLAY_FORCE_HEADLESS=1 DESK_DISPLAY_OUTPUT=headless python main.py   # no hardware
```

The Rotation Config page also has a single-screen diagnostic playback selector
that switches the running renderer without editing the saved rotation; a
command-line `--screen` choice overrides it.

`scripts/cleanup.sh` is a manual maintenance utility only — extra hardware reset,
`__pycache__` removal, archiving leftover screenshots and video. It is
deliberately not wired up as an `ExecStop` handler, because `main.py` owns its
own shutdown.

---

## Changing what the display shows

The rotation lives in `screens_config.json`, and `config_ui_desk_display.service`
edits it through a web page at `http://<pi>:5002` (`SCREEN_CONFIG_PORT` overrides
the port). The page enables and disables screens, edits frequencies and hold
times, manages playlists and sequence order, and imports and exports rotation
payloads.

The file has three top-level pieces: `screens` (per-screen frequency),
`playlists` (named groups of steps), and `sequence` (the order those playlists
rotate in).

A frequency is an integer — `1` shows the screen every pass, `2` every other
pass, `0` disables it — or an object with `frequency` plus optional
`extra_seconds`, an `alt` screen, a `hide_after_at` retirement date, and a
per-screen `scroll.speed`.

From the command line:

```bash
./scripts/show_screen_rotation_config.sh
python scripts/load_default_screen_config.py large        # or small
python scripts/load_default_screen_config.py small --dry-run
python scripts/export_screen_rotation_config.py
python scripts/import_screen_rotation_config.py path/to/export.json
```

---

## Adding a new screen

A screen is four things: a renderer module, an entry in the catalog, a
registration, and a frequency. Miss the catalog entry and the config UI will not
list it; miss the registration and it will never render.

**1. Write the renderer** in `screens/draw_<name>.py`. It takes the display object
and returns a PIL `Image` or a `utils.ScreenImage`:

```python
@log_call
def draw_my_screen(display, transition: bool = False):
    img = Image.new("RGB", (WIDTH, HEIGHT), get_screen_background_color())
    ...
    return ScreenImage(img, displayed=False)
```

Take `WIDTH`, `HEIGHT`, fonts and colors from `config` rather than hard-coding
them, so the screen works across the 240×135 through 800×480 profiles. Fetching
belongs in `services/` or `data_fetch.py`, not in the renderer.

**2. Add the ID to `screens_catalog.py`**, appended to the end of its section in
`RAW_SCREEN_IDS`. That list is the single source of truth: `main.py
--list-screens`, the config UI, and `scripts/render_screens.py` all read it.

**3. Register it in `screens/registry.py`.** Bind the renderer lazily near the top
of the file, so importing the registry does not drag in a heavy module for a
disabled screen:

```python
draw_my_screen = _lazy_callable("screens.draw_my_screen.draw_my_screen")
```

Then add a `register(...)` call inside `build_screen_registry()`:

```python
register(
    "my screen",
    lambda: draw_my_screen(context.display, transition=True),
    available=config.ENABLE_MY_SCREEN,
)
```

`available=False` keeps the screen in the catalog but out of the rotation — that
is how screens gate on a config flag, on cached data being present, or on the
Wi-Fi outage state in `context`.

**4. Give it a frequency** in `screens_config.json`, and add it to
`default_screens_large.json` and `default_screens_small.json` (plus a playlist
step) if it should be on by default for new installs.

**5. If it needs a toggle or credentials**, add the `ENABLE_*` flag and any keys
to `config.py` and document them in `.env.example`.

Then run the checks before pushing:

```bash
python scripts/test_all.py                    # the canonical CI check: Ruff, then pytest
python scripts/validate_required_files.py     # catches a module nothing imports
python scripts/render_screens.py              # renders screens for a visual look
```

`tests/test_screens_catalog.py`, `tests/test_screen_registry.py` and
`tests/test_default_screen_configs.py` are where screen-level tests go. CI runs
`python scripts/test_all.py` on every pull request.
