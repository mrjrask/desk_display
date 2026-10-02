# Desk Display Operator Guide

The operator's runbook. A standalone display is `main.py` running as
`desk_display.service` under systemd, installed by `Installers/install.sh`
and reconfigured through the web UI on port 5002. A render server and its
display clients are covered in [Installation modes](#installation-modes-server-and-client)
and [Server and client operations](#server-and-client-operations).

`README.md` is the full reference; this file is the operator's path through it.

---

## How it runs on the Pi

In a standalone install one systemd unit does the work: `desk_display.service` runs
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

### Raspberry Pi 5 (kernel output)

Kernel output works the same on a Pi 5, with these differences:

- Use 64-bit Raspberry Pi OS **with desktop** (Bookworm or Trixie). Kernel mode
  draws into the labwc/Wayland session, so Pi OS Lite never gets past
  `prepare_kernel_session_env.sh`. Turn on desktop autologin with
  `sudo raspi-config` → System Options → Boot / Auto Login → Desktop Autologin.
- `/boot/firmware/config.txt` must load KMS and the panel overlay for every
  board, not only the Pi 4. A card moved from a Pi 4 often has them under
  `[pi4]`, which the Pi 5 skips. Put them under `[all]`:
  - HyperPixel 4 Square: `dtoverlay=vc4-kms-v3d` and
    `dtoverlay=vc4-kms-dpi-hyperpixel4sq` (add `,rotate=…` as before).
    Pimoroni's legacy `hyperpixel4` driver and its init service do not work on
    a Pi 5. The panel appears as a `DPI-1` connector on `card1` or `card2`.
  - HDMI: `dtoverlay=vc4-kms-v3d`. `hdmi_force_hotplug`, `hdmi_group` and
    `hdmi_mode` are ignored; force a mode on a screen with no EDID with
    `video=HDMI-A-1:1920x1080@60D` in `/boot/firmware/cmdline.txt`.
- The Pi 5 renumbers DRM cards (`card0` is the render-only GPU) and I2C buses.
  If the indoor sensor stops reading, run `i2cdetect -l` and set
  `INSIDE_I2C_BUSES` in the panel's env file to the new bus.
- RPi.GPIO does not run on the Pi 5. Buttons and LEDs go through gpiozero
  and lgpio (`requirements/hw-gpio.txt`).

A backlight that comes on at boot but shows nothing usually means the panel
overlay loaded its backlight pin but no DPI/HDMI connector came up. When
`wait_for_display_ready.sh` times out it logs the board model and every DRM
connector's status and mode count; `scripts/check_hyperpixel_setup.sh` prints
the same plus the config.txt section headers. `No DRM connectors` means the
overlay did not load; `status=disconnected` on HDMI means the EDID was not read.

Up to ten project units can be installed. `scripts/restart_services.sh`
documents and enforces the dependency order between them:

| Unit | Role |
| --- | --- |
| `desk_display_adsb_collector.service` | Writes the ADS-B cache the aircraft screens read. |
| `feed_server_desk_display.service` | Hosts feed pages built from screenshots other Pis push. |
| `desk_display_server.service` | The render server (server and combined installs). |
| `desk_display.service` | The standalone renderer. |
| `desk_display_client.service` | The display client (client and combined installs). |
| `waveshare-fbcp.service` | Mirrors the framebuffer onto a Waveshare panel. |
| `desk_display_waveshare_oled.service` | Side status OLED helper. |
| `screenshot_uploader_desk_display.service` | POSTs new screenshots to a feed server. |
| `config_ui_desk_display.service` | The web config UI on port 5002 (only the Screenshots, Feed and Stats pages on a client). |
| `airplay_desk_display.service` | Optional AirPlay receiver add-on. |

---

## Installing on a fresh Pi

Clone the repository, create `.env` from `.env.example` and fill in the keys you
need (`.env` is gitignored, so it never leaves the Pi), then run the single entry
point from the repository root:

```bash
bash ./Installers/install.sh
```

It first asks for the installation mode (`standalone`, the default, or
`server`, `client` or `combined`: see "Installation modes" below, or pass
`--mode`). A standalone install then asks three questions, each of which
can be answered on the command line instead:

1. **Which hardware profile** — `display_hat_mini` (default),
   `adafruit_minipitft`, `hyperpixel`, `kernel`, `macos_window`, `pi_window`,
   `win_window`, or `waveshare_oled_lcd_hat_a`.
2. **Which default rotation** — `small` or `large` (default), loaded by
   `scripts/load_default_screen_config.py`.
3. **Whether to install the ADS-B collector** — defaults to no; needs
   `ADSB_DEVICE_1_HOST` in `.env`.

So a fully unattended install is:

```bash
bash ./Installers/install.sh --mode standalone display_hat_mini large n
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
| `install.sh` | The menu. Asks for the installation mode (or takes `--mode`), resolves a profile to one of the scripts below, runs it, then loads the chosen default rotation and optionally the ADS-B collector. |
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
| `uninstall.sh` | Destructive, in every mode. Stops and disables every project unit, removes the venv, copies the mode's backed-up data (see [What each mode keeps](#what-each-mode-keeps)) into `~/desk_display_uninstalled`, then deletes the project directory. Requires confirmation, or `CONFIRM_UNINSTALL=yes` non-interactively. |

## What each script in scripts/ does

| Script | What it does |
| --- | --- |
| `upgrade.sh` | Upgrades an install in any mode: `git pull`, dependencies, units, restart (see [Running, restarting, and logs](#running-restarting-and-logs)). `--no-pull` skips the pull. |
| `update_dependencies.sh` | Installs the requirements file for the installed mode and output; `--upgrade-outdated`, `--clean-caches`, `--print-requirements`. |
| `update_services.sh` | Rewrites or patches the installed units for the installed mode; `--dry-run`, `--no-restart`, `--mode`. |
| `restart_services.sh` | Restarts the installed project units in dependency order; `--list` prints them. |
| `clear-caches.sh` | Empties the pip download cache and runs `apt-get clean`, and reports the space freed. |
| `reset_screenshots.sh` | Empties the screenshot and screenshot archive folders (`SCREENSHOT_DIR`, `SCREENSHOT_ARCHIVE_BASE`). |
| `cleanup.sh` | Manual maintenance: stops and blanks the panel, removes `__pycache__`, archives leftover screenshots and video. |
| `convert_env.py` | Converts a `.env` to the server or client role (see [Converting an existing .env](#converting-an-existing-env)). |
| `migrate_standalone_config.py` | Moves a standalone rotation onto the server as a playlist (see [Moving a standalone rotation onto the server](#moving-a-standalone-rotation-onto-the-server)). |
| `collect_client_screenshots.py` | Gathers every display's latest screenshots into one HTML page (see [Comparing screens across displays](#comparing-screens-across-displays)). |
| `measure_scroll_fps.py` | Measures scroll smoothness on a client's panel (see [Checking scroll smoothness](#checking-scroll-smoothness)). |
| `test_led.py` | Cycles the Display HAT Mini's RGB LED and reports whether it responded; stop the display service first. |
| `test_api_connections.py` | Probes every external API family; `--json` for machine-readable output. |
| `load_default_screen_config.py`, `update_screen_config.py` | Load the small or large default rotation, by argument or by prompt. |
| `export_screen_rotation_config.py`, `import_screen_rotation_config.py`, `show_screen_rotation_config.sh` | Export, import and print the rotation configuration. |
| `logo_getter.py` | Downloads NCAA team logos for review or for committing to `images/ncaa/`. The render server already downloads missing FBS logos each week into `images/cache/ncaa/`. |
| `render_screens.py` | Renders screens to PNG and archives them in a dated ZIP for visual review. |
| `adsb_collector.py`, `screenshot_uploader.py`, `waveshare_oled_status.py` | The ADS-B collector, the Feed uploader and the Waveshare side-OLED helper, each run by its own unit. |
| `check_hyperpixel_setup.sh`, `check_waveshare_setup.sh`, `restore_desktop.sh` | Panel setup checks, and putting the Pi desktop back after a kernel or framebuffer install. |
| `launch_kernel_display.sh`, `launch_framebuffer.sh`, `framebuffer_service.sh`, `prepare_kernel_session_env.sh`, `wait_for_display_ready.sh`, `launch_macos_window_perf.sh` | Launchers and unit helpers for the kernel, framebuffer and macOS window outputs. |
| `test_all.py`, `validate_required_files.py`, `check_image_assets.py`, `check_lint_baseline.py`, `lint_cleanup_report.py`, `font_audit.py` | Development checks; `test_all.py` is what CI runs. |
| `make_v01_references.py`, `v01_parity_probe.py` | Record the v0.1 reference renders `tests/test_v01_parity.py` compares against. |
| `adjust_image_assets.py`, `render_bears_next_season_png.py`, `uninstall_airplay.sh` | One-off asset tools and the AirPlay add-on's uninstaller. |

---

## Installation modes (server and client)

Besides the standalone display, Desk Display can run as a render server,
a display client, or both on one machine. `service_units.py` defines the
systemd services for each mode:

| Mode | Services | Env file |
| --- | --- | --- |
| standalone | `desk_display.service` (main.py), config UI | `.env` |
| server | `desk_display_server.service`, config UI | `.env` |
| client | `desk_display_client.service`, config UI (Screenshots, Feed and Stats pages) | `.env.client` |
| combined | server, client and config UI | `.env` (server), `.env.client` (panel) |

In a combined installation the attached panel is an ordinary client of
its own server on the loopback address. It has its own client ID,
profile, assigned playlist, cache, rotation and touch settings in
`.env.client`, for example
`DESK_DISPLAY_SERVER_URL=http://127.0.0.1:8765` (plain HTTP is allowed on
loopback). The client unit does not depend on the server unit, so it
starts from its cache before the server is up, and restarting the server
leaves the panel playing. `main.py` refuses to start with a server or
client role, and `desk_display.service` conflicts with the client service,
so nothing draws to the panel outside the manifests.

Each mode has one installer command:

| Mode | Command |
| --- | --- |
| standalone | `bash Installers/install.sh [profile]` (as before) |
| server | `bash Installers/install.sh --mode server` |
| client | `bash Installers/install.sh --mode client --credentials office.env.client <profile>` |
| combined | `bash Installers/install.sh --mode combined <profile>` |

`<profile>` is a panel profile from the list above; the desktop window
profiles install no service, so they only run standalone.

A server install sets up no panel hardware and installs
`requirements/server.txt` (the full application, no GPIO or panel
drivers). A client installs `requirements/client-<output>.txt`: the
client core (`requests`, `pytz`, `Pillow`), Flask and Waitress for its
local config UI pages, and its panel driver, with no upstream provider
library or SVG renderer.

The installer prepares the env files before it starts anything:

- A client's `.env.client` starts from the panel settings in `.env`
  (profile, rotation, backlight, sensors), converted with
  `scripts/convert_env.py` so no server or provider setting survives, plus
  the identity and credential from `--credentials` (the file the server's
  "Add a display" setup gives you, including
  `DESK_DISPLAY_ALLOW_INSECURE_TRANSPORT` for a plain-HTTP server). Without `--credentials`, fill in the
  placeholders it lists.
- In a combined install the server provisions the panel itself as
  `<hostname>-panel` on `http://127.0.0.1:8765`.
- A server's `.env` is converted to the server role in place, with a
  `.env.bak-<time>` copy of the original.
- An existing `.env.client` is never replaced: it is the client's stable
  identity. Edit it to change panel settings.

The installer then writes the mode's units, disables the other project
units, and enables and starts the mode's services, server first. It
records the mode, output driver, service user and unit environment in
`.runtime/install_mode`, so `scripts/upgrade.sh`, `uninstall.sh` and
`cleanup.sh` act on the right services and files.
`python3 install_modes.py plan --mode combined` prints everything a mode
installs, starts, disables, keeps and backs up.

### What each mode keeps

Upgrades keep all of this exactly as it was. The uninstaller copies the
items marked "backed up" to `~/desk_display_uninstalled` (env files with
mode 600) and then deletes the rest with the project directory.

| Data | Modes | Uninstall |
| --- | --- | --- |
| `.env` (configuration, provider credentials) | standalone, server, combined | backed up |
| `.env.client` (client ID, credential, panel settings) | client, combined | backed up |
| `~/keys` (WeatherKit key files) | standalone, server, combined | backed up |
| `screens_config.local.json` | standalone, server, combined | backed up |
| `.runtime/server/playlists.json` (playlists, assignments) | server, combined | backed up |
| `.runtime/server/provisioned_clients.json` (credential hashes) | server, combined | backed up |
| `.runtime/server/clients.json` (known clients) | server, combined | backed up |
| `.runtime/server/migrations/`, `.runtime/server/backups/` | server, combined | backed up |
| `cache/artifacts/` (rendered artifacts) | server, combined | removed; re-rendered |
| `cache/client/` (offline cache) | client, combined | removed |
| `cache/` (feed caches, weather history) | standalone, server, combined | removed |

Before upgrading a server or combined install, `scripts/upgrade.sh`
copies its env files (`.env`, plus `.env.client` when combined),
`screens_config.local.json`, playlists, assignments, credential hashes and
client list to `.runtime/server/backups/upgrade-<time>/` (mode 700). `cleanup.sh`
stops and blanks whichever process drives the panel (`main.py` or
`display_client.py`) and does nothing to the panel on a server.

To write the unit files by hand:

```bash
python3 install_modes.py units --mode combined --output kernel --dir /tmp/units --user "$USER"
```

### Moving a standalone rotation onto the server

`scripts/migrate_standalone_config.py` turns the rotation this display
already plays into a shared server playlist, so nothing has to be rebuilt
by hand. It reads the active screen config (the local override when there
is one) and never changes it:

```bash
python3 scripts/migrate_standalone_config.py --assign office          # preview
python3 scripts/migrate_standalone_config.py --assign office --apply  # do it
```

The preview lists every change before anything happens: legacy screen IDs
that were renamed, retired or unknown screens that are dropped, settings
with no server equivalent, screens remote clients cannot show, and clients
already playing another playlist (those are only reassigned with
`--force-assign`). Frequencies, extra seconds, alternates, playlists and
sequence carry over unchanged. `--install-style` also installs the
migrated style and quad layouts. Each run leaves a bundle in
`.runtime/server/migrations/` that holds the original files and what was
changed. `--config FILE` migrates a rotation copied from another display instead of
this one's. `--export FILE` writes one as a backup without changing anything,
and `--rollback BUNDLE` undoes an applied run. Rerunning is safe: an
already-migrated rotation is reused, never duplicated.

### Converting an existing .env

`scripts/convert_env.py` edits a standalone `.env` into a server or client
configuration in place, so the values you already set carry over:

```bash
python3 scripts/convert_env.py --role server --dry-run   # show the change
python3 scripts/convert_env.py --role server             # write it
cp .env .env.client
python3 scripts/convert_env.py --role client --env-file .env.client \
    --credentials office.env.client                      # the file from "Add a display"
```

It removes every setting the role does not read, keeps the last line of a
setting that appears twice, sets `DESK_DISPLAY_ROLE`, and appends the
role's required and role-only settings under one marked block. Comments
stay where they are. A client also loses names that are not Desk Display
settings and commented-out credentials, because a client holds no provider
keys; `--keep-unknown` / `--no-keep-unknown` overrides that default. Each
removal and addition is listed, and the dry run shows a diff. Secret values
are always printed as `[redacted]`.

The result must pass the role's startup checks before it is written. A
missing client value becomes a placeholder, and the file is left alone
unless you pass `--allow-invalid`. The client credential comes from
`--credentials` or `--token-file`, never the command line. Before writing,
the original is copied to `.env.bak-<timestamp>` (mode 600). The new file
replaces it atomically, so an interrupted run leaves the old file in
place. Rerunning on a converted file changes nothing.

## Server and client operations

Commands below run on the server host unless they say otherwise. The admin
API needs `DESK_DISPLAY_SERVER_ADMIN_TOKEN` set in the server's `.env`.
Export it in your shell as `ADMIN` for the `curl` examples.

### Service lifecycle

| Mode | Start, stop, restart | Logs |
| --- | --- | --- |
| server | `sudo systemctl restart desk_display_server.service` | `sudo journalctl -u desk_display_server.service -f` |
| client | `sudo systemctl restart desk_display_client.service` | `sudo journalctl -u desk_display_client.service -f` |
| combined | both of the above; `./scripts/restart_services.sh` does them in order | both |
| any | `sudo systemctl restart config_ui_desk_display.service` | `sudo journalctl -u config_ui_desk_display.service -f` |

Restarting the server never stops a client: clients keep playing from their
cache and reconnect on their own.

### Provisioning, rotation and revocation

The easiest way is the configuration UI's `/clients` page. **Add a display**
opens a guided setup (`/clients/add`):

1. **Check the server.** It shows whether another device can reach the
   render server (bind address, advertised URL, enrollment mode, and a live
   connection to this Pi's LAN address), with the exact `.env` lines to fix
   anything that is not ready.
2. **Describe the display.** Pick its screen type, give it a name (the client
   ID is filled in from it), a playlist, and the server address it will use.
   Over plain HTTP it adds `DESK_DISPLAY_ALLOW_INSECURE_TRANSPORT=1` to the
   display's settings unless you untick it.
   A Waveshare OLED/LCD HAT (A) is one choice: the installer sets up the
   LCD as the client's panel and `desk_display_waveshare_oled.service` for
   the two side OLEDs, which show the date, time and weather (and live
   Cubs/Blackhawks scores) from the client's heartbeat. Reboot once it
   finishes.
3. **Install.** Paste one command on the new Pi:

   ```bash
   bash -c "$(curl -fsS -d code=<one-time code> http://square.local:8765/api/v1/join)"
   ```

   The code works once, for 30 minutes. The render server then issues the
   display's credential and returns a setup script that clones Desk Display
   (from the server's own git remote) when `~/desk_display` is missing,
   runs `git pull` in a checkout still on v0.1, moves
   any old `.env.client` aside, writes the settings with mode 600 and runs
   `Installers/install.sh --mode client` for that panel. The credential never
   appears on the web page. **Set it up by hand instead** shows the
   `.env.client` and a longer paste-able command once, and cancels the code.
4. **Connect.** The page watches for the display's first heartbeat and says
   when it is online, with log commands if it is not.

Each display's **Maintenance** tab on `/clients` has Rename, Rotate
credential, Revoke, Disable and Enable, and a display that has never
connected also has **Setup command**, which makes a new one-time
command (any earlier one stops working). From the shell:

```bash
python3 -m remote_display.provisioning provision office --profile hyperpixel4_square \
    --server-url https://render.lan:8765 > office.env.client      # shown once
python3 -m remote_display.provisioning list
python3 -m remote_display.provisioning rotate office > office.env.client
python3 -m remote_display.provisioning disable office             # or enable
python3 -m remote_display.provisioning revoke office
```

Copy the `.env.client` to the display and install it with
`bash Installers/install.sh --mode client --credentials office.env.client <profile>`.
For a display that is already installed, use
`python3 scripts/convert_env.py --role client --env-file .env.client --credentials office.env.client`
and restart `desk_display_client.service`. Rotating, revoking or disabling
ends the client's lease at once. Rotation takes effect when the client has
the new credential.

### Letting other displays reach the server

A server or combined install listens on `127.0.0.1:8765`, which only its own
panel can reach. Before adding a display on another device, set these in the
server's `.env` and restart with `./scripts/restart_services.sh`:

```bash
DESK_DISPLAY_SERVER_HOST=0.0.0.0
DESK_DISPLAY_SERVER_PUBLIC_URL=http://square.local:8765   # the server's LAN name or IP
```

The **Add a display** wizard's first step checks exactly this and shows these
lines when they are missing. `DESK_DISPLAY_SERVER_PUBLIC_URL` is the address
written into each `.env.client`; without it the wizard uses the name your
browser used to reach the config UI. Over plain
HTTP the server warns at startup and each client needs
`DESK_DISPLAY_ALLOW_INSECURE_TRANSPORT=1`, which is fine on a trusted home
network. For HTTPS see "Transport security" in
[CONFIGURATION.md](CONFIGURATION.md#transport-security).

### Assigning playlists

Playlists are edited on `/playlists` and assigned on `/clients`. Each
playlist card in the library reads "N active of M screens": the screens the
playlist actually plays (a frequency above zero, or an alternate with an
alternate frequency above zero, and not past its hide-after time) out of all
the screens it lists. Each assignment is checked against what the client
last saw, so two operators cannot overwrite each other. The display's
**Delivery** tab shows whether the new revision has been delivered and
acknowledged, and the display's card warns about screens it can show only
as a still, or not at all.

The Clients page shows one card per display with its state, playlist and a
one-line summary. Its **Settings**, **Delivery** and **Maintenance** tabs
hold the rest; which tab is open is remembered per browser.

### Scroll speed per display

The Rotation Config page's **Synchronized vertical scroll adjustment** is the
default for every display. To give one display its own, type a value in the
**Vertical scroll** field in the display's **Settings** tab on `/clients` and press **Save** (0.25 is
25% faster than normal, -0.25 is 25% slower, from -0.9 to 3). Leave it empty
or press **Use global** to follow the Rotation Config value again. The
setting is stored in the server's playlist store and reaches the display in
its manifest on its next sync, so nothing on the Pi needs editing. It changes
standings, scoreboards, On This Day, schedules and playoff screens, not
tickers or logo animations.

The server still renders each screen once per screen type, paced with the
global value, and records that value in the scroll package; a display with
its own value re-paces playback from it. Screenshots are unaffected, since
a scroll's screenshot is its whole canvas. A display running software older
than this feature ignores the setting until it is updated. A standalone
install has only the Rotation Config value, which is already per device.

### Weather location per display

Every display shows the weather for the server's `WEATHER_LATITUDE` /
`WEATHER_LONGITUDE`. A display somewhere else can have its own: type the
**Lat** and **Lon** (decimal degrees, west and south negative) in the
display's **Settings** tab on `/clients` and press **Save**. Clear both fields or press **Use server's** to
follow the server again. Nothing on the Pi needs editing, and the display
does not need an update.

The server fetches weather (and air quality, when enabled) once per distinct
location, on the same interval as its own, and renders that location's
screens once for every display there. It covers the weather screens, the
weather quad, weather radar (centred on the location), air quality and Sun &
Moon (which also prints the location's coordinates). The display's
side-display (OLED) temperature follows it too. Right after saving, that
display's weather screens are skipped until the first fetch for the new
location arrives, usually within a minute. Each location keeps its own
pressure trend, weather history and AQI history files beside the server's
(`pressure_history.<lat>_<lon>.json` and so on).

A display set to exactly the server's own location shares the server's
renders, so it costs no extra fetches.

The first time the render server starts after this was added, it filled in
the locations once for every display it knew (hyper at 41.9037, -87.6357,
every other display at 42.1373, -87.8446; see
`remote_display/location_seed.py`). A display that already had a location
kept it, and it never runs again, so later edits on the Clients page stick.
A display added later uses the server's `WEATHER_LATITUDE` /
`WEATHER_LONGITUDE` until you set its own.

Times on these screens (sunrise, hourly forecast) are still shown in the
server's time zone. A generic **quad** page with weather tiles uses the
server's location.

### Updating or restarting a client from the Clients page

Each display's **Maintenance** tab on `/clients` has **Update (git pull)**,
**Upgrade**, **Restart client**, **Reset Screenshots** and **Clear caches**
buttons. Each asks for confirmation. The config UI queues the command; the
client collects it with its next heartbeat (about once a minute), runs it,
and reports back on a later heartbeat, so the result and the command's output
show in the tab a minute or two after pressing.

- **Update (git pull)** runs `git pull --ff-only` in the display's own
  desk_display folder as the service user. It only changes the code on disk:
  press **Restart client** afterwards to run it. It does not install new
  dependencies or rewrite service units; **Upgrade** does. A pull that cannot
  fast-forward (local edits, files owned by root) fails with git's own message.
- **Upgrade** runs `scripts/upgrade.sh` on the display, exactly as you would
  by hand: pull, dependencies, service units, then a restart of every Desk
  Display service on it (the server too on a combined install). It runs in a
  transient systemd unit (`sudo -n systemd-run`, as the service user) so the
  restart it ends with cannot kill it, and its log is kept in the client
  cache's `command_jobs/` folder until the client reports it. It needs the
  passwordless sudo `upgrade.sh` already relies on; without it the tab shows
  "Could not start the upgrade". Upgrades are given two hours to report.
- **Restart client** makes `desk_display_client` exit; systemd starts it again
  after 5 seconds (`Restart=always`), so no sudo is needed. The tab shows
  "done" only once the restarted client has reported in.
- **Reset Screenshots** runs `scripts/reset_screenshots.sh`, which empties the
  display's screenshots and screenshot archive folders.
- **Clear caches** runs `scripts/clear-caches.sh` on the display, which
  empties the pip download cache and runs `apt-get clean`. It does not touch
  the display's own content cache (`cache/client/`).

Both scripts run as the service user with no terminal, so a step that falls
back to `sudo` works only where sudo needs no password; anything they could
not do shows in their output.

A client only runs these fixed actions, never a command line sent over
the network, and only when the command reached it through its own
authenticated heartbeat. Commands nobody answers expire after 15 minutes,
which is also what a client running software older than this feature shows:
run `bash scripts/upgrade.sh` on such a display once. Queued commands and
their output are kept in `.runtime/server/client_commands.json`
(`DESK_DISPLAY_CLIENT_COMMANDS_PATH`). If the config UI has a login, the
buttons need it too.

### Indoor sensor on a client

The `inside` screen is drawn by each display from its own sensor; the server
renders nothing for it. On a client with a sensor, set the same keys a
standalone display uses in `.env.client`: `INSIDE_SENSOR` (the sensor type,
for example `pimoroni_bme280`), `INSIDE_I2C_BUSES` (the bus or buses to
probe) and, optionally, `INSIDE_I2C_ADDRESS` (for example `0x76`; leave it
empty to try the sensor's usual addresses). Install the sensor drivers with
`bash scripts/update_dependencies.sh` after setting
`INSIDE_SENSOR`, add `inside` to the display's playlist, and restart the
client service. A client without a sensor skips the screen.

To check it, run `i2cdetect -y <bus>` and look for the address, then watch
the client log (`journalctl -u desk_display_client.service -f`) for
`Indoor sensor found; this display draws the inside screen` or
`No indoor sensor on this display`. The sensor is probed the first time the
playlist includes `inside`.

### Diagnosing a stale client

On `/clients`, a client is *stale* when its last heartbeat is older than
1.5 heartbeat intervals (the server's interval is a third of
`DESK_DISPLAY_CLIENT_LEASE_SECONDS`, so 150 seconds by default), and
*expired* when its lease has lapsed. A client runs a full sync every
`DESK_DISPLAY_SYNC_INTERVAL_SECONDS` (30 by default) and heartbeats every
`DESK_DISPLAY_HEARTBEAT_INTERVAL_SECONDS` (60 by default); the intervals the
server advertises cap both. From the shell:

```bash
curl -s -H "Authorization: Bearer $ADMIN" http://127.0.0.1:8765/api/v1/admin/status | python3 -m json.tool
```

This shows each client's lease state, its last status (current screen,
playback state, sync and cache age, recent errors) and its demand. On the
client itself:

```bash
sudo journalctl -u desk_display_client.service -n 100
```

Look for `Sync failed (...); retrying in Ns` (network or authentication
trouble) and `Not activating yet: artifacts not yet usable` (the server has
not finished rendering something the new playlist needs). A 401 in the log
usually means the credential was rotated or revoked: install the new
`.env.client`. A 409 `incompatible_protocol_version` means the client and
server need the same release: upgrade the older one.

### Checking delivery speed

Each heartbeat carries timings the client measured itself, so they include
the network between it and the server (a VPN, Wi-Fi). The *Delivery* column
on `/clients` shows them:

- *last sync N ago*: time since the last full sync succeeded. It should stay
  under the sync interval (30 s by default).
- *on screen rendered N ago*: how long ago the server rendered the screen now
  on the panel. This is the end-to-end freshness: render cadence plus
  delivery.
- *heartbeat*, *manifest*, *sync*: round trip of the last heartbeat, the
  last manifest fetch, and the whole last full sync.
- *last download*: files, bytes and time the last sync that fetched
  anything took. Bytes divided by time is the client's effective throughput.
- *failed syncs before the last success*: how many passes failed in a row
  before the connection recovered.

The same values are under `telemetry` in `/api/v1/admin/status`. The client
journal logs each sync that downloads something:

```bash
sudo journalctl -u desk_display_client.service -f | grep -E "Downloaded|Sync failed"
```

The server journal logs every request with its time on the server (for
example `GET /api/v1/clients/<id>/manifest -> 304 3ms`); that number
excludes the network, so a gap between it and the client's figure is the
link. Clients older than this release report no timings ("no timings
reported").

### Checking CPU, data and storage

The **Stats** page (`/stats` on the config UI, port 5002) refreshes every
10 seconds:

- *CPU by purpose*: a stacked chart (last hour, or 5-minute averages over
  24 hours) of what the project's CPU goes to. *Rendering screens* is the
  render worker processes, one set per display profile (each listed in the
  process table as "Render worker · <profile>"). Inside the render server
  each thread is attributed by name: *Serving displays* (the HTTP API),
  *Render scheduling*, *Feed refresh*, *Artifact cleanup*; *Short-lived
  threads* is work in threads that started and ended between two samples,
  mostly feed downloads. The config UI, the local panel client (combined
  installs), a standalone `main.py` and the feed server appear as their own
  processes. Percentages are of one core, like `top`: 100% is a full core,
  and a Pi 4 or 5 has four. The *Project CPU* tile also gives the share of
  the whole machine.
- *Data transferred*: bytes the display API sent to and received from each
  display (now, since the server started, and all time), what
  it was mostly for (artifacts, manifests, heartbeats), and the machine's
  network interfaces. The per-display figures count request and response
  bodies, not HTTP headers or TLS.
- *Storage*: free space on the project's filesystem and the sizes of the
  artifact store (against `DESK_DISPLAY_ARTIFACT_MAX_MB`) and caches.
- *Displays*: each display's own client CPU, machine CPU, free memory,
  temperature, free disk, cache use against `DESK_DISPLAY_CLIENT_CACHE_MAX_MB`
  and traffic since the client started, from its heartbeat. Displays need
  this release to report them.

The render server samples every 10 seconds (a few small `/proc` reads) and
writes the page's data to `/dev/shm`. The charts' history (last hour and
24 hours) and the all-time data totals are saved to
`.runtime/server/stats_history.json` every 5 minutes and when the server
stops, so they survive restarts and reboots; a power cut loses at most the
last 5 minutes, and the chart shows the time the server was down as a gap.
**Reset stats…** (next to the tiles) clears that history and the totals and
starts counting again; the server applies it at its next sample. Set `DESK_DISPLAY_STATS_ENABLED=0` in `.env` to turn
it off. When no render server is publishing (standalone and client installs,
or a stopped server) the page samples the machine itself while it is open:
processes and storage, without per-thread and per-display figures.

### Comparing screens across displays

To see every version of each screen side by side, run this from any
computer on the LAN (only Python 3 is needed):

```bash
python3 scripts/collect_client_screenshots.py --server square.local
```

It asks the server's config UI (port 5002) for the display clients, then
fetches the latest screenshots from each online client's own config UI
(port 5002 on the client, which client-only installs run in screenshots-only
mode). The page, saved to the Desktop as
`desk_display_screenshots_<date-time>.html` with every image embedded, has
one heading per screen and each display's screenshot under it, labelled
with the display's name, profile and size. Displays that are offline or
can't be reached are listed at the top as skipped.

Clients are reached at the address they last connected to the server from
(a combined server's own panel at the server's address). A client that has
not sent a heartbeat since the server was upgraded has no recorded address,
so the script tries `<client id>.local`; `--client-host ID=HOST` sets one by
hand. If the config UI has a password (`SCREEN_UI_PASSWORD`), pass
`--password` or type it when asked; the same password is tried on every
display. `--output` picks the file, `--include-inactive` also tries offline
clients.

#### Displays on another network

A display that is not on the server's network (it reaches the server over a
VPN or through a forwarded port) can't be fetched from directly. Have it
upload its screenshots to the server instead, over the connection it already
uses:

1. On the display, add to `.env.client` and restart the client:

   ```bash
   DESK_DISPLAY_CLIENT_UPLOAD_SCREENSHOTS=1
   # Optional: upload each screen at most every N minutes (default 10).
   DESK_DISPLAY_CLIENT_SCREENSHOT_UPLOAD_MINUTES=10
   ```

   ```bash
   sudo systemctl restart desk_display_client.service
   ```

   It needs `ENABLE_SCREENSHOTS=1` (the default). After each heartbeat the
   client sends up to 10 screenshots that are due, authenticated with its own
   client credential, so no extra password, port or SSH key is needed.
2. On the server nothing needs setting: uploads are on by default and land in
   `.runtime/server/client_screenshots/<client id>/`, one PNG per screen
   (the latest only) plus an `index.json`. The total is capped at 64 MB, and a
   screen not re-uploaded for 7 days is deleted (see the
   `DESK_DISPLAY_SCREENSHOT_UPLOAD_*` settings in `.env.server.example`).
   The server must be running this release or later; restart it after
   upgrading so it advertises uploads.
3. Run the collector as usual. When a display can't be reached it uses the
   screenshots that display uploaded (listed on the server's config UI at
   `/api/clients/<id>/uploaded-screenshots`), and it also fills in any
   screen a reachable display did not return. Uploaded screenshots are
   captioned "uploaded to server".

To check uploads are arriving, look on the server:

```bash
ls -l ~/desk_display/.runtime/server/client_screenshots/<client id>/
sudo journalctl -u desk_display_server.service | grep screenshots
```

### Checking scroll smoothness

A client plays scrolling screens itself, one step per frame at the rate the
panel can take, as v0.1 did; a slow panel (a Pi Zero 2 W pushing 320x240
over SPI) scrolls a little slower but never skips pixels. To measure it,
stop the client and play a test scroll through the real display driver:

```bash
sudo systemctl stop desk_display_client.service
python3 scripts/measure_scroll_fps.py            # current playback
python3 scripts/measure_scroll_fps.py --legacy   # the older, choppier loop
bash scripts/restart_services.sh
```

It prints frames per second, frame intervals and *px per frame*. Smooth
playback shows a single value there (1px on a Display HAT Mini); several
values mean the picture jumped unevenly.

### Diagnosing stale renders

```bash
curl -s -H "Authorization: Bearer $ADMIN" http://127.0.0.1:8765/api/v1/admin/render-status | python3 -m json.tool
```

This shows the render queue with the reason and wait time for each job, jobs
in flight, and per-artifact state: last success, last failure and error,
and consecutive failures. It also shows `data_health`, which marks a feed
stale when it has not updated for twice its interval. A failing render keeps
serving its last good output (manifest state `fallback`) and backs off up to
15 minutes. The server journal logs `Render of <screen> for <profile>
failed (...)`. Check the provider credential, or the feed that
`data_health` names.

The server draws each display profile in its own child process, started the
first time a client with that profile needs a render and configured the way
the v0.1 installer configured that display, so fonts and layouts match v0.1.
Its log lines carry `[render <pid>]`. `ps -ef | grep rendering.profile_process`
lists them (up to `DESK_DISPLAY_RENDER_WORKERS` per profile in use, so renders for one
profile run side by side). A worker that dies is started again on
the next render; restarting `desk_display_server.service` restarts them all.

### Logs and caches

| Where | What |
| --- | --- |
| `sudo journalctl -u <unit>` | All logs; secrets are redacted before they are written |
| `.runtime/server/` | Playlists, assignments, known clients, credential hashes, migration bundles and upgrade snapshots |
| `.runtime/server/stats_history.json` | The Stats page's 1-hour and 24-hour history and all-time data totals (`DESK_DISPLAY_SERVER_STATS_HISTORY_PATH`; saved every 5 minutes and on stop; **Reset stats…** clears it) |
| `.runtime/server/client_screenshots/` | Screenshots displays on another network uploaded (`DESK_DISPLAY_SCREENSHOT_UPLOAD_MAX_MB`, default 64; safe to delete, not backed up) |
| `cache/artifacts/` | Rendered artifacts (`DESK_DISPLAY_ARTIFACT_MAX_MB`, default 1024; safe to delete, the server re-renders) |
| `cache/` | Feed caches and history (`DESK_DISPLAY_SERVER_CACHE_MAX_MB`, default 1536; going over only logs a warning) |
| `images/cache/` (server) | Radar map tiles, downloaded logos and headshots (`DESK_DISPLAY_IMAGE_CACHE_MAX_MB`, default 256; least recently used files are deleted and downloaded again when needed) |
| `cache/client/` (client) | The offline cache: playlists, manifests, artifacts, lease credential (`DESK_DISPLAY_CLIENT_CACHE_MAX_MB`, default 512) |

Deleting `cache/client/` makes a client start from its diagnostic screen and
download everything again.

### Backup and restore

`scripts/upgrade.sh` snapshots a server before each upgrade. To take a
snapshot now:

```bash
python3 install_modes.py snapshot          # prints .runtime/server/backups/upgrade-<time>/
```

A snapshot holds `.env` (and `.env.client` in a combined install),
`screens_config.local.json`, and `playlists.json`,
`provisioned_clients.json` and `clients.json` from `.runtime/server/`, with
`/` replaced by `__` in their names. It does not include `~/keys`, the
migration bundles or older snapshots. To restore one, stop the services,
restore it, and start them again. The restore takes a snapshot of the
current state first, so it can be undone the same way. It restores only the
files the installed mode keeps (pass `--mode` to override the detected
mode):

```bash
sudo systemctl stop desk_display_server.service config_ui_desk_display.service
# combined installs: also stop desk_display_client.service
python3 install_modes.py restore .runtime/server/backups/upgrade-<time>
./scripts/restart_services.sh
```

Clients keep their credentials across a restore and enroll again on their
own. A client provisioned or rotated after the snapshot was taken is not in
the restored credential list; rotate it again and install its new
`.env.client`.

For a client, the only thing to keep is `.env.client`. The uninstaller keeps
everything listed in [What each mode keeps](#what-each-mode-keeps).

### Release validation

Before a release, `tests/test_end_to_end.py` runs a real render server and
clients over the wire with fixture data and no network, and
`soak.py` records a multi-day soak on the real devices and judges it
against release gates and rollback triggers. The procedure is in
[docs/soak-and-release.md](docs/soak-and-release.md):

```bash
python -m pytest -q tests/test_end_to_end.py
python3 soak.py sample --out soak/$(hostname).jsonl --hours 48
python3 soak.py gates soak/*.jsonl --clients office,den
```

### Operating through an outage

- **Server down:** clients keep playing their cached rotation, and clocks
  keep time. Clients reconnect with backoff (up to 5 minutes between
  tries). Set `DESK_DISPLAY_OFFLINE_MAX_AGE_HOURS` on a client to make it
  stop showing content older than that.
- **Provider down:** the server keeps the last good data and artifacts, and
  `render-status` shows the feed as stale in `data_health`.
- **Client rebooted while the server is down:** it starts from its cache.
  With an empty cache it shows its diagnostic screen until the server
  returns.

### Rolling back

- **Configuration:** restore the `.env.bak-<time>` that
  `scripts/convert_env.py` or the installer left, or a snapshot as described
  above.
- **Migrated rotation:** undo it with
  `python3 scripts/migrate_standalone_config.py --rollback <bundle>`.
- **Application:** check out the previous release and run
  `bash scripts/upgrade.sh --no-pull`, which reinstalls its dependencies and
  units and keeps all data:

```bash
git fetch --tags
git switch --detach <previous-release>
bash scripts/upgrade.sh --no-pull
```

### Upgrading a device from v0.1

A device still on `v0.1` runs the standalone `desk_display.service`. After
`git pull` it can stay standalone, or become a thin client of a server Pi.
Run everything from the project directory with your normal login.

**0. Pull cleanly.** If `git pull` fails with `Permission denied` or
`insufficient permission for adding an object`, an earlier `sudo` left files
owned by root. Give them back, then pull again:

```bash
sudo chown -R "$USER:$USER" ~/desk_display
git pull
```

If it refuses because of local changes to a tracked file, `git stash` first.

**To keep the device standalone**, reinstall its dependencies, rewrite its
units and restart:

```bash
bash scripts/upgrade.sh --no-pull
```

`.env`, `screens_config.local.json` and `screens_style.json` are kept as they
are.

**To make it a client of the server**, use the server's **Add a display**
setup (see [Provisioning, rotation and revocation](#provisioning-rotation-and-revocation)):

1. Keep a copy of this device's settings: `cp .env ~/env.v0.1.bak`.
2. Optional: to keep this device's own rotation, copy its active rotation
   file (`screens_config.local.json`, or `screens_config.json` when there is
   no local file) to the server and turn it into a playlist there, preview
   first and then with `--apply`:

   ```bash
   scp screens_config.local.json <server>:~/den-rotation.json      # on this device
   python3 scripts/migrate_standalone_config.py --config ~/den-rotation.json --name Den  # on the server
   ```

3. On the server's `/clients` page, click **Add a display**. Its first step
   shows the `.env` lines the server needs before other devices can reach it
   (see [Letting other displays reach the server](#letting-other-displays-reach-the-server)).
   Then pick this device's screen type, name and playlist.
4. Paste the one-time command it shows on this device. If the checkout is
   still v0.1, the setup script runs `git pull` first; if that fails it stops
   with the fix (the `chown` above, or `git stash`), and the display's row
   then offers a new **Setup command**. It then writes `.env.client` from the
   panel settings in `.env` plus the new credential, installs the client
   requirements, and disables `desk_display.service` and
   `config_ui_desk_display.service`.
5. The setup page says when the display comes online. If it does not, run
   `sudo journalctl -u desk_display_client.service -f` on the device.

**Set it up by hand instead** on the setup page gives a `.env.client`; save it
on the device as, say, `~/den.env.client` and run
`bash Installers/install.sh --mode client --credentials ~/den.env.client <profile>`.

### Restoring v0.1

`v0.1` (commit `9e193dc`) is the standalone release before the server/client
work. To return a device to it:

1. Keep your data. On a server, run `python3 install_modes.py snapshot`. On
   any device, copy `.env` or `.env.client` somewhere safe.
2. Stop and disable the server/client units, so only the standalone unit
   remains:

   ```bash
   sudo systemctl disable --now desk_display_server.service desk_display_client.service
   ```

3. Check out the release on a branch:

   ```bash
   git fetch --tags
   git switch -c restore-v0.1 v0.1
   ```

4. Put back a standalone `.env`. A server's pre-conversion original is its
   `.env.bak-<time>`. Otherwise start from `.env.example` and drop
   `DESK_DISPLAY_ROLE`, which v0.1 does not read.
5. Reinstall with v0.1's own installer, which has no `--mode` flag and
   installs the standalone service:

   ```bash
   bash ./Installers/install.sh <profile>
   ```

To go back to standalone on the current release instead, skip step 3 and run
`bash ./Installers/install.sh --mode standalone <profile>`.

Server playlists do not exist in v0.1. The rotation it plays is
`screens_config.json`, or `screens_config.local.json` when present. Neither
is changed by the server/client work, so the v0.1 display plays what it
played before. Test the rollback on a spare device first. `tests/test_docs.py`
checks that these steps stay documented.

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

It only ever touches the ten units listed above, skips the ones that are not
installed here, and restarts in its own dependency order regardless of the order
you type.

To upgrade, in any mode:

```bash
bash scripts/upgrade.sh              # git pull, dependencies, units, restart
```

It detects the installed mode and then does the equivalent of:

```bash
git pull --ff-only
./scripts/update_dependencies.sh --requirements <the mode's file>
./scripts/update_services.sh --no-restart  # standalone: patch the unit in place
                                           # other modes: rewrite the units as installed
./scripts/restart_services.sh
```

An upgrade never changes the data the mode keeps (see "What each mode keeps"),
and a server or combined install first snapshots its state.

`update_services.sh` is the safe way to repair installed units in any mode (add
`--dry-run` to preview). On a server, client or combined install it rewrites the
mode's units from `service_units.py` with the user, display output and
`Environment=` overrides recorded at install, adds missing ones, and disables
units that belong to another mode. On a standalone install it rewrites script
paths that moved in the repository and applies the current shutdown settings,
but leaves the display profile and every `Environment=` override alone. It ends
by listing every project unit with whether it is enabled and running.
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

On a server or combined install, displays play the playlists on the
**Playlists** page (`/playlists`) instead, edited with the same editor and
assigned on `/clients` (see [Assigning playlists](#assigning-playlists)).
The Rotation Config page still sets the scroll speeds every display uses.

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
them, so the screen works across the 240×135 through 1920×1080 profiles. Fetching
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
to `config.py`, document them in `.env.example`, and add them to the settings
catalog in `deployment_config.py` (role, and whether they are secret), then
regenerate `.env.server.example`, `.env.client.example` and the
CONFIGURATION.md reference table as
[CONFIGURATION.md](CONFIGURATION.md#checking-a-configuration) describes.

Then run the checks before pushing:

```bash
python scripts/test_all.py                    # the canonical CI check: Ruff, then pytest
python scripts/validate_required_files.py     # catches a module nothing imports
python scripts/render_screens.py              # renders screens for a visual look
```

**6. For the render server**, give the screen a class in `_TABLE` in
`rendering/screen_classes.py` (`static`, `scrolling_canvas` and so on; see
[CONFIGURATION.md](CONFIGURATION.md#screen-classes-and-render-packages)), or
`tests/test_render_packages.py` fails. If it shows data from a feed the
server already refreshes, add it to that feed's set in `services/feeds.py`
(`FEED_DEPENDENCIES`, or `LEAGUE_STANDINGS_DEPENDENCIES` for league tables
and brackets) so a refresh re-renders it.

Sizes for the 1080p HDMI profile go behind `config.is_hdmi_1080p_layout()`
as their own constants; the v0.1 profiles' fonts and layout constants are
checked against recorded references by `tests/test_v01_parity.py`, so leave
them alone.

`tests/test_screens_catalog.py`, `tests/test_screen_registry.py` and
`tests/test_default_screen_configs.py` are where screen-level tests go. CI runs
`python scripts/test_all.py` on every pull request.
