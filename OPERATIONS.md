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

---

## Installation modes (server and client)

Besides the standalone display, Desk Display can run as a render server,
a display client, or both on one machine. `service_units.py` defines the
systemd services for each mode:

| Mode | Services | Env file |
| --- | --- | --- |
| standalone | `desk_display.service` (main.py), config UI | `.env` |
| server | `desk_display_server.service`, config UI | `.env` |
| client | `desk_display_client.service` | `.env.client` |
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
client core (`requests`, `pytz`, `Pillow`) plus its panel driver, and no
upstream provider library, web server or SVG renderer.

The installer prepares the env files before it starts anything:

- A client's `.env.client` starts from the panel settings in `.env`
  (profile, rotation, backlight, sensors), converted with
  `scripts/convert_env.py` so no server or provider setting survives, plus
  the identity and credential from `--credentials` (the file the server's
  "Add a display" form gives you). Without `--credentials`, fill in the
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
changed. `--export FILE` writes one as a backup without changing anything,
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
issues a credential and shows the client's `.env.client` once, and each
client row has Rotate, Revoke, Disable and Enable. From the shell:

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

### Assigning playlists

Playlists are edited on `/playlists` and assigned on `/clients`. Each
assignment is checked against what the client last saw, so two operators
cannot overwrite each other. The row shows whether the new revision has been
delivered and acknowledged, and warns about screens this client can show
only as a still, or not at all.

### Diagnosing a stale client

On `/clients`, a client is *stale* when its last heartbeat is older than
1.5 heartbeat intervals (the server's interval is a third of
`DESK_DISPLAY_CLIENT_LEASE_SECONDS`, so 150 seconds by default), and
*expired* when its lease has lapsed. A client sends one heartbeat per sync,
every `DESK_DISPLAY_SYNC_INTERVAL_SECONDS` (30 by default); it does not use
the intervals the server advertises, so a sync interval longer than half the
lease makes a healthy client look stale. From the shell:

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

### Logs and caches

| Where | What |
| --- | --- |
| `sudo journalctl -u <unit>` | All logs; secrets are redacted before they are written |
| `.runtime/server/` | Playlists, assignments, known clients, credential hashes, migration bundles and upgrade snapshots |
| `cache/artifacts/` | Rendered artifacts (`DESK_DISPLAY_ARTIFACT_MAX_MB`, default 512; safe to delete, the server re-renders) |
| `cache/` | Feed caches and history |
| `cache/client/` (client) | The offline cache: playlists, manifests, artifacts, lease credential (`DESK_DISPLAY_CLIENT_CACHE_MAX_MB`, default 256) |

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
