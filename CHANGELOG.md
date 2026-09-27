# Changelog

All notable changes to Desk Display are documented in this file.

## Unreleased — server/client architecture

Desk Display can now run as a render server with any number of thin display
clients, or both on one machine, alongside the unchanged standalone install.
See the README's [deployment modes](README.md#deployment-modes) and
[OPERATIONS.md](OPERATIONS.md) for installing, upgrading and rolling back.

- Role-specific configuration (`.env.server.example`, `.env.client.example`)
  validated at startup, with secrets excluded from every client-facing payload.
- `display_server.py`: registration and leases, per-client credentials
  (provisioned by default), rate limits, manifests and content-addressed
  artifacts, and a render coordinator that renders each artifact once for
  every client that shares it.
- Server-managed playlists and assignments in the configuration UI, with
  per-client capability warnings, provisioning, rotation and revocation.
- `display_client.py`: offline-first playback from a local cache, render
  packages (static, animated and client-timed clocks), touch focus on quads,
  and physical rotation applied only at presentation.
- Migration: `scripts/migrate_standalone_config.py` moves a standalone rotation
  onto the server, and `scripts/convert_env.py` converts an existing `.env`.
- Installers for the server, client and combined modes, `scripts/upgrade.sh`,
  and mode-aware uninstall and cleanup. `install_modes.py` records the
  installed mode and answers what each mode installs, keeps and backs up.
- Operator documentation for every mode: the README's deployment modes,
  the server and client runbook in [OPERATIONS.md](OPERATIONS.md), the
  [wire protocol](docs/remote-display-protocol.md) and
  [render packages](docs/render-packages.md). `tests/test_docs.py` checks that
  links, named files and service names in these documents stay correct.
- A display client loads `.env.client` (it falls back to `.env` only when
  there is no `.env.client`), so in a combined install it never loads the
  server's `.env` and its provider credentials.
- End-to-end validation (`tests/test_end_to_end.py`): a real render server and
  real clients over the wire, offline, checking pixel parity with the
  standalone display, shared and different playlists and profiles, playlist
  acknowledgment, rotations, restarts, a long outage, stale data, failed
  renders, a corrupt client cache, revocation, incompatible versions, that no
  provider credential reaches a client, and a server backup and restore.
- The maintenance scripts follow the installed mode: `scripts/update_services.sh`
  rewrites a server, client or combined install's units (keeping the recorded
  user, output and overrides), adds missing ones, disables another mode's, and
  lists every unit's state; `scripts/update_dependencies.sh` installs the mode's
  requirements; `scripts/reset_screenshots.sh` follows `SCREENSHOT_DIR` and
  `SCREENSHOT_ARCHIVE_BASE`; and the setup checks and LED test look at the
  client's panel service. Every script in `scripts/` is executable.
- `python3 install_modes.py restore <snapshot>` puts back an upgrade snapshot,
  after taking a snapshot of the current state so the restore can be undone.
- `soak.py` records soak samples and judges them against release gates and
  rollback triggers; [docs/soak-and-release.md](docs/soak-and-release.md) is
  the runbook for the hardware soak.
- Fix: on 1-bit displays, scroll canvases, ticker strips and quad tiles were
  dithered once as a whole, so scrolled frames differed from the standalone
  display. Packages now keep these images in colour and the client dithers
  each frame it shows.
- Fix: unassigning a client now stops the server rendering for it. The client
  keeps playing its cached playlist but no longer reports it as demand, and
  the server ignores the demand an unassigned client last reported.
- Fix: the client honours the `heartbeat_interval_seconds` and
  `sync_interval_seconds` the server advertises as caps on its own settings,
  and uses `DESK_DISPLAY_HEARTBEAT_INTERVAL_SECONDS` for heartbeat-only
  passes between syncs, so its lease cannot lapse between syncs.
- Fix: the client honours the `Retry-After` header as well as
  `retry_after_seconds`, waiting for the larger (at most an hour).
- Fix: a cached artifact is checked against its SHA-256, not only its length,
  before a sync treats it as present, so a same-length corruption is
  downloaded again in that sync.
- Fix: at startup a cached playlist is only paired with the manifest it was
  activated with. When that manifest is missing, the previous playlist and
  its manifest play instead of an unrelated newer manifest.
- Fix: display clients apply `DARK_HOURS`, `DESK_DISPLAY_BACKLIGHT_LEVEL`,
  `DESK_DISPLAY_DARK_HOURS_MODE` and `DESK_DISPLAY_DARK_HOURS_BACKLIGHT_LEVEL`,
  and report `playback_state: dark` while blanked.
- Fix: `DESK_DISPLAY_CONTENT_TIMEZONE` is used. The server renders dates,
  schedules and clock packages in it, and the standalone display and clients
  read dark hours in it (default America/Chicago, as before).
- Fix: `DESK_DISPLAY_OFFLINE_START=0` makes a client wait for its first sync
  after start-up before playing its cache.
- Fix: fonts and layouts match v0.1 again on every display. The server drew
  every profile in one process and substituted each display's sizes at render
  time, which missed most of the values screens derive from the display size
  when they load, so text came out too big or too small. It now composes each
  profile in its own worker process (`rendering/profile_process.py`),
  configured the way the v0.1 installer configured that display, and a
  display client configures itself for its panel before drawing its clock.
  `tests/test_v01_parity.py` compares fonts, layout constants and renders for
  every v0.1 display with references recorded from the `v0.1` tag
  (`scripts/make_v01_references.py`).
- Fix: a display client's `date` clock cycles its colours again as v0.1 did,
  drawing fresh colours at the display's own pace after the screen appears
  and then holding them, instead of keeping one pair of colours.
- Fix: a feed refresh that returns unchanged data (for example from a TTL
  cache) no longer bumps its data revision, so it no longer re-renders the
  screens that read it.

### Known limitations

- `DESK_DISPLAY_CLIENT_NAME` is ignored; the server identifies clients by
  client ID.
- The `inside` screen reads a sensor attached to the display and is not
  available on remote clients.
- Physical display, touch, button and systemd behavior must still be checked
  on the real hardware in the Phase 20b soak.

## v0.1 — 2026-09-25

This release preserves the pre-server/client standalone Desk Display baseline.

- Commit: `9e193dc22ce0caa88a66493dafa42f90ae3b0db9`.
- Tag: `v0.1` is a lightweight tag (it points straight at the commit, with no
  tag message or signature) and has a hosted GitHub release. It is never moved
  or reused. [Restoring v0.1](OPERATIONS.md#restoring-v01) explains how to go
  back to it.

### Included

- A standalone, always-on dashboard application for Raspberry Pi, Linux, macOS,
  and Windows.
- Output profiles for Pimoroni Display HAT Mini, Adafruit miniPiTFT, Waveshare
  OLED/LCD HAT (A), HyperPixel/kernel displays, HDMI, SDL windows, direct
  framebuffers, and hardware-independent headless rendering.
- Configurable weather, date/time, indoor sensor, finance, travel, news, sports
  schedule, scoreboard, standings, playoff, ADS-B, and quad-layout screens.
- A browser-based configuration UI for screen rotation, playlists, layouts,
  import/export, diagnostics, and screenshot review.
- The standalone Feed service for publishing screenshot feeds uploaded by Desk
  Display devices.

### Known limitations

- Physical display, GPIO/button/touch, framebuffer, I2C sensor, and systemd
  installer behavior must be validated on the corresponding Raspberry Pi and
  display hardware; these checks cannot be completed in headless CI.
- Live weather, maps, finance, news, travel, sports, and Feed upload behavior
  depends on network availability, third-party services, and any required API
  credentials.
- The application remains a standalone deployment in this baseline. It does
  not include the later server/client architecture.
