# Changelog

All notable changes to Desk Display are documented in this file.

## Unreleased — server/client architecture

Desk Display can now run as a render server with any number of thin display
clients, or both on one machine, alongside the unchanged standalone install.
See the README's [deployment modes](README.md#deployment-modes) and
[OPERATIONS.md](OPERATIONS.md) for installing, upgrading and rolling back.

- New **Live** page on the server's config UI (`/live`): every online
  display's latest screenshot of each screen, grouped by screen or by
  display, at each display's own shape, with capture ages, refreshed every
  5 seconds without reloading. It shows what
  `scripts/collect_client_screenshots.py` collects, kept current: the server
  fetches each display's screenshot list from its config UI (reused for
  4 seconds, and a display that does not answer is retried after 20),
  proxies the images, cached by capture time, reads its own panel directly,
  and falls back to the screenshots a display on another network uploads.
  Linked from every page's navigation on servers.
- New **pi remote temp** screen: the CPU temperatures of the LAN's
  Raspberry Pis, drawn like the MMM-RemoteTempMonitor MagicMirror module
  (its uppercase header, the Device / °C / °F table, hottest first, names
  with Pi model and RAM, temperatures in the module's green → yellow-green →
  orange → red → purple scale with its glow). The render server reads the
  module's aggregate endpoint, `http://REMOTE_TEMP_MONITOR_HOST:REMOTE_TEMP_MONITOR_PORT/temps`
  (default `192.168.1.201:9877`), every minute as the `remote_temps` feed; an
  unreachable MagicMirror keeps the last good temperatures and the screen says
  they are cached. Laid out for every display profile, including 1080p. Off
  (frequency 0) in the default configs, in the sensors playlist after inside.
- The MagicMirror² module (MMM-desk_display) now animates every moving
  screen: weather radar and the league overviews loop their frames, news
  tickers scroll, and quads step their tiles, as on a desk_display client.
  It registers with `supports_animation: true`, so the Clients page no
  longer warns that its screens show their still image.
- Upgrades keep a display client you switched off off:
  `update_services.sh` (run by `upgrade.sh` and the Clients page's Upgrade)
  no longer re-enables a disabled `desk_display_client.service` (or any other
  disabled mode unit) on a recorded install, never restarts it, and leaves a
  masked one alone. Use `sudo systemctl disable --now desk_display_client.service`.
- Display clients use less CPU: a quad whose tiles do not change is drawn
  once instead of 10 times a second, each panel frame is copied once instead
  of up to three times, and a screenshot is encoded once (fast PNG) instead
  of twice. New `DESK_DISPLAY_CLIENT_MAX_FPS` (`.env.client`, default `0`,
  no cap) caps animation frames per second; see OPERATIONS.md "Reducing a
  client's CPU use".
- `LED_INDICATOR_PULSE` (`.env` / `.env.client`, default `0`): the notification
  border pulses gently instead of staying static, on standalone, client and
  combined installs. Saved screenshots keep the full-brightness border.
- Role-specific configuration (`.env.server.example`, `.env.client.example`)
  validated at startup, with secrets excluded from every client-facing payload.
- `display_server.py`: registration and leases, per-client credentials
  (provisioned by default), rate limits, manifests and content-addressed
  artifacts, and a render coordinator that renders each artifact once for
  every client that shares it.
- Server-managed playlists and assignments in the configuration UI, with
  per-client capability warnings, provisioning, rotation and revocation.
- `scripts/upgrade.sh` asks at the end whether to start
  `desk_display_client.service`, defaulting to its state before the upgrade
  (used as is without a terminal); `restart_services.sh --skip <unit>`.
- The Clients page's Maintenance tab can remove a display: its credential,
  assignment, settings and the server's record go, and its card leaves the page.
- The Playlists page edits each client playlist with the Screen Rotation
  Config page's editor (screen and playlist order, frequency, hide-after,
  alternates, extra seconds, Collapse all / Expand all); raw JSON editing
  moved under **Advanced**.
- A guided **Add a display** setup on the Clients page: it checks the render
  server is reachable from the LAN (with the `.env` lines to fix it), names
  the display and picks its screen type and playlist, gives one one-time
  command (`POST /api/v1/join`) that installs and configures the new Pi, and
  shows when the display comes online.
  On a Pi still on v0.1 the setup script updates the checkout first.
  The Waveshare OLED/LCD HAT (A) is one choice there: its LCD plays the
  playlist and its two side OLEDs run the status helper from the client's
  heartbeat.
- **Upgrade**, **Reset Screenshots** and **Clear caches** buttons in each
  display's Maintenance tab on the Clients page. They run `scripts/upgrade.sh`
  (in a transient systemd unit, so it survives the restart it ends with),
  `scripts/reset_screenshots.sh` and `scripts/clear-caches.sh` on that display
  and show the output like Update and Restart.
- **Update (git pull)** and **Restart client** buttons on each row of the
  Clients page. The client collects the command on its next authenticated
  heartbeat, runs one of those two fixed actions (`git pull --ff-only` in its
  checkout, or exiting so systemd restarts it) and reports the result, which
  shows in the row (see OPERATIONS.md, "Updating or restarting a client from
  the Clients page").
- Per-display **weather location** on each row of the Clients page: a display
  in another place can have its own latitude and longitude instead of the
  server's `WEATHER_LATITUDE` / `WEATHER_LONGITUDE` (both empty, or **Use
  server's**, follows the server). The server fetches weather and air quality
  once per location and renders that location's weather, radar, air quality
  and Sun & Moon screens once for all displays there (see OPERATIONS.md,
  "Weather location per display"). On its first start after the update the
  server fills in each known display's location once: hyper 41.9037,
  -87.6357, every other display 42.1373, -87.8446.
- Per-display **Vertical scroll** adjustment on each row of the Clients page:
  a display can have its own "Synchronized vertical scroll adjustment"
  instead of the Rotation Config page's global value (empty or **Use
  global** follows the global one). It is stored on the server and reaches
  the display in its manifest on the next sync, so no SSH is needed; the
  display re-paces the server's scroll packages to it (see OPERATIONS.md,
  "Scroll speed per display").
- Every display (standalone, client or combined) starts at the top of the
  playlist labelled "Starter" whenever it restarts, instead of resuming a
  saved position.
- Smooth scrolling on slow client panels (Pi Zero 2 W with a Display HAT
  Mini): each frame is pushed to the panel once instead of twice, and motion
  advances one step per frame drawn, as v0.1 did, instead of skipping pixels
  to keep up with the clock. `scripts/measure_scroll_fps.py` measures it on
  the device (see OPERATIONS.md, "Checking scroll smoothness").
- Delivery telemetry: clients report heartbeat and manifest round trips,
  full-sync time, what the last sync downloaded (files, bytes, time), how
  old the screen on the panel is and failed passes in a row. It shows in a
  **Delivery** column on the Clients page and in `/api/v1/admin/status`
  (see OPERATIONS.md, "Checking delivery speed").
- A **Stats** page in the configuration UI (`/stats`): the project's CPU use
  broken down by purpose (rendering per display profile, feed refresh,
  serving displays, render scheduling, config UI, local panel client) with
  per-process and per-thread figures and a 1-hour / 24-hour history; data
  sent to and received from each display (now, since start, all time) and
  the machine's network throughput; free disk space and the artifact store
  and caches against their limits; and each display's own CPU, memory,
  temperature, disk and cache, reported in a new optional heartbeat
  `resources` document (see OPERATIONS.md, "Checking CPU, data and storage").
  Its history and totals survive restarts (saved every 5 minutes and on
  shutdown), and a **Reset stats…** button starts them over.
- `scripts/collect_client_screenshots.py` gathers the latest screenshot of
  every screen from every active display client into one self-contained HTML
  page, grouped by screen, for comparing how each screen looks on each
  display. The server now records the address each client last connected
  from (Clients API and `/api/v1/admin/status`) so the script can find them
  (see OPERATIONS.md, "Comparing screens across displays").
- A display on another network can upload its latest screenshots to the
  server (`DESK_DISPLAY_CLIENT_UPLOAD_SCREENSHOTS=1` in `.env.client`) over its
  existing authenticated connection, and the collector uses them when it
  can't reach that display. The server keeps one image per screen per
  display under `.runtime/server/client_screenshots/`, with size and age
  limits (see OPERATIONS.md, "Displays on another network").
- New **MLB Playoffs** screen: the postseason bracket laid out like
  mlb.com/postseason (AL left, NL right, World Series in the middle, seeds and
  series scores), sized for every display profile, with the current round's
  pairings, series scores and next game or result under it. Before the
  postseason it shows the bracket projected from the standings. The render
  server fetches it as the `mlb_postseason` feed (every 10 minutes, every
  30 seconds during a live game). Off (frequency 0) in the default configs, in
  the mlb playlist after MLB Scoreboard.
- On a **Display HAT Mini**, the **MLB Playoffs** screen leaves out the
  bracket: its seven columns are too small to read on the 320×240 panel. The
  screen keeps the header and the series list, which stays readable at that
  size; every other display still shows the bracket.
- New **traffic** screen: Edens and Kennedy travel times and speeds from
  Travel Midwest's Chicago Quick Traffic report (no key). Every display shows
  the four inbound segments except hyper, which shows the four outbound ones
  (`TRAFFIC_OUTBOUND_DISPLAYS` on the server, `TRAFFIC_DIRECTION` on a
  standalone display). Rows are grouped by road, with travel time large,
  speed beside it, a red bar when Travel Midwest flags a segment as over its
  normal time, and N/A (row kept) for a closed reversible lane. The render
  server fetches the report once every 5 minutes as the `traffic` feed for
  all displays; a failed refresh keeps the last good report and the footer
  says it is cached and how old it is. Laid out for every display profile,
  including 1080p. Off (frequency 0) in the default configs, at the end of
  the weather playlist.
- **NHL Playoffs** and **NBA Playoffs** now use the MLB Playoffs design: a
  16-team bracket (West left, East right, the final in the middle, conference
  logos for the NHL) with the current round's series, scores and next game or
  result under it, sized for every display profile. The render server fetches
  them as the `nhl_playoffs` and `nba_playoffs` feeds (every 10 minutes,
  every 30 seconds during a live game), so they work on clients too. Before
  the NHL playoffs the first round is projected from the standings; out of
  season both show the last playoffs. Their default frequencies are unchanged.
- `display_client.py`: offline-first playback from a local cache, render
  packages (static, animated and client-timed clocks), touch focus on quads,
  and physical rotation applied only at presentation.
- Clients light the notification LED and indicator border as v0.1 did
  (`LED_INDICATOR_ENABLED`, `LED_INDICATOR_BORDER_ENABLED`,
  `LED_INDICATOR_BORDER_WIDTH`): the manifest carries each screen's alert or
  game-result color, and clock screens run the GitHub/apt update check and
  draw v0.1's GitHub update icon when that check finds new commits.
- Clients behave like v0.1 on the panel: each screen holds for 4 seconds plus
  its extra seconds, the A/B/X/Y buttons keep their v0.1 actions (next screen,
  display on/off, update indicator, restart), and the Wi-Fi monitor and
  recovery run (`ENABLE_WIFI_MONITOR`, `ENABLE_WIFI_RECOVERY`).
- Live scores are fresher: during a live game the server refreshes that
  team's feed every 30 seconds (every 2 minutes otherwise). While the server
  is unreachable, a client skips scoreboards and live screens once their
  refresh deadline passes, as v0.1 skipped them during an outage, so a frozen
  score never looks current.
- Helpers follow the install mode: the kernel and framebuffer desktop
  launchers drive `desk_display_client.service` and `display_client.py` on a
  client or combined install; the Waveshare OLED helper and the screenshot
  uploader take the panel's settings from `.env.client`; and each heartbeat
  response carries the weather and Cubs/Blackhawks summary the OLED helper
  shows, so the side displays work on clients.
- On a server or combined install, the config UI's Screens page says that
  displays play playlists, and single-screen diagnostic playback (which only
  `main.py` performs) is hidden and refused. Scroll-speed changes saved there
  now re-render every display. Settings a role accepts but never reads (for
  example `ESC_DOUBLE_PRESS_ACTION` or `DESK_DISPLAY_TEST_SCREEN` on a client)
  log a startup warning, are marked "no effect" in CONFIGURATION.md and are
  left out of the role's example file. `ESC_DOUBLE_PRESS_ACTION` now accepts
  `main.py`'s actual choices (`stop`, `restart`, `toggle`).
- Migration: `scripts/migrate_standalone_config.py` moves a standalone rotation
  onto the server, and `scripts/convert_env.py` converts an existing `.env`.
  `--config FILE` migrates a rotation copied from another display, and a
  client's credentials file may carry its transport settings
  (`DESK_DISPLAY_ALLOW_INSECURE_TRANSPORT`, `DESK_DISPLAY_TLS_VERIFY`,
  `DESK_DISPLAY_SERVER_CA_BUNDLE`). OPERATIONS.md walks a v0.1 device through
  the upgrade.
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
- Fix: the render server's artifact store now honours
  `DESK_DISPLAY_ARTIFACT_MAX_MB`. It used to only log a warning when over
  budget, so `cache/artifacts/` grew to a full day of renders (several GB on
  a busy server). Unreferenced artifacts are now deleted oldest first until
  the store fits; anything a screen or client manifest uses stays.
- Storage limits shown on the Stats page: the artifact store defaults to 1 GB
  (`DESK_DISPLAY_ARTIFACT_MAX_MB=1024`). `images/cache/` now has an
  enforced 256 MB limit (`DESK_DISPLAY_IMAGE_CACHE_MAX_MB`; least recently
  used tiles, logos and headshots are deleted and downloaded again when
  needed). `cache/` as a whole has a 1.5 GB limit
  (`DESK_DISPLAY_SERVER_CACHE_MAX_MB=1536`) that logs a warning when exceeded.
- The display client's local cache (`DESK_DISPLAY_CLIENT_CACHE_MAX_MB`) now
  defaults to 512 MB instead of 256 MB. `scripts/clear-caches.sh` (and the
  Clients page's **Clear caches**) clears only the pip and apt download
  caches, never this cache.
- Fix: a feed refresh that returns unchanged data (for example from a TTL
  cache) no longer bumps its data revision, so it no longer re-renders the
  screens that read it.
- Fix: the config UI's Screenshots and Feed pages work on client and combined
  installs. The display client now saves each screen it shows (and the display
  heartbeat) where the config UI reads them, honouring `ENABLE_SCREENSHOTS`,
  and the service banner reports `desk_display_client.service` there instead
  of the disabled standalone `desk_display.service`.
- The NCAA FBS Scoreboard shows the whole Monday–Sunday week of Top 25
  games, matching ESPN's college football weeks. It reads ESPN's
  site.web.api host one day at a time (site.api.espn.com refuses the server
  Pi with 403), falls back to ESPN's other hosts, and keeps the last good
  games for the week if every host fails. Each ranked team's poll ranking is drawn as a small superscript
  number before its logo (or abbreviation); unranked teams get none.
- NCAA FBS Scoreboard team logos are trimmed of padding and scaled to the
  same area inside wider logo columns, so wordmarks and round marks read at
  the same size. A team with no saved logo uses ESPN's logo from the game
  data before falling back to its abbreviation.
- NCAA FBS Scoreboard games are listed by their best-ranked team (#1 first),
  then by the other team's rank, then by kickoff time.
- NCAA FBS Scoreboard rank superscripts are a little larger (rank font 11 → 13
  before profile scaling) on every display profile.
- The render server downloads missing NCAA FBS team logos itself. Whenever
  its scoreboard feed loads the week's games, it saves ESPN's logo, trimmed
  and capped at 128 px like `scripts/logo_getter.py`, for each team with no
  saved logo, into the untracked `images/cache/ncaa/`. Only the server does
  this (clients get finished images; standalone displays keep using ESPN's
  logo in memory). Logos committed to `images/ncaa/` always win and are never
  touched, so a new week needs no logo run, commit or client `git pull`.
- The `inside` screen works on display clients with an indoor sensor. The
  client reads its own sensor, set by `INSIDE_SENSOR`, `INSIDE_I2C_BUSES` and
  the new optional `INSIDE_I2C_ADDRESS` in its `.env.client`, and draws the
  screen with the v0.1 layout; the server renders nothing for it. A client
  without a sensor skips the screen, as the standalone display did.
- `systemctl stop` (and restart) of `desk_display_client.service` now stops
  the client within a second or two. SDL, which HyperPixel and window panels
  use, had taken over SIGTERM and only queued a quit event nobody read, so
  systemd waited out its 10 s timeout and SIGKILLed the client.
- A display client no longer refuses to start over a plain server setting in
  its `.env.client` (for example `WEATHER_LATITUDE` copied over from a v0.1
  `.env`); it logs a warning and ignores it. Provider credentials and other
  secrets on a client still stop it.
- A screen that runs out of things to show on the render server (Sox or Cubs
  Live once the game goes final, an expired alert) now drops out of client
  rotations. The server had treated "nothing to show" as a render failure and
  kept serving the last render as a fallback, so the live box score stayed
  on screen, frozen, until the server restarted.
- Each playlist card on the Playlists page reads "N active of M screens",
  counting only screens the playlist actually plays (a frequency above zero,
  or an alternate with an alternate frequency above zero, and not past its
  hide-after time).
- The Rotation Config, Playlists and Clients pages are less cluttered: the
  shared screen editor has compact rows, a sticky column header, a "Find a
  screen" search and a More menu for rare playlist actions; the Clients page
  shows one card per display with **Settings**, **Delivery** and
  **Maintenance** tabs.
- On a client or combined install, the Screenshots and Feed pages list
  screens in the order the display's playlist editor shows them, alternates
  and frequency-0 rows included.
- The NHL standings overview screens are titled "NHL West" and "NHL East" on
  every display (they read "NHL Western/Eastern Conference" everywhere but
  the HyperPixel 4 Square).
- Wolves screens: the StanzaCal schedule calendar (`AHL_SCHEDULE_ICS_URL`, in
  the server's `.env` on a server or combined install) is parsed correctly
  (escaped commas and semicolons, opponent abbreviations and logos such as
  Milwaukee), a started game with no result shows as live, a live game stays
  live past midnight (until 4 AM), and a live Wolves game no longer breaks the
  screen registry.
- Radar is sharp on large displays: the base map is stitched from
  OpenStreetMap tiles at zoom 9 (1080p), 8 (720 px square and 800×480) or 7
  (small panels) and cached for 30 days in `images/cache/radar_basemap/`,
  radar uses RainViewer's 512 px tiles on displays larger than 256 px, the
  map widens on wide displays instead of stretching, both layers line up to
  the pixel, and the map carries an "© OpenStreetMap" credit.
- 1080p HDMI layouts fill the screen on Bears Next, Hawks Next/Next Home,
  Hawks and Bulls Last, Bulls Next, every team's Stand 1/2/3, Cubs/Sox series,
  the NFC/AFC overviews and ADS-B Live; the NCAA FBS league logo is no longer
  scaled twice, and the Cubs/Sox R/H/E labels clear the title. Other profiles
  are unchanged.
- Sun & Moon keeps its Rise and Set rows together on square displays, and
  shows the latitude and longitude centred under the cards on HyperPixel 4
  Square, where the title always crowded them out.
- Sun & Moon sizes its sun and moon icons and its text to fill each card on
  every display (it used v0.1's fixed sizes and left the lower part of the
  cards empty). The moon phase name shrinks on its own to fit, and the
  Mini PiTFT draws the icons beside the text instead of on top of it.
- On This Day draws the emoji in Hebcal holiday titles in colour with Noto
  Color Emoji (`fonts-noto-color-emoji`), and drops them, instead of drawing
  empty boxes, where that font is missing.
- Cubs/Sox Next keep today's game until first pitch (they showed tomorrow's
  game whenever there was one), and count warmup and a delayed start as not
  yet started.
- Fix: the render server refreshed live scoreboards only once a day, so a
  finished game could stay on screen mid-inning; its live-game check works
  again.
- Fix: the Hawks schedule quad's standings tile was black on the render
  server.
- `scripts/upgrade.sh` runs pip without prompts or a version check, and
  every pip step and the service restart print the elapsed time and the
  processes still running every 60 seconds
  (`DESK_DISPLAY_HEARTBEAT_SECONDS`), so a stalled step names itself.
- `Installers/install_hyperpixel.sh` keeps an existing `INSIDE_I2C_BUSES`
  instead of resetting it to 13, and a failed sensor probe says which buses
  it tried and why.
- The fullscreen kernel output window has no title, so the Pi taskbar's
  "Desk Display" tooltip no longer floats over the picture.
- `scripts/logo_getter.py` downloads NCAA team logos named the way the
  scoreboards look them up, for review or for committing to `images/ncaa/`.

### Known limitations

- `DESK_DISPLAY_CLIENT_NAME` is ignored; the server identifies clients by
  client ID.
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
