# Configuration by deployment role

Desk Display reads its settings from environment variables, usually loaded
from a `.env` file. `DESK_DISPLAY_ROLE` selects which settings apply:

| Role | Example file | What it configures |
| --- | --- | --- |
| `standalone` (default) | [`.env.example`](.env.example) | The legacy single-process install: content, provider credentials, and display hardware in one `.env`. Still fully supported during the migration. |
| `server` | [`.env.server.example`](.env.server.example) | The render server: bind address and client authentication, location and content timezone, provider credentials, sports, news, ADS-B, styles and layouts, refresh intervals, render workers, artifact storage, client leases, static clients, the configuration UI, logs, and diagnostics. |
| `client` | [`.env.client.example`](.env.client.example) | One display: stable client ID, server URL and token, display profile, output driver, physical rotation, framebuffer and panel hardware, local cache, sync and heartbeat intervals, backlight and dark hours, buttons, touch, offline startup, logs, and diagnostics. |

A client needs no upstream API key or provider URL. The server fetches all
upstream data and sends clients rendered artifacts only.

> The render server API runs as `display_server.py` (see
> [Render server API](#render-server-api)). The display client process that
> consumes `.env.client.example` lands separately; its settings are already
> declared, parsed, and validated.

## Checking a configuration

```bash
python3 -m deployment_config check --role client --env-file .env
python3 -m deployment_config check --env-file .env   # role from DESK_DISPLAY_ROLE
```

The check exits non-zero on errors. The same validation runs at startup in
`main.py`, `config_ui.py`, and `feed_server.py`. Server and client processes
refuse to start on errors; the standalone role logs them as warnings so
existing installs keep running.

Startup validation reports:

- **Misplaced server settings on a client.** Any server-only or provider
  setting with a value in a client's environment is an error (for example
  `OWM_API_KEY` or `SCREEN_UI_PASSWORD`). Remove it and set it on the server.
- **Missing identity or server settings.** A client needs
  `DESK_DISPLAY_CLIENT_ID`, `DESK_DISPLAY_SERVER_URL`,
  `DESK_DISPLAY_CLIENT_TOKEN`, and `DESK_DISPLAY_PROFILE`.
- **Unknown profiles** and **invalid rotations** (0, 90, 180, 270 degrees, or
  0 to 3 quarter turns).
- **Insecure authentication.** A server needs a
  `DESK_DISPLAY_SERVER_AUTH_TOKEN` of at least 32 characters; running
  without one is only allowed on a loopback bind with
  `DESK_DISPLAY_SERVER_ALLOW_UNAUTHENTICATED=1`. A client using plain HTTP to
  another host, or with TLS verification off, fails unless
  `DESK_DISPLAY_ALLOW_INSECURE_TRANSPORT=1`. `SCREEN_AUTH_ENABLED=1` without
  `SCREEN_UI_PASSWORD` fails.
- **Incompatible driver and profile combinations.** `displayhatmini` drives
  only `display_hat_mini`, `minipitft` drives only `adafruit_minipitft_114`,
  and those two SPI panels cannot use the `kernel` or `framebuffer` outputs.
- Malformed values of any kind, unpaired latitude and longitude, and unknown
  timezones.

Regenerate the example files and the reference table below after changing
the catalog in `deployment_config.py`:

```bash
python3 -m deployment_config example --role server > .env.server.example
python3 -m deployment_config example --role client > .env.client.example
python3 -m deployment_config reference   # paste between the markers below
```

Tests fail when the committed files drift from the catalog.

## Render server API

`DESK_DISPLAY_ROLE=server python3 display_server.py` serves a versioned API
under `/api/v1`, separate from the screenshot Feed server, whose endpoints
are unchanged.

- `POST /api/v1/register` takes the client's capabilities (and optional
  demand) with `Authorization: Bearer <DESK_DISPLAY_SERVER_AUTH_TOKEN>`. It
  returns the accepted versions, server version, assignment state, assigned
  playlist and revision, manifest revision, lease expiry, recommended
  heartbeat and sync intervals, and a per-client `client_credential`.
- `/api/v1/clients/<id>/heartbeat`, `/config`, `/manifest`, and
  `/artifacts/<sha256>.<ext>` need `Authorization: Bearer <client_credential>`
  for that client ID, so one client cannot read another's configuration or
  status. A client can download only artifacts its own manifests listed.
- `/api/v1/health` is public and only reports that the service is up.
- `/api/v1/admin/status`, `/admin/prerender/<name>`, and
  `/admin/clients/<id>/disable|enable` need
  `DESK_DISPLAY_SERVER_ADMIN_TOKEN`; the admin API is off when it is unset.

A lease lasts `DESK_DISPLAY_CLIENT_LEASE_SECONDS`; the server recommends a
heartbeat every third of that. A client that misses its deadline expires:
its demand is dropped, its credential stops working, and it must register
again. While a lease is active, a second registration for the same client ID
is refused with `409 client_id_in_use` unless it presents the current
credential, so a client should keep its credential across restarts or wait
for the lease to lapse. Clients listed in `DESK_DISPLAY_STATIC_CLIENTS`
contribute their assigned playlist's screens even when not connected, must
register with their configured profile, and merge with dynamic clients and
administrator pre-render demand into one de-duplicated render plan.

## Client playlists

On a render server, the configuration UI's **Playlists** page (`/playlists`)
is the source of truth for what each display plays; the `.env` files never
hold playlists or assignments. Playlists use the same document format as the
rotation config (frequency, extra seconds, alternate screens, playlists and
sequence), are validated against `screens_catalog.py` and the scheduler, and
are stored with stable `pl-…` IDs and content revisions in
`DESK_DISPLAY_PLAYLIST_STORE_PATH`. Writes are atomic and locked, every
change is recorded in an audit log, and every edit, rename, reorder, delete,
and assignment must name the revision (or current playlist) it was based on,
so a concurrent change is refused with `409 revision_conflict` instead of
being overwritten. A playlist still assigned to a client cannot be deleted.
Exports and imports carry only the playlist document, never credentials.

The **Clients** page (`/clients`) lists static and registered displays from
the snapshot the render server writes to `DESK_DISPLAY_CLIENT_REGISTRY_PATH`
(it holds no credentials). Each client has exactly one assigned playlist;
many clients may share one, and "Clone for this client" gives a client its
own copy to edit. The page shows the client's state (online, stale, expired,
disabled, or never connected), profile, dimensions, capabilities, rotation,
software version, current screen, cache age, and three revisions: **saved**
on the server, **delivered** to the client, and **acknowledged** by the
client's heartbeat. The demand preview on the Playlists page lists what each
assigned profile must render and warns when a playlist uses animation, touch,
or color that a client cannot show.

When `SCREEN_UI_PASSWORD` is set these pages and their `/api/` endpoints
require a login, and every change also needs the page's
`X-Requested-With: desk-display` header.

## Manifests and artifacts

Rendered output lives in `DESK_DISPLAY_ARTIFACT_DIR`, stored by the SHA-256
of its bytes, so an artifact URL never changes content. Output is written to
a staging file, checked (PNG format, the profile's exact dimensions and
color mode, checksum, and render-package schema) and only then renamed into
place, so no client can download a partial file. Output that fails these
checks, or a render that fails outright, never replaces the last good
artifact: the manifest keeps listing it with `state: "fallback"` and the
failure, or `state: "stale"` once its refresh deadline passes.

A client's manifest (`GET /api/v1/clients/<id>/manifest`) lists its
assigned playlist and revision, profile and dimensions, requested and
touch-dependency screens, and for each artifact its immutable URL, SHA-256,
length, media type, dimensions, generation time, refresh deadline, state,
animation metadata and required capabilities. `manifest_revision` changes
only when that content changes. The manifest and artifacts both return an
`ETag`, answer `If-None-Match` with `304 Not Modified`, and artifacts support
`Range` requests so an interrupted download can resume. Artifacts are served
with `Cache-Control: immutable`, so a client downloads only artifacts it has
not seen.

The render server renders only what current demand needs: screens of
connected clients, of static clients (from their assigned playlist), of
administrator pre-render entries, and their touch dependencies. Equal
requests from many clients render once. A screen is rerendered when its
style, data or renderer revision changes or its output passes its refresh
deadline, but never more often than `DESK_DISPLAY_RENDER_MIN_INTERVAL_SECONDS`;
a failing screen backs off from that interval, doubling up to 15 minutes.
Missing output renders before refreshes, and connected clients before
static clients and pre-render entries. At most `DESK_DISPLAY_RENDER_WORKERS`
renders run at once; one that exceeds `DESK_DISPLAY_RENDER_TIMEOUT_SECONDS`
counts as failed. Work for a client whose lease lapses is cancelled.
`GET /api/v1/admin/render-status` (admin token) shows clients, profiles,
render keys, the queue and running renders, each screen's state, last
success, failures and durations, data health, artifact disk use, and each
client's playlist delivery and acknowledgment.

Each screen keeps its current and three previous good artifacts. Artifacts
listed in any client's current or previous manifest are never deleted;
others are deleted `DESK_DISPLAY_ARTIFACT_RETENTION_HOURS` after they stop
being referenced, by a background task every ten minutes.

## Secrets

Settings marked secret below (provider keys, the server and client tokens,
the config UI password and session secret, the feed upload token) are never
sent to clients or browsers:

- Manifests built by `protocol.build_manifest` drop keys named after a secret
  setting and redact configured secret values.
- The configuration UI and feed server redact configured secret values from
  every text and JSON response, and scrub stored client status payloads.
- Log records in the display, config UI, and feed server processes have
  configured secret values replaced with `[redacted]`.

Value redaction applies to secrets of at least 8 characters, so short
placeholder values are not matched against unrelated text.

## Hot-reloadable settings versus restart

Every environment variable is read once when the process starts. **Changing
any value in `.env` requires a restart** of the affected service:

```bash
bash scripts/restart_services.sh
```

These documents are re-read automatically when their file changes, so edits
to their contents take effect without a restart (changing the path variable
that points to them still needs one):

| Document | Path setting | Reloaded by |
| --- | --- | --- |
| Screen schedule and playlists, including NHL break windows | `SCREENS_CONFIG_PATH` (`screens_config.json`) | `main.py` on the next rotation pass |
| Per-screen style overrides | `SCREENS_STYLE_PATH` (`screens_style.json`) | `config.py` style loader |
| Quad layouts | `SCREENS_LAYOUTS_PATH` (`screens_layouts.json`) | `screens/registry.py` |
| News topics and feeds | `NEWS_FEEDS_CONFIG_PATH` (`news_feeds.json`) | `services/news_feeds.py` |
| Second news screen's feeds | `NEWS_FEEDS_CONFIG_PATH_2` (`news_feeds_2.json`) | `services/news_feeds.py` |
| Client playlists, assignments, and friendly names | `DESK_DISPLAY_PLAYLIST_STORE_PATH` (`.runtime/server/playlists.json`) | `display_server.py` and the configuration UI on the next request |

The configuration UI writes these documents, which is why its changes appear
on the display without a restart.

## Settings reference

"Provider" marks an upstream data provider setting: servers and standalone
installs only, never clients.

<!-- BEGIN GENERATED SETTINGS REFERENCE -->
<!-- BEGIN GENERATED SETTINGS REFERENCE -->
| Setting | Roles | Takes effect | Secret |
| --- | --- | --- | --- |
| `DESK_DISPLAY_ROLE` | server, client, standalone | restart |  |
| `CONFIG_LOAD_DOTENV` | server, client, standalone | restart |  |
| `TZ` | server, client, standalone | restart |  |
| `DESK_DISPLAY_SERVER_HOST` | server | restart |  |
| `DESK_DISPLAY_SERVER_PORT` | server | restart |  |
| `DESK_DISPLAY_SERVER_PUBLIC_URL` | server | restart |  |
| `DESK_DISPLAY_SERVER_AUTH_TOKEN` | server | restart | yes |
| `DESK_DISPLAY_SERVER_ADMIN_TOKEN` | server | restart | yes |
| `DESK_DISPLAY_SERVER_ALLOW_UNAUTHENTICATED` | server | restart |  |
| `DESK_DISPLAY_SERVER_TLS_CERT` | server | restart |  |
| `DESK_DISPLAY_SERVER_TLS_KEY` | server | restart |  |
| `DESK_DISPLAY_CLIENT_ID` | client | restart |  |
| `DESK_DISPLAY_CLIENT_NAME` | client | restart |  |
| `DESK_DISPLAY_SERVER_URL` | client | restart |  |
| `DESK_DISPLAY_CLIENT_TOKEN` | client | restart | yes |
| `DESK_DISPLAY_SERVER_CA_BUNDLE` | client | restart |  |
| `DESK_DISPLAY_TLS_VERIFY` | client | restart |  |
| `DESK_DISPLAY_ALLOW_INSECURE_TRANSPORT` | client | restart |  |
| `WEATHER_LATITUDE` | server, standalone | restart |  |
| `WEATHER_LONGITUDE` | server, standalone | restart |  |
| `DESK_DISPLAY_CONTENT_TIMEZONE` | server | restart |  |
| `WEATHERKIT_TEAM_ID` | server, standalone | restart | provider |
| `WEATHERKIT_KEY_ID` | server, standalone | restart | provider |
| `WEATHERKIT_SERVICE_ID` | server, standalone | restart | provider |
| `WEATHERKIT_KEY_PATH` | server, standalone | restart | provider |
| `WEATHERKIT_PRIVATE_KEY` | server, standalone | restart | yes |
| `WEATHERKIT_LANGUAGE` | server, standalone | restart |  |
| `WEATHERKIT_TIMEZONE` | server, standalone | restart |  |
| `OWM_API_KEY` | server, standalone | restart | yes |
| `OWM_API_KEY_DEFAULT` | server, standalone | restart | yes |
| `OWM_API_KEY_WIFFY` | server, standalone | restart | yes |
| `OWM_API_KEY_VERANO` | server, standalone | restart | yes |
| `OWM_UNITS` | server, standalone | restart |  |
| `OWM_LANGUAGE` | server, standalone | restart |  |
| `HOURLY_FORECAST_HOURS` | server, standalone | restart |  |
| `WEATHER_USE_EMOJI_ICONS` | server, standalone | restart |  |
| `AIR_QUALITY_PROVIDER` | server, standalone | restart |  |
| `AIRNOW_API_KEY` | server, standalone | restart | yes |
| `AIR_QUALITY_LATITUDE` | server, standalone | restart |  |
| `AIR_QUALITY_LONGITUDE` | server, standalone | restart |  |
| `AIR_QUALITY_ENABLE_POLLEN` | server, standalone | restart |  |
| `NCAAM_SCOREBOARD_MODE` | server, standalone | restart |  |
| `NHL_BREAK_WINDOWS_JSON` | server, standalone | restart |  |
| `NHL_SCHEDULE_ICS_URL` | server, standalone | restart | provider |
| `WORLD_CUP_PREGAME_SCORE_DISPLAY` | server, standalone | restart |  |
| `TEAM_STANDINGS_DISPLAY_SECONDS` | server, standalone | restart |  |
| `MLB_SCOREBOARD_SCROLL_DELAY` | server, standalone | restart |  |
| `SCOREBOARD_STANDINGS_BOTTOM_PADDING` | server, standalone | restart |  |
| `SMALL_RESULT_FLAG_H` | server, standalone | restart |  |
| `NIXIE_TIME_FORMAT` | server, standalone | restart |  |
| `AHL_API_BASE_URL` | server, standalone | restart | provider |
| `AHL_API_KEY` | server, standalone | restart | yes |
| `AHL_CLIENT_CODE` | server, standalone | restart |  |
| `AHL_LEAGUE_ID` | server, standalone | restart |  |
| `AHL_SITE_ID` | server, standalone | restart |  |
| `AHL_SEASON_ID` | server, standalone | restart |  |
| `AHL_SCHEDULE_ICS_URL` | server, standalone | restart | yes |
| `AHL_TEAM_ID` | server, standalone | restart |  |
| `AHL_TEAM_TRICODE` | server, standalone | restart |  |
| `AHL_TEAM_NAME` | server, standalone | restart |  |
| `ENABLE_NEWS_HEADLINES` | server, standalone | restart |  |
| `ENABLE_NEWS_HEADLINES_2` | server, standalone | restart |  |
| `NEWS_FEEDS_CONFIG_PATH` | server, standalone | restart (file contents hot-reload) |  |
| `NEWS_FEEDS_CONFIG_PATH_2` | server, standalone | restart (file contents hot-reload) |  |
| `NEWS_HEADLINES_DISPLAY_SECONDS` | server, standalone | restart |  |
| `NEWS_HEADLINES_SHOW_IMAGES` | server, standalone | restart |  |
| `NEWS_TICKER_BASE_SPEED` | server, standalone | restart |  |
| `NEWS_ARTICLE_FETCH_TIMEOUT_SECONDS` | server, standalone | restart |  |
| `ENABLE_STOCK_TICKER` | server, standalone | restart |  |
| `STOCK_TICKER_CACHE_TTL_SECONDS` | server, standalone | restart |  |
| `ON_THIS_DAY_FEED_BUILD_TIMEOUT_SECONDS` | server, standalone | restart |  |
| `ON_THIS_DAY_INCOMPLETE_FEED_RETRY_SECONDS` | server, standalone | restart |  |
| `ON_THIS_DAY_OFFLINE_FALLBACK_RETRY_SECONDS` | server, standalone | restart |  |
| `ON_THIS_DAY_LIVE_THUMBNAILS` | server, standalone | restart |  |
| `WIKIMEDIA_USER_AGENT` | server, standalone | restart | provider |
| `ADSB_DEVICE_1_HOST` | server, standalone | restart | provider |
| `ADSB_DEVICE_1_LABEL` | server, standalone | restart |  |
| `ADSB_DEVICE_2_HOST` | server, standalone | restart | provider |
| `ADSB_DEVICE_2_LABEL` | server, standalone | restart |  |
| `ADSB_HOME_LATITUDE` | server, standalone | restart |  |
| `ADSB_HOME_LONGITUDE` | server, standalone | restart |  |
| `ADSB_DISTANCE_UNIT` | server, standalone | restart |  |
| `ADSB_POLL_INTERVAL_SECONDS` | server, standalone | restart |  |
| `ADSB_REQUEST_TIMEOUT_SECONDS` | server, standalone | restart |  |
| `ADSB_RETENTION_DAYS` | server, standalone | restart |  |
| `ADSB_DB_PATH` | server, standalone | restart |  |
| `ADSB_TYPE_DB_ENABLED` | server, standalone | restart |  |
| `ADSB_TYPE_DB_URL` | server, standalone | restart | provider |
| `ADSB_TYPE_DB_PATH` | server, standalone | restart |  |
| `ADSB_TYPE_DB_REFRESH_DAYS` | server, standalone | restart |  |
| `SCREENS_CONFIG_PATH` | server, standalone | restart (file contents hot-reload) |  |
| `SCREENS_CONFIG_LOCAL_PATH` | server, standalone | restart |  |
| `SCREENS_STYLE_PATH` | server, standalone | restart (file contents hot-reload) |  |
| `SCREENS_LAYOUTS_PATH` | server, standalone | restart (file contents hot-reload) |  |
| `DEFAULT_SCREENS_PATH` | server, standalone | restart |  |
| `DEFAULT_SCREENS_LARGE_PATH` | server, standalone | restart |  |
| `DEFAULT_SCREENS_SMALL_PATH` | server, standalone | restart |  |
| `DESK_DISPLAY_PLAYLIST_STORE_PATH` | server, standalone | restart (file contents hot-reload) |  |
| `WEATHER_REFRESH_SECONDS` | server, standalone | restart |  |
| `STARTUP_CRITICAL_FEED_TIMEOUT_SECONDS` | server, standalone | restart |  |
| `HTTP_CLIENT_FORBIDDEN_COOLDOWN_SECONDS` | server, standalone | restart |  |
| `HTTP_CLIENT_USE_SYSTEM_PROXIES` | server, client, standalone | restart |  |
| `DESK_DISPLAY_RENDER_WORKERS` | server | restart |  |
| `DESK_DISPLAY_RENDER_TIMEOUT_SECONDS` | server | restart |  |
| `DESK_DISPLAY_RENDER_MIN_INTERVAL_SECONDS` | server | restart |  |
| `DESK_DISPLAY_ARTIFACT_DIR` | server | restart |  |
| `DESK_DISPLAY_ARTIFACT_RETENTION_HOURS` | server | restart |  |
| `DESK_DISPLAY_ARTIFACT_MAX_MB` | server | restart |  |
| `DESK_DISPLAY_CLIENT_REGISTRY_PATH` | server, standalone | restart |  |
| `DESK_DISPLAY_CLIENT_LEASE_SECONDS` | server | restart |  |
| `DESK_DISPLAY_STATIC_CLIENTS` | server | restart |  |
| `SCREEN_CONFIG_HOST` | server, standalone | restart |  |
| `SCREEN_CONFIG_PORT` | server, standalone | restart |  |
| `SCREEN_UI_USERNAME` | server, standalone | restart |  |
| `SCREEN_UI_PASSWORD` | server, standalone | restart | yes |
| `SCREEN_SESSION_SECRET` | server, standalone | restart | yes |
| `SCREEN_AUTH_ENABLED` | server, standalone | restart |  |
| `FEED_SERVER_HOST` | server, standalone | restart |  |
| `FEED_SERVER_PORT` | server, standalone | restart |  |
| `FEED_STORAGE_DIR` | server, standalone | restart |  |
| `FEED_MAX_UPLOAD_BYTES` | server, standalone | restart |  |
| `FEED_STALE_SECONDS` | server, standalone | restart |  |
| `FEED_HEARTBEAT_STALE_SECONDS` | server, standalone | restart |  |
| `FEED_SCREEN_STALE_SECONDS` | server, standalone | restart |  |
| `PRESSURE_HISTORY_PATH` | server, standalone | restart |  |
| `WEATHER_METRIC_HISTORY_PATH` | server, standalone | restart |  |
| `AIR_QUALITY_HISTORY_PATH` | server, standalone | restart |  |
| `ON_THIS_DAY_CACHE_PATH` | server, standalone | restart |  |
| `DESK_DISPLAY_PROFILE` | client, standalone | restart |  |
| `DISPLAY_WIDTH` | client, standalone | restart |  |
| `DISPLAY_HEIGHT` | client, standalone | restart |  |
| `DISPLAY_RESOLUTION` | standalone | restart |  |
| `HYPERPIXEL_PANEL` | client, standalone | restart |  |
| `DESK_DISPLAY_LOW_POWER` | client, standalone | restart |  |
| `DESK_DISPLAY_OUTPUT` | client, standalone | restart |  |
| `DESK_DISPLAY_FORCE_HEADLESS` | client, standalone | restart |  |
| `DISPLAY_FADE_IN_ENABLED` | client, standalone | restart |  |
| `DISPLAY_FADE_IN_DISPLAY_HAT_MINI_STEPS` | client, standalone | restart |  |
| `DISPLAY_FADE_IN_HYPERPIXEL_STEPS` | client, standalone | restart |  |
| `DISPLAY_FADE_IN_HDMI_1080P_STEPS` | client, standalone | restart |  |
| `DISPLAY_ROTATION` | client, standalone | restart |  |
| `DISPLAY_ROTATION_STRICT` | client, standalone | restart |  |
| `DISPLAY_FB_DEVICE` | client, standalone | restart |  |
| `DISPLAY_FB_PIXEL_FORMAT` | client, standalone | restart |  |
| `DISPLAY_FB_PIXEL_ORDER` | client, standalone | restart |  |
| `DISPLAY_FB_HIDE_CONSOLE_CURSOR` | client, standalone | restart |  |
| `DISPLAY_FB_CONSOLE_GRAPHICS` | client, standalone | restart |  |
| `DISPLAY_HAT_MINI_LED_LEVEL` | client, standalone | restart |  |
| `DISPLAY_HAT_MINI_REINIT_SECONDS` | client, standalone | restart |  |
| `DISPLAY_HAT_MINI_IO_TIMEOUT_SECONDS` | client, standalone | restart |  |
| `DISPLAY_HAT_MINI_MAX_REFRESH_FAILURES` | client, standalone | restart |  |
| `LED_INDICATOR_ENABLED` | client, standalone | restart |  |
| `LED_INDICATOR_BORDER_ENABLED` | client, standalone | restart |  |
| `LED_INDICATOR_BORDER_WIDTH` | client, standalone | restart |  |
| `MINIPITFT_BAUDRATE` | client, standalone | restart |  |
| `MINIPITFT_DRIVER_ROTATION` | client, standalone | restart |  |
| `MINIPITFT_DRIVER_WIDTH` | client, standalone | restart |  |
| `MINIPITFT_DRIVER_HEIGHT` | client, standalone | restart |  |
| `MINIPITFT_X_OFFSET` | client, standalone | restart |  |
| `MINIPITFT_Y_OFFSET` | client, standalone | restart |  |
| `DESK_DISPLAY_WINDOW_SCALE` | client, standalone | restart |  |
| `DESK_DISPLAY_WINDOW_RESIZABLE` | client, standalone | restart |  |
| `DESK_DISPLAY_SDL_FULLSCREEN` | client, standalone | restart |  |
| `DESK_DISPLAY_SDL_DRIVERS` | client, standalone | restart |  |
| `SDL_VIDEODRIVER` | client, standalone | restart |  |
| `DESK_DISPLAY_SESSION_USER` | client, standalone | restart |  |
| `DISPLAY` | client, standalone | restart |  |
| `WAYLAND_DISPLAY` | client, standalone | restart |  |
| `XAUTHORITY` | client, standalone | restart |  |
| `XDG_RUNTIME_DIR` | client, standalone | restart |  |
| `DESK_DISPLAY_CLIENT_CACHE_DIR` | client | restart |  |
| `DESK_DISPLAY_CLIENT_CACHE_MAX_MB` | client | restart |  |
| `DESK_DISPLAY_SYNC_INTERVAL_SECONDS` | client | restart |  |
| `DESK_DISPLAY_HEARTBEAT_INTERVAL_SECONDS` | client | restart |  |
| `DARK_HOURS` | client, standalone | restart |  |
| `DESK_DISPLAY_BACKLIGHT_LEVEL` | client | restart |  |
| `DESK_DISPLAY_DARK_HOURS_MODE` | client | restart |  |
| `DESK_DISPLAY_DARK_HOURS_BACKLIGHT_LEVEL` | client | restart |  |
| `BUTTON_A` | client, standalone | restart |  |
| `BUTTON_B` | client, standalone | restart |  |
| `BUTTON_X` | client, standalone | restart |  |
| `BUTTON_Y` | client, standalone | restart |  |
| `DESK_DISPLAY_RPI_GPIO_FALLBACK` | client, standalone | restart |  |
| `TOUCH_DOUBLE_TAP_MAX_INTERVAL_SECONDS` | client, standalone | restart |  |
| `ESC_DOUBLE_PRESS_ACTION` | client, standalone | restart |  |
| `ESC_DOUBLE_PRESS_MAX_INTERVAL_SECONDS` | client, standalone | restart |  |
| `DESK_DISPLAY_OFFLINE_START` | client | restart |  |
| `DESK_DISPLAY_OFFLINE_MAX_AGE_HOURS` | client | restart |  |
| `ENABLE_SCREENSHOTS` | client, standalone | restart |  |
| `ENABLE_VIDEO` | client, standalone | restart |  |
| `SCREENSHOT_DIR` | client, standalone | restart |  |
| `SCREENSHOT_ARCHIVE_BASE` | client, standalone | restart |  |
| `FEED_UPLOAD_URL` | client, standalone | restart |  |
| `FEED_UPLOAD_TOKEN` | server, client, standalone | restart | yes |
| `FEED_SOURCE_NAME` | client, standalone | restart |  |
| `FEED_UPLOAD_INTERVAL_SECONDS` | client, standalone | restart |  |
| `FEED_UPLOAD_TIMEOUT_SECONDS` | client, standalone | restart |  |
| `FEED_UPLOAD_CYCLE_TIMEOUT_SECONDS` | client, standalone | restart |  |
| `FEED_UPLOAD_MAX_BACKOFF_SECONDS` | client, standalone | restart |  |
| `INSIDE_SENSOR` | client, standalone | restart |  |
| `INDOOR_SENSOR` | client, standalone | restart |  |
| `INSIDE_I2C_BUSES` | client, standalone | restart |  |
| `INSIDE_HISTORY_PATH` | client, standalone | restart |  |
| `ENABLE_WIFI_MONITOR` | client, standalone | restart |  |
| `ENABLE_WIFI_RECOVERY` | client, standalone | restart |  |
| `WIFI_INTERFACE` | client, standalone | restart |  |
| `WIFI_RECOVERY_LOG` | client, standalone | restart |  |
| `WIFI_TCP_PROBE_URLS` | client, standalone | restart |  |
| `WIFI_TCP_PROBE_URL` | client, standalone | restart |  |
| `WIFI_HTTPS_PROBE_URL` | client, standalone | restart |  |
| `WIFI_TCP_PROBE_HOSTS` | client, standalone | restart |  |
| `WIFI_TCP_PROBE_HOST` | client, standalone | restart |  |
| `WIFI_TCP_PROBE_PORT` | client, standalone | restart |  |
| `RPI_CONNECT_CONTROL_HOST` | client, standalone | restart |  |
| `WAVESHARE_OLED_LCD_HAT_A_INSTALLED` | client, standalone | restart |  |
| `WAVESHARE_OLED_I2C_BUS` | client, standalone | restart |  |
| `WAVESHARE_OLED_TEMP_ADDR` | client, standalone | restart |  |
| `WAVESHARE_OLED_TIME_ADDR` | client, standalone | restart |  |
| `WAVESHARE_OLED_WIDTH` | client, standalone | restart |  |
| `WAVESHARE_OLED_HEIGHT` | client, standalone | restart |  |
| `WAVESHARE_OLED_MIN_VALUE_FONT_SIZE` | client, standalone | restart |  |
| `WAVESHARE_OLED_MAX_VALUE_FONT_SIZE` | client, standalone | restart |  |
| `WAVESHARE_OLED_MIN_TIME_FONT_SIZE` | client, standalone | restart |  |
| `WAVESHARE_OLED_MAX_TIME_FONT_SIZE` | client, standalone | restart |  |
| `WAVESHARE_OLED_REFRESH_SECONDS` | client, standalone | restart |  |
| `WAVESHARE_OLED_STATUS_MAX_AGE_SECONDS` | client, standalone | restart |  |
| `WAVESHARE_OLED_SWAP_INTERVAL_MIN_SECONDS` | client, standalone | restart |  |
| `WAVESHARE_OLED_SWAP_INTERVAL_MAX_SECONDS` | client, standalone | restart |  |
| `WAVESHARE_OLED_FADE_STEPS` | client, standalone | restart |  |
| `WAVESHARE_OLED_FADE_STEP_MS` | client, standalone | restart |  |
| `WAVESHARE_OLED_TEMP_SOURCE` | client, standalone | restart |  |
| `WAVESHARE_OLED_TEMP_COMMAND` | client, standalone | restart |  |
| `WAVESHARE_OLED_TEMP_UNIT` | client, standalone | restart |  |
| `WAVESHARE_OLED_WAIT_FOR_WEATHER2` | client, standalone | restart |  |
| `WAVESHARE_OLED_CUBS_FINAL_STATE_PATH` | client, standalone | restart |  |
| `WAVESHARE_OLED_HAWKS_FINAL_STATE_PATH` | client, standalone | restart |  |
| `WAVESHARE_OLED_DISPLAY_STATUS_PATH` | client, standalone | restart |  |
| `WAVESHARE_OLED_SCREENSHOT_DIR` | client, standalone | restart |  |
| `WAVESHARE_OLED_FONT_PATH` | client, standalone | restart |  |
| `WAVESHARE_OLED_LOG_LEVEL` | client, standalone | restart |  |
| `SCREEN_CONFIG_AUTOSTART` | standalone | restart |  |
| `IP_WITH_TIME` | standalone | restart |  |
| `DESK_DISPLAY_LOG_LEVEL` | server, client, standalone | restart |  |
| `FEED_UPLOAD_LOG_LEVEL` | client, standalone | restart |  |
| `DESK_DISPLAY_GC_INTERVAL_SECONDS` | server, client, standalone | restart |  |
| `DESK_DISPLAY_DIAGNOSTIC_CONTROL_PATH` | server, client, standalone | restart |  |
| `DESK_DISPLAY_TEST_SCREEN` | client, standalone | restart |  |
| `DESK_DISPLAY_TEST_SCREEN_DELAY` | client, standalone | restart |  |
| `RES_OPTIONS` | server, client, standalone | restart |  |
| `LOCALDOMAIN` | server, client, standalone | restart |  |
| `HOSTALIASES` | server, client, standalone | restart |  |
<!-- END GENERATED SETTINGS REFERENCE -->
<!-- END GENERATED SETTINGS REFERENCE -->
<!-- END GENERATED SETTINGS REFERENCE -->
