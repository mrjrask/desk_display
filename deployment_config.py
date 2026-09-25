"""Role-specific Desk Display configuration: catalog, parsing and validation.

Desk Display runs in one of three deployment roles, selected by
``DESK_DISPLAY_ROLE``:

``standalone``
    The legacy single-process install (``main.py`` plus its config UI).  It
    reads every content *and* hardware setting and remains documented by
    ``.env.example``.
``server``
    Fetches upstream data, renders artifacts and serves them to display
    clients.  It owns every provider credential.  See ``.env.server.example``.
``client``
    Drives one physical display from artifacts the server publishes.  It needs
    only its identity, the server URL and its own hardware settings, never an
    upstream API key or provider URL.  See ``.env.client.example``.

This module is deliberately dependency-light (standard library plus
:mod:`display_profiles`), so a process can validate its environment before it
loads configuration, rendering libraries or display hardware.  It also owns the
secret-exclusion helpers used on every payload returned to clients or browsers.

Run ``python -m deployment_config --help`` for the command-line checker and the
example/docs generators.
"""
from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import logging
import math
import os
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

ROLE_ENV = "DESK_DISPLAY_ROLE"
REDACTED = "[redacted]"
# Value-based redaction only applies to secrets at least this long, so a short
# placeholder cannot blank out unrelated text.
_MIN_REDACTABLE_SECRET_LENGTH = 8
_MIN_SERVER_TOKEN_LENGTH = 32


class Role(str, Enum):
    SERVER = "server"
    CLIENT = "client"
    STANDALONE = "standalone"


class Reload(str, Enum):
    """When a changed value takes effect."""

    RESTART = "restart"
    HOT = "hot"


class ConfigurationError(RuntimeError):
    """Startup validation failed for a server or client process."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        lines = [f"{issue.name or ROLE_ENV}: {issue.message}" for issue in report.errors]
        super().__init__(
            f"Invalid {report.role.value} configuration:\n  " + "\n  ".join(lines)
        )


_ALL = frozenset(Role)
_SERVER = frozenset({Role.SERVER, Role.STANDALONE})
_CLIENT = frozenset({Role.CLIENT, Role.STANDALONE})
_SERVER_ONLY = frozenset({Role.SERVER})
_CLIENT_ONLY = frozenset({Role.CLIENT})
_STANDALONE_ONLY = frozenset({Role.STANDALONE})


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    note: str = ""


@dataclass(frozen=True)
class Setting:
    """One environment variable and the roles that may set it."""

    name: str
    kind: str
    roles: frozenset[Role]
    section: str
    description: str
    default: str = ""
    example: str | None = None
    secret: bool = False
    # A credential, private URL or endpoint for an upstream data provider.
    # Clients never need these: the server fetches all upstream data.
    provider: bool = False
    reload: Reload = Reload.RESTART
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None

    @property
    def example_value(self) -> str:
        return self.default if self.example is None else self.example


SECTIONS: tuple[Section, ...] = (
    Section("role", "Deployment role and .env loading"),
    Section("server", "Server bind address and client authentication"),
    Section("client_identity", "Client identity"),
    Section("client_server", "Server URL and client authentication"),
    Section("location", "Location and content timezone"),
    Section("weather", "Weather providers and forecast"),
    Section("air_quality", "Outdoor air quality"),
    Section("sports", "Sports and teams"),
    Section("news", "News, stocks, and On This Day"),
    Section("adsb", "ADS-B receiver stats"),
    Section(
        "maps",
        "Maps and travel",
        note=(
            "The weather radar map is centred on WEATHER_LATITUDE and "
            "WEATHER_LONGITUDE, and its tile and radar endpoints are built in. "
            "No map or travel-route provider credential is read by this release, "
            "so there is nothing else to set here."
        ),
    ),
    Section("styles", "Screen configuration, styles, and layouts"),
    Section("refresh", "Refresh intervals and upstream requests"),
    Section("render", "Render workers"),
    Section("artifacts", "Artifact storage"),
    Section("leases", "Client leases and static clients"),
    Section("config_ui", "Configuration UI"),
    Section("feed_server", "Screenshot feed server"),
    Section("history", "Server caches and history files"),
    Section("profile", "Display profile and render size"),
    Section("output", "Output driver"),
    Section("rotation", "Physical rotation"),
    Section("hardware", "Framebuffer and panel hardware"),
    Section("window", "SDL window and desktop session"),
    Section("cache", "Local artifact cache"),
    Section("sync", "Synchronization and heartbeat"),
    Section("backlight", "Backlight and dark hours"),
    Section("buttons", "Buttons"),
    Section("touch", "Touch and keyboard"),
    Section("offline", "Offline startup"),
    Section("screenshots", "Screenshots and feed upload"),
    Section("sensors", "Indoor sensor"),
    Section("wifi", "Wi-Fi monitor and recovery"),
    Section("waveshare", "Waveshare OLED/LCD HAT (A) status helper"),
    Section("standalone", "Standalone-only runtime behavior"),
    Section("logs", "Logs"),
    Section("diagnostics", "Diagnostics"),
)
_SECTION_KEYS = {section.key for section in SECTIONS}


def _s(name: str, kind: str, roles: frozenset[Role], section: str, description: str, **kw: Any) -> Setting:
    return Setting(name=name, kind=kind, roles=roles, section=section, description=description, **kw)


_ROTATIONS = ("0", "90", "180", "270", "1", "2", "3")
_OUTPUT_ALIASES = {
    "auto": "auto", "": "auto",
    "displayhatmini": "displayhatmini", "display-hat-mini": "displayhatmini",
    "hatmini": "displayhatmini", "hat": "displayhatmini",
    "minipitft": "minipitft", "mini-pitft": "minipitft",
    "adafruit-minipitft": "minipitft", "pitft": "minipitft",
    "kernel": "kernel", "kms": "kernel", "drm": "kernel", "sdl": "kernel", "fullscreen": "kernel",
    "window": "window", "windowed": "window", "macos_window": "window",
    "mac-window": "window", "sdl-window": "window",
    "framebuffer": "framebuffer", "fb": "framebuffer", "framebuffer-device": "framebuffer",
    "headless": "headless", "none": "headless", "off": "headless",
}
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

SETTINGS: tuple[Setting, ...] = (
    # ── Role and .env loading ────────────────────────────────────────────────
    _s(ROLE_ENV, "choice", _ALL, "role",
       "Deployment role: standalone (legacy single process), server, or client. "
       "Selects which settings are valid and how strictly startup validates them.",
       default="standalone", choices=tuple(role.value for role in Role)),
    _s("CONFIG_LOAD_DOTENV", "bool", _ALL, "role",
       "Load .env from the project root and working directory at startup.", default="1"),
    _s("TZ", "str", _ALL, "role",
       "Process timezone for logs and system time, usually inherited from the OS."),

    # ── Server bind and authentication (new split runtime) ───────────────────
    _s("DESK_DISPLAY_SERVER_HOST", "str", _SERVER_ONLY, "server",
       "Interface the render server listens on. Keep 127.0.0.1 behind a reverse "
       "proxy; use 0.0.0.0 only with authentication and TLS.", default="127.0.0.1"),
    _s("DESK_DISPLAY_SERVER_PORT", "int", _SERVER_ONLY, "server",
       "TCP port for client registration, manifests and packages.",
       default="8765", minimum=1, maximum=65535),
    _s("DESK_DISPLAY_SERVER_PUBLIC_URL", "url", _SERVER_ONLY, "server",
       "Externally reachable base URL advertised to clients, when it differs "
       "from the bind address (for example behind a reverse proxy)."),
    _s("DESK_DISPLAY_SERVER_AUTH_TOKEN", "str", _SERVER_ONLY, "server",
       f"Shared bearer token every client must present (at least {_MIN_SERVER_TOKEN_LENGTH} "
       "characters). Generate one with: python3 -c 'import secrets; print(secrets.token_urlsafe(32))'",
       secret=True),
    _s("DESK_DISPLAY_SERVER_ADMIN_TOKEN", "str", _SERVER_ONLY, "server",
       f"Bearer token for the /api/v1/admin endpoints (at least {_MIN_SERVER_TOKEN_LENGTH} "
       "characters). The admin API is disabled when empty. Never give it to clients.",
       secret=True),
    _s("DESK_DISPLAY_SERVER_ALLOW_UNAUTHENTICATED", "bool", _SERVER_ONLY, "server",
       "Accept clients without a token. Only allowed while the server is bound "
       "to a loopback address.", default="0"),
    _s("DESK_DISPLAY_SERVER_TLS_CERT", "path", _SERVER_ONLY, "server",
       "PEM certificate for serving HTTPS directly. Set together with DESK_DISPLAY_SERVER_TLS_KEY."),
    _s("DESK_DISPLAY_SERVER_TLS_KEY", "path", _SERVER_ONLY, "server",
       "PEM private key for DESK_DISPLAY_SERVER_TLS_CERT."),

    # ── Client identity and server connection (new split runtime) ────────────
    _s("DESK_DISPLAY_CLIENT_ID", "client_id", _CLIENT_ONLY, "client_identity",
       "Stable, unique ID for this display (letters, digits, '.', '_' or '-', up to "
       "64 characters). Keep it across reinstalls so the server keeps its playlist.",
       example="office-display"),
    _s("DESK_DISPLAY_CLIENT_NAME", "str", _CLIENT_ONLY, "client_identity",
       "Optional human-readable name shown in the server UI; defaults to the client ID."),
    _s("DESK_DISPLAY_SERVER_URL", "url", _CLIENT_ONLY, "client_server",
       "Base URL of the render server.", example="https://desk-display.lan:8765"),
    _s("DESK_DISPLAY_CLIENT_TOKEN", "str", _CLIENT_ONLY, "client_server",
       "Bearer token matching the server's DESK_DISPLAY_SERVER_AUTH_TOKEN.", secret=True),
    _s("DESK_DISPLAY_SERVER_CA_BUNDLE", "path", _CLIENT_ONLY, "client_server",
       "CA bundle used to verify a server certificate from a private CA."),
    _s("DESK_DISPLAY_TLS_VERIFY", "bool", _CLIENT_ONLY, "client_server",
       "Verify the server's TLS certificate.", default="1"),
    _s("DESK_DISPLAY_ALLOW_INSECURE_TRANSPORT", "bool", _CLIENT_ONLY, "client_server",
       "Permit plain HTTP to a non-loopback server, or TLS without verification, "
       "on a trusted network. Startup then warns instead of failing.", default="0"),

    # ── Location and content timezone ────────────────────────────────────────
    _s("WEATHER_LATITUDE", "latitude", _SERVER, "location",
       "Latitude for weather, the radar map centre and default AQI/ADS-B location."),
    _s("WEATHER_LONGITUDE", "longitude", _SERVER, "location",
       "Longitude for weather, the radar map centre and default AQI/ADS-B location."),
    _s("DESK_DISPLAY_CONTENT_TIMEZONE", "timezone", _SERVER_ONLY, "location",
       "IANA timezone the server renders dates, schedules and dark hours in.",
       default="America/Chicago"),

    # ── Weather providers ────────────────────────────────────────────────────
    _s("WEATHERKIT_TEAM_ID", "str", _SERVER, "weather",
       "Apple Developer team ID for WeatherKit.", provider=True),
    _s("WEATHERKIT_KEY_ID", "str", _SERVER, "weather",
       "WeatherKit key ID.", provider=True),
    _s("WEATHERKIT_SERVICE_ID", "str", _SERVER, "weather",
       "WeatherKit service ID.", provider=True),
    _s("WEATHERKIT_KEY_PATH", "path", _SERVER, "weather",
       "Path to the WeatherKit .p8 private key (alternative to WEATHERKIT_PRIVATE_KEY).",
       provider=True),
    _s("WEATHERKIT_PRIVATE_KEY", "str", _SERVER, "weather",
       "Inline WeatherKit .p8 private key contents.", secret=True, provider=True),
    _s("WEATHERKIT_LANGUAGE", "str", _SERVER, "weather",
       "WeatherKit response language.", default="en"),
    _s("WEATHERKIT_TIMEZONE", "timezone", _SERVER, "weather",
       "Timezone requested from WeatherKit.", default="America/Chicago"),
    _s("OWM_API_KEY", "str", _SERVER, "weather",
       "OpenWeatherMap One Call API key (fallback provider).", secret=True, provider=True),
    _s("OWM_API_KEY_DEFAULT", "str", _SERVER, "weather",
       "Additional OpenWeatherMap key slot; one populated slot is chosen at random.",
       secret=True, provider=True),
    _s("OWM_API_KEY_WIFFY", "str", _SERVER, "weather",
       "Additional OpenWeatherMap key slot.", secret=True, provider=True),
    _s("OWM_API_KEY_VERANO", "str", _SERVER, "weather",
       "Additional OpenWeatherMap key slot.", secret=True, provider=True),
    _s("OWM_UNITS", "choice", _SERVER, "weather",
       "OpenWeatherMap units.", default="imperial", choices=("imperial", "metric", "standard")),
    _s("OWM_LANGUAGE", "str", _SERVER, "weather", "OpenWeatherMap language.", default="en"),
    _s("HOURLY_FORECAST_HOURS", "int", _SERVER, "weather",
       "Number of hourly forecast entries shown.", default="5", minimum=1),
    _s("WEATHER_USE_EMOJI_ICONS", "bool", _SERVER, "weather",
       "Draw emoji weather symbols where supported.", default="0"),

    # ── Air quality ──────────────────────────────────────────────────────────
    _s("AIR_QUALITY_PROVIDER", "choice", _SERVER, "air_quality",
       "Outdoor air-quality provider.", default="airnow", choices=("airnow",)),
    _s("AIRNOW_API_KEY", "str", _SERVER, "air_quality",
       "Free AirNow API key (https://docs.airnowapi.org/account/request/).",
       secret=True, provider=True),
    _s("AIR_QUALITY_LATITUDE", "latitude", _SERVER, "air_quality",
       "AQI latitude; defaults to WEATHER_LATITUDE. Set both AQI coordinates or neither."),
    _s("AIR_QUALITY_LONGITUDE", "longitude", _SERVER, "air_quality",
       "AQI longitude; defaults to WEATHER_LONGITUDE."),
    _s("AIR_QUALITY_ENABLE_POLLEN", "bool", _SERVER, "air_quality",
       "Include pollen data where the provider supplies it.", default="1"),

    # ── Sports and teams ─────────────────────────────────────────────────────
    _s("NCAAM_SCOREBOARD_MODE", "str", _SERVER, "sports",
       "NCAAM scoreboard mode.", default="top25"),
    _s("NHL_BREAK_WINDOWS_JSON", "json", _SERVER, "sports",
       "Optional JSON override for NHL break windows (All-Star, Olympics)."),
    _s("NHL_SCHEDULE_ICS_URL", "url", _SERVER, "sports",
       "NHL schedule calendar feed (https or webcal).",
       default="webcal://ics.ecal.com/ecal-sub/6a5e38cbff3cfc0002c15087/NHL.ics", provider=True),
    _s("WORLD_CUP_PREGAME_SCORE_DISPLAY", "choice", _SERVER, "sports",
       "What pregame World Cup score columns show.", default="dash",
       choices=("dash", "abbreviation", "abbr", "abbrev", "team", "team_abbreviation")),
    _s("TEAM_STANDINGS_DISPLAY_SECONDS", "float", _SERVER, "sports",
       "Hold time for each team standings page.", default="5", minimum=0),
    _s("MLB_SCOREBOARD_SCROLL_DELAY", "float", _SERVER, "sports",
       "MLB scoreboard scroll delay; defaults to the profile's scoreboard scroll delay.", minimum=0),
    _s("SCOREBOARD_STANDINGS_BOTTOM_PADDING", "int", _SERVER, "sports",
       "Bottom padding below scoreboard standings rows, in pixels.", default="30", minimum=0),
    _s("SMALL_RESULT_FLAG_H", "int", _SERVER, "sports",
       "Result flag height for MLB schedule screens.", default="48", minimum=1),
    _s("NIXIE_TIME_FORMAT", "choice", _SERVER, "sports",
       "Nixie clock format.", default="12", choices=("12", "24")),
    _s("AHL_API_BASE_URL", "url", _SERVER, "sports",
       "HockeyTech feed base URL for the AHL/Chicago Wolves.",
       default="https://lscluster.hockeytech.com/feed/", provider=True),
    _s("AHL_API_KEY", "str", _SERVER, "sports",
       "HockeyTech API key; AHL requests are skipped when empty.", secret=True, provider=True),
    _s("AHL_CLIENT_CODE", "str", _SERVER, "sports", "HockeyTech client code.", default="ahl"),
    _s("AHL_LEAGUE_ID", "int", _SERVER, "sports", "HockeyTech league ID.", default="4"),
    _s("AHL_SITE_ID", "int", _SERVER, "sports", "HockeyTech site ID.", default="1"),
    _s("AHL_SEASON_ID", "int", _SERVER, "sports",
       "HockeyTech season ID; detected automatically when empty."),
    _s("AHL_SCHEDULE_ICS_URL", "url", _SERVER, "sports",
       "Private StanzaCal (or compatible https/webcal) ICS URL for the Wolves schedule.",
       secret=True, provider=True),
    _s("AHL_TEAM_ID", "int", _SERVER, "sports", "HockeyTech team ID.", default="624"),
    _s("AHL_TEAM_TRICODE", "str", _SERVER, "sports", "AHL team tricode.", default="CHI"),
    _s("AHL_TEAM_NAME", "str", _SERVER, "sports", "AHL team name.", default="Chicago Wolves"),

    # ── News, stocks, On This Day ────────────────────────────────────────────
    _s("ENABLE_NEWS_HEADLINES", "bool", _SERVER, "news",
       "Enable the news headlines screen.", default="1"),
    _s("ENABLE_NEWS_HEADLINES_2", "bool", _SERVER, "news",
       "Enable the independently configured second news screen.", default="1"),
    _s("NEWS_FEEDS_CONFIG_PATH", "path", _SERVER, "news",
       "News feed topics file; defaults to news_feeds.json. Its contents hot-reload."),
    _s("NEWS_FEEDS_CONFIG_PATH_2", "path", _SERVER, "news",
       "Second news screen's feed file; defaults to news_feeds_2.json. Its contents hot-reload."),
    _s("NEWS_HEADLINES_DISPLAY_SECONDS", "float", _SERVER, "news",
       "On-screen time for the news ticker.", default="30", minimum=0),
    _s("NEWS_HEADLINES_SHOW_IMAGES", "bool", _SERVER, "news",
       "Fetch and draw headline thumbnails.", default="1"),
    _s("NEWS_TICKER_BASE_SPEED", "float", _SERVER, "news",
       "Base ticker scroll speed in pixels per frame.", default="2.3", minimum=0),
    _s("NEWS_ARTICLE_FETCH_TIMEOUT_SECONDS", "float", _SERVER, "news",
       "Timeout for fetching an article opened from the ticker.", default="6", minimum=0),
    _s("ENABLE_STOCK_TICKER", "bool", _SERVER, "news",
       "Append the stock quote row to the news ticker.", default="1"),
    _s("STOCK_TICKER_CACHE_TTL_SECONDS", "int", _SERVER, "news",
       "Stock quote cache lifetime.", default="900", minimum=0),
    _s("ON_THIS_DAY_FEED_BUILD_TIMEOUT_SECONDS", "float", _SERVER, "news",
       "Time budget for building the On This Day feed.", default="3.5", minimum=0.5),
    _s("ON_THIS_DAY_INCOMPLETE_FEED_RETRY_SECONDS", "float", _SERVER, "news",
       "Retry interval after an incomplete Wikimedia response.", default="300", minimum=30),
    _s("ON_THIS_DAY_OFFLINE_FALLBACK_RETRY_SECONDS", "float", _SERVER, "news",
       "Retry interval after every On This Day feed fails.", default="900", minimum=60),
    _s("ON_THIS_DAY_LIVE_THUMBNAILS", "bool", _SERVER, "news",
       "Download live Wikimedia thumbnails.", default="0"),
    _s("WIKIMEDIA_USER_AGENT", "str", _SERVER, "news",
       "User agent sent to Wikimedia; include your own contact URL when redistributing.",
       default="DeskDisplay/1.0 (https://github.com/mrjrask/desk_display)", provider=True),

    # ── ADS-B ────────────────────────────────────────────────────────────────
    _s("ADSB_DEVICE_1_HOST", "str", _SERVER, "adsb",
       "Host or IP of the first dump1090-fa receiver.", provider=True),
    _s("ADSB_DEVICE_1_LABEL", "str", _SERVER, "adsb", "Label for receiver 1.", example=""),
    _s("ADSB_DEVICE_2_HOST", "str", _SERVER, "adsb",
       "Optional second receiver; leave empty for one receiver.", provider=True),
    _s("ADSB_DEVICE_2_LABEL", "str", _SERVER, "adsb", "Label for receiver 2."),
    _s("ADSB_HOME_LATITUDE", "latitude", _SERVER, "adsb",
       "Receiver site latitude for distances; defaults to WEATHER_LATITUDE."),
    _s("ADSB_HOME_LONGITUDE", "longitude", _SERVER, "adsb",
       "Receiver site longitude; defaults to WEATHER_LONGITUDE."),
    _s("ADSB_DISTANCE_UNIT", "choice", _SERVER, "adsb",
       "Distance unit shown on screen.", default="nm", choices=("nm", "mi")),
    _s("ADSB_POLL_INTERVAL_SECONDS", "int", _SERVER, "adsb",
       "How often the collector polls each receiver.", default="10", minimum=1),
    _s("ADSB_REQUEST_TIMEOUT_SECONDS", "int", _SERVER, "adsb",
       "Collector HTTP timeout per request.", default="5", minimum=1),
    _s("ADSB_RETENTION_DAYS", "int", _SERVER, "adsb",
       "Days of raw sightings kept before pruning.", default="7", minimum=1),
    _s("ADSB_DB_PATH", "path", _SERVER, "adsb",
       "SQLite database path; defaults to cache/adsb_stats.db."),
    _s("ADSB_TYPE_DB_ENABLED", "bool", _SERVER, "adsb",
       "Download the aircraft type database for type lookups.", default="1"),
    _s("ADSB_TYPE_DB_URL", "url", _SERVER, "adsb",
       "Aircraft type database download URL.",
       default="https://raw.githubusercontent.com/wiedehopf/tar1090-db/csv/aircraft.csv.gz",
       provider=True),
    _s("ADSB_TYPE_DB_PATH", "path", _SERVER, "adsb",
       "Aircraft type database path; defaults to cache/aircraft_types.db."),
    _s("ADSB_TYPE_DB_REFRESH_DAYS", "int", _SERVER, "adsb",
       "Days between aircraft type database refreshes.", default="30", minimum=0),

    # ── Screen configuration, styles and layouts ─────────────────────────────
    _s("SCREENS_CONFIG_PATH", "path", _SERVER, "styles",
       "Screen schedule/playlist file; defaults to screens_config.json. Its contents hot-reload."),
    _s("SCREENS_CONFIG_LOCAL_PATH", "path", _SERVER, "styles",
       "Local screen-config override file layered over SCREENS_CONFIG_PATH."),
    _s("SCREENS_STYLE_PATH", "path", _SERVER, "styles",
       "Screen style overrides; defaults to screens_style.json. Its contents hot-reload."),
    _s("SCREENS_LAYOUTS_PATH", "path", _SERVER, "styles",
       "Quad screen layouts; defaults to screens_layouts.json. Its contents hot-reload."),
    _s("DEFAULT_SCREENS_PATH", "path", _SERVER, "styles",
       "Default screen bundle used by the config UI; defaults to default_screens_large.json."),
    _s("DEFAULT_SCREENS_LARGE_PATH", "path", _SERVER, "styles",
       "Large-display default bundle; defaults to DEFAULT_SCREENS_PATH."),
    _s("DEFAULT_SCREENS_SMALL_PATH", "path", _SERVER, "styles",
       "Small-display default bundle; defaults to default_screens_small.json."),

    # ── Refresh intervals ────────────────────────────────────────────────────
    _s("WEATHER_REFRESH_SECONDS", "int", _SERVER, "refresh",
       "Weather refresh interval.", default="1800", minimum=60),
    _s("STARTUP_CRITICAL_FEED_TIMEOUT_SECONDS", "float", _SERVER, "refresh",
       "How long startup waits for critical feeds before rendering.", default="8", minimum=0),
    _s("HTTP_CLIENT_FORBIDDEN_COOLDOWN_SECONDS", "float", _SERVER, "refresh",
       "Back-off after an upstream answers 403 Forbidden.", default="300", minimum=0),
    _s("HTTP_CLIENT_USE_SYSTEM_PROXIES", "bool", _ALL, "refresh",
       "Honour HTTP(S)_PROXY and system proxy settings for outgoing requests.", default="0"),

    # ── Render workers, artifacts, leases, static clients (split runtime) ────
    _s("DESK_DISPLAY_RENDER_WORKERS", "int", _SERVER_ONLY, "render",
       "Parallel render workers; each renders one (screen, profile) artifact at a time.",
       default="2", minimum=1, maximum=64),
    _s("DESK_DISPLAY_RENDER_TIMEOUT_SECONDS", "float", _SERVER_ONLY, "render",
       "Abandon a single artifact render after this long.", default="30", minimum=1),
    _s("DESK_DISPLAY_ARTIFACT_DIR", "path", _SERVER_ONLY, "artifacts",
       "Directory for rendered artifacts and packages; defaults to cache/artifacts."),
    _s("DESK_DISPLAY_ARTIFACT_RETENTION_HOURS", "float", _SERVER_ONLY, "artifacts",
       "Delete artifacts no manifest has referenced for this long.", default="24", minimum=1),
    _s("DESK_DISPLAY_ARTIFACT_MAX_MB", "int", _SERVER_ONLY, "artifacts",
       "Upper bound for the artifact store; the oldest unreferenced artifacts go first.",
       default="512", minimum=16),
    _s("DESK_DISPLAY_CLIENT_LEASE_SECONDS", "int", _SERVER_ONLY, "leases",
       "How long a client registration stays active without a heartbeat.",
       default="300", minimum=30),
    _s("DESK_DISPLAY_STATIC_CLIENTS", "static_clients", _SERVER_ONLY, "leases",
       "Clients pre-registered at startup, as comma-separated client_id:profile pairs, "
       "so their artifacts render before they first connect.",
       example=""),

    # ── Config UI ────────────────────────────────────────────────────────────
    _s("SCREEN_CONFIG_HOST", "str", _SERVER, "config_ui",
       "Config UI bind address.", default="0.0.0.0"),
    _s("SCREEN_CONFIG_PORT", "int", _SERVER, "config_ui",
       "Config UI port.", default="5002", minimum=1, maximum=65535),
    _s("SCREEN_UI_USERNAME", "str", _SERVER, "config_ui", "Optional login username."),
    _s("SCREEN_UI_PASSWORD", "str", _SERVER, "config_ui",
       "Password required for UI pages and APIs when set.", secret=True),
    _s("SCREEN_SESSION_SECRET", "str", _SERVER, "config_ui",
       "Dedicated Flask session-signing secret; otherwise derived from the password "
       "or a per-process random value.", secret=True),
    _s("SCREEN_AUTH_ENABLED", "bool", _SERVER, "config_ui",
       "Force authentication on; startup fails when SCREEN_UI_PASSWORD is empty.", default="0"),

    # ── Screenshot feed server ───────────────────────────────────────────────
    _s("FEED_SERVER_HOST", "str", _SERVER, "feed_server",
       "Screenshot feed server bind address.", default="0.0.0.0"),
    _s("FEED_SERVER_PORT", "int", _SERVER, "feed_server",
       "Screenshot feed server port.", default="5003", minimum=1, maximum=65535),
    _s("FEED_STORAGE_DIR", "path", _SERVER, "feed_server",
       "Where uploaded screenshots are kept; defaults to feed_uploads/."),
    _s("FEED_MAX_UPLOAD_BYTES", "int", _SERVER, "feed_server",
       "Largest accepted screenshot upload.", default="8388608", minimum=1024),
    _s("FEED_STALE_SECONDS", "int", _SERVER, "feed_server",
       "Mark a feed source stale after this long without uploads.", default="120", minimum=0),
    _s("FEED_HEARTBEAT_STALE_SECONDS", "int", _SERVER, "feed_server",
       "Mark a source's heartbeat stale after this long.", default="600", minimum=0),
    _s("FEED_SCREEN_STALE_SECONDS", "int", _SERVER, "feed_server",
       "Hide a screen's screenshot from the Feed page after this long.", default="1200", minimum=0),

    # ── Server caches and history ────────────────────────────────────────────
    _s("PRESSURE_HISTORY_PATH", "path", _SERVER, "history",
       "Pressure trend history; defaults to cache/pressure_history.json."),
    _s("WEATHER_METRIC_HISTORY_PATH", "path", _SERVER, "history",
       "Weather chart history; defaults to cache/weather_metric_history.json."),
    _s("AIR_QUALITY_HISTORY_PATH", "path", _SERVER, "history",
       "Air-quality history; defaults to cache/air_quality_history.json."),
    _s("ON_THIS_DAY_CACHE_PATH", "path", _SERVER, "history",
       "Daily On This Day cache; defaults to the application cache directory."),

    # ── Display profile ──────────────────────────────────────────────────────
    _s("DESK_DISPLAY_PROFILE", "profile", _CLIENT, "profile",
       "Display profile: display_hat_mini, adafruit_minipitft_114, hyperpixel4, "
       "hyperpixel4_square, waveshare_lcd_320x240, waveshare_oled_128x64, hdmi_1080p, "
       "fallback_hd or fallback_default. Required on clients; standalone detects it "
       "from DISPLAY_WIDTH/DISPLAY_HEIGHT when empty.",
       example="hyperpixel4"),
    _s("DISPLAY_WIDTH", "int", _CLIENT, "profile",
       "Render width in pixels.", default="800", minimum=1),
    _s("DISPLAY_HEIGHT", "int", _CLIENT, "profile",
       "Render height in pixels.", default="480", minimum=1),
    _s("DISPLAY_RESOLUTION", "str", _STANDALONE_ONLY, "profile",
       "WIDTHxHEIGHT shorthand read by scripts/render_screens.py."),
    _s("HYPERPIXEL_PANEL", "choice", _CLIENT, "profile",
       "HyperPixel panel hint; hyperpixel4 normalizes 480x800 to 800x480.",
       choices=("hyperpixel4",)),
    _s("DESK_DISPLAY_LOW_POWER", "bool", _CLIENT, "profile",
       "Low-power preset: fewer radar frames and no optional screenshot/video/Wi-Fi "
       "work. Auto-enabled on Pi Zero boards; set 0 to force off."),

    # ── Output driver ────────────────────────────────────────────────────────
    _s("DESK_DISPLAY_OUTPUT", "output", _CLIENT, "output",
       "Output driver: auto, displayhatmini, minipitft, kernel, window, framebuffer "
       "or headless (aliases accepted).", default="auto"),
    _s("DESK_DISPLAY_FORCE_HEADLESS", "bool", _CLIENT, "output",
       "Force headless output regardless of DESK_DISPLAY_OUTPUT.", default="0"),
    _s("DISPLAY_FADE_IN_ENABLED", "bool", _CLIENT, "output",
       "Fade new frames in where supported.", default="1"),
    _s("DISPLAY_FADE_IN_DISPLAY_HAT_MINI_STEPS", "int", _CLIENT, "output",
       "Fade-in steps on Display HAT Mini; 0 disables.", default="10", minimum=0),
    _s("DISPLAY_FADE_IN_HYPERPIXEL_STEPS", "int", _CLIENT, "output",
       "Fade-in steps on HyperPixel; 0 disables.", default="0", minimum=0),
    _s("DISPLAY_FADE_IN_HDMI_1080P_STEPS", "int", _CLIENT, "output",
       "Fade-in steps on HDMI 1080p; 0 disables.", default="0", minimum=0),

    # ── Physical rotation ────────────────────────────────────────────────────
    _s("DISPLAY_ROTATION", "rotation", _CLIENT, "rotation",
       "Physical rotation applied at presentation: 0, 90, 180 or 270 degrees, or 0-3 "
       "quarter turns.", default="0"),
    _s("DISPLAY_ROTATION_STRICT", "bool", _CLIENT, "rotation",
       "Reject invalid rotations and avoid double rotation when a kernel overlay "
       "already rotates.", default="1"),

    # ── Framebuffer and panel hardware ───────────────────────────────────────
    _s("DISPLAY_FB_DEVICE", "path", _CLIENT, "hardware",
       "Framebuffer device.", default="/dev/fb0"),
    _s("DISPLAY_FB_PIXEL_FORMAT", "choice", _CLIENT, "hardware",
       "Framebuffer pixel format override; auto-detected when empty.",
       choices=("rgb565", "bgr565", "rgb888", "bgr888", "xrgb8888", "xbgr8888",
                "argb8888", "abgr8888", "rgba8888", "bgra8888")),
    _s("DISPLAY_FB_PIXEL_ORDER", "choice", _CLIENT, "hardware",
       "Framebuffer channel order override.", choices=("rgb", "bgr")),
    _s("DISPLAY_FB_HIDE_CONSOLE_CURSOR", "bool", _CLIENT, "hardware",
       "Hide the Linux console cursor in framebuffer mode.", default="1"),
    _s("DISPLAY_FB_CONSOLE_GRAPHICS", "bool", _CLIENT, "hardware",
       "Switch the console to graphics mode while the framebuffer is open.", default="1"),
    _s("DISPLAY_HAT_MINI_LED_LEVEL", "float", _CLIENT, "hardware",
       "Display HAT Mini LED brightness; driver default when empty.", minimum=0),
    _s("DISPLAY_HAT_MINI_REINIT_SECONDS", "float", _CLIENT, "hardware",
       "Periodic Display HAT Mini re-initialization; 0 disables (recommended).",
       default="0", minimum=0),
    _s("DISPLAY_HAT_MINI_IO_TIMEOUT_SECONDS", "float", _CLIENT, "hardware",
       "Exit for systemd recovery when an SPI write blocks this long; 0 disables.",
       default="15", minimum=0),
    _s("DISPLAY_HAT_MINI_MAX_REFRESH_FAILURES", "int", _CLIENT, "hardware",
       "Consecutive refresh errors before exiting for recovery; 0 disables.",
       default="3", minimum=0),
    _s("LED_INDICATOR_ENABLED", "bool", _CLIENT, "hardware",
       "Drive the Display HAT Mini RGB notification LED.", default="1"),
    _s("LED_INDICATOR_BORDER_ENABLED", "bool", _CLIENT, "hardware",
       "Draw the notification colour as a frame border on any display.", default="1"),
    _s("LED_INDICATOR_BORDER_WIDTH", "int", _CLIENT, "hardware",
       "Notification border width in pixels.", default="2", minimum=0),
    _s("MINIPITFT_BAUDRATE", "int", _CLIENT, "hardware",
       "miniPiTFT SPI baud rate.", default="64000000", minimum=1),
    _s("MINIPITFT_DRIVER_ROTATION", "int", _CLIENT, "hardware",
       "miniPiTFT driver rotation.", default="90"),
    _s("MINIPITFT_DRIVER_WIDTH", "int", _CLIENT, "hardware",
       "miniPiTFT driver width.", default="135", minimum=1),
    _s("MINIPITFT_DRIVER_HEIGHT", "int", _CLIENT, "hardware",
       "miniPiTFT driver height.", default="240", minimum=1),
    _s("MINIPITFT_X_OFFSET", "int", _CLIENT, "hardware", "miniPiTFT X offset.", default="53"),
    _s("MINIPITFT_Y_OFFSET", "int", _CLIENT, "hardware", "miniPiTFT Y offset.", default="40"),

    # ── SDL window and desktop session ───────────────────────────────────────
    _s("DESK_DISPLAY_WINDOW_SCALE", "float", _CLIENT, "window",
       "SDL window scale factor.", default="1", minimum=0.1),
    _s("DESK_DISPLAY_WINDOW_RESIZABLE", "bool", _CLIENT, "window",
       "Allow resizing the SDL window.", default="0"),
    _s("DESK_DISPLAY_SDL_FULLSCREEN", "bool", _CLIENT, "window",
       "Start SDL fullscreen.", default="0"),
    _s("DESK_DISPLAY_SDL_DRIVERS", "csv", _CLIENT, "window",
       "Comma-separated SDL video drivers to try in order."),
    _s("SDL_VIDEODRIVER", "str", _CLIENT, "window", "Force a single SDL video driver."),
    _s("DESK_DISPLAY_SESSION_USER", "str", _CLIENT, "window",
       "Desktop user whose session a window/kernel output attaches to."),
    _s("DISPLAY", "str", _CLIENT, "window",
       "X11 display, usually inherited; set only when launching manually."),
    _s("WAYLAND_DISPLAY", "str", _CLIENT, "window", "Wayland display, usually inherited."),
    _s("XAUTHORITY", "path", _CLIENT, "window", "X11 authority file, usually inherited."),
    _s("XDG_RUNTIME_DIR", "path", _CLIENT, "window", "Session runtime directory, usually inherited."),

    # ── Local cache, sync, offline startup (split runtime) ───────────────────
    _s("DESK_DISPLAY_CLIENT_CACHE_DIR", "path", _CLIENT_ONLY, "cache",
       "Where the client keeps its manifest, playlist and packages; defaults to cache/client."),
    _s("DESK_DISPLAY_CLIENT_CACHE_MAX_MB", "int", _CLIENT_ONLY, "cache",
       "Upper bound for cached packages.", default="256", minimum=16),
    _s("DESK_DISPLAY_SYNC_INTERVAL_SECONDS", "int", _CLIENT_ONLY, "sync",
       "How often the client checks the server for a new manifest.", default="30", minimum=5),
    _s("DESK_DISPLAY_HEARTBEAT_INTERVAL_SECONDS", "int", _CLIENT_ONLY, "sync",
       "How often the client reports status; keep well under the server lease.",
       default="60", minimum=5),
    _s("DESK_DISPLAY_OFFLINE_START", "bool", _CLIENT_ONLY, "offline",
       "Start from a complete, compatible local cache when the server is unreachable.",
       default="1"),
    _s("DESK_DISPLAY_OFFLINE_MAX_AGE_HOURS", "float", _CLIENT_ONLY, "offline",
       "Stop showing cached content older than this while offline; 0 keeps it indefinitely.",
       default="0", minimum=0),

    # ── Backlight and dark hours ─────────────────────────────────────────────
    _s("DARK_HOURS", "str", _CLIENT, "backlight",
       "Dark-hours schedule, e.g. 'Mon-Fri 22:00-06:30; Sat-Sun 23:00-08:00'."),
    _s("DESK_DISPLAY_BACKLIGHT_LEVEL", "int", _CLIENT_ONLY, "backlight",
       "Backlight brightness percentage outside dark hours.", default="100",
       minimum=0, maximum=100),
    _s("DESK_DISPLAY_DARK_HOURS_MODE", "choice", _CLIENT_ONLY, "backlight",
       "What the client does during DARK_HOURS.", default="off", choices=("off", "dim")),
    _s("DESK_DISPLAY_DARK_HOURS_BACKLIGHT_LEVEL", "int", _CLIENT_ONLY, "backlight",
       "Backlight percentage when DESK_DISPLAY_DARK_HOURS_MODE=dim.", default="10",
       minimum=0, maximum=100),

    # ── Buttons, touch and keyboard ──────────────────────────────────────────
    _s("BUTTON_A", "int", _CLIENT, "buttons", "GPIO pin override for button A.", minimum=0),
    _s("BUTTON_B", "int", _CLIENT, "buttons", "GPIO pin override for button B.", minimum=0),
    _s("BUTTON_X", "int", _CLIENT, "buttons", "GPIO pin override for button X.", minimum=0),
    _s("BUTTON_Y", "int", _CLIENT, "buttons", "GPIO pin override for button Y.", minimum=0),
    _s("DESK_DISPLAY_RPI_GPIO_FALLBACK", "bool", _CLIENT, "buttons",
       "Fall back to RPi.GPIO when the preferred GPIO library is unavailable.", default="1"),
    _s("TOUCH_DOUBLE_TAP_MAX_INTERVAL_SECONDS", "float", _CLIENT, "touch",
       "Maximum gap between taps of a double tap.", default="0.45", minimum=0),
    _s("ESC_DOUBLE_PRESS_ACTION", "choice", _CLIENT, "touch",
       "What a double Escape press does in window/SDL outputs.", default="stop",
       choices=("stop", "exit", "none")),
    _s("ESC_DOUBLE_PRESS_MAX_INTERVAL_SECONDS", "float", _CLIENT, "touch",
       "Maximum gap between Escape presses.", default="1.0", minimum=0),

    # ── Screenshots and feed upload ──────────────────────────────────────────
    _s("ENABLE_SCREENSHOTS", "bool", _CLIENT, "screenshots",
       "Save screenshots of presented frames.", default="1"),
    _s("ENABLE_VIDEO", "bool", _CLIENT, "screenshots",
       "Record presented frames to video.", default="0"),
    _s("SCREENSHOT_DIR", "path", _CLIENT, "screenshots", "Screenshot directory."),
    _s("SCREENSHOT_ARCHIVE_BASE", "path", _CLIENT, "screenshots", "Screenshot archive directory."),
    _s("FEED_UPLOAD_URL", "url", _CLIENT, "screenshots",
       "Screenshot feed server URL for scripts/screenshot_uploader.py."),
    _s("FEED_UPLOAD_TOKEN", "str", _ALL, "screenshots",
       "Shared upload token; the feed server rejects uploads without it.", secret=True),
    _s("FEED_SOURCE_NAME", "str", _CLIENT, "screenshots",
       "Name this display uploads under; defaults to the hostname."),
    _s("FEED_UPLOAD_INTERVAL_SECONDS", "float", _CLIENT, "screenshots",
       "Seconds between upload passes.", default="5", minimum=1),
    _s("FEED_UPLOAD_TIMEOUT_SECONDS", "float", _CLIENT, "screenshots",
       "Timeout per upload request.", default="10", minimum=1),
    _s("FEED_UPLOAD_CYCLE_TIMEOUT_SECONDS", "float", _CLIENT, "screenshots",
       "Time budget for one upload pass.", default="30", minimum=1),
    _s("FEED_UPLOAD_MAX_BACKOFF_SECONDS", "float", _CLIENT, "screenshots",
       "Longest retry back-off after failed uploads.", default="300", minimum=1),

    # ── Indoor sensor ────────────────────────────────────────────────────────
    _s("INSIDE_SENSOR", "choice", _CLIENT, "sensors",
       "Indoor sensor override; auto-detected when empty.",
       choices=("pim_sensor_stick", "pimoroni_bme280", "adafruit_bme280", "pimoroni_bme680",
                "pimoroni_bme68x", "adafruit_bme680", "adafruit_sht41", "adafruit_sht4x")),
    _s("INDOOR_SENSOR", "str", _CLIENT, "sensors", "Backward-compatible alias for INSIDE_SENSOR."),
    _s("INSIDE_I2C_BUSES", "csv", _CLIENT, "sensors",
       "I2C buses probed for the indoor sensor.", default="1,2,10,11,13,14,15"),
    _s("INSIDE_HISTORY_PATH", "path", _CLIENT, "sensors",
       "Indoor sensor history; defaults to cache/inside_history.json."),

    # ── Wi-Fi ────────────────────────────────────────────────────────────────
    _s("ENABLE_WIFI_MONITOR", "bool", _CLIENT, "wifi", "Monitor Wi-Fi connectivity.", default="1"),
    _s("ENABLE_WIFI_RECOVERY", "bool", _CLIENT, "wifi",
       "Attempt Wi-Fi recovery after sustained loss.", default="1"),
    _s("WIFI_INTERFACE", "str", _CLIENT, "wifi", "Wi-Fi interface; auto-detected when empty."),
    _s("WIFI_RECOVERY_LOG", "path", _CLIENT, "wifi", "Wi-Fi recovery log file."),
    _s("WIFI_TCP_PROBE_URLS", "csv", _CLIENT, "wifi", "Comma-separated connectivity probe URLs."),
    _s("WIFI_TCP_PROBE_URL", "url", _CLIENT, "wifi", "Single connectivity probe URL."),
    _s("WIFI_HTTPS_PROBE_URL", "url", _CLIENT, "wifi", "HTTPS connectivity probe URL."),
    _s("WIFI_TCP_PROBE_HOSTS", "csv", _CLIENT, "wifi", "Comma-separated TCP probe hosts."),
    _s("WIFI_TCP_PROBE_HOST", "str", _CLIENT, "wifi", "Single TCP probe host."),
    _s("WIFI_TCP_PROBE_PORT", "int", _CLIENT, "wifi",
       "TCP probe port.", default="443", minimum=1, maximum=65535),
    _s("RPI_CONNECT_CONTROL_HOST", "str", _CLIENT, "wifi",
       "Raspberry Pi Connect control host used for reachability checks."),

    # ── Waveshare OLED/LCD HAT helper ────────────────────────────────────────
    _s("WAVESHARE_OLED_LCD_HAT_A_INSTALLED", "bool", _CLIENT, "waveshare",
       "Set by the Waveshare installer; enables the HAT's layout tweaks.", default="0"),
    _s("WAVESHARE_OLED_I2C_BUS", "int", _CLIENT, "waveshare", "OLED I2C bus.", default="1"),
    _s("WAVESHARE_OLED_TEMP_ADDR", "int", _CLIENT, "waveshare",
       "Temperature OLED I2C address.", default="0x3C"),
    _s("WAVESHARE_OLED_TIME_ADDR", "int", _CLIENT, "waveshare",
       "Time OLED I2C address.", default="0x3D"),
    _s("WAVESHARE_OLED_WIDTH", "int", _CLIENT, "waveshare", "OLED width.", default="128", minimum=1),
    _s("WAVESHARE_OLED_HEIGHT", "int", _CLIENT, "waveshare", "OLED height.", default="64", minimum=1),
    _s("WAVESHARE_OLED_MIN_VALUE_FONT_SIZE", "int", _CLIENT, "waveshare",
       "Smallest value font size.", default="8", minimum=6),
    _s("WAVESHARE_OLED_MAX_VALUE_FONT_SIZE", "int", _CLIENT, "waveshare",
       "Largest value font size.", minimum=6),
    _s("WAVESHARE_OLED_MIN_TIME_FONT_SIZE", "int", _CLIENT, "waveshare",
       "Smallest time font size; defaults to the minimum value size.", minimum=6),
    _s("WAVESHARE_OLED_MAX_TIME_FONT_SIZE", "int", _CLIENT, "waveshare",
       "Largest time font size.", minimum=6),
    _s("WAVESHARE_OLED_REFRESH_SECONDS", "int", _CLIENT, "waveshare",
       "OLED refresh interval.", default="5", minimum=1),
    _s("WAVESHARE_OLED_STATUS_MAX_AGE_SECONDS", "int", _CLIENT, "waveshare",
       "Treat display status older than this as stale.", default="300", minimum=1),
    _s("WAVESHARE_OLED_SWAP_INTERVAL_MIN_SECONDS", "int", _CLIENT, "waveshare",
       "Minimum seconds between OLED content swaps.", default="60", minimum=1),
    _s("WAVESHARE_OLED_SWAP_INTERVAL_MAX_SECONDS", "int", _CLIENT, "waveshare",
       "Maximum seconds between OLED content swaps.", default="240", minimum=1),
    _s("WAVESHARE_OLED_FADE_STEPS", "int", _CLIENT, "waveshare",
       "Fade steps between OLED frames.", default="8", minimum=1),
    _s("WAVESHARE_OLED_FADE_STEP_MS", "int", _CLIENT, "waveshare",
       "Milliseconds per fade step.", default="35", minimum=5),
    _s("WAVESHARE_OLED_TEMP_SOURCE", "str", _CLIENT, "waveshare",
       "Temperature source screen for the OLED.", default="weather1"),
    _s("WAVESHARE_OLED_TEMP_COMMAND", "str", _CLIENT, "waveshare",
       "Optional shell command that prints a temperature."),
    _s("WAVESHARE_OLED_TEMP_UNIT", "choice", _CLIENT, "waveshare",
       "OLED temperature unit.", default="C", choices=("C", "F", "c", "f")),
    _s("WAVESHARE_OLED_WAIT_FOR_WEATHER2", "bool", _CLIENT, "waveshare",
       "Wait for the weather2 screenshot before showing temperature.", default="1"),
    _s("WAVESHARE_OLED_CUBS_FINAL_STATE_PATH", "path", _CLIENT, "waveshare",
       "Cubs final-state marker file.", default="/var/tmp/desk_display_cubs_final_state.json"),
    _s("WAVESHARE_OLED_HAWKS_FINAL_STATE_PATH", "path", _CLIENT, "waveshare",
       "Blackhawks final-state marker file.", default="/var/tmp/desk_display_hawks_final_state.json"),
    _s("WAVESHARE_OLED_DISPLAY_STATUS_PATH", "path", _CLIENT, "waveshare",
       "Display status file the helper reads; defaults under the screenshot directory."),
    _s("WAVESHARE_OLED_SCREENSHOT_DIR", "path", _CLIENT, "waveshare",
       "Screenshot directory the helper reads; defaults to SCREENSHOT_DIR."),
    _s("WAVESHARE_OLED_FONT_PATH", "path", _CLIENT, "waveshare", "Font file for the OLEDs."),
    _s("WAVESHARE_OLED_LOG_LEVEL", "log_level", _CLIENT, "waveshare",
       "Helper log level.", default="INFO"),

    # ── Standalone-only runtime behavior ─────────────────────────────────────
    _s("SCREEN_CONFIG_AUTOSTART", "bool", _STANDALONE_ONLY, "standalone",
       "Start the config UI from main.py.", default="1"),
    _s("IP_WITH_TIME", "bool", _STANDALONE_ONLY, "standalone",
       "Show this device's IP address on the date/time screen.", default="1"),

    # ── Logs ─────────────────────────────────────────────────────────────────
    _s("DESK_DISPLAY_LOG_LEVEL", "log_level", _ALL, "logs",
       "Log level for the display, config UI and feed server processes.", default="INFO"),
    _s("FEED_UPLOAD_LOG_LEVEL", "log_level", _CLIENT, "logs",
       "Log level for scripts/screenshot_uploader.py.", default="INFO"),

    # ── Diagnostics ──────────────────────────────────────────────────────────
    _s("DESK_DISPLAY_GC_INTERVAL_SECONDS", "float", _ALL, "diagnostics",
       "Periodic garbage-collection interval.", default="30", minimum=0),
    _s("DESK_DISPLAY_DIAGNOSTIC_CONTROL_PATH", "path", _ALL, "diagnostics",
       "Control file for diagnostic single-screen playback from the config UI."),
    _s("DESK_DISPLAY_TEST_SCREEN", "str", _CLIENT, "diagnostics",
       "Repeat just this screen ID instead of the normal rotation."),
    _s("DESK_DISPLAY_TEST_SCREEN_DELAY", "float", _CLIENT, "diagnostics",
       "Pause between repeats of DESK_DISPLAY_TEST_SCREEN.", default="0.5", minimum=0),
    _s("RES_OPTIONS", "str", _ALL, "diagnostics", "Resolver options reported in NHL diagnostics."),
    _s("LOCALDOMAIN", "str", _ALL, "diagnostics", "Resolver search domain reported in diagnostics."),
    _s("HOSTALIASES", "path", _ALL, "diagnostics", "Resolver host aliases file reported in diagnostics."),
)

SETTINGS_BY_NAME: dict[str, Setting] = {setting.name: setting for setting in SETTINGS}

# Environment values are read once at process start, so every setting above is
# ``Reload.RESTART``.  The documents these paths point to are re-read when their
# mtime changes; those are the hot-reloadable pieces of configuration.
HOT_RELOAD_DOCUMENTS: dict[str, str] = {
    "SCREENS_CONFIG_PATH": "Screen schedule and playlists (main.py reloads on change).",
    "SCREENS_STYLE_PATH": "Per-screen style overrides (config.py reloads on change).",
    "SCREENS_LAYOUTS_PATH": "Quad layouts (screens/registry.py reloads on change).",
    "NEWS_FEEDS_CONFIG_PATH": "News topics and feeds (services/news_feeds.py reloads on change).",
    "NEWS_FEEDS_CONFIG_PATH_2": "Second news screen's feeds (reloaded on change).",
}


# ─── Parsing ────────────────────────────────────────────────────────────────


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _known_profiles() -> frozenset[str]:
    from display_profiles import PROFILE_PRESETS

    return frozenset(PROFILE_PRESETS)


def normalize_output(value: str) -> str | None:
    """Return the canonical output driver for *value*, or ``None`` if unknown."""

    return _OUTPUT_ALIASES.get(value.strip().lower())


def parse_value(setting: Setting, raw: str) -> Any:
    """Parse *raw* for *setting*; an empty value means "use the default".

    Raises :class:`ValueError` with a human-readable reason when *raw* is not a
    valid value.
    """

    value = raw.strip()
    if value == "":
        return None
    kind = setting.kind
    if kind in {"str", "path"}:
        return value
    if kind == "bool":
        lowered = value.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ValueError("expected 1/0, true/false, yes/no or on/off")
    if kind in {"int", "float"}:
        try:
            number: float = int(value, 0) if kind == "int" else float(value)
        except ValueError:
            if kind == "int":
                try:
                    number = int(value)
                except ValueError:
                    raise ValueError("expected a whole number") from None
            else:
                raise ValueError("expected a number") from None
        if kind == "float" and not math.isfinite(number):
            raise ValueError("expected a finite number")
        if setting.minimum is not None and number < setting.minimum:
            raise ValueError(f"must be at least {_format_number(setting.minimum)}")
        if setting.maximum is not None and number > setting.maximum:
            raise ValueError(f"must be at most {_format_number(setting.maximum)}")
        return number
    if kind == "choice":
        if value not in setting.choices and value.lower() not in setting.choices:
            raise ValueError("expected one of " + ", ".join(setting.choices))
        return value if value in setting.choices else value.lower()
    if kind == "url":
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https", "webcal"} or not parts.netloc:
            raise ValueError("expected an http://, https:// or webcal:// URL")
        return value
    if kind == "csv":
        return [item.strip() for item in value.split(",") if item.strip()]
    if kind == "json":
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON ({exc.msg})") from None
    if kind in {"latitude", "longitude"}:
        limit = 90.0 if kind == "latitude" else 180.0
        try:
            number = float(value)
        except ValueError:
            raise ValueError("expected decimal degrees") from None
        if not math.isfinite(number) or abs(number) > limit:
            raise ValueError(f"must be between -{limit:g} and {limit:g}")
        return number
    if kind == "timezone":
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("unknown IANA timezone") from None
        return value
    if kind == "rotation":
        if value not in _ROTATIONS:
            raise ValueError("expected 0, 90, 180 or 270 degrees, or 0-3 quarter turns")
        number = int(value)
        return number * 90 if number in (1, 2, 3) else number
    if kind == "profile":
        profile = value.lower()
        if profile not in _known_profiles():
            raise ValueError("unknown display profile; expected one of " + ", ".join(sorted(_known_profiles())))
        return profile
    if kind == "output":
        output = normalize_output(value)
        if output is None:
            raise ValueError(
                "unknown output driver; expected auto, displayhatmini, minipitft, kernel, "
                "window, framebuffer or headless"
            )
        return output
    if kind == "log_level":
        level = value.upper()
        if level not in _LOG_LEVELS:
            raise ValueError("expected one of " + ", ".join(_LOG_LEVELS))
        return level
    if kind == "client_id":
        if not _CLIENT_ID_RE.match(value):
            raise ValueError(
                "use 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit"
            )
        return value
    if kind == "static_clients":
        clients: dict[str, str] = {}
        for entry in value.split(","):
            entry = entry.strip()
            if not entry:
                continue
            client_id, sep, profile = entry.partition(":")
            if not sep or not _CLIENT_ID_RE.match(client_id.strip()):
                raise ValueError(f"entry {entry!r} is not client_id:profile")
            profile = profile.strip().lower()
            if profile not in _known_profiles():
                raise ValueError(f"entry {entry!r} names unknown profile {profile!r}")
            clients[client_id.strip()] = profile
        return clients
    raise ValueError(f"unsupported setting kind {kind!r}")  # pragma: no cover - catalog bug


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def parse_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """Parse a dotenv file the same way ``config._load_env_file`` does."""

    values: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.lower().startswith("export "):
            key = key[7:].strip()
        value = value.strip()
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        else:
            value = re.sub(r"\s+#.*$", "", value).strip()
        if key:
            values[key] = value
    return values


def resolve_role(env: Mapping[str, str] | None = None) -> Role:
    """Return the configured role; raise :class:`ValueError` when unknown."""

    source = os.environ if env is None else env
    raw = (source.get(ROLE_ENV) or "").strip().lower()
    if not raw:
        return Role.STANDALONE
    try:
        return Role(raw)
    except ValueError:
        raise ValueError(
            f"{ROLE_ENV}={raw!r} is not a role; expected server, client or standalone"
        ) from None


def settings_for_role(role: Role) -> tuple[Setting, ...]:
    return tuple(setting for setting in SETTINGS if role in setting.roles)


def load_settings(role: Role, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return every setting for *role*, parsed, with defaults applied.

    Invalid values raise :class:`ValueError`; call :func:`validate` first for a
    full report.
    """

    source = os.environ if env is None else env
    parsed: dict[str, Any] = {}
    for setting in settings_for_role(role):
        raw = source.get(setting.name)
        value = parse_value(setting, raw) if raw is not None else None
        if value is None and setting.default:
            value = parse_value(setting, setting.default)
        parsed[setting.name] = value
    return parsed


def resolve_log_level(env: Mapping[str, str] | None = None, default: int = logging.INFO) -> int:
    """Return the numeric level for ``DESK_DISPLAY_LOG_LEVEL``."""

    source = os.environ if env is None else env
    raw = (source.get("DESK_DISPLAY_LOG_LEVEL") or "").strip().upper()
    return getattr(logging, raw) if raw in _LOG_LEVELS else default


# ─── Validation ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Issue:
    level: str  # "error" or "warning"
    name: str | None
    message: str


@dataclass
class ValidationReport:
    role: Role
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, name: str | None, message: str) -> None:
        self.issues.append(Issue("error", name, message))

    def warning(self, name: str | None, message: str) -> None:
        self.issues.append(Issue("warning", name, message))


# Output drivers bound to one panel family, and outputs a panel cannot use.
_DRIVER_PROFILES = {
    "displayhatmini": frozenset({"display_hat_mini"}),
    "minipitft": frozenset({"adafruit_minipitft_114"}),
}
_SPI_ONLY_PROFILES = frozenset({"display_hat_mini", "adafruit_minipitft_114"})
_FRAMEBUFFER_OUTPUTS = frozenset({"framebuffer", "kernel"})


def _is_loopback_host(host: str) -> bool:
    host = host.strip().strip("[]").lower()
    if host in {"localhost", ""}:
        return host == "localhost"
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate(
    role: Role,
    env: Mapping[str, str],
    *,
    check_unknown: bool = True,
    check_files: bool = True,
) -> ValidationReport:
    """Validate *env* for *role*.

    ``check_unknown`` reports names that are not Desk Display settings; turn it
    off for a full process environment, which legitimately contains ``PATH``
    and friends.  ``check_files`` verifies that referenced TLS/CA files exist.
    """

    report = ValidationReport(role)
    values: dict[str, Any] = {}

    for name, raw in env.items():
        setting = SETTINGS_BY_NAME.get(name)
        if setting is None:
            if check_unknown:
                report.warning(name, "not a Desk Display setting; it is ignored")
            continue
        if role not in setting.roles:
            if raw is None or not str(raw).strip():
                continue
            owner = ", ".join(sorted(r.value for r in setting.roles))
            if role is Role.CLIENT:
                what = "provider credential" if setting.provider or setting.secret else "server setting"
                report.error(
                    name,
                    f"{what} does not belong on a client (used by: {owner}); remove it "
                    "and configure it on the server",
                )
            else:
                report.warning(name, f"ignored in the {role.value} role (used by: {owner})")
            continue
        try:
            values[name] = parse_value(setting, str(raw))
        except ValueError as exc:
            report.error(name, f"invalid value {str(raw)!r}: {exc}")

    def get(name: str) -> Any:
        if name in values and values[name] is not None:
            return values[name]
        setting = SETTINGS_BY_NAME[name]
        if role in setting.roles and setting.default:
            try:
                return parse_value(setting, setting.default)
            except ValueError:  # pragma: no cover - catalog bug
                return None
        return None

    if role in (Role.SERVER, Role.STANDALONE):
        _validate_content(report, env, get)
        _validate_config_ui(report, get)
    if role is Role.SERVER:
        _validate_server(report, get, check_files=check_files)
    if role in (Role.CLIENT, Role.STANDALONE):
        _validate_presentation(report, role, values, get)
    if role is Role.CLIENT:
        _validate_client(report, get, check_files=check_files)
    return report


def _validate_content(report: ValidationReport, env: Mapping[str, str], get) -> None:
    for lat, lon in (
        ("WEATHER_LATITUDE", "WEATHER_LONGITUDE"),
        ("AIR_QUALITY_LATITUDE", "AIR_QUALITY_LONGITUDE"),
        ("ADSB_HOME_LATITUDE", "ADSB_HOME_LONGITUDE"),
    ):
        has_lat = bool(str(env.get(lat) or "").strip())
        has_lon = bool(str(env.get(lon) or "").strip())
        if has_lat != has_lon:
            report.error(lon if has_lat else lat, f"set both {lat} and {lon}, or neither")


def _validate_config_ui(report: ValidationReport, get) -> None:
    password = get("SCREEN_UI_PASSWORD")
    if get("SCREEN_AUTH_ENABLED") and not password:
        report.error("SCREEN_UI_PASSWORD", "SCREEN_AUTH_ENABLED=1 requires a password")
    elif not password and not _is_loopback_host(get("SCREEN_CONFIG_HOST") or ""):
        report.warning(
            "SCREEN_UI_PASSWORD",
            "the config UI listens beyond this machine without a password; set "
            "SCREEN_UI_PASSWORD or bind SCREEN_CONFIG_HOST to 127.0.0.1",
        )


def _validate_server(report: ValidationReport, get, *, check_files: bool) -> None:
    host = get("DESK_DISPLAY_SERVER_HOST") or ""
    token = get("DESK_DISPLAY_SERVER_AUTH_TOKEN")
    allow_unauthenticated = bool(get("DESK_DISPLAY_SERVER_ALLOW_UNAUTHENTICATED"))
    loopback = _is_loopback_host(host)

    if not token:
        if not allow_unauthenticated:
            report.error(
                "DESK_DISPLAY_SERVER_AUTH_TOKEN",
                "required; clients authenticate with it (or set "
                "DESK_DISPLAY_SERVER_ALLOW_UNAUTHENTICATED=1 on a loopback-only server)",
            )
        elif not loopback:
            report.error(
                "DESK_DISPLAY_SERVER_ALLOW_UNAUTHENTICATED",
                f"insecure: unauthenticated access is only allowed on loopback, but the "
                f"server binds {host!r}",
            )
    elif len(token) < _MIN_SERVER_TOKEN_LENGTH:
        report.error(
            "DESK_DISPLAY_SERVER_AUTH_TOKEN",
            f"insecure: use at least {_MIN_SERVER_TOKEN_LENGTH} random characters",
        )

    admin = get("DESK_DISPLAY_SERVER_ADMIN_TOKEN")
    if admin and len(admin) < _MIN_SERVER_TOKEN_LENGTH:
        report.error(
            "DESK_DISPLAY_SERVER_ADMIN_TOKEN",
            f"insecure: use at least {_MIN_SERVER_TOKEN_LENGTH} random characters",
        )
    elif admin and token and hmac.compare_digest(admin, token):
        report.error(
            "DESK_DISPLAY_SERVER_ADMIN_TOKEN",
            "must differ from DESK_DISPLAY_SERVER_AUTH_TOKEN, which every client knows",
        )

    cert, key = get("DESK_DISPLAY_SERVER_TLS_CERT"), get("DESK_DISPLAY_SERVER_TLS_KEY")
    if bool(cert) != bool(key):
        report.error(
            "DESK_DISPLAY_SERVER_TLS_CERT" if not cert else "DESK_DISPLAY_SERVER_TLS_KEY",
            "set DESK_DISPLAY_SERVER_TLS_CERT and DESK_DISPLAY_SERVER_TLS_KEY together",
        )
    if check_files:
        for name, path in (("DESK_DISPLAY_SERVER_TLS_CERT", cert), ("DESK_DISPLAY_SERVER_TLS_KEY", key)):
            if path and not Path(path).expanduser().is_file():
                report.error(name, f"file not found: {path}")
    if not cert and not loopback:
        report.warning(
            "DESK_DISPLAY_SERVER_HOST",
            "the server listens beyond this machine without TLS; terminate TLS at a "
            "reverse proxy or set DESK_DISPLAY_SERVER_TLS_CERT/KEY",
        )


def _validate_presentation(report: ValidationReport, role: Role, values: Mapping[str, Any], get) -> None:
    profile = get("DESK_DISPLAY_PROFILE")
    output = get("DESK_DISPLAY_OUTPUT") or "auto"
    if get("DESK_DISPLAY_FORCE_HEADLESS"):
        output = "headless"
    if role is Role.CLIENT and not profile and "DESK_DISPLAY_PROFILE" not in report_error_names(report):
        report.error(
            "DESK_DISPLAY_PROFILE",
            "required on clients; the server renders artifacts for this profile",
        )
    if profile:
        allowed = _DRIVER_PROFILES.get(output)
        if allowed is not None and profile not in allowed:
            report.error(
                "DESK_DISPLAY_OUTPUT",
                f"output {output!r} drives only the {', '.join(sorted(allowed))} profile, "
                f"not {profile!r}",
            )
        if output in _FRAMEBUFFER_OUTPUTS and profile in _SPI_ONLY_PROFILES:
            report.error(
                "DESK_DISPLAY_OUTPUT",
                f"profile {profile!r} is an SPI panel without a framebuffer; use "
                f"{'displayhatmini' if profile == 'display_hat_mini' else 'minipitft'} or auto",
            )


def report_error_names(report: ValidationReport) -> set[str | None]:
    return {issue.name for issue in report.errors}


def _validate_client(report: ValidationReport, get, *, check_files: bool) -> None:
    errored = report_error_names(report)
    if not get("DESK_DISPLAY_CLIENT_ID") and "DESK_DISPLAY_CLIENT_ID" not in errored:
        report.error("DESK_DISPLAY_CLIENT_ID", "required; a stable, unique ID for this display")
    url = get("DESK_DISPLAY_SERVER_URL")
    if not url and "DESK_DISPLAY_SERVER_URL" not in errored:
        report.error("DESK_DISPLAY_SERVER_URL", "required; the render server's base URL")
    if not get("DESK_DISPLAY_CLIENT_TOKEN"):
        report.error(
            "DESK_DISPLAY_CLIENT_TOKEN",
            "required; copy the server's DESK_DISPLAY_SERVER_AUTH_TOKEN",
        )

    allow_insecure = bool(get("DESK_DISPLAY_ALLOW_INSECURE_TRANSPORT"))

    def insecure(name: str, message: str) -> None:
        if allow_insecure:
            report.warning(name, message + " (allowed by DESK_DISPLAY_ALLOW_INSECURE_TRANSPORT)")
        else:
            report.error(
                name,
                message + "; use https, or set DESK_DISPLAY_ALLOW_INSECURE_TRANSPORT=1 "
                "on a trusted network",
            )

    if url:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"}:
            report.error("DESK_DISPLAY_SERVER_URL", "must be an http:// or https:// URL")
        elif parts.scheme == "http" and not _is_loopback_host(parts.hostname or ""):
            insecure("DESK_DISPLAY_SERVER_URL", "insecure: the client token would travel in plain HTTP")
    if get("DESK_DISPLAY_TLS_VERIFY") is False:
        insecure("DESK_DISPLAY_TLS_VERIFY", "insecure: TLS certificate verification is disabled")
    bundle = get("DESK_DISPLAY_SERVER_CA_BUNDLE")
    if check_files and bundle and not Path(bundle).expanduser().is_file():
        report.error("DESK_DISPLAY_SERVER_CA_BUNDLE", f"file not found: {bundle}")


def startup_check(
    component: str,
    env: Mapping[str, str] | None = None,
    *,
    logger: logging.Logger | None = None,
) -> ValidationReport:
    """Validate the process environment for its role and log the outcome.

    Server and client processes fail closed: any error raises
    :class:`ConfigurationError`.  The legacy standalone role keeps its lenient
    startup and logs errors as warnings.
    """

    log = logger or logging.getLogger("desk_display.config")
    source = os.environ if env is None else env
    try:
        role = resolve_role(source)
    except ValueError as exc:
        report = ValidationReport(Role.STANDALONE)
        report.error(ROLE_ENV, str(exc))
        raise ConfigurationError(report) from None
    report = validate(role, source, check_unknown=False)
    for issue in report.warnings:
        log.warning("%s config: %s: %s", component, issue.name, issue.message)
    if report.errors:
        if role is Role.STANDALONE:
            for issue in report.errors:
                log.warning("%s config: %s: %s", component, issue.name, issue.message)
        else:
            raise ConfigurationError(report)
    return report


# ─── Secret exclusion ───────────────────────────────────────────────────────


SECRET_SETTING_NAMES: frozenset[str] = frozenset(s.name for s in SETTINGS if s.secret)
_SECRET_KEYS_LOWER = frozenset(name.lower() for name in SECRET_SETTING_NAMES)


def secret_values(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Configured secret values long enough to redact safely, longest first."""

    source = os.environ if env is None else env
    found = {
        str(source.get(name) or "").strip()
        for name in SECRET_SETTING_NAMES
    }
    usable = [value for value in found if len(value) >= _MIN_REDACTABLE_SECRET_LENGTH]
    return tuple(sorted(usable, key=len, reverse=True))


def redact_text(text: str, env: Mapping[str, str] | None = None) -> str:
    """Replace every configured secret value in *text* with ``[redacted]``."""

    for value in secret_values(env):
        if value in text:
            text = text.replace(value, REDACTED)
    return text


def scrub_secrets(payload: Any, env: Mapping[str, str] | None = None) -> Any:
    """Return a copy of *payload* safe to send to a client or browser.

    Mapping keys naming a secret setting (case-insensitive) are dropped, and
    any configured secret value inside a string is replaced with
    ``[redacted]``.  Tuples and sets come back as lists, matching JSON.
    """

    values = secret_values(env)

    def scrub(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: scrub(item)
                for key, item in value.items()
                if not (isinstance(key, str) and key.lower() in _SECRET_KEYS_LOWER)
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [scrub(item) for item in value]
        if isinstance(value, str):
            for secret in values:
                if secret in value:
                    value = value.replace(secret, REDACTED)
            return value
        return value

    return scrub(payload)


def find_secrets(payload: Any, env: Mapping[str, str] | None = None) -> list[str]:
    """Return JSON-style paths in *payload* that expose a secret."""

    values = secret_values(env)
    leaks: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{path}.{key}" if path else str(key)
                if isinstance(key, str) and key.lower() in _SECRET_KEYS_LOWER:
                    leaks.append(child)
                walk(item, child)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, str) and any(secret in value for secret in values):
            leaks.append(path or "$")

    walk(payload, "")
    return leaks


def redact_response(response: Any) -> Any:
    """Flask ``after_request`` hook removing secret values from text responses."""

    mimetype = getattr(response, "mimetype", "") or ""
    if getattr(response, "direct_passthrough", False) or not (
        mimetype.startswith("text/") or mimetype in {"application/json", "application/javascript"}
    ):
        return response
    if not secret_values():
        return response
    body = response.get_data(as_text=True)
    redacted = redact_text(body)
    if redacted != body:
        response.set_data(redacted)
    return response


_ORIGINAL_RECORD_FACTORY = None


def install_secret_log_redaction() -> None:
    """Redact configured secret values from every log record in this process."""

    global _ORIGINAL_RECORD_FACTORY
    if _ORIGINAL_RECORD_FACTORY is not None:
        return
    original = logging.getLogRecordFactory()
    _ORIGINAL_RECORD_FACTORY = original

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = original(*args, **kwargs)
        values = secret_values()
        if not values:
            return record
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - leave malformed records to logging
            return record
        redacted = redact_text(message)
        if redacted != message:
            record.msg, record.args = redacted, None
        return record

    logging.setLogRecordFactory(factory)


# ─── Examples and docs generation ───────────────────────────────────────────

_EXAMPLE_FILES = {
    Role.SERVER: ".env.server.example",
    Role.CLIENT: ".env.client.example",
}
_EXAMPLE_INTROS = {
    Role.SERVER: (
        "Desk Display render server environment example.",
        "Copy to .env on the server host and fill in the values you need. The server",
        "fetches all upstream data, so every provider credential lives here and never",
        "on a display client. Check a file with:",
        "  python3 -m deployment_config check --role server --env-file .env",
    ),
    Role.CLIENT: (
        "Desk Display display-client environment example.",
        "Copy to .env on each display. A client needs only its identity, the server",
        "URL and token, and its own hardware settings: never an upstream API key or",
        "provider URL. Check a file with:",
        "  python3 -m deployment_config check --role client --env-file .env",
    ),
}


def render_example(role: Role) -> str:
    """Return the canonical ``.env.<role>.example`` contents."""

    if role not in _EXAMPLE_INTROS:
        raise ValueError("examples exist for the server and client roles")
    lines = [f"# {line}" if line else "#" for line in _EXAMPLE_INTROS[role]]
    lines += [
        "#",
        "# Booleans accept 1/0, true/false, yes/no or on/off. An empty value uses the",
        "# built-in default. Environment values are read at startup: restart the",
        "# process after changing them (see CONFIGURATION.md for what hot-reloads).",
    ]
    for section in SECTIONS:
        settings = [s for s in SETTINGS if s.section == section.key and role in s.roles]
        if not settings and not (section.note and role is Role.SERVER):
            continue
        lines += ["", "# " + "-" * 77, f"# {section.title}", "# " + "-" * 77]
        if section.note:
            lines += [f"# {line}" for line in _wrap(section.note)]
        for setting in settings:
            description = setting.description
            if setting.secret:
                description += " Secret: keep it out of version control."
            lines += [f"# {line}" for line in _wrap(description)]
            value = ROLE_ENV == setting.name and role.value or setting.example_value
            lines.append(f"{setting.name}={value}")
    return "\n".join(lines) + "\n"


def _wrap(text: str, width: int = 77) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines


DOCS_BEGIN = "<!-- BEGIN GENERATED SETTINGS REFERENCE -->"
DOCS_END = "<!-- END GENERATED SETTINGS REFERENCE -->"


def render_settings_reference() -> str:
    """Return the generated Markdown settings table for CONFIGURATION.md."""

    order = (Role.SERVER, Role.CLIENT, Role.STANDALONE)
    rows = [
        "| Setting | Roles | Takes effect | Secret |",
        "| --- | --- | --- | --- |",
    ]
    for section in SECTIONS:
        for setting in (s for s in SETTINGS if s.section == section.key):
            roles = ", ".join(role.value for role in order if role in setting.roles)
            effect = "restart" if setting.reload is Reload.RESTART else "hot reload"
            if setting.name in HOT_RELOAD_DOCUMENTS:
                effect = "restart (file contents hot-reload)"
            secret = "yes" if setting.secret else ("provider" if setting.provider else "")
            rows.append(f"| `{setting.name}` | {roles} | {effect} | {secret} |")
    return "\n".join([DOCS_BEGIN, *rows, DOCS_END])


def _check_catalog() -> None:
    names = [setting.name for setting in SETTINGS]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:  # pragma: no cover - catalog bug
        raise AssertionError(f"duplicate settings: {sorted(duplicates)}")
    unknown = {setting.section for setting in SETTINGS} - _SECTION_KEYS
    if unknown:  # pragma: no cover - catalog bug
        raise AssertionError(f"unknown sections: {sorted(unknown)}")


_check_catalog()


# ─── Command line ───────────────────────────────────────────────────────────


def _cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m deployment_config",
        description="Validate Desk Display configuration and generate its examples.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="validate an env file or the current environment")
    check.add_argument("--role", choices=[role.value for role in Role])
    check.add_argument("--env-file", help="dotenv file to check instead of the environment")
    example = sub.add_parser("example", help="print .env.server.example or .env.client.example")
    example.add_argument("--role", choices=[Role.SERVER.value, Role.CLIENT.value], required=True)
    sub.add_parser("reference", help="print the generated settings reference table")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "example":
        sys.stdout.write(render_example(Role(args.role)))
        return 0
    if args.command == "reference":
        print(render_settings_reference())
        return 0

    env: Mapping[str, str] = parse_env_file(args.env_file) if args.env_file else os.environ
    try:
        role = Role(args.role) if args.role else resolve_role(env)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = validate(role, env, check_unknown=bool(args.env_file))
    for issue in report.issues:
        print(f"{issue.level}: {issue.name}: {issue.message}")
    print(f"{role.value}: {len(report.errors)} error(s), {len(report.warnings)} warning(s)")
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
