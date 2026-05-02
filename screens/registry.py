"""Screen registry utilities for mapping screen IDs to render callables."""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from PIL import Image

from config import CENTRAL_TIME, HEIGHT, NBA_TEAM_TRICODE, WIDTH, is_display_profile
from paths import resolve_layouts_config_path, resolve_screens_config_paths
from utils import ScreenImage, animate_scroll, timestamp_to_datetime
from screens.draw_bears_schedule import show_bears_next_game, show_bears_next_season
from screens.draw_bulls_schedule import (
    draw_bulls_next_home_game,
    draw_last_bulls_game,
    draw_live_bulls_game,
    draw_sports_screen_bulls,
)
from screens.draw_hawks_schedule import (
    draw_hawks_next_home_game,
    draw_last_hawks_game,
    draw_live_hawks_game,
    draw_sports_screen_hawks,
)
from screens.draw_wolves_schedule import (
    draw_last_wolves_game,
    draw_live_wolves_game,
    draw_sports_screen_wolves,
    draw_wolves_next_home_game,
)
from screens.draw_inside import draw_inside, is_inside_sensor_available
from screens.draw_vrnof import draw_vrnof_screen
from screens.draw_weather import (
    _pop_pct_from,
    draw_weather_astronomical,
    draw_weather_daily,
    draw_weather_hourly,
    draw_weather_radar,
    draw_weather_screen_1,
    draw_weather_screen_2,
)
from screens.draw_nixie import draw_nixie
from screens.draw_date_time import draw_date
from screens.draw_quad import _TileSpec, draw_quad_screen
from screens.mlb_schedule import (
    draw_box_score,
    draw_cubs_result,
    draw_last_game,
    draw_series_screen,
    draw_next_home_game,
    draw_sports_screen,
)
from screens.mlb_scoreboard import render_mlb_scoreboard
from screens.mlb_scoreboard_v2 import render_mlb_scoreboard_v2
from screens.mlb_league_standings import (
    draw_AL_Overview,
    draw_NL_Overview,
    draw_mlb_al_standings,
    draw_mlb_nl_standings,
)
from screens.mlb_team_standings import (
    draw_standings_screen1,
    draw_standings_screen2,
    draw_standings_screen3,
)
from screens.nba_team_standings import draw_nba_standings_screen1
from screens.nfl_team_standings import (
    draw_nfl_standings_screen1,
    draw_nfl_standings_screen2,
)
from screens.nhl_team_standings import draw_nhl_standings_screen1
from screens.nba_scoreboard import render_nba_scoreboard
from screens.nba_scoreboard_v2 import render_nba_scoreboard_v2
from screens.ncaam_scoreboard import render_ncaam_scoreboard
from screens.nfl_scoreboard import render_nfl_scoreboard
from screens.nfl_scoreboard_v2 import render_nfl_scoreboard_v2
from screens.nfl_standings import (
    draw_nfl_overview_afc,
    draw_nfl_overview_nfc,
    draw_nfl_standings_afc,
    draw_nfl_standings_nfc,
)
from screens.nhl_scoreboard import render_nhl_scoreboard
from screens.nhl_scoreboard_v2 import render_nhl_scoreboard_v2
from screens.nhl_playoffs import render_nhl_playoffs
from screens.nba_playoffs import render_nba_playoffs
from screens.nhl_standings import (
    draw_nhl_standings_east,
    draw_nhl_standings_overview_east,
    draw_nhl_standings_overview_west,
    draw_nhl_standings_west,
)
from screens.nhl_standings_v2 import (
    draw_nhl_standings_east_v2,
    draw_nhl_standings_west_v2,
)
from services.sports.nhl import prepare_scoreboard_data as prepare_nhl_scoreboard_data

RenderCallable = Callable[[], Optional[Image.Image | ScreenImage]]

_LAYOUTS_CONFIG_PATH = str(resolve_layouts_config_path())


_QUAD_DEFAULT_PAGE = ["date", "weather1", "weather hourly", "inside"]
_QUAD_DEFAULT_SCROLL_SPEED = 1.0
_quad_page_index = 0
_quad_page_lock = threading.Lock()
_layouts_cache_lock = threading.Lock()
_layouts_payload_cache: Optional[dict[str, Any]] = None
_layouts_payload_mtime: Optional[float] = None

_QUAD_TILE_SAMPLE_FRAMES = 10
_QUAD_TILE_CAPTURE_FRAME_LIMIT = 120
_quad_tile_scroll_cursor: Dict[str, float] = {}
_quad_tile_scroll_lock = threading.Lock()


def _normalize_quad_scroll_speed(value: Any) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError):
        speed = _QUAD_DEFAULT_SCROLL_SPEED
    return min(3.0, max(0.25, speed))


def _quad_layout_from_layouts() -> tuple[bool, float, list[list[str]]]:
    enabled = False
    scroll_speed = _QUAD_DEFAULT_SCROLL_SPEED
    pages: list[list[str]] = []

    payload = _load_layouts_payload()
    if payload is None:
        return enabled, scroll_speed, [_QUAD_DEFAULT_PAGE.copy()]

    if not isinstance(payload, dict):
        return enabled, scroll_speed, [_QUAD_DEFAULT_PAGE.copy()]
    screens = payload.get("screens")
    if not isinstance(screens, dict):
        return enabled, scroll_speed, [_QUAD_DEFAULT_PAGE.copy()]
    quad = screens.get("quad")
    if not isinstance(quad, dict):
        return enabled, scroll_speed, [_QUAD_DEFAULT_PAGE.copy()]

    enabled = bool(quad.get("enabled", False))
    scroll_speed = _normalize_quad_scroll_speed(quad.get("scroll_speed", _QUAD_DEFAULT_SCROLL_SPEED))

    raw_pages = quad.get("pages")
    if not isinstance(raw_pages, list):
        raw_pages = [{"tiles": quad.get("tiles")}] if isinstance(quad.get("tiles"), list) else []

    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            continue
        raw_tiles = raw_page.get("tiles")
        if not isinstance(raw_tiles, list):
            continue
        tiles: list[str] = []
        for raw_tile in raw_tiles:
            if not isinstance(raw_tile, str):
                continue
            tile = raw_tile.strip()
            if not tile or tile == "quad":
                continue
            tiles.append(tile)
            if len(tiles) >= 4:
                break
        while len(tiles) < 4:
            tiles.append(_QUAD_DEFAULT_PAGE[len(tiles)])
        pages.append(tiles)

    if not pages:
        pages = [_QUAD_DEFAULT_PAGE.copy()]

    return enabled, scroll_speed, pages


def _load_layouts_payload() -> Optional[dict[str, Any]]:
    """Return cached quad layouts payload, reloading only when file mtime changes."""

    global _layouts_payload_cache, _layouts_payload_mtime

    try:
        mtime = os.path.getmtime(_LAYOUTS_CONFIG_PATH)
    except OSError:
        mtime = None

    with _layouts_cache_lock:
        if mtime == _layouts_payload_mtime:
            return _layouts_payload_cache

        try:
            with open(_LAYOUTS_CONFIG_PATH, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            _layouts_payload_cache = None
            _layouts_payload_mtime = mtime
            return None

        if not isinstance(payload, dict):
            _layouts_payload_cache = None
            _layouts_payload_mtime = mtime
            return None

        _layouts_payload_cache = payload
        _layouts_payload_mtime = mtime
        return _layouts_payload_cache


def _next_quad_page_tiles() -> tuple[bool, float, list[str]]:
    global _quad_page_index

    enabled, scroll_speed, pages = _quad_layout_from_layouts()
    with _quad_page_lock:
        page = pages[_quad_page_index % len(pages)]
        _quad_page_index += 1
    return enabled, scroll_speed, page


RADAR_LOOKAHEAD_HOURS = 8
WEATHER_CURRENT_TTL = _dt.timedelta(minutes=20)
WEATHER_HOURLY_TTL = _dt.timedelta(hours=1)
_screens_config_paths = resolve_screens_config_paths()
_SCREENS_CONFIG_DEFAULT_PATH = str(_screens_config_paths.default_path)
_SCREENS_CONFIG_LOCAL_PATH = str(_screens_config_paths.local_override_path)
_nhl_break_windows_cache_lock = threading.Lock()
_nhl_break_windows_cache_source: Optional[str] = None
_nhl_break_windows_cache_mtime: Optional[float] = None
_nhl_break_windows_cache: tuple[tuple[_dt.date, _dt.date], ...] = ()


def _active_screens_config_path() -> str:
    return str(resolve_screens_config_paths().active_path)


def _normalize_nhl_break_windows(raw: Any) -> tuple[tuple[_dt.date, _dt.date], ...]:
    windows: list[tuple[_dt.date, _dt.date]] = []

    if isinstance(raw, dict):
        iterable = raw.values()
    elif isinstance(raw, list):
        iterable = raw
    else:
        iterable = []

    for item in iterable:
        if not isinstance(item, dict):
            continue
        start_raw = item.get("start")
        end_raw = item.get("end")
        if not isinstance(start_raw, str) or not isinstance(end_raw, str):
            continue
        try:
            start = _dt.date.fromisoformat(start_raw)
            end = _dt.date.fromisoformat(end_raw)
        except ValueError:
            continue
        if start > end:
            start, end = end, start
        windows.append((start, end))

    windows.sort(key=lambda window: window[0])
    return tuple(windows)


def _load_nhl_break_windows() -> tuple[tuple[_dt.date, _dt.date], ...]:
    """Load NHL break windows from env or screens config, caching by source and mtime."""

    global _nhl_break_windows_cache_source, _nhl_break_windows_cache_mtime, _nhl_break_windows_cache

    env_payload = os.environ.get("NHL_BREAK_WINDOWS_JSON")
    if env_payload is not None:
        with _nhl_break_windows_cache_lock:
            if _nhl_break_windows_cache_source == f"env:{env_payload}":
                return _nhl_break_windows_cache
            try:
                payload = json.loads(env_payload)
            except Exception:
                logging.warning("Invalid NHL_BREAK_WINDOWS_JSON payload; NHL scoreboards will remain enabled.")
                payload = {}
            windows = _normalize_nhl_break_windows(payload)
            _nhl_break_windows_cache_source = f"env:{env_payload}"
            _nhl_break_windows_cache_mtime = None
            _nhl_break_windows_cache = windows
            return windows

    config_path = _active_screens_config_path()
    try:
        mtime = os.path.getmtime(config_path)
    except OSError:
        mtime = None

    with _nhl_break_windows_cache_lock:
        if (
            _nhl_break_windows_cache_source == config_path
            and _nhl_break_windows_cache_mtime == mtime
        ):
            return _nhl_break_windows_cache

        payload: Any = {}
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            payload = {}

        if isinstance(payload, dict):
            sports = payload.get("sports")
            nhl = sports.get("nhl") if isinstance(sports, dict) else payload.get("nhl")
            break_windows = nhl.get("break_windows") if isinstance(nhl, dict) else {}
        else:
            break_windows = {}

        windows = _normalize_nhl_break_windows(break_windows)
        _nhl_break_windows_cache_source = config_path
        _nhl_break_windows_cache_mtime = mtime
        _nhl_break_windows_cache = windows
        return windows


def _is_nhl_break_day(today: _dt.date) -> bool:
    for start, end in _load_nhl_break_windows():
        if start <= today <= end:
            return True
    return False


def _is_1080p_or_higher(width: int, height: int) -> bool:
    """Return True when layout is 1080p-class (or higher) regardless of orientation."""
    short_edge = min(int(width), int(height))
    return short_edge >= 1080


def _is_display_hat_mini_layout(width: int, height: int) -> bool:
    """Return True for Display HAT Mini dimensions regardless of orientation."""
    return sorted((int(width), int(height))) == [240, 320]


def _is_adafruit_minipitft_layout(width: int, height: int) -> bool:
    """Return True for Adafruit miniPiTFT 1.14" dimensions regardless of orientation."""
    return sorted((int(width), int(height))) == [135, 240]


def _is_waveshare_oled_lcd_hat() -> bool:
    """Return True when running on the Waveshare OLED/LCD HAT (A) install profile."""
    marker = os.environ.get("WAVESHARE_OLED_LCD_HAT_A_INSTALLED", "").strip().lower()
    if marker and marker not in {"0", "false", "no", "off"}:
        return True
    # The Waveshare installer writes these env vars for the main service.
    if os.environ.get("WAVESHARE_OLED_MAX_VALUE_FONT_SIZE") is not None:
        return True
    if os.environ.get("WAVESHARE_OLED_MAX_TIME_FONT_SIZE") is not None:
        return True
    return False


def _logo_scroll_speed_for_layout(width: int, height: int) -> float:
    base_speed = 2.2
    if _is_display_hat_mini_layout(width, height):
        return base_speed * 2.0
    return base_speed * (2.0 if _is_1080p_or_higher(width, height) else 1.0)


_LOGO_SCROLL_SPEED = _logo_scroll_speed_for_layout(WIDTH, HEIGHT)


@dataclass
class ScreenDefinition:
    """Represents one renderable screen."""

    id: str
    render: RenderCallable
    available: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScreenContext:
    """Runtime context required to build screen callables."""

    display: Any
    cache: Dict[str, Any]
    logos: Any
    image_dir: str
    now: _dt.datetime
    now_utc: _dt.datetime
    offline: bool
    weather_fetched_at: Optional[_dt.datetime]
    skip_scoreboards: bool


def _show_logo(display, image: Image.Image) -> Image.Image:
    animate_scroll(display, image, speed=_LOGO_SCROLL_SPEED)
    return image


def _extract_team_id(blob):
    if not isinstance(blob, dict):
        return None
    team = blob.get("team") if isinstance(blob.get("team"), dict) else blob
    if isinstance(team, dict):
        for key in ("id", "teamId", "team_id"):
            if team.get(key) is not None:
                return team.get(key)
    return None


def _extract_team_tricode(blob):
    if not isinstance(blob, dict):
        return None
    team = blob.get("team") if isinstance(blob.get("team"), dict) else blob
    if isinstance(team, dict):
        for key in ("triCode", "abbreviation", "abbrev", "teamAbbrev"):
            value = team.get(key)
            if value:
                return str(value).upper()
    return None


def _games_match(game_a, game_b):
    if not game_a or not game_b:
        return False

    for key in ("gamePk", "id", "gameId", "gameUUID"):
        a_val = game_a.get(key)
        b_val = game_b.get(key)
        if a_val and b_val and a_val == b_val:
            return True

    def _teams(game, prefix):
        teams = game.get("teams")
        if isinstance(teams, dict):
            return teams.get(prefix) or {}
        return game.get(f"{prefix}Team") or game.get(f"{prefix}_team") or {}

    date_a = (game_a.get("gameDate") or game_a.get("officialDate") or "")[:10]
    date_b = (game_b.get("gameDate") or game_b.get("officialDate") or "")[:10]
    if date_a and date_b and date_a == date_b:
        home_a = _extract_team_id(_teams(game_a, "home"))
        home_b = _extract_team_id(_teams(game_b, "home"))
        away_a = _extract_team_id(_teams(game_a, "away"))
        away_b = _extract_team_id(_teams(game_b, "away"))
        return home_a and home_a == home_b and away_a and away_a == away_b

    return False


def _is_bulls_home_game(game: Any) -> bool:
    if not isinstance(game, dict):
        return False
    teams = game.get("teams") or {}
    home = teams.get("home") or game.get("homeTeam") or game.get("home_team")
    tricode = _extract_team_tricode(home)
    if not tricode:
        return False
    return tricode == (NBA_TEAM_TRICODE or "CHI").upper()


def _format_time(value: Optional[_dt.time]) -> str:
    if isinstance(value, _dt.time):
        return value.strftime("%I:%M %p").lstrip("0").replace(" 0", " ")
    return "all day"


def _normalise_reference_time(now: Optional[_dt.datetime]) -> _dt.datetime:
    current = now or _dt.datetime.now(CENTRAL_TIME)
    if current.tzinfo is None:
        if hasattr(CENTRAL_TIME, "localize"):
            return CENTRAL_TIME.localize(current)  # type: ignore[arg-type]
        return current.replace(tzinfo=CENTRAL_TIME)
    return current.astimezone(CENTRAL_TIME)


def _has_precipitation_amount(hour: dict) -> bool:
    for key in ("rain", "snow"):
        amount = hour.get(key)
        if isinstance(amount, dict):
            for value in amount.values():
                try:
                    if float(value) > 0:
                        return True
                except Exception:
                    continue
        else:
            try:
                if amount is not None and float(amount) > 0:
                    return True
            except Exception:
                continue
    return False


def _precip_within_hours(weather: object, hours: int, *, now: Optional[_dt.datetime] = None) -> bool:
    if not isinstance(weather, dict) or hours <= 0:
        return False

    hourly = weather.get("hourly") if isinstance(weather.get("hourly"), list) else None
    if not hourly:
        return False

    current = _normalise_reference_time(now)
    cutoff = current + _dt.timedelta(hours=hours)

    for hour in hourly:
        if not isinstance(hour, dict):
            continue

        dt_val = timestamp_to_datetime(hour.get("dt"), CENTRAL_TIME)
        if dt_val is None or dt_val < current or dt_val > cutoff:
            continue

        pop_pct = _pop_pct_from(hour)
        if pop_pct is not None and pop_pct > 0:
            return True
        if _has_precipitation_amount(hour):
            return True

    return False


def build_screen_registry(context: ScreenContext) -> Tuple[Dict[str, ScreenDefinition], Dict[str, Any]]:
    """Create a registry mapping screen IDs to render callables."""

    registry: Dict[str, ScreenDefinition] = {}
    metadata: Dict[str, Any] = {}
    adafruit_minipitft_layout = _is_adafruit_minipitft_layout(WIDTH, HEIGHT)
    waveshare_oled_lcd_hat = _is_waveshare_oled_lcd_hat()
    hyperpixel4_layout = is_display_profile("hyperpixel4", WIDTH, HEIGHT)

    def _mlb_series_title(team_name: str, short_title: str) -> str:
        if hyperpixel4_layout:
            return f"{team_name} {short_title}"
        return short_title

    def register(screen_id: str, func: RenderCallable, available: bool = True, **extra):
        registry[screen_id] = ScreenDefinition(
            id=screen_id,
            render=func,
            available=available,
            metadata=extra,
        )

    mlb_rotation_cursors: Dict[str, int] = {
        "cubs last": 0,
        "cubs next": 0,
        "sox last": 0,
        "sox next": 0,
    }

    def _rotate_games(screen_id: str, primary: Any, alternate: Any) -> Any:
        if not primary:
            return alternate
        if not alternate:
            return primary
        cursor = mlb_rotation_cursors.get(screen_id, 0)
        mlb_rotation_cursors[screen_id] = (cursor + 1) % 2
        return alternate if cursor else primary

    # Date/time screens intentionally run outside transition mode so their
    # color-cycle threads can keep animating while those screens are visible.
    register("date", lambda: draw_date(context.display, transition=False))
    register("nixie", lambda: draw_nixie(context.display, transition=False))

    weather_data = context.cache.get("weather")

    class _QuadCaptureDisplay:
        def __init__(self, *, frame_limit: Optional[int] = None):
            self.width = WIDTH
            self.height = HEIGHT
            self._last: Optional[Image.Image] = None
            self._frame_id = 0
            self._frame_limit = frame_limit
            self._frames: list[Image.Image] = []

        def image(self, img: Image.Image):
            copied = img.copy()
            self._last = copied
            self._frame_id += 1
            if self._frame_limit is None or len(self._frames) < self._frame_limit:
                self._frames.append(copied)

        def show(self):
            return None

        def wait_for_skip(self, _duration: float) -> bool:
            # In quad mode we sample frames from animated/scrolling screens and
            # avoid per-frame sleeps so a single tile cannot stall the page.
            return False

        def skip_requested(self) -> bool:
            if self._frame_limit is None:
                return False
            return self._frame_id >= self._frame_limit

        def frame_id(self) -> int:
            return self._frame_id

        @property
        def last_image(self) -> Optional[Image.Image]:
            return self._last

        @property
        def frames(self) -> list[Image.Image]:
            return self._frames

    def _render_quad_tile(screen_id: str, *, scroll_speed: float) -> Optional[Image.Image | list[Image.Image]]:
        definition = registry.get(screen_id)
        if definition is None:
            return None

        with _quad_tile_scroll_lock:
            cursor = float(_quad_tile_scroll_cursor.get(screen_id, 0.0)) % _QUAD_TILE_SAMPLE_FRAMES
        capture = _QuadCaptureDisplay(frame_limit=_QUAD_TILE_CAPTURE_FRAME_LIMIT)
        original_display = context.display
        context.display = capture
        try:
            rendered = definition.render()
        finally:
            context.display = original_display

        sampled_frames = capture.frames
        if len(sampled_frames) > _QUAD_TILE_SAMPLE_FRAMES:
            span = len(sampled_frames) - 1
            sampled_frames = [
                sampled_frames[int(round((idx * span) / (_QUAD_TILE_SAMPLE_FRAMES - 1)))]
                for idx in range(_QUAD_TILE_SAMPLE_FRAMES)
            ]

        sampled_count = len(sampled_frames)
        if sampled_count > 1:
            start_index = int(cursor) % sampled_count
            ordered_frames = [
                sampled_frames[(start_index + offset) % sampled_count]
                for offset in range(sampled_count)
            ]
            next_cursor = (cursor + scroll_speed) % sampled_count
            with _quad_tile_scroll_lock:
                _quad_tile_scroll_cursor[screen_id] = next_cursor
            return ordered_frames

        with _quad_tile_scroll_lock:
            _quad_tile_scroll_cursor.pop(screen_id, None)

        if sampled_count:
            return sampled_frames[0]
        if isinstance(rendered, ScreenImage):
            return rendered.image
        if isinstance(rendered, Image.Image):
            return rendered
        return capture.last_image

    def _render_black_quad_tile() -> Image.Image:
        return Image.new("RGB", (WIDTH, HEIGHT), "black")

    weather_logo = context.logos.get("weather logo")
    # Keep weather screens visible whenever cached forecast data exists.
    # During offline periods we may be unable to refresh for hours, but hiding
    # weather pages entirely is more disruptive than showing the latest cache.
    weather_current_available = bool(weather_data)
    weather_hourly_available = bool(weather_data)
    if weather_logo is not None:
        register(
            "weather logo",
            lambda img=weather_logo: _show_logo(context.display, img),
            available=True,
        )
    register(
        "weather1",
        lambda data=weather_data: draw_weather_screen_1(context.display, data, transition=True),
        available=weather_current_available,
    )
    register(
        "weather2",
        lambda data=weather_data: draw_weather_screen_2(context.display, data, transition=True),
        available=weather_current_available,
    )
    register(
        "weather hourly",
        lambda data=weather_data: draw_weather_hourly(context.display, data, transition=True),
        available=weather_hourly_available,
    )
    register(
        "weather daily",
        lambda data=weather_data: draw_weather_daily(context.display, data, transition=True),
        available=weather_hourly_available,
    )
    register(
        "astronomical",
        lambda data=weather_data: draw_weather_astronomical(context.display, data, transition=True),
        available=weather_hourly_available,
    )
    register(
        "weather quad",
        lambda scroll_speed=_QUAD_DEFAULT_SCROLL_SPEED: draw_quad_screen(
            context.display,
            [
                _TileSpec("weather1", lambda speed=scroll_speed: _render_quad_tile("weather1", scroll_speed=speed)),
                _TileSpec("weather2", lambda speed=scroll_speed: _render_quad_tile("weather2", scroll_speed=speed)),
                _TileSpec("weather hourly", lambda speed=scroll_speed: _render_quad_tile("weather hourly", scroll_speed=speed)),
                _TileSpec("weather daily", lambda speed=scroll_speed: _render_quad_tile("weather daily", scroll_speed=speed)),
            ],
            transition=True,
            scroll_speed=scroll_speed,
        ),
        available=weather_current_available and weather_hourly_available,
        quad_tiles=["weather1", "weather2", "weather hourly", "weather daily"],
    )
    radar_available = weather_hourly_available and _precip_within_hours(
        weather_data, RADAR_LOOKAHEAD_HOURS, now=context.now
    )
    register(
        "weather radar",
        lambda: draw_weather_radar(context.display, weather_data, transition=True),
        available=radar_available,
    )
    register(
        "inside",
        lambda: draw_inside(context.display, transition=True),
        available=is_inside_sensor_available(),
    )
    quad_enabled, quad_scroll_speed, quad_tiles = _next_quad_page_tiles()
    register(
        "quad",
        lambda tiles=quad_tiles, scroll_speed=quad_scroll_speed: draw_quad_screen(
            context.display,
            [_TileSpec(tile_id, lambda tile_id=tile_id, speed=scroll_speed: _render_quad_tile(tile_id, scroll_speed=speed)) for tile_id in tiles],
            transition=True,
            scroll_speed=scroll_speed,
        ),
        available=quad_enabled,
        quad_tiles=list(quad_tiles),
    )

    verano_logo = context.logos.get("verano logo")
    if verano_logo is not None:
        register(
            "verano logo",
            lambda img=verano_logo: _show_logo(context.display, img),
            available=True,
        )
    register("vrnof", lambda: draw_vrnof_screen(context.display, "VRNO", transition=True))

    scoreboards_available = not (context.offline and context.skip_scoreboards)
    scoreboards = (context.cache.get("scoreboards") or {})
    mlb_scoreboard_games = scoreboards.get("mlb") or []
    nhl_scoreboard_games = prepare_nhl_scoreboard_data(scoreboards.get("nhl"))
    today = context.now.date()
    nhl_scoreboards_available = scoreboards_available and not _is_nhl_break_day(today)

    def _is_live_game_today(game: Any) -> bool:
        """Return True when *game* appears to be in progress today."""

        if not isinstance(game, dict):
            return False

        status_parts: list[str] = []
        status_blob = game.get("status")
        if isinstance(status_blob, dict):
            for key in (
                "detailedState",
                "abstractGameState",
                "gameStatus",
                "gameStatusText",
                "state",
                "gameState",
            ):
                value = status_blob.get(key)
                if value:
                    status_parts.append(str(value))
            coded = str(status_blob.get("codedGameState") or "").upper()
            status_code = str(status_blob.get("statusCode") or "").upper()
        else:
            coded = str(game.get("codedGameState") or "").upper()
            status_code = str(game.get("statusCode") or "").upper()

        for key in (
            "gameStatusText",
            "gameStatus",
            "detailedState",
            "abstractGameState",
            "status",
            "gameState",
        ):
            value = game.get(key)
            if value:
                status_parts.append(str(value))

        status_text = " ".join(
            part.strip().lower() for part in status_parts if str(part).strip()
        )
        warmup = "warmup" in status_text

        if not status_text and not coded and not status_code:
            return False

        negative_keywords = (
            "final",
            "postponed",
            "suspend",
            "cancel",
            "delay",
            "preview",
            "schedule",
            "pregame",
        )
        if any(word in status_text for word in negative_keywords) and not warmup:
            return False

        positive = any(
            token in status_text
            for token in (
                "live",
                "in progress",
                "in-progress",
                "playing",
                "1st",
                "2nd",
                "3rd",
                "4th",
                "5th",
                "6th",
                "7th",
                "8th",
                "9th",
                "ot",
                "quarter",
                "period",
                "half",
                "top",
                "bottom",
                "warmup",
            )
        )

        if warmup:
            positive = True

        if not positive:
            if coded == "I":
                positive = True
            elif status_code == "2":
                positive = True

        if not positive:
            return False

        today = context.now.date()
        date_candidates: list[str] = []
        for key in (
            "officialDate",
            "official_date",
            "gameDate",
            "game_date",
            "date",
        ):
            value = game.get(key)
            if isinstance(value, str) and value.strip():
                date_candidates.append(value.strip())

        for text in date_candidates:
            candidate = text[:10]
            try:
                game_date = _dt.date.fromisoformat(candidate)
            except ValueError:
                continue
            if game_date == today:
                return True
            return False

        return True

    def register_logo(screen_id: str):
        image = context.logos.get(screen_id)
        if image is None:
            return
        register(screen_id, lambda img=image: _show_logo(context.display, img), available=True)

    for base_logo in (
        "bears logo",
        "hawks logo",
        "bulls logo",
        "nfl logo",
        "nhl logo",
        "mlb logo",
        "nba logo",
    ):
        register_logo(base_logo)

    bears = context.cache.get("bears") or {}
    if bears.get("stand"):
        register(
            "bears stand1",
            lambda data=bears.get("stand"): draw_nfl_standings_screen1(
                context.display,
                data,
                os.path.join(context.image_dir, "nfl/chi.png"),
                "NFC North",
                screen_id="bears stand1",
                transition=True,
            ),
            available=True,
        )
        register(
            "bears stand2",
            lambda data=bears.get("stand"): draw_nfl_standings_screen2(
                context.display,
                data,
                os.path.join(context.image_dir, "nfl/chi.png"),
                screen_id="bears stand2",
                transition=True,
            ),
            available=True,
        )

    register("bears next", lambda: show_bears_next_game(context.display, transition=True))
    register("bears next season", lambda: show_bears_next_season(context.display, transition=True))
    register(
        "NFL Scoreboard",
        lambda: render_nfl_scoreboard(
            context.display,
            (context.cache.get("scoreboards") or {}).get("nfl") or [],
            transition=True,
        ),
        available=scoreboards_available,
    )
    register(
        "NFL Scoreboard v2",
        (
            lambda: render_nfl_scoreboard(
                context.display,
                (context.cache.get("scoreboards") or {}).get("nfl") or [],
                transition=True,
            )
            if adafruit_minipitft_layout or waveshare_oled_lcd_hat
            else render_nfl_scoreboard_v2(
                context.display,
                (context.cache.get("scoreboards") or {}).get("nfl") or [],
                transition=True,
            )
        ),
        available=scoreboards_available,
    )
    register("NFL Overview NFC", lambda: draw_nfl_overview_nfc(context.display, transition=True))
    register("NFL Overview AFC", lambda: draw_nfl_overview_afc(context.display, transition=True))
    register("NFL Standings NFC", lambda: draw_nfl_standings_nfc(context.display, transition=True))
    register("NFL Standings AFC", lambda: draw_nfl_standings_afc(context.display, transition=True))

    hawks = context.cache.get("hawks") or {}
    if any(hawks.values()):
        register_logo("hawks logo")
        hawks_next = hawks.get("next")
        hawks_next_home = hawks.get("next_home")
        if _games_match(hawks_next_home, hawks_next):
            hawks_next_home = None
        if hawks.get("stand"):
            register(
                "hawks stand1",
                lambda data=hawks.get("stand"): draw_nhl_standings_screen1(
                    context.display,
                    data,
                    os.path.join(context.image_dir, "nhl/CHI.png"),
                    "",
                    logo_scale=1.0,
                    screen_id="hawks stand1",
                    transition=True,
                ),
                available=True,
            )
            register(
            "hawks last",
            lambda data=hawks.get("last"): draw_last_hawks_game(
                context.display, data, transition=True
            ),
            available=bool(hawks.get("last")),
        )
        register(
            "hawks live",
            lambda data=hawks.get("live"): draw_live_hawks_game(
                context.display, data, transition=True
            ),
            available=_is_live_game_today(hawks.get("live")),
        )
        register(
            "hawks next",
            lambda data=hawks_next: draw_sports_screen_hawks(
                context.display, data, transition=True
            ),
            available=bool(hawks_next),
        )
        if hawks_next_home:
            register(
                "hawks next home",
                lambda data=hawks_next_home: draw_hawks_next_home_game(
                    context.display, data, transition=True
                ),
                available=True,
            )
        register(
            "hawks schedule quad",
            lambda scroll_speed=_QUAD_DEFAULT_SCROLL_SPEED: draw_quad_screen(
                context.display,
                [
                    _TileSpec("hawks stand1", lambda speed=scroll_speed: _render_quad_tile("hawks stand1", scroll_speed=speed)),
                    _TileSpec("hawks last", lambda speed=scroll_speed: _render_quad_tile("hawks last", scroll_speed=speed)),
                    _TileSpec("hawks next", lambda speed=scroll_speed: _render_quad_tile("hawks next", scroll_speed=speed)),
                    (
                        _TileSpec("hawks next home", lambda speed=scroll_speed: _render_quad_tile("hawks next home", scroll_speed=speed))
                        if hawks_next_home
                        else _TileSpec("blank", _render_black_quad_tile)
                    ),
                ],
                transition=True,
                scroll_speed=scroll_speed,
            ),
            available=bool(hawks.get("stand"))
            and bool(hawks.get("last"))
            and bool(hawks_next),
            quad_tiles=[
                "hawks stand1",
                "hawks last",
                "hawks next",
                "hawks next home" if hawks_next_home else "blank",
            ],
        )

        register_logo("nhl logo")
        register(
            "NHL Scoreboard",
            lambda: render_nhl_scoreboard(
                context.display,
                nhl_scoreboard_games,
                transition=True,
            ),
            available=nhl_scoreboards_available,
        )
        register(
            "NHL Scoreboard v2",
            (
                lambda: render_nhl_scoreboard(
                    context.display,
                    nhl_scoreboard_games,
                    transition=True,
                )
                if adafruit_minipitft_layout or waveshare_oled_lcd_hat
                else render_nhl_scoreboard_v2(
                    context.display,
                    nhl_scoreboard_games,
                    transition=True,
                )
            ),
            available=nhl_scoreboards_available,
        )
        register(
            "NHL Playoffs",
            lambda: render_nhl_playoffs(
                context.display,
                nhl_scoreboard_games,
                transition=True,
            ),
            available=nhl_scoreboards_available,
        )
        register(
            "NHL Standings Overview West",
            lambda: draw_nhl_standings_overview_west(context.display, transition=True),
        )
        register(
            "NHL Standings Overview East",
            lambda: draw_nhl_standings_overview_east(context.display, transition=True),
        )
        register(
            "NHL Standings West",
            lambda: draw_nhl_standings_west(context.display, transition=True),
        )
        register(
            "NHL Standings East",
            lambda: draw_nhl_standings_east(context.display, transition=True),
        )
        register(
            "NHL Standings West v2",
            lambda: draw_nhl_standings_west_v2(context.display, transition=True),
        )
        register(
            "NHL Standings East v2",
            lambda: draw_nhl_standings_east_v2(context.display, transition=True),
        )

    wolves = context.cache.get("wolves") or {}
    if any(wolves.values()):
        register_logo("wolves logo")
        wolves_next = wolves.get("next")
        wolves_next_home = wolves.get("next_home")
        if _games_match(wolves_next_home, wolves_next):
            wolves_next_home = None
        register(
            "wolves last",
            lambda data=wolves.get("last"): draw_last_wolves_game(
                context.display, data, transition=True
            ),
            available=bool(wolves.get("last")),
        )
        register(
            "wolves live",
            lambda data=wolves.get("live"): draw_live_wolves_game(
                context.display, data, transition=True
            ),
            available=_is_live_game_today(wolves.get("live")),
        )
        register(
            "wolves next",
            lambda data=wolves_next: draw_sports_screen_wolves(
                context.display, data, transition=True
            ),
            available=bool(wolves_next),
        )
        if wolves_next_home:
            register(
                "wolves next home",
                lambda data=wolves_next_home: draw_wolves_next_home_game(
                    context.display, data, transition=True
                ),
                available=True,
            )

    cubs = context.cache.get("cubs") or {}
    if any(cubs.values()):
        register_logo("cubs logo")
        cubs_next = cubs.get("next")
        cubs_next_alt = cubs.get("next_alt")
        cubs_current_series = cubs.get("current_series")
        cubs_next_series = cubs.get("next_series")
        cubs_next_home_series = cubs.get("next_home_series")
        cubs_next_home = cubs.get("next_home")
        if _games_match(cubs_next_home, cubs_next):
            cubs_next_home = None

        register(
            "cubs stand1",
            lambda data=cubs.get("stand"): draw_standings_screen1(
                context.display,
                data,
                os.path.join(context.image_dir, "mlb/CUBS.png"),
                "NL Central",
                screen_id="cubs stand1",
                transition=True,
            ),
            available=bool(cubs.get("stand")),
        )
        register(
            "cubs stand2",
            lambda data=cubs.get("stand"): draw_standings_screen2(
                context.display,
                data,
                os.path.join(context.image_dir, "mlb/CUBS.png"),
                screen_id="cubs stand2",
                transition=True,
            ),
            available=bool(cubs.get("stand")),
        )
        register(
            "cubs stand3",
            lambda data=cubs.get("stand"): draw_standings_screen3(
                context.display,
                data,
                os.path.join(context.image_dir, "mlb/CUBS.png"),
                "NL Central",
                screen_id="cubs stand3",
                transition=True,
            ),
            available=bool(cubs.get("stand")),
        )
        register(
            "cubs last",
            lambda primary=cubs.get("last"), alternate=cubs.get("last_alt"): draw_last_game(
                context.display,
                _rotate_games("cubs last", primary, alternate),
                "Last Cubs game...",
                screen_id="cubs last",
                transition=True,
            ),
            available=bool(cubs.get("last") or cubs.get("last_alt")),
        )
        register(
            "cubs result",
            lambda data=cubs.get("last"): draw_cubs_result(
                context.display, data, transition=True
            ),
            available=bool(cubs.get("last")),
        )
        register(
            "cubs live",
            lambda data=cubs.get("live"): draw_box_score(
                context.display,
                data,
                "Cubs Live...",
                screen_id="cubs live",
                transition=True,
            ),
            available=_is_live_game_today(cubs.get("live")),
        )
        register(
            "cubs next",
            lambda primary=cubs_next, alternate=cubs_next_alt: draw_sports_screen(
                context.display,
                _rotate_games("cubs next", primary, alternate),
                "Next Cubs game...",
                screen_id="cubs next",
                transition=True,
            ),
            available=bool(cubs_next or cubs_next_alt),
        )
        if cubs_next_home:
            register(
                "cubs next home",
                lambda data=cubs_next_home: draw_next_home_game(
                    context.display,
                    data,
                    transition=True,
                    screen_id="cubs next home",
                ),
                available=True,
            )
        register(
            "cubs current series",
            lambda data=cubs_current_series: draw_series_screen(
                context.display,
                data,
                _mlb_series_title("Cubs", "Current Series"),
                screen_id="cubs current series",
                transition=True,
            ),
            available=bool(cubs_current_series),
        )
        register(
            "cubs next series",
            lambda data=cubs_next_series: draw_series_screen(
                context.display,
                data,
                _mlb_series_title("Cubs", "Next Series"),
                screen_id="cubs next series",
                transition=True,
            ),
            available=bool(cubs_next_series),
        )
        register(
            "cubs next home series",
            lambda data=cubs_next_home_series: draw_series_screen(
                context.display,
                data,
                _mlb_series_title("Cubs", "Following Home Series"),
                screen_id="cubs next home series",
                transition=True,
            ),
            available=bool(cubs_next_home_series),
        )
        register(
            "cubs schedule quad",
            lambda scroll_speed=_QUAD_DEFAULT_SCROLL_SPEED: draw_quad_screen(
                context.display,
                [
                    _TileSpec("cubs next", lambda speed=scroll_speed: _render_quad_tile("cubs next", scroll_speed=speed)),
                    _TileSpec("cubs current series", lambda speed=scroll_speed: _render_quad_tile("cubs current series", scroll_speed=speed)),
                    _TileSpec("cubs next series", lambda speed=scroll_speed: _render_quad_tile("cubs next series", scroll_speed=speed)),
                    _TileSpec("cubs next home series", lambda speed=scroll_speed: _render_quad_tile("cubs next home series", scroll_speed=speed)),
                ],
                transition=True,
                scroll_speed=scroll_speed,
            ),
            available=bool(cubs_next or cubs_next_alt)
            and bool(cubs_current_series)
            and bool(cubs_next_series)
            and bool(cubs_next_home_series),
            quad_tiles=[
                "cubs next",
                "cubs current series",
                "cubs next series",
                "cubs next home series",
            ],
        )

    sox = context.cache.get("sox") or {}
    if any(sox.values()):
        register_logo("sox logo")
        sox_next = sox.get("next")
        sox_next_alt = sox.get("next_alt")
        sox_current_series = sox.get("current_series")
        sox_next_series = sox.get("next_series")
        sox_next_home_series = sox.get("next_home_series")
        sox_next_home = sox.get("next_home")
        if _games_match(sox_next_home, sox_next):
            sox_next_home = None

        register(
            "sox stand1",
            lambda data=sox.get("stand"): draw_standings_screen1(
                context.display,
                data,
                os.path.join(context.image_dir, "mlb/SOX.png"),
                "AL Central",
                screen_id="sox stand1",
                transition=True,
            ),
            available=bool(sox.get("stand")),
        )
        register(
            "sox stand2",
            lambda data=sox.get("stand"): draw_standings_screen2(
                context.display,
                data,
                os.path.join(context.image_dir, "mlb/SOX.png"),
                screen_id="sox stand2",
                transition=True,
            ),
            available=bool(sox.get("stand")),
        )
        register(
            "sox stand3",
            lambda data=sox.get("stand"): draw_standings_screen3(
                context.display,
                data,
                os.path.join(context.image_dir, "mlb/SOX.png"),
                "AL Central",
                screen_id="sox stand3",
                transition=True,
            ),
            available=bool(sox.get("stand")),
        )
        register(
            "sox last",
            lambda primary=sox.get("last"), alternate=sox.get("last_alt"): draw_last_game(
                context.display,
                _rotate_games("sox last", primary, alternate),
                "Last Sox game...",
                screen_id="sox last",
                transition=True,
            ),
            available=bool(sox.get("last") or sox.get("last_alt")),
        )
        register(
            "sox live",
            lambda data=sox.get("live"): draw_box_score(
                context.display,
                data,
                "Sox Live...",
                screen_id="sox live",
                transition=True,
            ),
            available=_is_live_game_today(sox.get("live")),
        )
        register(
            "sox next",
            lambda primary=sox_next, alternate=sox_next_alt: draw_sports_screen(
                context.display,
                _rotate_games("sox next", primary, alternate),
                "Next Sox game...",
                screen_id="sox next",
                transition=True,
            ),
            available=bool(sox_next or sox_next_alt),
        )
        if sox_next_home:
            register(
                "sox next home",
                lambda data=sox_next_home: draw_next_home_game(
                    context.display,
                    data,
                    transition=True,
                    screen_id="sox next home",
                ),
                available=True,
            )
        register(
            "sox current series",
            lambda data=sox_current_series: draw_series_screen(
                context.display,
                data,
                _mlb_series_title("Sox", "Current Series"),
                screen_id="sox current series",
                transition=True,
            ),
            available=bool(sox_current_series),
        )
        register(
            "sox next series",
            lambda data=sox_next_series: draw_series_screen(
                context.display,
                data,
                _mlb_series_title("Sox", "Next Series"),
                screen_id="sox next series",
                transition=True,
            ),
            available=bool(sox_next_series),
        )
        register(
            "sox next home series",
            lambda data=sox_next_home_series: draw_series_screen(
                context.display,
                data,
                _mlb_series_title("Sox", "Following Home Series"),
                screen_id="sox next home series",
                transition=True,
            ),
            available=bool(sox_next_home_series),
        )
        register(
            "sox schedule quad",
            lambda scroll_speed=_QUAD_DEFAULT_SCROLL_SPEED: draw_quad_screen(
                context.display,
                [
                    _TileSpec("sox next", lambda speed=scroll_speed: _render_quad_tile("sox next", scroll_speed=speed)),
                    _TileSpec("sox current series", lambda speed=scroll_speed: _render_quad_tile("sox current series", scroll_speed=speed)),
                    _TileSpec("sox next series", lambda speed=scroll_speed: _render_quad_tile("sox next series", scroll_speed=speed)),
                    _TileSpec("sox next home series", lambda speed=scroll_speed: _render_quad_tile("sox next home series", scroll_speed=speed)),
                ],
                transition=True,
                scroll_speed=scroll_speed,
            ),
            available=bool(sox_next or sox_next_alt)
            and bool(sox_current_series)
            and bool(sox_next_series)
            and bool(sox_next_home_series),
            quad_tiles=[
                "sox next",
                "sox current series",
                "sox next series",
                "sox next home series",
            ],
        )

    register(
        "MLB Scoreboard",
        lambda: render_mlb_scoreboard(
            context.display,
            mlb_scoreboard_games,
            transition=True,
        ),
        available=scoreboards_available,
    )
    register(
        "MLB Scoreboard v2",
        (
            lambda: render_mlb_scoreboard(
                context.display,
                mlb_scoreboard_games,
                transition=True,
            )
            if adafruit_minipitft_layout
            else render_mlb_scoreboard_v2(
                context.display,
                mlb_scoreboard_games,
                transition=True,
            )
        ),
        available=scoreboards_available,
    )
    register(
        "NBA Scoreboard",
        lambda: render_nba_scoreboard(
            context.display,
            (context.cache.get("scoreboards") or {}).get("nba") or [],
            transition=True,
        ),
        available=scoreboards_available,
    )
    register(
        "NBA Scoreboard v2",
        (
            lambda: render_nba_scoreboard(
                context.display,
                (context.cache.get("scoreboards") or {}).get("nba") or [],
                transition=True,
            )
            if adafruit_minipitft_layout or waveshare_oled_lcd_hat
            else render_nba_scoreboard_v2(
                context.display,
                (context.cache.get("scoreboards") or {}).get("nba") or [],
                transition=True,
            )
        ),
        available=scoreboards_available,
    )
    register(
        "NBA Playoffs",
        lambda: render_nba_playoffs(
            context.display,
            (context.cache.get("scoreboards") or {}).get("nba") or [],
            transition=True,
        ),
        available=scoreboards_available,
    )
    register(
        "NCAAM Scoreboard",
        lambda: render_ncaam_scoreboard(
            context.display,
            (context.cache.get("scoreboards") or {}).get("ncaam") or [],
            transition=True,
        ),
        available=scoreboards_available,
    )

    register("NL Overview", lambda: draw_NL_Overview(context.display, transition=True))
    register("AL Overview", lambda: draw_AL_Overview(context.display, transition=True))
    register("MLB AL Standings", lambda: draw_mlb_al_standings(context.display, transition=True))
    register("MLB NL Standings", lambda: draw_mlb_nl_standings(context.display, transition=True))

    bulls = context.cache.get("bulls") or {}
    register_logo("bulls logo")
    bulls_next = bulls.get("next")
    bulls_next_home = bulls.get("next_home")
    if not bulls_next_home and _is_bulls_home_game(bulls_next):
        bulls_next_home = bulls_next
    # Always show the Bulls "next home" card, even if the next game is at home.
    # This avoids dropping the screen when the next home matchup matches the
    # general "next" game entry.

    if bulls.get("stand"):
        register(
            "bulls stand1",
            lambda data=bulls.get("stand"): draw_nba_standings_screen1(
                context.display,
                data,
                os.path.join(context.image_dir, "nba/CHI.png"),
                "Western conf.",
                logo_scale=1.0,
                screen_id="bulls stand1",
                transition=True,
            ),
            available=True,
        )

    register(
        "bulls last",
        lambda data=bulls.get("last"): draw_last_bulls_game(
            context.display, data, transition=True
        ),
        available=True,
    )
    register(
        "bulls live",
        lambda data=bulls.get("live"): draw_live_bulls_game(
            context.display, data, transition=True
        ),
        available=_is_live_game_today(bulls.get("live")),
    )
    register(
        "bulls next",
        lambda data=bulls_next: draw_sports_screen_bulls(
            context.display, data, transition=True
        ),
        available=True,
    )
    register(
        "bulls next home",
        lambda data=bulls_next_home: draw_bulls_next_home_game(
            context.display, data, transition=True
        ),
        available=True,
    )
    register(
        "bulls schedule quad",
        lambda scroll_speed=_QUAD_DEFAULT_SCROLL_SPEED: draw_quad_screen(
            context.display,
            [
                _TileSpec("bulls stand1", lambda speed=scroll_speed: _render_quad_tile("bulls stand1", scroll_speed=speed)),
                _TileSpec("bulls last", lambda speed=scroll_speed: _render_quad_tile("bulls last", scroll_speed=speed)),
                _TileSpec("bulls next", lambda speed=scroll_speed: _render_quad_tile("bulls next", scroll_speed=speed)),
                _TileSpec("bulls next home", lambda speed=scroll_speed: _render_quad_tile("bulls next home", scroll_speed=speed)),
            ],
            transition=True,
            scroll_speed=scroll_speed,
        ),
        available=bool(bulls.get("stand"))
        and bool(bulls.get("last"))
        and bool(bulls_next)
        and bool(bulls_next_home),
        quad_tiles=[
            "bulls stand1",
            "bulls last",
            "bulls next",
            "bulls next home",
        ],
    )

    return registry, metadata
