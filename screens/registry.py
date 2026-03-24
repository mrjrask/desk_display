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

from config import CENTRAL_TIME, HEIGHT, NBA_TEAM_TRICODE, WIDTH
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
from screens.draw_sensors import draw_sensors
from screens.draw_vrnof import draw_vrnof_screen
from screens.draw_weather import (
    _pop_pct_from,
    draw_weather_daily,
    draw_weather_hourly,
    draw_weather_radar,
    draw_weather_screen_1,
    draw_weather_screen_2,
)
from screens.draw_nixie import draw_nixie
from screens.draw_date_time import draw_date, draw_time
from screens.draw_quad import _TileSpec, draw_quad_screen
from screens.mlb_schedule import (
    draw_box_score,
    draw_cubs_result,
    draw_last_game,
    draw_next_home_game,
    draw_sports_screen,
)
from screens.mlb_scoreboard import render_mlb_scoreboard
from screens.mlb_scoreboard_v2 import render_mlb_scoreboard_v2
from screens.mlb_standings import (
    draw_AL_Central,
    draw_AL_East,
    draw_AL_Overview,
    draw_AL_West,
    draw_AL_WildCard,
    draw_NL_Central,
    draw_NL_East,
    draw_NL_Overview,
    draw_NL_West,
    draw_NL_WildCard,
)
from screens.mlb_team_standings import draw_standings_screen1, draw_standings_screen2
from screens.nba_team_standings import (
    draw_nba_standings_screen1,
    draw_nba_standings_screen2,
)
from screens.nfl_team_standings import (
    draw_nfl_standings_screen1,
    draw_nfl_standings_screen2,
)
from screens.nhl_team_standings import (
    draw_nhl_standings_screen1,
    draw_nhl_standings_screen2,
)
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

RenderCallable = Callable[[], Optional[Image.Image | ScreenImage]]

_LAYOUTS_CONFIG_PATH = os.environ.get(
    "SCREENS_LAYOUTS_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "screens_layouts.json"),
)


_QUAD_DEFAULT_PAGE = ["date", "weather1", "weather hourly", "inside"]
_quad_page_index = 0
_quad_page_lock = threading.Lock()
_layouts_cache_lock = threading.Lock()
_layouts_payload_cache: Optional[dict[str, Any]] = None
_layouts_payload_mtime: Optional[float] = None


def _quad_layout_from_layouts() -> tuple[bool, list[list[str]]]:
    enabled = False
    pages: list[list[str]] = []

    payload = _load_layouts_payload()
    if payload is None:
        return enabled, [_QUAD_DEFAULT_PAGE.copy()]

    if not isinstance(payload, dict):
        return enabled, [_QUAD_DEFAULT_PAGE.copy()]
    screens = payload.get("screens")
    if not isinstance(screens, dict):
        return enabled, [_QUAD_DEFAULT_PAGE.copy()]
    quad = screens.get("quad")
    if not isinstance(quad, dict):
        return enabled, [_QUAD_DEFAULT_PAGE.copy()]

    enabled = bool(quad.get("enabled", False))

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

    return enabled, pages


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


def _next_quad_page_tiles() -> tuple[bool, list[str]]:
    global _quad_page_index

    enabled, pages = _quad_layout_from_layouts()
    with _quad_page_lock:
        page = pages[_quad_page_index % len(pages)]
        _quad_page_index += 1
    return enabled, page


RADAR_LOOKAHEAD_HOURS = 8
WEATHER_CURRENT_TTL = _dt.timedelta(minutes=20)
WEATHER_HOURLY_TTL = _dt.timedelta(hours=1)
NHL_BREAK_START = _dt.date(2026, 2, 6)
NHL_BREAK_END = _dt.date(2026, 2, 24)
def _is_1080p_or_higher(width: int, height: int) -> bool:
    """Return True when layout is 1080p-class (or higher) regardless of orientation."""
    short_edge = min(int(width), int(height))
    return short_edge >= 1080


def _is_display_hat_mini_layout(width: int, height: int) -> bool:
    """Return True for Display HAT Mini dimensions regardless of orientation."""
    return sorted((int(width), int(height))) == [240, 320]


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
    register("time", lambda: draw_time(context.display, transition=False))
    register("nixie", lambda: draw_nixie(context.display, transition=False))

    weather_data = context.cache.get("weather")

    class _QuadCaptureDisplay:
        def __init__(self):
            self.width = WIDTH
            self.height = HEIGHT
            self._last: Optional[Image.Image] = None
            self._frame_id = 0

        def image(self, img: Image.Image):
            self._last = img.copy()
            self._frame_id += 1

        def show(self):
            return None

        def frame_id(self) -> int:
            return self._frame_id

        @property
        def last_image(self) -> Optional[Image.Image]:
            return self._last

    def _render_quad_tile(screen_id: str) -> Optional[Image.Image]:
        definition = registry.get(screen_id)
        if definition is None:
            return None

        capture = _QuadCaptureDisplay()
        original_display = context.display
        context.display = capture
        try:
            rendered = definition.render()
        finally:
            context.display = original_display

        if isinstance(rendered, ScreenImage):
            return rendered.image
        if isinstance(rendered, Image.Image):
            return rendered
        return capture.last_image
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
    quad_enabled, quad_tiles = _next_quad_page_tiles()
    register(
        "quad",
        lambda tiles=quad_tiles: draw_quad_screen(
            context.display,
            [_TileSpec(tile_id, lambda tile_id=tile_id: _render_quad_tile(tile_id)) for tile_id in tiles],
            transition=True,
        ),
        available=quad_enabled,
    )
    register("sensors", lambda: draw_sensors(context, transition=True))

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
    today = context.now.date()
    nhl_scoreboards_available = scoreboards_available and not (
        NHL_BREAK_START <= today <= NHL_BREAK_END
    )

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
        lambda: render_nfl_scoreboard_v2(
            context.display,
            (context.cache.get("scoreboards") or {}).get("nfl") or [],
            transition=True,
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
                "hawks stand2",
                lambda data=hawks.get("stand"): draw_nhl_standings_screen2(
                    context.display,
                    data,
                    os.path.join(context.image_dir, "nhl/CHI.png"),
                    screen_id="hawks stand2",
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

        register_logo("nhl logo")
        register(
            "NHL Scoreboard",
            lambda: render_nhl_scoreboard(
                context.display,
                (context.cache.get("scoreboards") or {}).get("nhl") or [],
                transition=True,
            ),
            available=nhl_scoreboards_available,
        )
        register(
            "NHL Scoreboard v2",
            lambda: render_nhl_scoreboard_v2(
                context.display,
                (context.cache.get("scoreboards") or {}).get("nhl") or [],
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

    sox = context.cache.get("sox") or {}
    if any(sox.values()):
        register_logo("sox logo")
        sox_next = sox.get("next")
        sox_next_alt = sox.get("next_alt")
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
        lambda: render_mlb_scoreboard_v2(
            context.display,
            mlb_scoreboard_games,
            transition=True,
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
        lambda: render_nba_scoreboard_v2(
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
    register("NL East", lambda: draw_NL_East(context.display, transition=True))
    register("NL Central", lambda: draw_NL_Central(context.display, transition=True))
    register("NL West", lambda: draw_NL_West(context.display, transition=True))
    register("NL Wild Card", lambda: draw_NL_WildCard(context.display, transition=True))
    register("AL Overview", lambda: draw_AL_Overview(context.display, transition=True))
    register("AL East", lambda: draw_AL_East(context.display, transition=True))
    register("AL Central", lambda: draw_AL_Central(context.display, transition=True))
    register("AL West", lambda: draw_AL_West(context.display, transition=True))
    register("AL Wild Card", lambda: draw_AL_WildCard(context.display, transition=True))

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
            "bulls stand2",
            lambda data=bulls.get("stand"): draw_nba_standings_screen2(
                context.display,
                data,
                os.path.join(context.image_dir, "nba/CHI.png"),
                screen_id="bulls stand2",
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

    return registry, metadata
