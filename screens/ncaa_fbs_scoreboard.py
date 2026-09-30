#!/usr/bin/env python3
"""Render NCAA FBS football Top 25 scoreboard."""

from __future__ import annotations

import datetime
import io
import logging
import os
import re
import time
from typing import Any, Optional

from PIL import Image, ImageDraw

from config import (
    CENTRAL_TIME,
    FONT_STATUS,
    FONT_TEAM_SPORTS,
    FONT_TITLE_SPORTS,
    HEIGHT,
    IMAGES_DIR,
    SCOREBOARD_BACKGROUND_COLOR,
    SCOREBOARD_FINAL_LOSING_SCORE_COLOR,
    SCOREBOARD_FINAL_WINNING_SCORE_COLOR,
    SCOREBOARD_IN_PROGRESS_SCORE_COLOR,
    SCOREBOARD_SCROLL_DELAY,
    SCOREBOARD_SCROLL_PAUSE_BOTTOM,
    SCOREBOARD_SCROLL_PAUSE_TOP,
    SCOREBOARD_SCROLL_STEP,
    SCOREBOARD_STANDINGS_BOTTOM_PADDING,
    WIDTH,
    get_screen_background_color,
    get_screen_font,
    get_screen_image_scale,
    is_hdmi_1080p_layout,
    is_hyperpixel_4_square_layout,
    is_hyperpixel_next_layout,
    scale_value,
    scale_value_width,
)
from image_compat import LANCZOS
from screens.scoreboard_components import center_text as _center_text
from services.http_client import get_session
from utils import ScreenImage, clear_display, scroll_vertical_content

HYPERPIXEL_LAYOUT = is_hyperpixel_next_layout()
HYPERPIXEL_4_SQUARE = is_hyperpixel_4_square_layout()
HDMI_1080P_LAYOUT = is_hdmi_1080p_layout()


def _scale_y(value: int) -> int:
    return scale_value(value) if HYPERPIXEL_LAYOUT else scale_value_width(value)


REQUEST_TIMEOUT = 10
SCREEN_ID = "NCAA FBS Scoreboard"
LOGO_DIR = os.path.join(IMAGES_DIR, "ncaa")
# Team logos the render server downloads from ESPN for itself. Logos in LOGO_DIR (the
# ones committed to the repo) always win. images/cache is untracked, so these
# never block a git pull, even when the same logo is later committed.
AUTO_LOGO_DIR = os.path.join(IMAGES_DIR, "cache", "ncaa")
# Longest edge of a downloaded logo; the team-logo limit in images/README.md
# and scripts/logo_getter.py.
AUTO_LOGO_MAX_DIMENSION = 128
# site.web.api.espn.com is primary: site.api.espn.com answers 403 to the
# college football scoreboard from some networks (seen on the server Pi).
# site.web.api rejects ranged dates and large limits (400), so every request
# asks for a single day with ESPN's default limit.
ESPN_URL = "https://site.web.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
# Tried in order when a host refuses. Each is a separate host, so a 403
# cooldown on one does not skip the others.
ESPN_FALLBACK_URLS = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard",
    "https://cdn.espn.com/core/college-football/scoreboard",
)
MODE_TOP25 = "top25"

TITLE_GAP = scale_value(8)
BLOCK_SPACING = scale_value(10)
SCORE_ROW_H = scale_value(56)
STATUS_ROW_H = scale_value(18)
SEED_FONT = get_screen_font(SCREEN_ID, "seed", base_font=FONT_STATUS, default_size=13)
RANK_FONT = get_screen_font(SCREEN_ID, "rank", base_font=FONT_STATUS, default_size=11)
TEAM_ABBREVIATION_FONT = get_screen_font(
    SCREEN_ID, "team_abbreviation", base_font=FONT_TEAM_SPORTS, default_size=18
)
SEED_GAP = max(2, scale_value_width(3))
RANK_GAP = max(1, scale_value_width(2))

# Score, logo, "@", logo, score. The logo columns are wide enough for a
# wordmark (LSU) at the same visual weight as a round mark; the score columns
# still fit a three-digit score.
COL_WIDTHS = [
    scale_value_width(64),
    scale_value_width(80),
    scale_value_width(32),
    scale_value_width(80),
    scale_value_width(64),
]
_TOTAL_COL_WIDTH = sum(COL_WIDTHS)
_COL_LEFT = max(0, (WIDTH - _TOTAL_COL_WIDTH) // 2)
COL_X = [_COL_LEFT]
for w in COL_WIDTHS:
    COL_X.append(COL_X[-1] + w)

TEAM_LOGO_BASE_HEIGHT = scale_value_width(36) if HYPERPIXEL_LAYOUT else scale_value_width(52)
LEAGUE_LOGO_BASE_HEIGHT = TEAM_LOGO_BASE_HEIGHT
LEAGUE_LOGO_GAP = scale_value(4)

SCORE_FONT = get_screen_font(SCREEN_ID, "score", base_font=FONT_TEAM_SPORTS, default_size=39)
STATUS_FONT = get_screen_font(SCREEN_ID, "status", base_font=FONT_STATUS, default_size=28)
CENTER_FONT = get_screen_font(SCREEN_ID, "center", base_font=FONT_STATUS, default_size=28)

IN_PROGRESS_SCORE_COLOR = SCOREBOARD_IN_PROGRESS_SCORE_COLOR
IN_PROGRESS_STATUS_COLOR = IN_PROGRESS_SCORE_COLOR
FINAL_WINNING_SCORE_COLOR = SCOREBOARD_FINAL_WINNING_SCORE_COLOR
FINAL_LOSING_SCORE_COLOR = SCOREBOARD_FINAL_LOSING_SCORE_COLOR
BACKGROUND_COLOR = get_screen_background_color(SCREEN_ID, SCOREBOARD_BACKGROUND_COLOR)

_SESSION = get_session("ncaa_fbs")
_REMOTE_LOGO_CACHE: dict[tuple[str, int, int, int], Optional[Image.Image]] = {}
# Every team logo is scaled to the area of a square this fraction of the logo
# box's shorter side, so wide wordmarks do not dwarf round or square marks.
TEAM_LOGO_AREA_FACTOR = 0.75
# A logo that could not be found or downloaded is retried after this long.
LOGO_MISS_RETRY_SECONDS = 30 * 60
_LOGO_MISSES: dict[tuple[str, int, int, int], float] = {}
_LEAGUE_LOGO_CACHE: dict[tuple[str, int], Optional[Image.Image]] = {}
_TEAM_LOGO_URL_OVERRIDES: dict[str, str] = {
    "iowa": "https://brand.uiowa.edu/sites/brand.uiowa.edu/files/styles/widescreen__1920_x_1080/public/2020-05/Tigerhawk-gold%20on%20black%402x.png?h=e39f7b2b&itok=TdYKif5p",
}


def _scoreboard_mode() -> str:
    return MODE_TOP25


def _mode_title_and_logo() -> tuple[str, str]:
    return "Top 25 - FBS", "NCAA"

def _team_logo_height() -> int:
    scale = get_screen_image_scale(SCREEN_ID, "team_logo", 1.0)
    target = max(1, int(round(TEAM_LOGO_BASE_HEIGHT * scale)))
    if HYPERPIXEL_4_SQUARE:
        target = max(1, int(round(target * 0.6)))
    return min(target, max(1, SCORE_ROW_H - _scale_y(8)))


def _league_logo_height() -> int:
    team_scale = get_screen_image_scale(SCREEN_ID, "team_logo", 1.0)
    scale = get_screen_image_scale(SCREEN_ID, "league_logo", team_scale)
    if HDMI_1080P_LAYOUT:
        # The base height is already scaled to the panel, and each image scale
        # multiplies in the display scale again (27x at 1080p), which made the
        # NCAA logo thousands of pixels tall. Match the team logos instead.
        return _team_logo_height()
    return max(1, int(round(LEAGUE_LOGO_BASE_HEIGHT * scale)))



def _fetch_json(params: dict[str, Any], url: str = ESPN_URL) -> dict[str, Any]:
    if url.startswith("https://cdn.espn.com/"):
        params = {**params, "xhr": 1}
    resp = _SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, dict) else {}


def _events_from_payload(payload: Any) -> Optional[list[dict]]:
    """Find the event list in a Site API payload or a CDN ``content.sbData`` wrapper."""

    if not isinstance(payload, dict):
        return None
    events = payload.get("events")
    if isinstance(events, list):
        return [event for event in events if isinstance(event, dict)]
    for key in ("content", "sbData"):
        found = _events_from_payload(payload.get(key))
        if found is not None:
            return found
    return None


def _fetch_events(params: dict[str, Any]) -> list[dict]:
    """Return ESPN events from the first host that answers; raise if none do."""

    errors: list[str] = []
    for url in (ESPN_URL, *ESPN_FALLBACK_URLS):
        try:
            events = _events_from_payload(_fetch_json(params, url))
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue
        if events is None:
            errors.append(f"{url}: response had no events")
            continue
        if errors:
            logging.info("NCAA FBS scoreboard fetched from fallback %s", url)
        return events
    raise RuntimeError("; ".join(errors) or "no ESPN hosts configured")


def _ranked_games(raw_events: list[dict]) -> list[dict]:
    filtered: list[dict] = []
    for event in raw_events:
        comp = (event.get("competitions") or [{}])[0] or {}
        competitors = comp.get("competitors") or []
        if any(_extract_rank(team) is not None for team in competitors if isinstance(team, dict)):
            filtered.append(_normalize_event(event))
    return filtered


def fetch_games_for_day(day: datetime.date) -> list[dict]:
    """Return one day's Top 25 games; raise when no ESPN host answers."""

    return _ranked_games(_fetch_events({"dates": day.strftime("%Y%m%d"), "groups": 80}))


def _extract_seed(competitor: dict[str, Any]) -> str:
    def _parse_seed_value(value: Any) -> str:
        if isinstance(value, bool):
            return ""
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed.isdigit():
                return trimmed
            match = re.search(r"\d+", trimmed)
            return match.group(0) if match else ""
        if isinstance(value, dict):
            for nested_key in ("seed", "current", "value", "displayValue", "rank"):
                parsed = _parse_seed_value(value.get(nested_key))
                if parsed:
                    return parsed
        return ""

    # `curatedRank.current` is frequently 99 for unranked teams, which should
    # not be shown as a tournament seed.
    for source in (competitor, competitor.get("team")):
        if not isinstance(source, dict):
            continue
        for key in ("seed", "tournamentSeed", "playoffSeed"):
            parsed = _parse_seed_value(source.get(key))
            if parsed:
                return parsed
    return ""


def _extract_rank(competitor: dict[str, Any]) -> Optional[int]:
    rank = competitor.get("curatedRank")
    if isinstance(rank, dict):
        value = rank.get("current")
        try:
            parsed = int(value)
            return parsed if 1 <= parsed <= 25 else None
        except (TypeError, ValueError):
            pass
    for key in ("rank", "curatedRankCurrent"):
        try:
            parsed = int(competitor.get(key))
            return parsed if 1 <= parsed <= 25 else None
        except (TypeError, ValueError):
            continue
    return None


def _is_tournament_game(event: dict[str, Any]) -> bool:
    text_parts: list[str] = []
    for key in ("name", "shortName", "seasonSlug", "type"):
        value = event.get(key)
        if value:
            text_parts.append(str(value).lower())
    season = event.get("season")
    if isinstance(season, dict):
        for key in ("slug", "displayName"):
            val = season.get(key)
            if val:
                text_parts.append(str(val).lower())
        try:
            if int(season.get("type")) == 3:
                return True
        except (TypeError, ValueError):
            pass
    for comp in event.get("competitions") or []:
        if not isinstance(comp, dict):
            continue
        if comp.get("tournamentId"):
            return True
        for c in comp.get("competitors") or []:
            if _extract_seed(c):
                return True
    text = " ".join(text_parts)
    return "tournament" in text or "march madness" in text or ("ncaa" in text and "first four" in text)


def _normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    comp = (event.get("competitions") or [{}])[0] or {}
    competitors = comp.get("competitors") or []
    away = next((c for c in competitors if str(c.get("homeAway", "")).lower() == "away"), competitors[0] if competitors else {})
    home = next((c for c in competitors if str(c.get("homeAway", "")).lower() == "home"), competitors[1] if len(competitors) > 1 else {})

    status_blob = comp.get("status") or event.get("status") or {}
    status_type = status_blob.get("type") if isinstance(status_blob, dict) else {}
    state = str((status_type or {}).get("state") or "").lower()
    completed = bool((status_type or {}).get("completed"))

    return {
        "id": event.get("id"),
        "date": comp.get("date") or event.get("date"),
        "status": {
            "type": {
                "state": state,
                "completed": completed,
                "shortDetail": (status_type or {}).get("shortDetail") or (status_type or {}).get("description") or status_blob.get("type", {}).get("description", ""),
            },
            "displayClock": status_blob.get("displayClock", ""),
            "period": status_blob.get("period"),
        },
        "teams": {"away": away, "home": home},
    }


def _fetch_games_for_date(day: datetime.date, mode: Optional[str] = None) -> list[dict]:
    try:
        return fetch_games_for_day(day)
    except Exception as exc:
        logging.error("Failed to fetch NCAA FBS scoreboard for %s: %s", day, exc)
        return []

def _status_text(game: dict) -> str:
    status = (game or {}).get("status", {}) or {}
    type_info = status.get("type") or {}
    detail = str(type_info.get("shortDetail") or "").strip()
    state = str(type_info.get("state") or "").lower()
    if state == "pre":
        start = _parse_start_time_central(game)
        if start:
            return start
        return "Scheduled"
    return detail or "Scheduled"


def _parse_start_time_central(game: dict[str, Any]) -> str:
    raw = str((game or {}).get("date") or "").strip()
    if not raw:
        return ""
    try:
        dt = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.UTC)
        local = dt.astimezone(CENTRAL_TIME)
        day_of_week = local.strftime("%a")
        month = local.month
        day = local.day
        gametime = local.strftime("%-I:%M %p")
        return f"{day_of_week} {month}/{day} {gametime}"
    except Exception:
        return ""


def _is_in_progress(game: dict) -> bool:
    return str(((game.get("status") or {}).get("type") or {}).get("state") or "").lower() == "in"


def _is_final(game: dict) -> bool:
    status_type = (game.get("status") or {}).get("type") or {}
    state = str(status_type.get("state") or "").lower()
    return state == "post" or bool(status_type.get("completed"))


def _score_text(team: dict, *, show: bool) -> str:
    if not show:
        return "—"
    score = team.get("score")
    return str(score) if score not in (None, "") else "—"


def _should_display_scores(game: dict) -> bool:
    state = str((((game or {}).get("status") or {}).get("type") or {}).get("state") or "").lower()
    return state in {"in", "post"}


def _score_value(team: dict) -> Optional[int]:
    try:
        return int(str(team.get("score")))
    except Exception:
        return None


def _score_fill(team_key: str, *, in_progress: bool, final: bool, away: dict, home: dict) -> tuple[int, int, int]:
    if in_progress:
        return IN_PROGRESS_SCORE_COLOR
    if not final:
        return (255, 255, 255)
    away_score = _score_value(away)
    home_score = _score_value(home)
    if away_score is None or home_score is None or away_score == home_score:
        return (255, 255, 255)
    if team_key == "away":
        return FINAL_WINNING_SCORE_COLOR if away_score > home_score else FINAL_LOSING_SCORE_COLOR
    return FINAL_WINNING_SCORE_COLOR if home_score > away_score else FINAL_LOSING_SCORE_COLOR


def _team_logo_filename(team: dict[str, Any]) -> str:
    team_blob = team.get("team") if isinstance(team.get("team"), dict) else team
    abbreviation = str(team_blob.get("abbreviation") or "").strip().upper()
    if abbreviation:
        return f"{abbreviation}.png"
    team_id = str(team_blob.get("id") or team.get("id") or "unknown").strip()
    return f"{team_id}.png"


def _team_abbreviation(team: dict[str, Any]) -> str:
    """Return the exact filename stem used for this team's missing-logo fallback."""

    return os.path.splitext(_team_logo_filename(team))[0]


def _team_logo_box(height: int, column_width: int) -> tuple[int, int, int]:
    """Return (max width, max height, target area) for every team logo in a column.

    Logos are scaled to the same area, so a wide wordmark (LSU, Mississippi
    State) and a round mark (Georgia, BYU) carry the same visual weight, and
    never past the column (so they do not run into the "@" or the scores).
    """

    margin = max(2, scale_value_width(4))
    max_width = max(1, column_width - 2 * margin)
    area = int(round((min(height, max_width) * TEAM_LOGO_AREA_FACTOR) ** 2))
    return max_width, height, area


def _fit_logo(img: Image.Image, box_w: int, box_h: int, area: Optional[int] = None) -> Image.Image:
    """Trim transparent padding, then scale to *area* within the box."""

    img = img.convert("RGBA")
    bbox = img.getchannel("A").getbbox()
    if bbox:
        img = img.crop(bbox)
    w, h = max(1, img.width), max(1, img.height)
    scale = min(box_w / w, box_h / h)
    if area:
        scale = min(scale, (area / (w * h)) ** 0.5)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return img.resize(size, LANCZOS)


def _team_logo_url(team: dict[str, Any]) -> str:
    team_blob = team.get("team") if isinstance(team.get("team"), dict) else team
    for key in ("abbreviation", "shortDisplayName", "displayName", "name", "location"):
        override = _TEAM_LOGO_URL_OVERRIDES.get(str(team_blob.get(key) or "").strip().lower())
        if override:
            return override
    for source in (team_blob, team):
        if not isinstance(source, dict):
            continue
        logos = source.get("logos")
        if isinstance(logos, list):
            for logo in logos:
                if isinstance(logo, dict) and logo.get("href"):
                    return str(logo["href"])
        logo_url = source.get("logo")
        if isinstance(logo_url, str) and logo_url.strip():
            return logo_url.strip()
    return ""


def _saved_team_logo_path(filename: str) -> Optional[str]:
    for folder in (LOGO_DIR, AUTO_LOGO_DIR):
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path
    return None


def _prepare_downloaded_logo(data: bytes) -> Image.Image:
    """Trim transparent padding and cap the longest edge, like logo_getter.py."""

    img = Image.open(io.BytesIO(data)).convert("RGBA")
    bbox = img.getchannel("A").getbbox()
    if bbox:
        img = img.crop(bbox)
    width, height = img.size
    if max(width, height) > AUTO_LOGO_MAX_DIMENSION:
        ratio = AUTO_LOGO_MAX_DIMENSION / float(max(width, height))
        img = img.resize((max(1, round(width * ratio)), max(1, round(height * ratio))), LANCZOS)
    return img


def _save_downloaded_logo(filename: str, img: Image.Image) -> None:
    os.makedirs(AUTO_LOGO_DIR, exist_ok=True)
    path = os.path.join(AUTO_LOGO_DIR, filename)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    try:
        img.save(tmp_path, format="PNG", optimize=True)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _download_team_logo(team: dict[str, Any], filename: str, *, save: bool) -> Optional[Image.Image]:
    """Fetch ESPN's logo for *team*; with *save*, keep a copy in AUTO_LOGO_DIR."""

    url = _team_logo_url(team)
    if not url:
        return None
    try:
        resp = _SESSION.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        img = _prepare_downloaded_logo(resp.content)
    except Exception as exc:
        logging.warning("Unable to download NCAA FBS team logo %s: %s", url, exc)
        return None
    if save:
        try:
            _save_downloaded_logo(filename, img)
            logging.info("Saved ESPN logo for %s to %s", filename, AUTO_LOGO_DIR)
        except Exception as exc:
            logging.warning("Unable to save NCAA FBS team logo %s: %s", filename, exc)
    else:
        logging.info("Using ESPN logo for %s; save it as %s to keep it local", filename, filename)
    return img


def _open_team_logo(team: dict[str, Any]) -> Optional[Image.Image]:
    """Open the saved logo, or download ESPN's logo from the game payload."""

    filename = _team_logo_filename(team)
    path = _saved_team_logo_path(filename)
    if path:
        try:
            return Image.open(path).convert("RGBA")
        except Exception as exc:
            logging.warning("Unable to load NCAA FBS team logo %s: %s", path, exc)
    img = _download_team_logo(team, filename, save=False)
    if img is not None:
        return img
    logging.warning("Missing NCAA FBS team logo; expected filename: %s", filename)
    return None


# Filenames whose download failed, and when; retried after LOGO_MISS_RETRY_SECONDS.
_DOWNLOAD_FAILURES: dict[str, float] = {}


def download_missing_team_logos(games: list[dict]) -> list[str]:
    """Save ESPN's logo for every team in *games* that has no saved logo.

    Only the render server calls this, when its scoreboard feed refreshes, so
    each new week's teams are on disk before the board draws them and no
    other display downloads logos. Returns the filenames saved.
    """

    saved: list[str] = []
    seen: set[str] = set()
    for game in games or []:
        teams = game.get("teams") if isinstance(game, dict) else None
        if not isinstance(teams, dict):
            continue
        for team in (teams.get("away"), teams.get("home")):
            if not isinstance(team, dict) or not team:
                continue
            filename = _team_logo_filename(team)
            if filename in seen or _saved_team_logo_path(filename):
                continue
            seen.add(filename)
            failed_at = _DOWNLOAD_FAILURES.get(filename)
            if failed_at is not None and time.monotonic() - failed_at < LOGO_MISS_RETRY_SECONDS:
                continue
            if _download_team_logo(team, filename, save=True) is None or not _saved_team_logo_path(filename):
                _DOWNLOAD_FAILURES[filename] = time.monotonic()
                continue
            _DOWNLOAD_FAILURES.pop(filename, None)
            saved.append(filename)
    if saved:
        logging.info("Downloaded %d NCAA FBS team logo(s): %s", len(saved), ", ".join(saved))
    return saved


def _load_team_logo(
    team: dict[str, Any],
    height: int,
    max_width: Optional[int] = None,
    area: Optional[int] = None,
) -> Optional[Image.Image]:
    box_w = max_width if max_width is not None else max(1, height * 2)
    cache_key = (_team_logo_filename(team), box_w, height, area or 0)
    if cache_key in _REMOTE_LOGO_CACHE:
        return _REMOTE_LOGO_CACHE[cache_key]
    missed_at = _LOGO_MISSES.get(cache_key)
    if missed_at is not None and time.monotonic() - missed_at < LOGO_MISS_RETRY_SECONDS:
        return None
    img = _open_team_logo(team)
    if img is None:
        _LOGO_MISSES[cache_key] = time.monotonic()
        return None
    _LOGO_MISSES.pop(cache_key, None)
    fitted = _fit_logo(img, box_w, height, area)
    _REMOTE_LOGO_CACHE[cache_key] = fitted
    return fitted


def _seed_text_for_display(team: dict[str, Any]) -> str:
    seed = _extract_seed(team)
    rank = _extract_rank(team)
    if seed and rank is not None and seed == str(rank):
        return ""
    return seed


def _draw_seed(draw: ImageDraw.ImageDraw, seed: str, x_logo: int, y_logo: int, logo_h: int):
    if not seed:
        return
    text = f"({seed})"
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=SEED_FONT)
        tw, th = r - l, b - t
    except Exception:
        tw, th = draw.textsize(text, font=SEED_FONT)
        l = t = 0
    x = x_logo - SEED_GAP - tw - l
    y = y_logo + logo_h - th - t
    draw.text((x, y), text, font=SEED_FONT, fill=(210, 210, 210))


def _draw_rank(
    draw: ImageDraw.ImageDraw,
    rank: Optional[int],
    x_logo: int,
    y_logo: int,
    logo_w: int,
    logo_h: int,
):
    """Draw a poll ranking as a small superscript just before the team's logo."""

    if rank is None:
        return
    text = str(rank)
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=RANK_FONT)
        tw = r - l
    except Exception:
        tw, _ = draw.textsize(text, font=RANK_FONT)
        l = t = 0
    x = x_logo - RANK_GAP - tw - l
    y = y_logo - t
    draw.text((x, y), text, font=RANK_FONT, fill=(210, 210, 210))


def _rank_for_display(team: dict[str, Any], *, mode: Optional[str] = None) -> Optional[int]:
    return _extract_rank(team)

def _get_league_logo(mode: Optional[str] = None) -> Optional[Image.Image]:
    _, logo_key = _mode_title_and_logo()
    h = _league_logo_height()
    cache_key = (logo_key, h)
    if cache_key in _LEAGUE_LOGO_CACHE:
        return _LEAGUE_LOGO_CACHE[cache_key]
    path = os.path.join(LOGO_DIR, f"{logo_key}.png")
    if not os.path.exists(path):
        logging.warning(
            "Missing NCAA FBS league logo; expected filename: %s.png", logo_key
        )
        _LEAGUE_LOGO_CACHE[cache_key] = None
        return None
    try:
        img = Image.open(path).convert("RGBA")
        ratio = h / max(1, img.height)
        resized = img.resize((max(1, int(round(img.width * ratio))), h), LANCZOS)
        _LEAGUE_LOGO_CACHE[cache_key] = resized
        return resized
    except Exception:
        _LEAGUE_LOGO_CACHE[cache_key] = None
        return None


def _render_scoreboard(games: list[dict], *, mode: Optional[str] = None) -> Image.Image:
    selected_mode = mode or _scoreboard_mode()
    title, _ = _mode_title_and_logo()
    logo_height = _team_logo_height()

    block_h = SCORE_ROW_H + STATUS_ROW_H
    canvas_h = max(HEIGHT, len(games) * block_h + max(0, len(games) - 1) * BLOCK_SPACING)
    canvas = Image.new("RGB", (WIDTH, canvas_h), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    y = 0
    for idx, game in enumerate(games):
        teams = game.get("teams") or {}
        away = teams.get("away") or {}
        home = teams.get("home") or {}

        show_scores = _should_display_scores(game)
        away_score = _score_text(away, show=show_scores)
        home_score = _score_text(home, show=show_scores)
        in_progress = _is_in_progress(game)
        final = _is_final(game)

        for col_idx, text in ((0, away_score), (2, "@"), (4, home_score)):
            font = SCORE_FONT if col_idx != 2 else CENTER_FONT
            fill = (255, 255, 255)
            if col_idx == 0:
                fill = _score_fill("away", in_progress=in_progress, final=final, away=away, home=home)
            elif col_idx == 4:
                fill = _score_fill("home", in_progress=in_progress, final=final, away=away, home=home)
            _center_text(draw, text, font, COL_X[col_idx], COL_WIDTHS[col_idx], y, SCORE_ROW_H, fill=fill)

        for col_idx, team in ((1, away), (3, home)):
            box_w, box_h, box_area = _team_logo_box(logo_height, COL_WIDTHS[col_idx])
            logo = _load_team_logo(team, box_h, box_w, box_area)
            if not logo:
                abbreviation = _team_abbreviation(team)
                try:
                    left, top, right, bottom = draw.textbbox(
                        (0, 0), abbreviation, font=TEAM_ABBREVIATION_FONT
                    )
                    logo_width, logo_height_actual = right - left, bottom - top
                except Exception:
                    logo_width, logo_height_actual = draw.textsize(
                        abbreviation, font=TEAM_ABBREVIATION_FONT
                    )
                    left = top = 0
                x0 = COL_X[col_idx] + (COL_WIDTHS[col_idx] - logo_width) // 2
                y0 = y + (SCORE_ROW_H - logo_height_actual) // 2
                draw.text(
                    (x0 - left, y0 - top),
                    abbreviation,
                    font=TEAM_ABBREVIATION_FONT,
                    fill=(255, 255, 255),
                )
            else:
                logo_width, logo_height_actual = logo.width, logo.height
                x0 = COL_X[col_idx] + (COL_WIDTHS[col_idx] - logo_width) // 2
                y0 = y + (SCORE_ROW_H - logo_height_actual) // 2
                canvas.paste(logo, (x0, y0), logo)
            _draw_rank(
                draw,
                _rank_for_display(team, mode=selected_mode),
                x0,
                y0,
                logo_width,
                logo_height_actual,
            )
        status_fill = IN_PROGRESS_STATUS_COLOR if in_progress else (255, 255, 255)
        _center_text(draw, _status_text(game), STATUS_FONT, COL_X[0], sum(COL_WIDTHS), y + SCORE_ROW_H, STATUS_ROW_H, fill=status_fill)

        y += block_h
        if idx < len(games) - 1:
            sep_y = y + BLOCK_SPACING // 2
            draw.line((10, sep_y, WIDTH - 10, sep_y), fill=(45, 45, 45))
            y += BLOCK_SPACING

    dummy = Image.new("RGB", (WIDTH, 8), BACKGROUND_COLOR)
    dd = ImageDraw.Draw(dummy)
    try:
        l, t, r, b = dd.textbbox((0, 0), title, font=FONT_TITLE_SPORTS)
        title_h = b - t
    except Exception:
        _, title_h = dd.textsize(title, font=FONT_TITLE_SPORTS)

    league_logo = _get_league_logo(selected_mode)
    league_h = league_logo.height if league_logo else 0
    gap = LEAGUE_LOGO_GAP if league_logo else 0

    content_top = league_h + gap + title_h + TITLE_GAP
    total_h = max(HEIGHT, content_top + canvas.height + SCOREBOARD_STANDINGS_BOTTOM_PADDING)
    out = Image.new("RGB", (WIDTH, total_h), BACKGROUND_COLOR)
    draw_out = ImageDraw.Draw(out)

    if league_logo:
        out.paste(league_logo, ((WIDTH - league_logo.width) // 2, 0), league_logo)
    title_top = league_h + gap
    try:
        l, t, r, b = draw_out.textbbox((0, 0), title, font=FONT_TITLE_SPORTS)
        draw_out.text(((WIDTH - (r - l)) // 2 - l, title_top - t), title, font=FONT_TITLE_SPORTS, fill=(255, 255, 255))
    except Exception:
        tw, _ = draw_out.textsize(title, font=FONT_TITLE_SPORTS)
        draw_out.text(((WIDTH - tw) // 2, title_top), title, font=FONT_TITLE_SPORTS, fill=(255, 255, 255))

    out.paste(canvas, (0, content_top))
    return out


def _scroll_display(display, img: Image.Image):
    scroll_vertical_content(
        display=display,
        content_height=img.height,
        viewport_width=WIDTH,
        viewport_height=HEIGHT,
        render_at_offset=lambda offset: display.image(img.crop((0, offset, WIDTH, offset + HEIGHT))),
        base_step=SCOREBOARD_SCROLL_STEP,
        pause_start=SCOREBOARD_SCROLL_PAUSE_TOP,
        pause_end=SCOREBOARD_SCROLL_PAUSE_BOTTOM,
        min_frame_time=SCOREBOARD_SCROLL_DELAY,
    )


def _render_ncaam_scoreboard_v1(display, games: list[dict] | None, transition: bool = False) -> ScreenImage:
    games = games or []
    if not games:
        clear_display(display)
        title, _ = _mode_title_and_logo()
        img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(img)
        league_logo = _get_league_logo()
        title_top = 0
        title_height = 0
        if league_logo:
            img.paste(league_logo, ((WIDTH - league_logo.width) // 2, 0), league_logo)
            title_top = league_logo.height + LEAGUE_LOGO_GAP
        try:
            l, t, r, b = draw.textbbox((0, 0), title, font=FONT_TITLE_SPORTS)
            title_height = b - t
            draw.text(((WIDTH - (r - l)) // 2 - l, title_top - t), title, font=FONT_TITLE_SPORTS, fill=(255, 255, 255))
        except Exception:
            tw, th = draw.textsize(title, font=FONT_TITLE_SPORTS)
            title_height = th
            draw.text(((WIDTH - tw) // 2, title_top), title, font=FONT_TITLE_SPORTS, fill=(255, 255, 255))

        no_games_top = max(title_top + title_height + LEAGUE_LOGO_GAP, HEIGHT // 2 - STATUS_ROW_H // 2)
        _center_text(draw, "No games this week", STATUS_FONT, 0, WIDTH, no_games_top, STATUS_ROW_H)
        if transition:
            return ScreenImage(img, displayed=False)
        display.image(img)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
        return ScreenImage(img, displayed=True)

    full = _render_scoreboard(games)
    if transition:
        _scroll_display(display, full)
        return ScreenImage(full, displayed=True)

    if full.height <= HEIGHT:
        display.image(full)
        time.sleep(SCOREBOARD_SCROLL_PAUSE_BOTTOM)
    else:
        _scroll_display(display, full)
    return ScreenImage(full, displayed=True)


def render_ncaa_fbs_scoreboard(display, games: list[dict] | None, transition: bool = False) -> ScreenImage:
    return _render_ncaam_scoreboard_v1(display, games, transition=transition)


def _scoreboard_date(now: Optional[datetime.datetime] = None) -> datetime.date:
    if now is None:
        now = datetime.datetime.now(CENTRAL_TIME)
    cutoff = now.replace(hour=10, minute=10, second=0, microsecond=0)
    return (now.date() - datetime.timedelta(days=1)) if now < cutoff else now.date()
