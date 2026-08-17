#!/usr/bin/env python3
"""Render simple status content on the Waveshare OLED/LCD HAT (A) OLED displays.

Default behavior:
- Left OLED (0x3c): current date in M/D/YY format
- Right OLED (0x3d): local time in 12-hour format, no leading zero (small AM/PM)
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import logging
import os
import random
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import Event
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

try:
    from smbus import SMBus
except ImportError:  # pragma: no cover - environment specific fallback
    from smbus2 import SMBus


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw, 0)
    except ValueError:
        return default


I2C_BUS = _env_int("WAVESHARE_OLED_I2C_BUS", 1)
TEMP_ADDR = _env_int("WAVESHARE_OLED_TEMP_ADDR", 0x3C)
TIME_ADDR = _env_int("WAVESHARE_OLED_TIME_ADDR", 0x3D)
OLED_WIDTH = _env_int("WAVESHARE_OLED_WIDTH", 128)
OLED_HEIGHT = _env_int("WAVESHARE_OLED_HEIGHT", 64)
TEMP_SOURCE = os.getenv("WAVESHARE_OLED_TEMP_SOURCE", "weather1").strip().lower()
TEMP_COMMAND = os.getenv("WAVESHARE_OLED_TEMP_COMMAND", "")
TEMP_UNIT = os.getenv("WAVESHARE_OLED_TEMP_UNIT", "C").strip().upper()
REFRESH_SECONDS = max(1, _env_int("WAVESHARE_OLED_REFRESH_SECONDS", 5))
SWAP_INTERVAL_MIN_SECONDS = max(1, _env_int("WAVESHARE_OLED_SWAP_INTERVAL_MIN_SECONDS", 60))
SWAP_INTERVAL_MAX_SECONDS = max(
    SWAP_INTERVAL_MIN_SECONDS,
    _env_int("WAVESHARE_OLED_SWAP_INTERVAL_MAX_SECONDS", 240),
)
FADE_STEPS = max(1, _env_int("WAVESHARE_OLED_FADE_STEPS", 8))
FADE_STEP_MS = max(5, _env_int("WAVESHARE_OLED_FADE_STEP_MS", 35))
WAIT_FOR_WEATHER2 = os.getenv("WAVESHARE_OLED_WAIT_FOR_WEATHER2", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
MIN_VALUE_FONT_SIZE = max(6, _env_int("WAVESHARE_OLED_MIN_VALUE_FONT_SIZE", 8))
MAX_VALUE_FONT_SIZE = max(
    MIN_VALUE_FONT_SIZE,
    _env_int("WAVESHARE_OLED_MAX_VALUE_FONT_SIZE", 26),
)
MIN_TIME_FONT_SIZE = max(6, _env_int("WAVESHARE_OLED_MIN_TIME_FONT_SIZE", MIN_VALUE_FONT_SIZE))
MAX_TIME_FONT_SIZE = max(
    MIN_TIME_FONT_SIZE,
    _env_int("WAVESHARE_OLED_MAX_TIME_FONT_SIZE", MAX_VALUE_FONT_SIZE),
)


LOGGER = logging.getLogger("waveshare_oled_status")
_STOP_EVENT = Event()
_LAST_WEATHER_TEMP_F: float | None = None
_WEATHER2_RENDERED = False
_LAST_GITHUB_UPDATE_CHECK_AT = 0.0
_LAST_GITHUB_UPDATE_AVAILABLE = False
_CUBS_FINAL_GAME_PK: str | None = None
_CUBS_FINAL_HOLD_UNTIL_EPOCH = 0.0
_CUBS_FINAL_STATE_PATH = Path(os.getenv("WAVESHARE_OLED_CUBS_FINAL_STATE_PATH", "/var/tmp/desk_display_cubs_final_state.json"))
_HAWKS_FINAL_GAME_PK: str | None = None
_HAWKS_FINAL_HOLD_UNTIL_EPOCH = 0.0
_HAWKS_FINAL_STATE_PATH = Path(os.getenv("WAVESHARE_OLED_HAWKS_FINAL_STATE_PATH", "/var/tmp/desk_display_hawks_final_state.json"))


def _load_final_state(state_path: Path) -> tuple[str | None, float]:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None, 0.0

    game_pk = str(payload.get("game_pk") or "").strip() or None
    try:
        hold_until = float(payload.get("hold_until_epoch") or 0.0)
    except Exception:
        hold_until = 0.0
    return game_pk, hold_until


def _persist_final_state(state_path: Path, game_pk: str | None, hold_until_epoch: float) -> None:
    payload = {"game_pk": (game_pk or ""), "hold_until_epoch": float(hold_until_epoch)}
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        return


def _load_cubs_final_state() -> tuple[str | None, float]:
    return _load_final_state(_CUBS_FINAL_STATE_PATH)


def _persist_cubs_final_state(game_pk: str | None, hold_until_epoch: float) -> None:
    _persist_final_state(_CUBS_FINAL_STATE_PATH, game_pk, hold_until_epoch)


def _load_hawks_final_state() -> tuple[str | None, float]:
    return _load_final_state(_HAWKS_FINAL_STATE_PATH)


def _persist_hawks_final_state(game_pk: str | None, hold_until_epoch: float) -> None:
    _persist_final_state(_HAWKS_FINAL_STATE_PATH, game_pk, hold_until_epoch)


class SSD1306Display:
    def __init__(self, bus: SMBus, address: int, width: int = 128, height: int = 64) -> None:
        self.bus = bus
        self.address = address
        self.width = width
        self.height = height
        self.pages = height // 8

    def _cmd(self, value: int) -> None:
        self.bus.write_byte_data(self.address, 0x00, value)

    def _data(self, data: bytes) -> None:
        chunk_size = 16
        for idx in range(0, len(data), chunk_size):
            chunk = data[idx : idx + chunk_size]
            self.bus.write_i2c_block_data(self.address, 0x40, list(chunk))

    def initialize(self) -> None:
        init_sequence = [
            0xAE,
            0x20,
            0x00,
            0xB0,
            0xC8,
            0x00,
            0x10,
            0x40,
            0x81,
            0x7F,
            0xA1,
            0xA6,
            0xA8,
            self.height - 1,
            0xA4,
            0xD3,
            0x00,
            0xD5,
            0x80,
            0xD9,
            0xF1,
            0xDA,
            0x12 if self.height == 64 else 0x02,
            0xDB,
            0x40,
            0x8D,
            0x14,
            0xAF,
        ]
        for cmd in init_sequence:
            self._cmd(cmd)

    def set_contrast(self, value: int) -> None:
        contrast = max(0, min(255, int(value)))
        self._cmd(0x81)
        self._cmd(contrast)

    def clear(self) -> None:
        self.display_image(Image.new("1", (self.width, self.height), 0))

    def display_image(self, image: Image.Image) -> None:
        if image.mode != "1":
            image = image.convert("1")
        image = image.resize((self.width, self.height))
        pixels = image.load()

        for page in range(self.pages):
            self._cmd(0xB0 + page)
            self._cmd(0x00)
            self._cmd(0x10)
            buf = bytearray(self.width)
            for x in range(self.width):
                value = 0
                for bit in range(8):
                    y = page * 8 + bit
                    if y < self.height and pixels[x, y] != 0:
                        value |= 1 << bit
                buf[x] = value
            self._data(bytes(buf))


def _parse_temperature_value(text: str) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _read_cpu_temp_c() -> float | None:
    candidates = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/hwmon/hwmon0/temp1_input",
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            raw = open(path, encoding="utf-8").read().strip()
            value = float(raw)
            if value > 1000:
                value = value / 1000.0
            return value
        except Exception:
            continue

    try:
        output = subprocess.check_output(["vcgencmd", "measure_temp"], text=True, timeout=2)
        return _parse_temperature_value(output)
    except Exception:
        return None


def _read_weather1_temp_f() -> float | None:
    global _LAST_WEATHER_TEMP_F

    def _resolve_fetch_weather():
        try:
            module = importlib.import_module("data_fetch")
            fetch_weather = getattr(module, "fetch_weather", None)
            if callable(fetch_weather):
                return fetch_weather
            legacy = getattr(module, "get_weather_data", None)
            if callable(legacy):
                return legacy
        except Exception:
            pass

        repo_root = Path(__file__).resolve().parents[1]
        module_path = repo_root / "data_fetch.py"
        if not module_path.exists():
            return None

        spec = importlib.util.spec_from_file_location("data_fetch", module_path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("data_fetch", module)
        try:
            spec.loader.exec_module(module)
        except Exception:
            return None

        fetch_weather = getattr(module, "fetch_weather", None)
        if callable(fetch_weather):
            return fetch_weather
        legacy = getattr(module, "get_weather_data", None)
        return legacy if callable(legacy) else None

    fetch_weather = _resolve_fetch_weather()
    if fetch_weather is None:
        return _LAST_WEATHER_TEMP_F

    def _fetch_weather_payload(force_refresh: bool):
        try:
            return fetch_weather(force_refresh=force_refresh)
        except TypeError:
            # Backward compatibility for older data_fetch modules.
            if force_refresh:
                return None
            return fetch_weather()

    try:
        # Prefer cached weather to avoid API rate limits and transient misses on
        # the OLED loop cadence.
        weather = _fetch_weather_payload(force_refresh=False)
        if weather is None:
            weather = _fetch_weather_payload(force_refresh=True)
    except Exception:
        return _LAST_WEATHER_TEMP_F

    if not isinstance(weather, dict):
        return _LAST_WEATHER_TEMP_F

    current = weather.get("current")
    if not isinstance(current, dict):
        return _LAST_WEATHER_TEMP_F

    temp_f = (
        current.get("temp")
        or current.get("temp_f")
        or current.get("temperature")
    )
    try:
        _LAST_WEATHER_TEMP_F = float(temp_f)
        return _LAST_WEATHER_TEMP_F
    except (TypeError, ValueError):
        return _LAST_WEATHER_TEMP_F


def _display_status_path() -> Path:
    override = os.getenv("WAVESHARE_OLED_DISPLAY_STATUS_PATH")
    if override:
        return Path(override).expanduser()

    repo_root = Path(__file__).resolve().parents[1]
    screenshot_dir = os.getenv("SCREENSHOT_DIR", str(repo_root / "screenshots"))
    return Path(screenshot_dir).expanduser() / "current" / "display_status.json"


def _screenshot_current_dir() -> Path:
    override = os.getenv("WAVESHARE_OLED_SCREENSHOT_DIR")
    if override:
        return Path(override).expanduser()

    repo_root = Path(__file__).resolve().parents[1]
    screenshot_dir = os.getenv("SCREENSHOT_DIR", str(repo_root / "screenshots"))
    return Path(screenshot_dir).expanduser() / "current"


# Approximates the pixel color of the physical Waveshare OLED/LCD HAT (A)
# blue OLED panels so the web screenshot gallery matches what the hardware
# actually shows instead of rendering lit pixels as plain white.
OLED_SCREENSHOT_ON_COLOR = (63, 166, 255)


def _tint_oled_screenshot(image: Image.Image) -> Image.Image:
    """Recolor a 1-bit OLED frame to the panel's blue for the web preview."""
    mask = image if image.mode == "1" else image.convert("1")
    tinted = Image.new("RGB", mask.size, (0, 0, 0))
    tinted.paste(OLED_SCREENSHOT_ON_COLOR, (0, 0), mask)
    return tinted


def _save_oled_screenshot(name: str, image: Image.Image) -> None:
    """Persist the frame actually pushed to an OLED so the web screenshot
    gallery can show it (mirrors how the main display's screens are saved).
    """
    try:
        current_dir = _screenshot_current_dir()
        current_dir.mkdir(parents=True, exist_ok=True)
        target = current_dir / f"{name}.png"
        tmp_target = current_dir / f"{name}.png.tmp"
        _tint_oled_screenshot(image).save(tmp_target, format="PNG")
        os.replace(tmp_target, target)
    except Exception as exc:
        LOGGER.debug("Failed to save %s OLED screenshot: %s", name, exc)


@lru_cache(maxsize=1)
def _resolve_mlb_abbreviation() -> Optional[Callable[[str], str]]:
    try:
        from screens.mlb_schedule import (
            get_mlb_abbreviation,  # pylint: disable=import-outside-toplevel
        )

        return get_mlb_abbreviation
    except Exception:
        return None


def _read_display_status_payload() -> dict:
    status_path = _display_status_path()
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _is_cubs_team(team: dict) -> bool:
    if not isinstance(team, dict):
        return False
    team_id = team.get("id")
    if str(team_id) == "112":
        return True
    for key in ("teamName", "name", "clubName", "abbreviation", "triCode"):
        value = str(team.get(key) or "").strip().lower()
        if "cubs" in value or value == "chc":
            return True
    return False


def _team_label(team: dict) -> str:
    if _is_cubs_team(team):
        return "CUBS"
    helper = _resolve_mlb_abbreviation()
    if callable(helper):
        display_name = (
            team.get("name")
            or team.get("teamName")
            or team.get("clubName")
            or team.get("abbreviation")
            or ""
        )
        try:
            value = str(helper(display_name) or "").strip().upper()
            if value:
                return value
        except Exception:
            pass
    for key in ("triCode", "abbreviation", "teamCode", "abbrev"):
        value = str(team.get(key) or "").strip().upper()
        if value:
            return value
    return "TEAM"


def _mlb_live_state(game: dict) -> tuple[bool, bool]:
    status = (game or {}).get("status") or {}
    abstract = str(status.get("abstractGameState") or "").lower()
    detailed = str(status.get("detailedState") or "").lower()
    code = str(status.get("statusCode") or status.get("codedGameState") or "").upper()
    text = f"{abstract} {detailed}".strip()
    is_final = code in {"F", "O"} or "final" in text or "completed" in text
    is_live = (
        code in {"I", "2", "3"}
        or "live" in text
        or "in progress" in text
        or "top " in text
        or "bottom " in text
        or "middle " in text
    )
    return is_live, is_final


def _format_inning_text(game: dict) -> str:
    linescore = (game or {}).get("linescore") or {}
    inning_state = str(linescore.get("inningState") or "").strip()
    inning_ord = str(linescore.get("currentInningOrdinal") or "").strip()
    if inning_state and inning_ord:
        return f"{inning_state} {inning_ord}"
    if inning_ord:
        return inning_ord
    status = (game or {}).get("status") or {}
    detailed = str(status.get("detailedState") or "").strip()
    return detailed or "Live"


def _format_outs_text(game: dict) -> str:
    linescore = (game or {}).get("linescore") or {}
    raw_outs = linescore.get("outs")
    try:
        outs = int(raw_outs)
    except (TypeError, ValueError):
        return ""
    outs_label = "Out" if outs == 1 else "Outs"
    return f"{outs} {outs_label}"


def _render_score_panel(width: int, height: int, *, team: str, score: str, footer: str) -> Image.Image:
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    center_y = max(0, (height // 2) - 8)
    team_font_size = max(12, min(34, _best_value_font_size(int(width * 0.64), height, team, 2)))
    score_font_size = max(20, min(58, _best_value_font_size(int(width * 0.33), height, score, 0) + 8))
    footer_font_size = max(10, min(20, _best_value_font_size(width - 8, 24, footer or " ", 0)))
    team_font = _load_value_font(team_font_size)
    score_font = _load_value_font(score_font_size)
    footer_font = _load_value_font(footer_font_size)

    team_bbox = draw.textbbox((0, 0), team, font=team_font)
    score_bbox = draw.textbbox((0, 0), score, font=score_font)
    team_w = team_bbox[2] - team_bbox[0]
    team_h = team_bbox[3] - team_bbox[1]
    score_w = score_bbox[2] - score_bbox[0]
    score_h = score_bbox[3] - score_bbox[1]

    gap = 4
    team_x = 3
    team_y = center_y - (team_h // 2)
    score_x = width - score_w - 4
    score_y = center_y - (score_h // 2)
    if team_x + team_w + gap > score_x:
        available = max(10, score_x - team_x - gap)
        reduced_size = max(10, team_font_size - 2)
        while reduced_size >= 10:
            test_font = _load_value_font(reduced_size)
            test_bbox = draw.textbbox((0, 0), team, font=test_font)
            if (test_bbox[2] - test_bbox[0]) <= available:
                team_font = test_font
                team_w = test_bbox[2] - test_bbox[0]
                team_h = test_bbox[3] - test_bbox[1]
                team_y = center_y - (team_h // 2)
                break
            reduced_size -= 1

    draw.text((team_x, team_y), team, font=team_font, fill=255)
    draw.text((score_x, score_y), score, font=score_font, fill=255)

    if footer:
        footer_bbox = draw.textbbox((0, 0), footer, font=footer_font)
        footer_w = footer_bbox[2] - footer_bbox[0]
        footer_h = footer_bbox[3] - footer_bbox[1]
        footer_x = max(2, (width - footer_w) // 2)
        footer_y = max(0, height - footer_h - 2)
        draw.text((footer_x, footer_y), footer, font=footer_font, fill=255)

    return image


def _cubs_oled_frames() -> tuple[Image.Image, Image.Image] | None:
    global _CUBS_FINAL_GAME_PK, _CUBS_FINAL_HOLD_UNTIL_EPOCH

    payload = _read_display_status_payload()
    cubs = payload.get("cubs") if isinstance(payload, dict) else None
    if not isinstance(cubs, dict):
        return None

    live_game = cubs.get("live_game") if isinstance(cubs.get("live_game"), dict) else None
    last_game = cubs.get("last_game") if isinstance(cubs.get("last_game"), dict) else None

    selected_game = None
    is_final = False
    now_epoch = time.time()
    if _CUBS_FINAL_GAME_PK is None and _CUBS_FINAL_HOLD_UNTIL_EPOCH <= 0:
        _CUBS_FINAL_GAME_PK, _CUBS_FINAL_HOLD_UNTIL_EPOCH = _load_cubs_final_state()
    if isinstance(live_game, dict):
        live_live, live_final = _mlb_live_state(live_game)
        if live_live:
            selected_game = live_game
            is_final = False
        elif live_final:
            selected_game = live_game
            is_final = True

    if selected_game is None and isinstance(last_game, dict):
        _live, last_final = _mlb_live_state(last_game)
        if last_final:
            selected_game = last_game
            is_final = True

    if selected_game is None:
        return None

    game_pk = str(selected_game.get("gamePk") or selected_game.get("game_id") or "")
    if is_final:
        if game_pk and game_pk != _CUBS_FINAL_GAME_PK:
            _CUBS_FINAL_GAME_PK = game_pk
            _CUBS_FINAL_HOLD_UNTIL_EPOCH = now_epoch + (90 * 60)
            _persist_cubs_final_state(_CUBS_FINAL_GAME_PK, _CUBS_FINAL_HOLD_UNTIL_EPOCH)
        if now_epoch > _CUBS_FINAL_HOLD_UNTIL_EPOCH:
            return None
    else:
        _CUBS_FINAL_GAME_PK = game_pk or _CUBS_FINAL_GAME_PK
        _CUBS_FINAL_HOLD_UNTIL_EPOCH = 0.0
        _persist_cubs_final_state(_CUBS_FINAL_GAME_PK, _CUBS_FINAL_HOLD_UNTIL_EPOCH)

    teams = selected_game.get("teams") or {}
    away = (teams.get("away") or {}).get("team") or {}
    home = (teams.get("home") or {}).get("team") or {}
    away_score = (teams.get("away") or {}).get("score")
    home_score = (teams.get("home") or {}).get("score")
    away_label = _team_label(away)
    home_label = _team_label(home)
    away_score_text = str(away_score) if isinstance(away_score, int) else "-"
    home_score_text = str(home_score) if isinstance(home_score, int) else "-"
    # Consistent, fixed layout regardless of which side is batting: the away
    # panel always carries the inning, the home panel always carries the
    # outs (or "Final" once the game has ended).
    away_footer = "" if is_final else _format_inning_text(selected_game)
    home_footer = "Final" if is_final else _format_outs_text(selected_game)

    return (
        _render_score_panel(OLED_WIDTH, OLED_HEIGHT, team=away_label, score=away_score_text, footer=away_footer),
        _render_score_panel(OLED_WIDTH, OLED_HEIGHT, team=home_label, score=home_score_text, footer=home_footer),
    )


NHL_HAWKS_TEAM_ID = 16


def _is_hawks_team(team: dict) -> bool:
    if not isinstance(team, dict):
        return False
    team_id = team.get("id") or team.get("teamId")
    if str(team_id) == str(NHL_HAWKS_TEAM_ID):
        return True
    for key in ("commonName", "name", "teamName", "clubName", "abbrev", "abbreviation", "triCode"):
        value = team.get(key)
        if isinstance(value, dict):
            value = value.get("default") or value.get("en") or ""
        value = str(value or "").strip().lower()
        if "blackhawks" in value or value == "chi":
            return True
    return False


def _team_label_nhl(team: dict) -> str:
    if _is_hawks_team(team):
        return "HAWKS"
    for key in ("abbrev", "triCode", "abbreviation", "teamCode"):
        value = team.get(key) if isinstance(team, dict) else None
        if isinstance(value, dict):
            value = value.get("default") or value.get("en") or ""
        value = str(value or "").strip().upper()
        if value:
            return value
    return "TEAM"


def _nhl_live_state(game: dict) -> tuple[bool, bool]:
    state = str((game or {}).get("gameState") or "").strip().upper()
    is_live = state in {"LIVE", "CRIT"}
    is_final = state in {"OFF", "FINAL"}
    return is_live, is_final


def _format_period_text(game: dict, feed: dict | None) -> str:
    period = ""
    if isinstance(feed, dict):
        period = _normalize_nhl_period(feed.get("perOrdinal"))
    if not period:
        period_desc = (game or {}).get("periodDescriptor") or {}
        period = _normalize_nhl_period(
            period_desc.get("ordinalNum") or period_desc.get("number")
        )
    return f"{period} Period" if period else "Live"


def _normalize_nhl_period(period_val) -> str:
    if period_val is None:
        return ""
    text = str(period_val).strip()
    if not text:
        return ""
    if text.isdigit():
        num = int(text)
        suffix = "th" if 10 <= num % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(num % 10, "th")
        return f"{num}{suffix}"
    return text


def _format_clock_text(feed: dict | None) -> str:
    if not isinstance(feed, dict):
        return ""
    clock_state = str(feed.get("clockState") or "").strip()
    if clock_state.upper() == "INTERMISSION":
        return "Intermission"
    clock = str(feed.get("clock") or "").strip()
    return clock


def _hawks_oled_frames() -> tuple[Image.Image, Image.Image] | None:
    global _HAWKS_FINAL_GAME_PK, _HAWKS_FINAL_HOLD_UNTIL_EPOCH

    payload = _read_display_status_payload()
    hawks = payload.get("hawks") if isinstance(payload, dict) else None
    if not isinstance(hawks, dict):
        return None

    live_game = hawks.get("live_game") if isinstance(hawks.get("live_game"), dict) else None
    last_game = hawks.get("last_game") if isinstance(hawks.get("last_game"), dict) else None
    live_feed = hawks.get("live_feed") if isinstance(hawks.get("live_feed"), dict) else None

    selected_game = None
    is_final = False
    now_epoch = time.time()
    if _HAWKS_FINAL_GAME_PK is None and _HAWKS_FINAL_HOLD_UNTIL_EPOCH <= 0:
        _HAWKS_FINAL_GAME_PK, _HAWKS_FINAL_HOLD_UNTIL_EPOCH = _load_hawks_final_state()
    if isinstance(live_game, dict):
        live_live, live_final = _nhl_live_state(live_game)
        if live_live:
            selected_game = live_game
            is_final = False
        elif live_final:
            selected_game = live_game
            is_final = True

    if selected_game is None and isinstance(last_game, dict):
        _live, last_final = _nhl_live_state(last_game)
        if last_final:
            selected_game = last_game
            is_final = True

    if selected_game is None:
        return None

    game_pk = str(selected_game.get("id") or selected_game.get("gamePk") or "")
    if is_final:
        if game_pk and game_pk != _HAWKS_FINAL_GAME_PK:
            _HAWKS_FINAL_GAME_PK = game_pk
            _HAWKS_FINAL_HOLD_UNTIL_EPOCH = now_epoch + (90 * 60)
            _persist_hawks_final_state(_HAWKS_FINAL_GAME_PK, _HAWKS_FINAL_HOLD_UNTIL_EPOCH)
        if now_epoch > _HAWKS_FINAL_HOLD_UNTIL_EPOCH:
            return None
    else:
        _HAWKS_FINAL_GAME_PK = game_pk or _HAWKS_FINAL_GAME_PK
        _HAWKS_FINAL_HOLD_UNTIL_EPOCH = 0.0
        _persist_hawks_final_state(_HAWKS_FINAL_GAME_PK, _HAWKS_FINAL_HOLD_UNTIL_EPOCH)

    away = selected_game.get("awayTeam") or {}
    home = selected_game.get("homeTeam") or {}
    away_score = away.get("score")
    home_score = home.get("score")
    if isinstance(live_feed, dict):
        if isinstance(live_feed.get("awayScore"), int):
            away_score = live_feed["awayScore"]
        if isinstance(live_feed.get("homeScore"), int):
            home_score = live_feed["homeScore"]
    away_label = _team_label_nhl(away)
    home_label = _team_label_nhl(home)
    away_score_text = str(away_score) if isinstance(away_score, int) else "-"
    home_score_text = str(home_score) if isinstance(home_score, int) else "-"
    # Same fixed layout as the Cubs panels: the away panel always carries
    # the period, the home panel always carries the clock (or "Final").
    away_footer = "" if is_final else _format_period_text(selected_game, live_feed)
    home_footer = "Final" if is_final else _format_clock_text(live_feed)

    return (
        _render_score_panel(OLED_WIDTH, OLED_HEIGHT, team=away_label, score=away_score_text, footer=away_footer),
        _render_score_panel(OLED_WIDTH, OLED_HEIGHT, team=home_label, score=home_score_text, footer=home_footer),
    )


def _weather2_screen_has_rendered() -> bool:
    global _WEATHER2_RENDERED
    if _WEATHER2_RENDERED:
        return True
    if TEMP_SOURCE not in {"weather", "weather1"}:
        return True
    if not WAIT_FOR_WEATHER2:
        return True

    status_path = _display_status_path()
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if str(payload.get("screen_id", "")).strip().lower() == "weather2":
        _WEATHER2_RENDERED = True
        return True
    return False


def read_temperature() -> str:
    value_c: float | None = None
    value_f: float | None = None

    if TEMP_SOURCE == "command" and TEMP_COMMAND:
        try:
            output = subprocess.check_output(TEMP_COMMAND, shell=True, text=True, timeout=4)
            value_c = _parse_temperature_value(output)
        except Exception:
            value_c = None
    elif TEMP_SOURCE in {"weather", "weather1"}:
        value_f = _read_weather1_temp_f()
    else:
        value_c = _read_cpu_temp_c()

    if value_f is None and value_c is None:
        return ""

    if value_f is not None:
        return f"{round(value_f)}°F"

    if TEMP_UNIT == "F":
        return f"{(value_c * 9 / 5) + 32:.1f}°F"

    return f"{value_c:.1f}°C"


def current_time_12h() -> str:
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def current_date_mdy() -> str:
    now = datetime.now()
    return f"{now.month}/{now.day}/{now.strftime('%y')}"


def random_swap_interval_seconds() -> int:
    return random.randint(SWAP_INTERVAL_MIN_SECONDS, SWAP_INTERVAL_MAX_SECONDS)


def _github_updates_available(*, force: bool = False) -> bool:
    global _LAST_GITHUB_UPDATE_CHECK_AT, _LAST_GITHUB_UPDATE_AVAILABLE

    now = time.monotonic()
    if not force and now - _LAST_GITHUB_UPDATE_CHECK_AT < 60:
        return _LAST_GITHUB_UPDATE_AVAILABLE

    repo_root = Path(__file__).resolve().parents[1]
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        _LAST_GITHUB_UPDATE_CHECK_AT = now
        _LAST_GITHUB_UPDATE_AVAILABLE = False
        return False

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            text=True,
            timeout=5,
        ).strip()
        if not branch or branch == "HEAD":
            _LAST_GITHUB_UPDATE_AVAILABLE = False
            return False

        subprocess.run(
            ["git", "fetch", "--quiet", "origin", branch],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
        local_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            timeout=5,
        ).strip()
        remote_sha = subprocess.check_output(
            ["git", "rev-parse", f"origin/{branch}"],
            cwd=repo_root,
            text=True,
            timeout=5,
        ).strip()
        _LAST_GITHUB_UPDATE_AVAILABLE = bool(local_sha and remote_sha and local_sha != remote_sha)
    except Exception:
        _LAST_GITHUB_UPDATE_AVAILABLE = False
    finally:
        _LAST_GITHUB_UPDATE_CHECK_AT = now

    return _LAST_GITHUB_UPDATE_AVAILABLE


def _invert_for_update(image: Image.Image) -> Image.Image:
    return image.convert("L").point(lambda pixel: 255 - pixel).convert("1")


@lru_cache(maxsize=96)
def _load_value_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidate_paths = [
        os.getenv("WAVESHARE_OLED_FONT_PATH"),
        "/workspace/desk_display/fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidate_paths:
        if not path:
            continue
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _best_value_font_size(width: int, height: int, text: str, top_margin: int) -> int:
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    max_height = height - top_margin - 2
    best_size = MIN_VALUE_FONT_SIZE
    for size in range(MIN_VALUE_FONT_SIZE, MAX_VALUE_FONT_SIZE + 1):
        font = _load_value_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= width - 4 and text_h <= max_height:
            best_size = size
        else:
            break
    return best_size


def _best_time_font_size(width: int, height: int, time_text: str, top_margin: int) -> int:
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    max_height = height - top_margin - 2
    best_size = MIN_TIME_FONT_SIZE
    time_match = re.match(r"^(.*?)(?:\s+([AP]M))?$", time_text.strip(), re.IGNORECASE)
    base_time = time_match.group(1) if time_match else time_text
    meridiem = (time_match.group(2) or "").upper() if time_match else ""

    for size in range(MIN_TIME_FONT_SIZE, MAX_TIME_FONT_SIZE + 1):
        main_font = _load_value_font(size)
        meridiem_font = _load_value_font(max(8, size // 2))

        main_bbox = draw.textbbox((0, 0), base_time, font=main_font)
        main_w = main_bbox[2] - main_bbox[0]
        main_h = main_bbox[3] - main_bbox[1]

        gap = 3 if meridiem else 0
        meridiem_bbox = (
            draw.textbbox((0, 0), meridiem, font=meridiem_font)
            if meridiem
            else (0, 0, 0, 0)
        )
        meridiem_w = meridiem_bbox[2] - meridiem_bbox[0]
        meridiem_h = meridiem_bbox[3] - meridiem_bbox[1]

        total_w = main_w + gap + meridiem_w
        total_h = max(main_h, meridiem_h)
        if total_w <= width - 4 and total_h <= max_height:
            best_size = size
        else:
            break
    return best_size


def _title_top_margin(width: int, title: str | None) -> int:
    if not title:
        return 2
    image = Image.new("1", (width, OLED_HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_h = title_bbox[3] - title_bbox[1]
    return max(2, title_h + 6)


def render_centered_text(
    width: int,
    height: int,
    text: str,
    *,
    title: str | None = None,
    value_font_size: int | None = None,
) -> Image.Image:
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()

    y_offset = 0
    if title:
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_w = title_bbox[2] - title_bbox[0]
        title_h = title_bbox[3] - title_bbox[1]
        draw.text(((width - title_w) // 2, 2), title, font=title_font, fill=255)
        y_offset = title_h + 6

    top_margin = max(2, y_offset)
    if value_font_size is None:
        value_font_size = _best_value_font_size(width, height, text, top_margin)

    best_font = _load_value_font(value_font_size)
    best_bbox = draw.textbbox((0, 0), text, font=best_font)

    if best_bbox is None:
        best_bbox = draw.textbbox((0, 0), text, font=best_font)

    value_w = best_bbox[2] - best_bbox[0]
    value_h = best_bbox[3] - best_bbox[1]
    max_height = height - top_margin - 2
    value_x = (width - value_w) // 2
    value_y = top_margin + max(0, (max_height - value_h) // 2)
    draw.text((value_x, value_y), text, font=best_font, fill=255)
    return image


def render_centered_time_text(
    width: int,
    height: int,
    time_text: str,
    *,
    title: str | None = None,
    value_font_size: int | None = None,
) -> Image.Image:
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()

    y_offset = 0
    if title:
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_w = title_bbox[2] - title_bbox[0]
        title_h = title_bbox[3] - title_bbox[1]
        draw.text(((width - title_w) // 2, 2), title, font=title_font, fill=255)
        y_offset = title_h + 6

    top_margin = max(2, y_offset)
    time_match = re.match(r"^(.*?)(?:\s+([AP]M))?$", time_text.strip(), re.IGNORECASE)
    base_time = time_match.group(1) if time_match else time_text
    meridiem = (time_match.group(2) or "").upper() if time_match else ""

    if value_font_size is None:
        value_font_size = _best_value_font_size(width, height, base_time, top_margin)

    main_font = _load_value_font(value_font_size)
    meridiem_font = _load_value_font(max(8, value_font_size // 2))

    main_bbox = draw.textbbox((0, 0), base_time, font=main_font)
    main_w = main_bbox[2] - main_bbox[0]
    main_h = main_bbox[3] - main_bbox[1]

    gap = 3 if meridiem else 0
    meridiem_bbox = draw.textbbox((0, 0), meridiem, font=meridiem_font) if meridiem else (0, 0, 0, 0)
    meridiem_w = meridiem_bbox[2] - meridiem_bbox[0]
    meridiem_h = meridiem_bbox[3] - meridiem_bbox[1]

    total_w = main_w + gap + meridiem_w
    total_h = max(main_h, meridiem_h)
    max_height = height - top_margin - 2
    start_x = (width - total_w) // 2
    start_y = top_margin + max(0, (max_height - total_h) // 2)

    draw.text((start_x, start_y), base_time, font=main_font, fill=255)
    if meridiem:
        meridiem_y = start_y + max(0, main_h - meridiem_h)
        draw.text((start_x + main_w + gap, meridiem_y), meridiem, font=meridiem_font, fill=255)

    return image


def fade_transition(display: SSD1306Display, new_image: Image.Image) -> None:
    for step in range(FADE_STEPS, -1, -1):
        display.set_contrast(int(255 * step / FADE_STEPS))
        time.sleep(FADE_STEP_MS / 1000)

    display.display_image(new_image)

    for step in range(FADE_STEPS + 1):
        display.set_contrast(int(255 * step / FADE_STEPS))
        time.sleep(FADE_STEP_MS / 1000)


def _request_stop(signum: int, _frame: object) -> None:
    LOGGER.info("Received signal %s; stopping OLED helper loop.", signum)
    _STOP_EVENT.set()


def _safe_clear_display(display: SSD1306Display, name: str) -> None:
    try:
        display.clear()
    except Exception as exc:
        LOGGER.warning("Failed to clear %s OLED during shutdown: %s", name, exc)


def _safe_render(display: SSD1306Display, image: Image.Image, name: str) -> bool:
    try:
        fade_transition(display, image)
        return True
    except Exception as exc:
        LOGGER.warning("Failed to render frame on %s OLED: %s", name, exc)
        return False


def main() -> int:
    logging.basicConfig(
        level=os.getenv("WAVESHARE_OLED_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    bus = SMBus(I2C_BUS)
    temp_display = SSD1306Display(bus, TEMP_ADDR, OLED_WIDTH, OLED_HEIGHT)
    time_display = SSD1306Display(bus, TIME_ADDR, OLED_WIDTH, OLED_HEIGHT)

    temp_display.initialize()
    time_display.initialize()
    temp_display.clear()
    time_display.clear()

    show_date_on_left = True
    next_swap_at = time.monotonic() + random_swap_interval_seconds()
    try:
        while not _STOP_EVENT.is_set():
            sports_frames = _cubs_oled_frames() or _hawks_oled_frames()
            if sports_frames is not None:
                left_image, right_image = sports_frames
            else:
                time_text = current_time_12h()
                date_text = current_date_mdy()
                time_top_margin = _title_top_margin(OLED_WIDTH, "Time")
                date_top_margin = _title_top_margin(OLED_WIDTH, "Date")
                time_value_font_size = _best_time_font_size(
                    OLED_WIDTH,
                    OLED_HEIGHT,
                    time_text,
                    time_top_margin,
                )
                date_value_font_size = _best_value_font_size(
                    OLED_WIDTH,
                    OLED_HEIGHT,
                    date_text,
                    date_top_margin,
                )

                date_image = render_centered_text(
                    OLED_WIDTH,
                    OLED_HEIGHT,
                    date_text,
                    title="Date",
                    value_font_size=date_value_font_size,
                )
                time_image = render_centered_time_text(
                    OLED_WIDTH,
                    OLED_HEIGHT,
                    time_text,
                    title="Time",
                    value_font_size=time_value_font_size,
                )

                left_image, right_image = (
                    (date_image, time_image) if show_date_on_left else (time_image, date_image)
                )
            if _github_updates_available():
                left_image = _invert_for_update(left_image)
                right_image = _invert_for_update(right_image)
            left_ok = _safe_render(temp_display, left_image, "left")
            right_ok = _safe_render(time_display, right_image, "right")
            _save_oled_screenshot("oled_left", left_image)
            _save_oled_screenshot("oled_right", right_image)

            if not left_ok or not right_ok:
                LOGGER.info("Reinitializing OLED displays after render failure.")
                try:
                    temp_display.initialize()
                    time_display.initialize()
                except Exception as exc:
                    LOGGER.warning("OLED reinitialization failed: %s", exc)

            if sports_frames is None and time.monotonic() >= next_swap_at:
                show_date_on_left = not show_date_on_left
                next_swap_at = time.monotonic() + random_swap_interval_seconds()

            _STOP_EVENT.wait(REFRESH_SECONDS)
    finally:
        _safe_clear_display(temp_display, "left")
        _safe_clear_display(time_display, "right")
        with contextlib.suppress(Exception):
            bus.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
