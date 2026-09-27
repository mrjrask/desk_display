#!/usr/bin/env python3
"""Record what one display profile draws, for comparison with the v0.1 release.

Run it in a process configured for one profile (see
``rendering.profile_process.composition_env``).  It writes to OUT_DIR:

* ``constants.json``: every font size and upper-case numeric layout constant
  the screen modules and ``config`` hold after import;
* PNG stills of the MLB AL standings (first frame), the AL overview (settled
  frame), the NHL scoreboard with two and six games, the ``date`` and
  ``nixie`` clock faces at a fixed time and colour, the six weather pages
  (current, details, hourly, daily, sun & moon, alert) and the news ticker
  (first frame), drawn from the standings fixture in ``FIXTURE`` and the
  ``nhl_games.json`` and ``weather.json`` fixtures beside it.

With ``--v01`` it runs against a checkout of the v0.1 tag (the standalone
display, which drew everything in its own process); without it, against
this tree the way the render server now does.  ``scripts/make_v01_references.py``
records the v0.1 side into ``tests/fixtures/v01_reference`` and
``tests/test_v01_parity.py`` compares the current side with it.

Usage: v01_parity_probe.py ROOT PROFILE_ID FIXTURE OUT_DIR [--v01]
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import logging
import os
import pkgutil
import sys
import time

NOW = dt.datetime(2026, 9, 27, 14, 5, 9, tzinfo=dt.timezone.utc)
COLORS = ((255, 128, 0), (0, 200, 255))


def _constants(config) -> dict:
    from PIL import ImageFont

    import screens

    def snap(namespace: dict) -> dict:
        found = {}
        for name, value in namespace.items():
            if isinstance(value, ImageFont.FreeTypeFont):
                found[name] = ["font", os.path.basename(value.path), value.size]
            elif not name.isupper() or "MTIME" in name or isinstance(value, bool):
                continue
            elif isinstance(value, int | float):
                found[name] = value
            elif isinstance(value, tuple) and value and all(
                    isinstance(v, int | float) and not isinstance(v, bool) for v in value):
                found[name] = list(value)
        return found

    result = {"config": snap(vars(config))}
    for info in sorted(pkgutil.iter_modules(screens.__path__), key=lambda i: i.name):
        try:
            module = importlib.import_module(f"screens.{info.name}")
        except Exception as exc:  # noqa: BLE001 - recorded, so the comparison reports it
            result[f"screens.{info.name}"] = {"__error__": f"{type(exc).__name__}: {exc}"}
            continue
        result[f"screens.{info.name}"] = snap(vars(module))
    return result


class _Display:
    def __init__(self, width: int, height: int, *, skip: bool) -> None:
        self.width, self.height, self.skip, self.frames = width, height, skip, []

    def clear(self) -> None:
        pass

    def image(self, image) -> None:
        self.frames.append(image.copy())

    def show(self) -> None:
        pass

    def skip_requested(self) -> bool:
        return self.skip and bool(self.frames)

    def wait_for_skip(self, *_args, **_kwargs) -> bool:
        return False


def _nhl_scoreboards(config, fixture: str, out: str) -> None:
    """The NHL scoreboard, as the standalone display composed it, in this process."""

    import screens.nhl_scoreboard as nhl

    with open(os.path.join(os.path.dirname(fixture), "nhl_games.json"), encoding="utf-8") as handle:
        raw = json.load(handle)
    day = dt.date(2026, 10, 14)
    for count in (2, 6):
        games = nhl._hydrate_games([nhl._map_api_web_game(game, day) for game in raw[:count]])
        display = _Display(config.WIDTH, config.HEIGHT, skip=False)
        result = nhl.render_nhl_scoreboard(display, games, transition=True)
        image = getattr(result, "image", result)
        image.convert("RGB").save(os.path.join(out, f"nhl_scoreboard_{count}.png"))


# (reference name, v0.1 draw function in screens.draw_weather, screen id)
WEATHER_SCREENS = (
    ("weather1", "draw_weather_screen_1", "weather1"),
    ("weather2", "draw_weather_screen_2", "weather2"),
    ("weather_hourly", "draw_weather_hourly", "weather hourly"),
    ("weather_daily", "draw_weather_daily", "weather daily"),
    ("weather_astronomical", "draw_weather_astronomical", "astronomical"),
    ("weather_alert", "draw_weather_alert_screen", "weather alert"),
)
NEWS_TOPICS = (("world", "World"), ("tech", "Tech"), ("sports", "Sports"))


def _freeze_clock(module, name: str = "datetime") -> None:
    """Make *module*'s ``datetime.datetime.now()`` return ``NOW``."""

    import types

    class _Frozen(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)

    shim = types.ModuleType("datetime")
    shim.__dict__.update(vars(dt))
    shim.datetime = _Frozen
    setattr(module, name, shim)


def _weather_and_news_fixtures(config, fixture: str) -> dict:
    """Pin the weather and news screens to fixture data and a fixed clock.

    Returns the weather payload (``weather.json`` beside ``FIXTURE``).  The
    news ticker gets fixed headlines and no thumbnails, so nothing depends
    on the network or on when the probe runs.
    """

    import screens.draw_news_headlines as news
    import screens.draw_weather as weather_screens
    from services.news_feeds import NewsHeadline, NewsTopic

    with open(os.path.join(os.path.dirname(fixture), "weather.json"), encoding="utf-8") as handle:
        weather = json.load(handle)
    _freeze_clock(weather_screens)
    _freeze_clock(news, "dt")
    topics = [NewsTopic(id=topic_id, label=label, name=f"{label} News", url=f"https://example.com/{topic_id}.xml")
              for topic_id, label in NEWS_TOPICS]
    headlines = {
        topic.id: [NewsHeadline(topic_id=topic.id, title=f"{topic.label} headline {n}: parity fixture story",
                                link=f"https://example.com/{topic.id}/{n}") for n in range(1, 5)]
        for topic in topics
    }
    config.ENABLE_NEWS_HEADLINES = True
    config.ENABLE_STOCK_TICKER = False
    news.load_news_feed_config = lambda: (topics, 4, 20)
    news.fetch_all_headlines = lambda **_kwargs: headlines
    news._download_thumbnail = lambda *_args, **_kwargs: None
    news._download_hero_image = lambda *_args, **_kwargs: None
    return weather


def _v01(config, standings: dict, fixture: str, out: str) -> None:
    import screens.draw_date_time as date_time
    import screens.draw_nixie as nixie
    import screens.mlb_league_standings as mlb

    mlb._fetch_league_standings = lambda **_kwargs: standings
    display = _Display(config.WIDTH, config.HEIGHT, skip=True)
    mlb.draw_mlb_al_standings(display)
    display.frames[0].convert("RGB").save(os.path.join(out, "mlb_al_standings.png"))
    display = _Display(config.WIDTH, config.HEIGHT, skip=False)
    mlb.draw_AL_Overview(display)
    display.frames[-1].convert("RGB").save(os.path.join(out, "al_overview.png"))
    local = NOW.astimezone(config.CENTRAL_TIME)
    date_time.display_datetime = lambda *_args, **_kwargs: local
    date_time._compose_frame("date_time", *COLORS, False, "date").convert("RGB").save(
        os.path.join(out, "date.png"))
    nixie._compose_frame(NOW).convert("RGB").save(os.path.join(out, "nixie.png"))

    import screens.draw_news_headlines as news
    import screens.draw_weather as weather_screens

    weather = _weather_and_news_fixtures(config, fixture)
    for name, function, _screen_id in WEATHER_SCREENS:
        display = _Display(config.WIDTH, config.HEIGHT, skip=False)
        result = getattr(weather_screens, function)(display, weather, transition=True)
        getattr(result, "image", result).convert("RGB").save(os.path.join(out, f"{name}.png"))
    display = _Display(config.WIDTH, config.HEIGHT, skip=True)
    news.draw_news_headlines(display, transition=True)
    display.frames[0].convert("RGB").save(os.path.join(out, "news_headlines.png"))


def _current(config, profile_id: str, standings: dict, fixture: str, out: str) -> None:
    from display_profiles import resolve_display_profile_by_id
    from remote_display.models import RenderKey, ScreenRevisions
    from remote_display.server_rendering import compose_screen
    from rendering.clock_faces import render_clock
    from rendering.logos import ProfileLogos
    from screens.registry import set_native_profile
    from services.data_coordinator import DataCoordinator

    profile = resolve_display_profile_by_id(profile_id)
    if not set_native_profile(profile):
        raise SystemExit(f"process is not configured for {profile_id}")
    data = DataCoordinator()
    data.publish("mlb_league_standings", standings)
    data.publish("weather", _weather_and_news_fixtures(config, fixture))
    logos = ProfileLogos()
    screens = [("MLB AL Standings", "mlb_al_standings"), ("AL Overview", "al_overview"),
               *((screen_id, name) for name, _function, screen_id in WEATHER_SCREENS),
               ("news headlines", "news_headlines")]
    for screen_id, name in screens:
        key = RenderKey.for_screen(screen_id, profile_id, ScreenRevisions("s", "d", "r"))
        image, _metadata, _package = compose_screen(key, profile, data.snapshot(), logos)
        image.convert("RGB").save(os.path.join(out, f"{name}.png"))
    layout = {"face": "date", "time_zone": "America/Chicago", "time_format": "12", "show_ip": False}
    render_clock(layout, profile, NOW, colors=COLORS).convert("RGB").save(os.path.join(out, "date.png"))
    render_clock({**layout, "face": "nixie"}, profile, NOW).convert("RGB").save(os.path.join(out, "nixie.png"))


def main(argv: list[str]) -> None:
    root, profile_id, fixture, out = argv[:4]
    v01 = "--v01" in argv
    os.chdir(root)
    sys.path.insert(0, root)
    os.environ["CONFIG_LOAD_DOTENV"] = "0"
    os.environ["IP_WITH_TIME"] = "0"
    # Content comes from the fixtures, not from whatever location or forecast
    # timezone the calling environment carries (the sun & moon page prints
    # the coordinates when they are set).
    for name in ("WEATHER_LATITUDE", "WEATHER_LONGITUDE", "WEATHERKIT_TIMEZONE"):
        os.environ.pop(name, None)
    logging.disable(logging.CRITICAL)
    time.sleep = lambda *_args: None  # frames, not pacing, are compared
    with open(fixture, encoding="utf-8") as handle:
        standings = {int(league): value for league, value in json.load(handle).items()}
    os.makedirs(out, exist_ok=True)
    import config

    # Before anything renders: some renderers update module globals as they draw.
    constants = _constants(config)
    if not v01:
        # The registry now derives these from the render profile it is built
        # for instead of holding them as module constants; record what it derives.
        from screens import registry

        derived = constants.setdefault("screens.registry", {})
        derived.setdefault("WIDTH", config.WIDTH)
        derived.setdefault("HEIGHT", config.HEIGHT)
        derived.setdefault("_LOGO_SCROLL_SPEED", registry._logo_scroll_speed_for_layout(config.WIDTH, config.HEIGHT))
    if v01:
        _v01(config, standings, fixture, out)
    else:
        _current(config, profile_id, standings, fixture, out)
    # Both sides compose it in a process configured for the profile, which
    # is what the render server's worker for that profile is.
    _nhl_scoreboards(config, fixture, out)
    with open(os.path.join(out, "constants.json"), "w", encoding="utf-8") as handle:
        json.dump(constants, handle, indent=1, sort_keys=True)
        handle.write("\n")

if __name__ == "__main__":
    main(sys.argv[1:])
