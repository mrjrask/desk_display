"""Headless upstream data collection for the render server.

The standalone display loop (``main.py``) refreshes feeds into its own
in-memory cache.  A render server has no display loop, so this service does
the same work for it: it refreshes only the feeds that current render demand
needs, on the same intervals and with the same live-game rules as the
display loop (from :mod:`services.feeds`), and publishes each feed into
:mod:`services.data_coordinator` under the cache key screens already read.

Every feed keeps its last good value when a refresh fails or returns nothing,
and each has its own source revision, so the render coordinator can rerender
only the screens that use a feed that changed.
"""
from __future__ import annotations

import hashlib
import importlib
import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from remote_display.locations import LOCATION_FEEDS, Location, data_key
from services import feeds
from services.feed_state import FeedStateFile

LOGGER = logging.getLogger("desk_display.server_feeds")

# While a game is live (or about to start), refresh its feed this often.
# v0.1 fetched these fresh before every showing; a client syncs every 30 s,
# so this keeps a live score on screen within about a minute of the source.
LIVE_REFRESH_SECONDS = 30
# How often a team feed is checked for a game starting while its live
# screen is in a playlist but no game is on.
LIVE_WATCH_SECONDS = 120
# A feed is reported stale when its last success is older than this many intervals.
STALE_AFTER_INTERVALS = 2
TEAM_FEEDS = ("bears", "hawks", "wolves", "bulls", "cubs", "sox")
# Screens that read the scoreboard payload also read its metadata.
_FEED_KEYS: Mapping[str, tuple[str, ...]] = {
    "scoreboards": ("scoreboards", "scoreboard_metadata"),
    "nfl_standings": ("nfl_standings", "nfl_standings_meta"),
    "nhl_standings": ("nhl_standings", "nhl_wildcard_order"),
}
# A feed restores its health from saved state once these keys are saved.
_RESTORE_KEYS: Mapping[str, tuple[str, ...]] = {**_FEED_KEYS, "nhl_standings": ("nhl_standings",)}
# Displays with their own location: {location: screens those displays play}.
LocationDemand = Mapping[Location, Iterable[str]]


def _base_feed(feed: str) -> str:
    """``weather`` for ``weather@loc-…`` (a location's copy of a feed)."""

    return feed.split("@", 1)[0]


@dataclass
class FeedHealth:
    last_attempt: float | None = None
    last_success: float | None = None
    last_error: str | None = None
    consecutive_failures: int = 0


class ServerFeedService:
    """Refresh demanded feeds and publish them into the data coordinator."""

    def __init__(
        self,
        data: Any = None,
        provider: Any = None,
        *,
        fetch_air_quality: Callable[..., Any] | None = None,
        standings_fetchers: Mapping[str, Callable[..., Any]] | None = None,
        settings: Any = None,
        history_path: str | None = None,
        state_path: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        seed: bool = True,
        download_ncaa_fbs_logos: Callable[[list[dict]], Any] | None = None,
        fetch_traffic: Callable[..., Any] | None = None,
    ) -> None:
        if data is None:
            from services.data_coordinator import coordinator as data
        if provider is None:
            from services.data_provider import provider
        if fetch_air_quality is None:
            from services.air_quality import fetch_air_quality
        if settings is None:
            import config as settings
        if history_path is None:
            from paths import resolve_cache_file_path

            history_path = str(resolve_cache_file_path("AIR_QUALITY_HISTORY_PATH", "air_quality_history.json"))
        self.data = data
        self.provider = provider
        self.fetch_air_quality = fetch_air_quality
        if fetch_traffic is None:
            from services.traffic import fetch_report as fetch_traffic
        self.fetch_traffic = fetch_traffic
        self.standings_fetchers = dict(standings_fetchers or _default_standings_fetchers())
        self.settings = settings
        self.history_path = history_path
        # Only the render server passes this, so one machine downloads team
        # logos rather than every display.
        self.download_ncaa_fbs_logos = download_ncaa_fbs_logos
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._health: dict[str, FeedHealth] = {feed: FeedHealth() for feed in feeds.SERVER_FEED_DEPENDENCIES}
        self._scoreboard_dates: dict[str, Any] = {}
        # The wildcard order is only fetched for v2 screens, so it ages apart
        # from the NHL standings it is refreshed with.
        self._wildcard_fetched_at: float | None = None
        self._wildcard_unsaved = False
        self._stop = threading.Event()
        self._state = FeedStateFile(state_path) if state_path else None
        self._saved: dict[str, dict[str, Any]] = {}
        if seed:
            # Screens expect the standalone cache's shape before the first refresh.
            snapshot = self.data.snapshot()
            for key, value in feeds.default_cache().items():
                if key not in snapshot.values:
                    self.data.publish(key, value)
        if self._state is not None:
            self._restore()

    # ── Saved state ────────────────────────────────────────────────────────

    def _restore(self) -> None:
        """Reload last good data so a restarted server renders at once.

        Each restored feed's last success is set to when it was saved, so
        feeds refresh on their normal intervals rather than all at once.
        """

        saved = self._state.load()
        if not saved:
            return
        self._saved = saved
        self.data.restore({key: (entry["value"], entry["source_revision"]) for key, entry in saved.items()})
        now, wall = self._clock(), self._wall_clock()
        located = [key for key in saved if "@" in key and _base_feed(key) in LOCATION_FEEDS]
        for feed in [*feeds.SERVER_FEED_DEPENDENCIES, *located]:
            keys = _RESTORE_KEYS.get(feed, (feed,))
            if all(key in saved for key in keys):
                age = max(0.0, wall - min(saved[key]["saved_at"] for key in keys))
                self._health.setdefault(feed, FeedHealth()).last_success = now - age
        if "nhl_wildcard_order" in saved:
            self._wildcard_fetched_at = now - max(0.0, wall - saved["nhl_wildcard_order"]["saved_at"])
        LOGGER.info("Restored saved data for %s", ", ".join(sorted(saved)))

    def _save(self, feeds_refreshed: Iterable[str]) -> None:
        """Save the refreshed feeds' data, once, when any of it changed.

        One write per refresh pass rather than one per feed, and none when
        every refresh returned the same data: the file holds every feed, so
        each write is large, and during live games a pass runs every 30 s.
        """

        if self._state is None:
            return
        snapshot = self.data.snapshot()
        wall = self._wall_clock()
        changed = False
        for feed in feeds_refreshed:
            for key in _FEED_KEYS.get(feed, (feed,)):
                if key == "nhl_wildcard_order":
                    # Keep its saved time unless this refresh actually fetched it.
                    if not self._wildcard_unsaved:
                        continue
                    self._wildcard_unsaved = False
                if key not in snapshot.values:
                    continue
                revision = snapshot.source_revisions.get(key, 0)
                saved = self._saved.get(key)
                if saved is not None and saved["source_revision"] == revision:
                    continue
                self._saved[key] = {"value": snapshot.values[key], "source_revision": revision, "saved_at": wall}
                changed = True
        if not changed:
            return
        try:
            self._state.save(self._saved)
        except OSError as exc:
            LOGGER.warning("Could not save feed data: %s", exc)

    # ── Selection ──────────────────────────────────────────────────────────

    def enabled(self, feed: str) -> bool:
        feed = _base_feed(feed)
        if feed == "weather":
            return bool(getattr(self.settings, "ENABLE_WEATHER", False))
        if feed == "air_quality":
            return bool(getattr(self.settings, "ENABLE_AIR_QUALITY", False))
        return True

    def required_feeds(self, screens: Iterable[str]) -> set[str]:
        wanted = set(screens)
        return {
            feed for feed, dependents in feeds.SERVER_FEED_DEPENDENCIES.items()
            if wanted & dependents and self.enabled(feed)
        }

    def location_feeds(self, locations: LocationDemand | None) -> dict[str, tuple[str, Location]]:
        """``{"weather@loc-…": ("weather", location)}`` for each location's demanded feeds."""

        result: dict[str, tuple[str, Location]] = {}
        for location, screens in (locations or {}).items():
            wanted = set(screens)
            for feed in LOCATION_FEEDS:
                if wanted & feeds.SERVER_FEED_DEPENDENCIES.get(feed, set()) and self.enabled(feed):
                    result[data_key(feed, location.scope)] = (feed, location)
        return result

    def due_feeds(self, screens: Iterable[str], locations: LocationDemand | None = None) -> dict[str, bool]:
        """Return ``{feed: fresh}`` for every required feed that should refresh now.

        A location's feeds (``weather@loc-…``) refresh on their feed's interval.
        """

        due = self._due_global_feeds(screens)
        now = self._clock()
        for name, (feed, _location) in sorted(self.location_feeds(locations).items()):
            health = self._health.setdefault(name, FeedHealth())
            interval = feeds.SERVER_FEED_REFRESH_INTERVALS.get(feed, feeds.SCHEDULE_UPDATE_INTERVAL)
            last = health.last_success if health.consecutive_failures == 0 else health.last_attempt
            if last is None or now - last >= interval:
                due[name] = False
        return due

    def _due_global_feeds(self, screens: Iterable[str]) -> dict[str, bool]:
        screens = set(screens)
        now = self._clock()
        snapshot = self.data.snapshot()
        due: dict[str, bool] = {}
        for feed in sorted(self.required_feeds(screens)):
            health = self._health.setdefault(feed, FeedHealth())
            interval = feeds.SERVER_FEED_REFRESH_INTERVALS.get(feed, feeds.SCHEDULE_UPDATE_INTERVAL)
            fresh = False
            if feed == "scoreboards":
                if feeds.scoreboards_in_live_window(snapshot.values.get("scoreboards")):
                    interval, fresh = LIVE_REFRESH_SECONDS, True
                elif self._scoreboard_dates_changed(screens):
                    interval = 0
            elif any(feeds.LIVE_TEAM_SCREEN_TO_FEED.get(s) == feed for s in screens):
                team = snapshot.values.get(feed)
                live = isinstance(team, Mapping) and bool(team.get("live"))
                interval, fresh = (LIVE_REFRESH_SECONDS if live else LIVE_WATCH_SECONDS), True
            elif feed in feeds.POSTSEASON_FEED_MODULES and _has_live_postseason_game(feed, snapshot.values.get(feed)):
                interval, fresh = LIVE_REFRESH_SECONDS, True
            elif (
                feed == "nhl_standings"
                and screens & feeds.NHL_WILDCARD_SCREEN_IDS
                and health.consecutive_failures == 0
                and (self._wildcard_fetched_at is None or now - self._wildcard_fetched_at >= interval)
            ):
                # A v2 screen joined demand and its wildcard order is missing
                # or older than the feed interval; fetch it now.
                interval = 0
            last = health.last_success if health.consecutive_failures == 0 else health.last_attempt
            if last is None or now - last >= interval:
                due[feed] = fresh
        return due

    def _scoreboard_dates_changed(self, screens: set[str]) -> bool:
        return any(
            self._scoreboard_dates.get(league) != feeds.scoreboard_date_for_league(league)
            for league in feeds.scoreboard_leagues_for_screens(screens)
        )

    # ── Refresh ────────────────────────────────────────────────────────────

    def refresh(self, screens: Iterable[str], *, force: bool = False,
                locations: LocationDemand | None = None) -> dict[str, bool]:
        """Refresh the demanded feeds that are due; return ``{feed: succeeded}``.

        ``locations`` names the displays' own locations and the screens they
        play; each location's weather feeds are fetched once for all of them.
        """

        screens = set(screens)
        located = self.location_feeds(locations)
        if force:
            due = {f: True for f in [*self.required_feeds(screens), *located]}
        else:
            due = self.due_feeds(screens, locations)
        results: dict[str, bool] = {}
        for feed, fresh in due.items():
            health = self._health.setdefault(feed, FeedHealth())
            health.last_attempt = self._clock()
            try:
                if feed in located:
                    self._refresh_location(feed, *located[feed], fresh=fresh)
                else:
                    self._refresh_one(feed, screens, fresh=fresh)
            except Exception as exc:  # noqa: BLE001 - one feed must not stop the others
                health.last_error = f"{type(exc).__name__}: {exc}"[:300]
                health.consecutive_failures += 1
                results[feed] = False
                LOGGER.warning("Refreshing %s failed; keeping its last good data: %s", feed, exc)
                continue
            health.last_success = self._clock()
            health.last_error = None
            health.consecutive_failures = 0
            results[feed] = True
        self._save(feed for feed, ok in results.items() if ok)
        return results

    def _refresh_one(self, feed: str, screens: set[str], *, fresh: bool) -> None:
        if feed == "weather":
            ttl = int(getattr(self.settings, "WEATHER_REFRESH_SECONDS", 1800))
            value = self.provider.read_weather(ttl_seconds=0 if fresh else ttl)
            if not value:
                raise RuntimeError("weather provider returned no data")
            self.data.publish("weather", value)
        elif feed == "air_quality":
            self._refresh_air_quality()
        elif feed in TEAM_FEEDS:
            # read_legacy_team publishes the payload under the team's cache key.
            self.data.read_legacy_team(feed, force=fresh)
        elif feed == "scoreboards":
            self._refresh_scoreboards(screens, fresh=fresh)
        elif feed in feeds.LEAGUE_STANDINGS_DEPENDENCIES:
            self._refresh_standings(feed, screens, fresh=fresh)
        elif feed == "traffic":
            # Raises on a failed or malformed report, so the last good
            # report stays published and the screen shows its age.
            self.data.publish("traffic", self.fetch_traffic(force=True))
        else:  # pragma: no cover - every catalogued feed is handled above
            raise KeyError(f"no server refresher for feed {feed!r}")

    def _refresh_location(self, name: str, feed: str, location: Location, *, fresh: bool) -> None:
        coordinates = (location.latitude, location.longitude)
        if feed == "weather":
            ttl = int(getattr(self.settings, "WEATHER_REFRESH_SECONDS", 1800))
            value = self.provider.read_weather(ttl_seconds=0 if fresh else ttl, location=coordinates)
            if not value:
                raise RuntimeError(f"weather provider returned no data for {location.scope}")
            self.data.publish(name, value)
        elif feed == "air_quality":
            self._refresh_air_quality(location)
        else:  # pragma: no cover - LOCATION_FEEDS are handled above
            raise KeyError(f"no location refresher for feed {feed!r}")

    def _refresh_air_quality(self, location: Location | None = None) -> None:
        settings = self.settings
        if location is None:
            latitude = getattr(settings, "AIR_QUALITY_LATITUDE", None)
            longitude = getattr(settings, "AIR_QUALITY_LONGITUDE", None)
        else:
            latitude, longitude = location.latitude, location.longitude
        key = data_key("air_quality", None if location is None else location.scope)
        if latitude is None or longitude is None:
            raise RuntimeError("air quality coordinates are not configured")
        report = self.fetch_air_quality(
            latitude,
            longitude,
            api_key=getattr(settings, "AIRNOW_API_KEY", None),
            include_pollen=getattr(settings, "AIR_QUALITY_ENABLE_POLLEN", False),
        )
        if report is None:
            raise RuntimeError("air quality provider returned no data")
        if all(hasattr(report, f) for f in ("us_aqi_pm2_5", "us_aqi_pm10", "us_aqi_ozone")):
            now = self._wall_clock()
            previous = self.data.snapshot().values.get(key)
            history_path = self.history_path if location is None else _location_path(self.history_path, location)
            history = list(getattr(previous, "component_history", ()))
            if not history:
                history = feeds.load_air_quality_history(history_path, now)
            history = feeds.append_air_quality_sample(history, report, now)
            feeds.save_air_quality_history(history_path, history)
            report = replace(report, component_history=tuple(history))
        self.data.publish(key, report)

    def _refresh_standings(self, feed: str, screens: set[str], *, fresh: bool) -> None:
        wildcard = bool(screens & feeds.NHL_WILDCARD_SCREEN_IDS)
        kwargs = {"include_wildcard_order": wildcard} if feed == "nhl_standings" else {}
        values = self.standings_fetchers[feed](force=fresh, **kwargs)
        for key, value in values.items():
            self.data.publish(key, value)
        if "nhl_wildcard_order" in values:
            self._wildcard_fetched_at = self._clock()
            self._wildcard_unsaved = True
        if wildcard and feed == "nhl_standings" and "nhl_wildcard_order" not in values:
            # The standings still publish, but the refresh counts as failed so
            # the wildcard order is retried and the failure shows in health.
            raise RuntimeError("NHL wildcard order returned no data")

    def _refresh_scoreboards(self, screens: set[str], *, fresh: bool) -> None:
        leagues = feeds.scoreboard_leagues_for_screens(screens)
        values = self.data.snapshot().values
        previous = dict(values.get("scoreboards") or {})
        metadata = dict(values.get("scoreboard_metadata") or {})
        force_leagues: set[str] = set()
        if fresh and feeds.scoreboards_in_live_window({"nfl": previous.get("nfl") or []}):
            force_leagues = {"nfl"}
        payloads = self.provider.read_sports_payloads(
            ttl_seconds=0 if fresh else 120,
            leagues=leagues,
            force_refresh_leagues=force_leagues,
        ) or {}
        previous.update(payloads.get("scoreboards") or {})
        metadata.update(payloads.get("scoreboard_metadata") or {})
        self.data.publish("scoreboards", previous)
        self.data.publish("scoreboard_metadata", metadata)
        for league in leagues:
            self._scoreboard_dates[league] = feeds.scoreboard_date_for_league(league)
        fbs_games = (payloads.get("scoreboards") or {}).get("ncaa_fbs")
        if self.download_ncaa_fbs_logos is not None and fbs_games:
            try:
                self.download_ncaa_fbs_logos(list(fbs_games))
            except Exception as exc:  # a logo problem must never cost the scores
                logging.warning("NCAA FBS logo download failed: %s", exc)

    # ── Revisions and health ───────────────────────────────────────────────

    def data_revision(self, screen: str, source_revisions: Mapping[str, int] | None = None,
                      scope: str | None = None) -> str | None:
        """A data revision covering exactly the feeds *screen* reads.

        Returns ``None`` for a screen that no catalogued feed serves, so the
        caller can fall back to a conservative whole-snapshot revision.  With a
        location ``scope`` the location feeds are that location's copies.
        """

        screen_feeds = feeds.feeds_for_screen(screen, feeds.SERVER_FEED_DEPENDENCIES)
        if not screen_feeds:
            return None
        if source_revisions is None:
            source_revisions = self.data.snapshot().source_revisions
        parts = []
        for feed in sorted(screen_feeds):
            for key in _FEED_KEYS.get(feed, (feed,)):
                if scope and feed in LOCATION_FEEDS:
                    key = data_key(key, scope)
                parts.append(f"{key}:{source_revisions.get(key, 0)}")
        return "f-" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    def health(self) -> dict[str, Any]:
        now = self._clock()
        snapshot = self.data.snapshot()
        report: dict[str, Any] = {}
        for feed, health in sorted(self._health.items()):
            interval = feeds.SERVER_FEED_REFRESH_INTERVALS.get(_base_feed(feed), feeds.SCHEDULE_UPDATE_INTERVAL)
            age = None if health.last_success is None else round(now - health.last_success, 1)
            report[feed] = {
                "enabled": self.enabled(feed),
                "source_revision": snapshot.source_revisions.get(feed, 0),
                "last_attempt_seconds_ago": None if health.last_attempt is None
                else round(now - health.last_attempt, 1),
                "last_success_seconds_ago": age,
                "consecutive_failures": health.consecutive_failures,
                "last_error": health.last_error,
                "stale": age is None or age > interval * STALE_AFTER_INTERVALS,
            }
        return report

    # ── Background loop ────────────────────────────────────────────────────

    def start(
        self,
        demanded_screens: Callable[[], Iterable[str]],
        interval_seconds: float = 30,
        demanded_locations: Callable[[], LocationDemand] | None = None,
    ) -> threading.Thread:
        """Refresh feeds for the current demand every *interval_seconds*."""

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.refresh(demanded_screens(),
                                 locations=None if demanded_locations is None else demanded_locations())
                except Exception:  # pragma: no cover - logged and retried
                    LOGGER.exception("Feed refresh pass failed")
                self._stop.wait(interval_seconds)

        thread = threading.Thread(target=loop, name="server-feeds", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()


def _location_path(path: str, location: Location) -> str:
    """*path* with the location's scope before its extension."""

    root, dot, ext = path.rpartition(".")
    return f"{root}.{location.scope}.{ext}" if dot and "/" not in ext else f"{path}.{location.scope}"


def _fetch_nfl_standings(*, force: bool = False) -> dict[str, Any]:
    from screens import nfl_standings

    standings, fallback_message, season_note = nfl_standings._fetch_standings_data(force=True)
    if fallback_message == nfl_standings.FALLBACK_MESSAGE_UNAVAILABLE:
        # The fetch failed, even when it hands back cached rows.
        raise RuntimeError(fallback_message)
    if not any((standings or {}).values()) and fallback_message != nfl_standings.FALLBACK_MESSAGE_OFFSEASON:
        # Only the offseason legitimately has no rows.
        raise RuntimeError(fallback_message or "NFL standings returned no data")
    return {
        "nfl_standings": standings,
        "nfl_standings_meta": {"fallback_message": fallback_message, "season_note": season_note},
    }


def _fetched_since(cache: Mapping[str, Any], started: float) -> bool:
    """Whether a standings module stored a successful fetch at or after *started*."""

    return float(cache.get("timestamp", 0.0)) >= started


def _fetch_nhl_standings(*, force: bool = False, include_wildcard_order: bool = False) -> dict[str, Any]:
    from screens import nhl_standings

    # Always fetch: the feed interval sets the cadence, and a fresh fetch
    # lets a failure that falls back to the module's cached rows be seen.
    started = time.time()
    standings = nhl_standings._fetch_standings_data(force=True)
    if not standings or not _fetched_since(nhl_standings._STANDINGS_CACHE, started):
        raise RuntimeError("NHL standings fetch failed")
    values: dict[str, Any] = {"nhl_standings": standings}
    if include_wildcard_order:
        wildcard_order = nhl_standings._fetch_wildcard_order_api_web()
        if wildcard_order:
            values["nhl_wildcard_order"] = wildcard_order
    return values


def _fetch_mlb_league_standings(*, force: bool = False) -> dict[str, Any]:
    from screens import mlb_league_standings

    started = time.time()
    standings = mlb_league_standings._fetch_league_standings(force=True)
    # A failed fetch returns cached rows, or every division empty.
    if not _fetched_since(mlb_league_standings._STANDINGS_CACHE, started) or not any(
        rows for league in (standings or {}).values() for rows in league.values()
    ):
        raise RuntimeError("MLB league standings fetch failed")
    return {"mlb_league_standings": standings}


def _postseason_fetcher(feed: str) -> Callable[..., dict[str, Any]]:
    def fetch(*, force: bool = False) -> dict[str, Any]:
        module = importlib.import_module(feeds.POSTSEASON_FEED_MODULES[feed])
        # Raises when none of the league's bracket sources answered.
        return {feed: module.fetch_postseason(force=True)}

    return fetch


def _has_live_postseason_game(feed: str, value: Any) -> bool:
    return importlib.import_module(feeds.POSTSEASON_FEED_MODULES[feed]).has_live_series(value)


def _default_standings_fetchers() -> dict[str, Callable[..., dict[str, Any]]]:
    """Fetch league standings the way the standalone runtime does, keyed by feed."""

    return {
        "nfl_standings": _fetch_nfl_standings,
        "nhl_standings": _fetch_nhl_standings,
        "mlb_league_standings": _fetch_mlb_league_standings,
        **{feed: _postseason_fetcher(feed) for feed in feeds.POSTSEASON_FEED_MODULES},
    }


__all__ = ["LIVE_REFRESH_SECONDS", "LIVE_WATCH_SECONDS", "ServerFeedService"]
