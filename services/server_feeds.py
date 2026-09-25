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
import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from services import feeds

LOGGER = logging.getLogger("desk_display.server_feeds")

# While a game is live (or about to start), refresh its feed this often.
LIVE_REFRESH_SECONDS = 120
# A feed is reported stale when its last success is older than this many intervals.
STALE_AFTER_INTERVALS = 2
TEAM_FEEDS = ("bears", "hawks", "wolves", "bulls", "cubs", "sox")
# Screens that read the scoreboard payload also read its metadata.
_FEED_KEYS: Mapping[str, tuple[str, ...]] = {"scoreboards": ("scoreboards", "scoreboard_metadata")}


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
        settings: Any = None,
        history_path: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        seed: bool = True,
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
        self.settings = settings
        self.history_path = history_path
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._health: dict[str, FeedHealth] = {feed: FeedHealth() for feed in feeds.FEED_DEPENDENCIES}
        self._scoreboard_dates: dict[str, Any] = {}
        self._stop = threading.Event()
        if seed:
            # Screens expect the standalone cache's shape before the first refresh.
            snapshot = self.data.snapshot()
            for key, value in feeds.default_cache().items():
                if key not in snapshot.values:
                    self.data.publish(key, value)

    # ── Selection ──────────────────────────────────────────────────────────

    def enabled(self, feed: str) -> bool:
        if feed == "weather":
            return bool(getattr(self.settings, "ENABLE_WEATHER", False))
        if feed == "air_quality":
            return bool(getattr(self.settings, "ENABLE_AIR_QUALITY", False))
        return True

    def required_feeds(self, screens: Iterable[str]) -> set[str]:
        wanted = set(screens)
        return {
            feed for feed, dependents in feeds.FEED_DEPENDENCIES.items()
            if wanted & dependents and self.enabled(feed)
        }

    def due_feeds(self, screens: Iterable[str]) -> dict[str, bool]:
        """Return ``{feed: fresh}`` for every required feed that should refresh now."""

        screens = set(screens)
        now = self._clock()
        snapshot = self.data.snapshot()
        due: dict[str, bool] = {}
        for feed in sorted(self.required_feeds(screens)):
            health = self._health.setdefault(feed, FeedHealth())
            interval = feeds.FEED_REFRESH_INTERVALS.get(feed, feeds.SCHEDULE_UPDATE_INTERVAL)
            fresh = False
            if feed == "scoreboards":
                if feeds.scoreboards_in_live_window(snapshot.values.get("scoreboards")):
                    interval, fresh = LIVE_REFRESH_SECONDS, True
                elif self._scoreboard_dates_changed(screens):
                    interval = 0
            elif any(feeds.LIVE_TEAM_SCREEN_TO_FEED.get(s) == feed for s in screens):
                interval, fresh = LIVE_REFRESH_SECONDS, True
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

    def refresh(self, screens: Iterable[str], *, force: bool = False) -> dict[str, bool]:
        """Refresh the demanded feeds that are due; return ``{feed: succeeded}``."""

        screens = set(screens)
        due = {f: True for f in self.required_feeds(screens)} if force else self.due_feeds(screens)
        results: dict[str, bool] = {}
        for feed, fresh in due.items():
            health = self._health.setdefault(feed, FeedHealth())
            health.last_attempt = self._clock()
            try:
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
        else:  # pragma: no cover - every catalogued feed is handled above
            raise KeyError(f"no server refresher for feed {feed!r}")

    def _refresh_air_quality(self) -> None:
        settings = self.settings
        latitude = getattr(settings, "AIR_QUALITY_LATITUDE", None)
        longitude = getattr(settings, "AIR_QUALITY_LONGITUDE", None)
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
            previous = self.data.snapshot().values.get("air_quality")
            history = list(getattr(previous, "component_history", ()))
            if not history:
                history = feeds.load_air_quality_history(self.history_path, now)
            history = feeds.append_air_quality_sample(history, report, now)
            feeds.save_air_quality_history(self.history_path, history)
            report = replace(report, component_history=tuple(history))
        self.data.publish("air_quality", report)

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

    # ── Revisions and health ───────────────────────────────────────────────

    def data_revision(self, screen: str, source_revisions: Mapping[str, int] | None = None) -> str | None:
        """A data revision covering exactly the feeds *screen* reads.

        Returns ``None`` for a screen that no catalogued feed serves, so the
        caller can fall back to a conservative whole-snapshot revision.
        """

        screen_feeds = feeds.feeds_for_screen(screen)
        if not screen_feeds:
            return None
        if source_revisions is None:
            source_revisions = self.data.snapshot().source_revisions
        parts = []
        for feed in sorted(screen_feeds):
            for key in _FEED_KEYS.get(feed, (feed,)):
                parts.append(f"{key}:{source_revisions.get(key, 0)}")
        return "f-" + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    def health(self) -> dict[str, Any]:
        now = self._clock()
        snapshot = self.data.snapshot()
        report: dict[str, Any] = {}
        for feed, health in sorted(self._health.items()):
            interval = feeds.FEED_REFRESH_INTERVALS.get(feed, feeds.SCHEDULE_UPDATE_INTERVAL)
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

    def start(self, demanded_screens: Callable[[], Iterable[str]], interval_seconds: float = 30) -> threading.Thread:
        """Refresh feeds for the current demand every *interval_seconds*."""

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.refresh(demanded_screens())
                except Exception:  # pragma: no cover - logged and retried
                    LOGGER.exception("Feed refresh pass failed")
                self._stop.wait(interval_seconds)

        thread = threading.Thread(target=loop, name="server-feeds", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()


__all__ = ["LIVE_REFRESH_SECONDS", "ServerFeedService"]
