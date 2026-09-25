"""Central ownership of refreshable upstream data.

Renderers consume immutable :class:`DataSnapshot` objects.  They never call an
upstream API, which makes rendering multiple profiles from one snapshot cheap
and deterministic.
"""
from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

import data_fetch
from services.data_provider import DataProvider, provider


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    # Isolate a snapshot from mutable objects still held by provider caches.
    try:
        return copy.deepcopy(value)
    except (TypeError, ValueError):
        return value


@dataclass(frozen=True)
class DataSnapshot(Mapping[str, Any]):
    """An immutable point-in-time view of coordinated data."""

    revision: int
    created_at: datetime
    values: Mapping[str, Any]
    source_revisions: Mapping[str, int]

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class DataSource:
    fetcher: Callable[[], Any]
    ttl_seconds: int = 300


class DataCoordinator:
    """Refresh registered sources once and publish revisioned snapshots."""

    def __init__(self, data_provider: DataProvider | None = None) -> None:
        self.provider = data_provider or provider
        self._lock = threading.RLock()
        self._sources: dict[str, DataSource] = {}
        self._values: dict[str, Any] = {}
        self._source_revisions: dict[str, int] = {}
        self._revision = 0

    def register_source(
        self, name: str, fetcher: Callable[[], Any], *, ttl_seconds: int = 300
    ) -> None:
        if not name or not callable(fetcher):
            raise ValueError("A source requires a name and callable fetcher")
        with self._lock:
            self._sources[name] = DataSource(fetcher, max(0, int(ttl_seconds)))

    def refresh(self, sources: set[str] | None = None, *, force: bool = False) -> DataSnapshot:
        """Refresh selected sources and atomically publish one new revision.

        ``DataProvider`` provides per-source single-flight behavior, so
        concurrent refresh requests and multiple display profiles share the
        same logical upstream request.
        """

        with self._lock:
            names = tuple(sorted(sources if sources is not None else self._sources))
            specs = {name: self._sources[name] for name in names}

        refreshed: dict[str, Any] = {}
        for name, source in specs.items():
            refreshed[name] = self.provider.read(
                name, source.fetcher, ttl_seconds=source.ttl_seconds, force=force
            )

        with self._lock:
            for name, value in refreshed.items():
                self._values[name] = value
                self._source_revisions[name] = self._source_revisions.get(name, 0) + 1
            if refreshed:
                self._revision += 1
            return self.snapshot()

    def publish(self, name: str, value: Any) -> DataSnapshot:
        """Publish already-acquired local data through the same revision model."""

        with self._lock:
            self._values[name] = value
            self._source_revisions[name] = self._source_revisions.get(name, 0) + 1
            self._revision += 1
            return self.snapshot()

    def snapshot(self) -> DataSnapshot:
        with self._lock:
            return DataSnapshot(
                revision=self._revision,
                created_at=datetime.now(UTC),
                values=_freeze(self._values),
                source_revisions=MappingProxyType(dict(self._source_revisions)),
            )

    def read_legacy_team(self, team: str, *, ttl_seconds: int = 120, force: bool = False) -> dict[str, Any]:
        """Coordinate the legacy team feeds while screens migrate to snapshots."""

        def fetch() -> dict[str, Any]:
            if team == "bears":
                return {"stand": data_fetch.fetch_bears_standings()}
            if team == "hawks":
                live = data_fetch.fetch_blackhawks_live_game()
                return {
                    "stand": data_fetch.fetch_blackhawks_standings(),
                    "last": data_fetch.fetch_blackhawks_last_game(),
                    "live": live,
                    "next": data_fetch.fetch_blackhawks_next_game(),
                    "next_home": data_fetch.fetch_blackhawks_next_home_game(),
                }
            if team == "wolves":
                games = data_fetch.fetch_wolves_games() or {}
                return {
                    "last": games.get("last_game"), "live": games.get("live_game"),
                    "next": games.get("next_game"), "next_home": games.get("next_home_game"),
                }
            if team == "bulls":
                return {
                    "stand": data_fetch.fetch_bulls_standings(),
                    "last": data_fetch.fetch_bulls_last_game(),
                    "live": data_fetch.fetch_bulls_live_game(),
                    "next": data_fetch.fetch_bulls_next_game(),
                    "next_home": data_fetch.fetch_bulls_next_home_game(),
                }
            if team in {"cubs", "sox"}:
                games = (data_fetch.fetch_cubs_games() if team == "cubs" else data_fetch.fetch_sox_games()) or {}
                standings = data_fetch.fetch_cubs_standings() if team == "cubs" else data_fetch.fetch_sox_standings()
                return {
                    "stand": standings, "last": games.get("last_game"),
                    "last_alt": games.get("last_game_alt"), "live": games.get("live_game"),
                    "next": games.get("next_game"), "next_alt": games.get("next_game_alt"),
                    "current_series": games.get("current_series_games"),
                    "next_series": games.get("next_series_games"),
                    "next_home_series": games.get("next_home_series_games"),
                    "next_home": games.get("next_home_game"),
                    "schedule_covers_today": bool(games.get("schedule_covers_today")),
                }
            raise KeyError(f"Unknown legacy team source: {team}")

        value = self.provider.read(f"team:{team}", fetch, ttl_seconds=ttl_seconds, force=force)
        self.publish(team, value)
        return value

    def read_mlb_league_standings(
        self, *, ttl_seconds: int = 300, force: bool = False
    ) -> dict[int, dict[str, list[dict[str, Any]]]]:
        """Acquire MLB league standings before snapshot-based rendering."""

        from screens.mlb_league_standings import _fetch_league_standings

        value = self.provider.read(
            "mlb_league_standings", _fetch_league_standings,
            ttl_seconds=ttl_seconds, force=force,
        )
        self.publish("mlb_league_standings", value)
        return value

    @staticmethod
    def weather_cache_timestamp() -> datetime | None:
        return data_fetch.get_weather_cache_timestamp()


coordinator = DataCoordinator()
