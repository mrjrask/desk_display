"""The feed summary carried in ``display_status.json`` for side displays.

The Waveshare OLED helper (``scripts/waveshare_oled_status.py``) shows the
outdoor temperature and Cubs/Blackhawks games from the display heartbeat
instead of calling the APIs itself. ``main.py`` builds this summary from its
own cache; a render server builds it from its feed data and returns it in
each heartbeat response, and the client writes it into its heartbeat file,
so the helper works the same in every mode.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _plain(value: Any) -> Any:
    """A JSON-ready copy (feed snapshots hold read-only mappings and tuples)."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def feed_summary(cache: Mapping[str, Any] | None) -> dict[str, Any]:
    """``cubs``, ``hawks`` and ``weather`` entries of the heartbeat, when known."""

    summary: dict[str, Any] = {}
    if not isinstance(cache, Mapping):
        return summary
    cubs = cache.get("cubs")
    if isinstance(cubs, Mapping):
        summary["cubs"] = {"live_game": _plain(cubs.get("live")), "last_game": _plain(cubs.get("last"))}
    hawks = cache.get("hawks")
    if isinstance(hawks, Mapping):
        summary["hawks"] = {
            "live_game": _plain(hawks.get("live")),
            "live_feed": _plain(hawks.get("live_feed")),
            "last_game": _plain(hawks.get("last")),
        }
    weather = cache.get("weather")
    current = weather.get("current") if isinstance(weather, Mapping) else None
    if isinstance(current, Mapping):
        temp_f = current.get("temp") or current.get("temp_f") or current.get("temperature")
        condition = None
        conditions = current.get("weather")
        if isinstance(conditions, (list, tuple)) and conditions and isinstance(conditions[0], Mapping):
            description = conditions[0].get("description")
            if isinstance(description, str) and description.strip():
                condition = description.strip()
        summary["weather"] = {"temp_f": _plain(temp_f), "condition": condition}
    return summary


__all__ = ["feed_summary"]
