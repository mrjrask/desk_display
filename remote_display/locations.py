"""Per-display locations for weather and astronomy.

Each display normally shows the weather for the server's ``WEATHER_LATITUDE``
/ ``WEATHER_LONGITUDE``.  A display in another place can be given its own
latitude and longitude on the Clients page.  Displays that share a location
share its data and renders:

* The server fetches the location feeds (weather and air quality) once per
  distinct location and publishes them under ``"<feed>@<scope>"`` beside the
  global keys.
* Location screens for such a display are rendered with the location's
  ``scope`` as their render key's ``client_scope``, so each location has its
  own artifact lineage and every display at that location shares it.
* :func:`scoped_values` hands a render the location's data under the plain
  feed keys the screens read, so drawing code is unchanged.

The scope encodes the rounded coordinates (``loc-41.8781_-87.6298``), so a
render worker can recover the location from the render key alone.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# Feeds whose data depends on where the display is.
LOCATION_FEEDS = ("weather", "air_quality")
# Four decimal places is about 11 m: plenty for weather, and it keeps scopes short.
COORDINATE_DECIMALS = 4
SCOPE_PREFIX = "loc-"
_SCOPE_RE = re.compile(r"^loc-(-?\d{1,3}\.\d{4})_(-?\d{1,3}\.\d{4})$")


class LocationError(ValueError):
    """An invalid latitude/longitude pair."""

    def __init__(self, message: str, field: str) -> None:
        super().__init__(message)
        self.field = field


def _coordinate(value: Any, field: str, limit: float) -> float:
    try:
        if isinstance(value, bool):
            raise TypeError
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = math.nan
    if not -limit <= parsed <= limit:
        raise LocationError(f"{field} must be a number from {-limit:g} to {limit:g}", field)
    # Normalise -0.0 so equal places always get the same scope.
    return round(parsed, COORDINATE_DECIMALS) + 0.0


@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float

    @classmethod
    def parse(cls, latitude: Any, longitude: Any) -> Location | None:
        """Validate a pair from a form or the store; both blank means none."""

        blank = [v is None or (isinstance(v, str) and not v.strip()) for v in (latitude, longitude)]
        if all(blank):
            return None
        if blank[0]:
            raise LocationError("latitude is required with a longitude", "latitude")
        if blank[1]:
            raise LocationError("longitude is required with a latitude", "longitude")
        return cls(_coordinate(latitude, "latitude", 90), _coordinate(longitude, "longitude", 180))

    @classmethod
    def from_scope(cls, scope: str | None) -> Location | None:
        """The location a render scope names, or None for any other scope."""

        match = _SCOPE_RE.match(scope or "")
        if match is None:
            return None
        return cls(float(match.group(1)) + 0.0, float(match.group(2)) + 0.0)

    @property
    def scope(self) -> str:
        return (f"{SCOPE_PREFIX}{self.latitude:.{COORDINATE_DECIMALS}f}"
                f"_{self.longitude:.{COORDINATE_DECIMALS}f}")

    def as_dict(self) -> dict[str, float]:
        return {"latitude": self.latitude, "longitude": self.longitude}


def data_key(feed: str, scope: str | None) -> str:
    """The data coordinator key of *feed* for a location scope (None: global)."""

    return feed if not scope else f"{feed}@{scope}"


def location_screens() -> frozenset[str]:
    """Screens whose output depends on the display's location."""

    from services import feeds

    screens: set[str] = set()
    for feed in LOCATION_FEEDS:
        screens.update(feeds.FEED_DEPENDENCIES.get(feed, ()))
    return frozenset(screens)


def screen_scopes(location: Location | None, screens: Iterable[str]) -> dict[str, str]:
    """``{screen: scope}`` for the location screens among *screens*."""

    if location is None:
        return {}
    located = location_screens()
    return {screen: location.scope for screen in screens if screen in located}


def scoped_values(values: Mapping[str, Any], scope: str | None) -> dict[str, Any]:
    """*values* with each location feed replaced by its data for *scope*.

    A location whose data has not arrived yet has None for it, so its screens
    are unavailable rather than showing another place's weather.  The weather
    payload also carries the location, for screens that show or centre on it.
    """

    result = dict(values)
    location = Location.from_scope(scope)
    if location is None:
        return result
    for feed in LOCATION_FEEDS:
        result[feed] = values.get(data_key(feed, scope))
    weather = result.get("weather")
    if isinstance(weather, Mapping):
        result["weather"] = {**weather, "location": location.as_dict()}
    return result


__all__ = [
    "LOCATION_FEEDS",
    "Location",
    "LocationError",
    "data_key",
    "location_screens",
    "scoped_values",
    "screen_scopes",
]
