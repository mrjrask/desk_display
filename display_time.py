"""Display timezone helpers that do not depend on the rendering stack."""
from __future__ import annotations

import datetime
import logging
import os
from collections.abc import Mapping
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

CONTENT_TIMEZONE_ENV = "DESK_DISPLAY_CONTENT_TIMEZONE"
DEFAULT_CONTENT_TIMEZONE = "America/Chicago"


class LocalizableZoneInfo(datetime.tzinfo):
    """ZoneInfo wrapper that provides a pytz-compatible ``localize`` helper."""

    def __init__(self, key: str) -> None:
        self._zone = ZoneInfo(key)

    def _coerce(self, dt: Optional[datetime.datetime]) -> Optional[datetime.datetime]:
        if dt is None:
            return None
        return dt.replace(tzinfo=self._zone)

    def utcoffset(self, dt: Optional[datetime.datetime]) -> Optional[datetime.timedelta]:
        return self._zone.utcoffset(self._coerce(dt))

    def dst(self, dt: Optional[datetime.datetime]) -> Optional[datetime.timedelta]:
        return self._zone.dst(self._coerce(dt))

    def tzname(self, dt: Optional[datetime.datetime]) -> Optional[str]:
        return self._zone.tzname(self._coerce(dt))

    def fromutc(self, dt: datetime.datetime) -> datetime.datetime:
        coerced = dt.replace(tzinfo=self._zone)
        converted = self._zone.fromutc(coerced)
        return converted.replace(tzinfo=self)

    def localize(self, dt: datetime.datetime) -> datetime.datetime:
        if dt.tzinfo is not None:
            return dt.astimezone(self)
        return dt.replace(tzinfo=self)


def content_timezone_name(env: Mapping[str, str] | None = None) -> str:
    """The configured IANA content timezone, or the default when unset or unknown."""

    source = os.environ if env is None else env
    name = (source.get(CONTENT_TIMEZONE_ENV) or "").strip()
    if not name:
        return DEFAULT_CONTENT_TIMEZONE
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        logging.warning("Unknown %s %r; using %s", CONTENT_TIMEZONE_ENV, name, DEFAULT_CONTENT_TIMEZONE)
        return DEFAULT_CONTENT_TIMEZONE
    return name


class ContentZone(LocalizableZoneInfo):
    """The configured content timezone (``DESK_DISPLAY_CONTENT_TIMEZONE``).

    Dates, schedules, clocks and dark hours all use this one zone.  It is
    resolved on first use rather than at import, because ``config`` imports
    this module before it loads the dotenv file that may set the zone.
    """

    def __init__(self) -> None:  # noqa: D107 - the zone is resolved lazily
        self._resolved: ZoneInfo | None = None

    @property
    def _zone(self) -> ZoneInfo:  # type: ignore[override]
        if self._resolved is None:
            self._resolved = ZoneInfo(content_timezone_name())
        return self._resolved

    @property
    def key(self) -> str:
        return self._zone.key

    def reset(self) -> None:
        """Read the setting again on next use (tests and configuration reloads)."""

        self._resolved = None

    def __repr__(self) -> str:
        return f"ContentZone({self.key!r})"


# Historical name: this was fixed to America/Chicago before the content
# timezone became configurable.
CENTRAL_TIME = CONTENT_TIME = ContentZone()
