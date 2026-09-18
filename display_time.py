"""Display timezone helpers that do not depend on the rendering stack."""
from __future__ import annotations

import datetime
from typing import Optional
from zoneinfo import ZoneInfo


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


CENTRAL_TIME = LocalizableZoneInfo("America/Chicago")
