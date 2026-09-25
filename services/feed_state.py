"""Save and restore the render server's last good feed data.

Without this, a restarted server has no data: it renders nothing new until
every feed has been fetched again, and it fetches them all at once. The
feed service saves each feed's last good value, source revision and success
time here after every successful refresh, and reloads them at start-up.

The file is JSON with a small, closed set of tagged types (datetimes and
the dataclasses in :data:`DATACLASSES`), so loading it never constructs an
arbitrary object. Keys named after secret settings and configured secret
values are removed before writing.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import secrets
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("desk_display.feed_state")

STATE_SCHEMA_VERSION = 1


def _dataclasses() -> dict[str, type]:
    from services.air_quality import AirQualityReport

    return {"AirQualityReport": AirQualityReport}


class UnsupportedValue(TypeError):
    pass


def encode(value: Any) -> Any:
    """JSON-safe form of a feed value; raise :class:`UnsupportedValue`."""

    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, datetime):
        return {"__type__": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"__type__": "date", "value": value.isoformat()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if _dataclasses().get(name) is not type(value):
            raise UnsupportedValue(f"cannot save {name}")
        fields = {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)}
        return {"__type__": "dataclass", "name": name, "fields": fields}
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            if all(isinstance(k, int) and not isinstance(k, bool) for k in value):
                return {"__type__": "intmap", "items": [[k, encode(v)] for k, v in value.items()]}
            raise UnsupportedValue("mapping keys must be strings or integers")
        if "__type__" in value:
            raise UnsupportedValue("reserved key __type__")
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [encode(v) for v in value]
    raise UnsupportedValue(f"cannot save {type(value).__name__}")


def _tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuples(v) for v in value)
    return value


def decode(value: Any) -> Any:
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("__type__")
    if kind is None:
        return {k: decode(v) for k, v in value.items()}
    if kind == "datetime":
        return datetime.fromisoformat(value["value"])
    if kind == "date":
        return date.fromisoformat(value["value"])
    if kind == "intmap":
        return {int(k): decode(v) for k, v in value["items"]}
    if kind == "dataclass":
        cls = _dataclasses().get(value.get("name"))
        if cls is None:
            raise ValueError(f"unknown saved type {value.get('name')!r}")
        known = {f.name for f in dataclasses.fields(cls)}
        fields = {k: _tuples(decode(v)) for k, v in value["fields"].items() if k in known}
        return cls(**fields)
    raise ValueError(f"unknown saved type {kind!r}")


class FeedStateFile:
    """Atomic JSON file of ``{key: {value, source_revision, saved_at}}``."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser()

    def load(self) -> dict[str, dict[str, Any]]:
        """Return the saved entries; an unreadable file yields nothing."""

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            LOGGER.warning("Ignoring unreadable feed state %s: %s", self.path, exc)
            return {}
        if not isinstance(raw, dict) or raw.get("schema_version") != STATE_SCHEMA_VERSION:
            LOGGER.warning("Ignoring feed state %s with an unsupported schema", self.path)
            return {}
        result: dict[str, dict[str, Any]] = {}
        for key, entry in (raw.get("feeds") or {}).items():
            try:
                revision = entry["source_revision"]
                saved_at = float(entry["saved_at"])
                if type(revision) is not int or revision < 0:
                    raise ValueError("bad source revision")
                result[key] = {"value": decode(entry["value"]), "source_revision": revision,
                               "saved_at": saved_at}
            except (KeyError, TypeError, ValueError) as exc:
                LOGGER.warning("Ignoring saved %s data: %s", key, exc)
        return result

    def save(self, entries: Mapping[str, Mapping[str, Any]]) -> None:
        import deployment_config

        feeds: dict[str, Any] = {}
        for key, entry in entries.items():
            try:
                encoded = encode(entry["value"])
            except UnsupportedValue as exc:
                LOGGER.debug("Not saving %s: %s", key, exc)
                continue
            feeds[key] = {"value": encoded, "source_revision": int(entry["source_revision"]),
                          "saved_at": float(entry["saved_at"])}
        payload = deployment_config.scrub_secrets({"schema_version": STATE_SCHEMA_VERSION, "feeds": feeds})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{secrets.token_hex(6)}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise


__all__ = ["FeedStateFile", "UnsupportedValue", "decode", "encode"]
