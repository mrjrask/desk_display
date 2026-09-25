"""Explicit migrations for persisted playlist and configuration schemas."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from protocol_versions import (
    CLIENT_CONFIG_SCHEMA_VERSION,
    PLAYLIST_SCHEMA_VERSION,
    SERVER_CONFIG_SCHEMA_VERSION,
)
from schedule_migrations import migrate_config as migrate_playlist_v1_to_v2


class UnsupportedSchemaVersion(ValueError):
    """The document is newer than this software or has no migration path."""


def _migrate_versioned(
    document: Mapping[str, Any],
    *,
    target: int,
    migrations: Mapping[int, Callable[[dict[str, Any]], dict[str, Any]]],
) -> dict[str, Any]:
    result = dict(document)
    version = result.get("schema_version", 0)
    if type(version) is not int or version < 0 or version > target:
        raise UnsupportedSchemaVersion(f"unsupported schema version {version!r}")
    while version < target:
        migration = migrations.get(version)
        if migration is None:
            raise UnsupportedSchemaVersion(f"no migration from schema version {version}")
        result = migration(result)
        version += 1
        result["schema_version"] = version
    return result


def _adopt_standalone_v01(document: dict[str, Any]) -> dict[str, Any]:
    """Add a schema marker to an unversioned standalone v0.1 config.

    v0.1 configuration keys remain at the top level, so migration is lossless
    and existing paths and environment-derived overrides keep their meaning.
    """

    result = dict(document)
    result["schema_version"] = 1
    result.setdefault("migrated_from", "standalone-v0.1")
    return result


def migrate_server_config(document: Mapping[str, Any]) -> dict[str, Any]:
    return _migrate_versioned(
        document, target=SERVER_CONFIG_SCHEMA_VERSION, migrations={0: _adopt_standalone_v01}
    )


def migrate_client_config(document: Mapping[str, Any]) -> dict[str, Any]:
    return _migrate_versioned(
        document, target=CLIENT_CONFIG_SCHEMA_VERSION, migrations={0: _adopt_standalone_v01}
    )


def migrate_playlist(document: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade legacy playlists sequentially and expose a schema_version key."""

    version = document.get("schema_version", document.get("version", 1))
    if version == PLAYLIST_SCHEMA_VERSION:
        result = dict(document)
    elif version == 1:
        result = migrate_playlist_v1_to_v2(dict(document)).config
    else:
        raise UnsupportedSchemaVersion(f"unsupported playlist schema version {version!r}")
    result["schema_version"] = PLAYLIST_SCHEMA_VERSION
    return result
