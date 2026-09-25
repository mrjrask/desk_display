"""Dependency-light client/server compatibility and payload helpers.

The functions here operate only on JSON-compatible dictionaries.  They are
safe to use from the server and from a display client before either process
loads configuration, rendering libraries, or display hardware.
"""
from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

from deployment_config import scrub_secrets
from protocol_versions import (
    APPLICATION_VERSION,
    CLIENT_CONFIG_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    NETWORK_PROTOCOL_VERSION,
    PLAYLIST_SCHEMA_VERSION,
    RENDER_PACKAGE_SCHEMA_VERSION,
    SERVER_CONFIG_SCHEMA_VERSION,
)

# Compatibility is explicit rather than inferred from ordering.  A future
# server may add an older version here while it retains the matching codec.
SERVER_ACCEPTED_CLIENT_PROTOCOL_VERSIONS = frozenset({NETWORK_PROTOCOL_VERSION})
CLIENT_SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset({MANIFEST_SCHEMA_VERSION})
CLIENT_SUPPORTED_RENDER_PACKAGE_SCHEMA_VERSIONS = frozenset(
    {RENDER_PACKAGE_SCHEMA_VERSION}
)


@dataclass(frozen=True)
class IncompatibleClientError(ValueError):
    """Registration rejection suitable for serialization as an HTTP 409."""

    received_protocol_version: int

    @property
    def code(self) -> str:
        return "incompatible_protocol_version"

    def as_response(self) -> dict[str, Any]:
        return {
            "error": self.code,
            "message": "Client protocol version is not supported by this server",
            "client_protocol_version": self.received_protocol_version,
            "accepted_protocol_versions": sorted(SERVER_ACCEPTED_CLIENT_PROTOCOL_VERSIONS),
            "server_software_version": APPLICATION_VERSION,
        }


def client_registration(client_id: str, *, software_version: str = APPLICATION_VERSION) -> dict:
    """Build the required client registration fields."""

    if not isinstance(client_id, str) or not client_id.strip():
        raise ValueError("client_id must not be empty")
    if not isinstance(software_version, str) or not software_version.strip():
        raise ValueError("software_version must not be empty")
    return {
        "client_id": client_id,
        "protocol_version": NETWORK_PROTOCOL_VERSION,
        "client_software_version": software_version,
    }


def registration_response(registration: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a registration and return the server's version advertisement.

    Callers exposing HTTP should serialize :class:`IncompatibleClientError`
    with ``as_response()`` and status 409.  No manifest or package should be
    sent after that rejection.
    """

    protocol_version = registration.get("protocol_version")
    if type(protocol_version) is not int or (
        protocol_version not in SERVER_ACCEPTED_CLIENT_PROTOCOL_VERSIONS
    ):
        raise IncompatibleClientError(
            protocol_version if type(protocol_version) is int else -1
        )
    software_version = registration.get("client_software_version")
    if not isinstance(software_version, str) or not software_version.strip():
        raise ValueError("client_software_version is required")
    client_id = registration.get("client_id")
    if not isinstance(client_id, str) or not client_id.strip():
        raise ValueError("client_id is required")
    return {
        "accepted": True,
        "protocol_version": NETWORK_PROTOCOL_VERSION,
        "server_software_version": APPLICATION_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "render_package_schema_version": RENDER_PACKAGE_SCHEMA_VERSION,
        "playlist_schema_version": PLAYLIST_SCHEMA_VERSION,
        "server_config_schema_version": SERVER_CONFIG_SCHEMA_VERSION,
        "client_config_schema_version": CLIENT_CONFIG_SCHEMA_VERSION,
    }


def build_manifest(**contents: Any) -> dict[str, Any]:
    """Return a manifest with all versions needed to interpret its contents.

    Server API credentials never reach a client: keys named after a secret
    setting are dropped and configured secret values are redacted.
    """

    return {
        **scrub_secrets(contents),
        "server_software_version": APPLICATION_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "render_package_schema_version": RENDER_PACKAGE_SCHEMA_VERSION,
        "playlist_schema_version": PLAYLIST_SCHEMA_VERSION,
        "server_config_schema_version": SERVER_CONFIG_SCHEMA_VERSION,
        "client_config_schema_version": CLIENT_CONFIG_SCHEMA_VERSION,
    }


def client_supports_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_versions: Collection[int] = CLIENT_SUPPORTED_MANIFEST_SCHEMA_VERSIONS,
    render_package_versions: Collection[int] = CLIENT_SUPPORTED_RENDER_PACKAGE_SCHEMA_VERSIONS,
) -> bool:
    """Whether a client can consume both the manifest and referenced package."""

    manifest_version = manifest.get("manifest_schema_version")
    render_package_version = manifest.get("render_package_schema_version")
    return (
        type(manifest_version) is int
        and manifest_version in manifest_versions
        and type(render_package_version) is int
        and render_package_version in render_package_versions
    )


def cached_manifest_usable_offline(
    manifest: Mapping[str, Any],
    *,
    manifest_versions: Collection[int] = CLIENT_SUPPORTED_MANIFEST_SCHEMA_VERSIONS,
    render_package_versions: Collection[int] = CLIENT_SUPPORTED_RENDER_PACKAGE_SCHEMA_VERSIONS,
) -> bool:
    """Return whether an already-downloaded manifest can be used offline.

    Cache validity depends on the cached document schemas, not on the current
    server software or protocol version.  Thus an old client rejected by an
    upgraded server may keep displaying a compatible, fully downloaded cache;
    it simply cannot register or receive new content.
    """

    return client_supports_manifest(
        manifest,
        manifest_versions=manifest_versions,
        render_package_versions=render_package_versions,
    ) and manifest.get("cache_complete") is True
