#!/usr/bin/env python3
"""Desk Display render server API (``/api/v1``).

A separate service from the screenshot Feed server (``feed_server.py``),
whose endpoints and semantics are unchanged.  Run it with
``DESK_DISPLAY_ROLE=server python3 display_server.py``.

Authentication
    ``POST /api/v1/register`` needs ``Authorization: Bearer <enrollment
    credential>``.  By default (``DESK_DISPLAY_SERVER_ENROLLMENT=provisioned``)
    that is the client's own provisioned credential (see
    ``remote_display/provisioning.py``), so each client can be rotated,
    revoked or disabled on its own; ``shared`` mode instead accepts the one
    ``DESK_DISPLAY_SERVER_AUTH_TOKEN`` every client knows.  A successful
    registration returns a short-lived lease ``client_credential``; every ``/api/v1/clients/<id>/...``
    endpoint needs ``Authorization: Bearer <client credential>`` for *that*
    client ID, so one client can never read another's configuration, status
    or manifest.  ``/api/v1/admin/...`` needs ``DESK_DISPLAY_SERVER_ADMIN_TOKEN``
    and is disabled when it is unset.  ``/api/v1/health`` is public and says
    only whether the service is up.  ``/api/v1/join`` needs a one-time code
    from the config UI's "Add a display" wizard; redeeming it issues that
    display's credential inside the setup script it returns.

Endpoints
    ``POST /api/v1/register``                      register or renew a lease
    ``POST /api/v1/clients/<id>/heartbeat``        report status, renew lease, collect
                                                   queued update/restart commands
    ``GET  /api/v1/clients/<id>/config``           client configuration
    ``GET  /api/v1/clients/<id>/manifest``         current manifest (ETag)
    ``GET  /api/v1/clients/<id>/artifacts/<sha256>.<ext>``
                                                   immutable artifact (ETag, Range)
    ``GET  /api/v1/health``                        liveness
    ``POST /api/v1/join``                          redeem a one-time join code (form field
                                                   ``code``) for a new display's setup script
    ``GET  /api/v1/admin/status``                  clients, leases and demand
    ``PUT|DELETE /api/v1/admin/prerender/<name>``  explicit pre-render demand
    ``POST /api/v1/admin/clients/<id>/disable|enable|rotate|revoke|remove``
    ``POST /api/v1/admin/clients``                 provision a client (credential shown once)
    ``GET  /api/v1/admin/clients``                 provisioned clients, never their credentials

Requests are rate limited per client and per address (``429`` with
``Retry-After``); see ``remote_display/rate_limit.py``.
    ``GET  /api/v1/admin/render-status``           queue, render and data health
"""
from __future__ import annotations

import atexit
import hmac
import io
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, g, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

import deployment_config
from deployment_config import Role
from protocol import (
    CLIENT_SUPPORTED_RENDER_PACKAGE_SCHEMA_VERSIONS,
    SERVER_ACCEPTED_CLIENT_PROTOCOL_VERSIONS,
    IncompatibleClientError,
    registration_response,
)
from protocol_versions import CLIENT_CONFIG_SCHEMA_VERSION, PLAYLIST_SCHEMA_VERSION
from remote_display.artifact_store import ArtifactStore
from remote_display.client_commands import (
    WIRE_VERSION as CLIENT_COMMAND_VERSION,
    CommandStore,
    commands_path,
    parse_heartbeat_commands,
)
from remote_display.manifest import build_client_manifest, referenced_hashes
from remote_display.render_coordinator import RenderCoordinator, Renderer, RevisionSource, ScopedRevisionSource
from remote_display.models import (
    ClientCapabilities,
    ClientDemand,
    ClientResources,
    ClientStatus,
    ClientTelemetry,
    ModelValidationError,
    UnsupportedCapabilitiesError,
    identifier,
)
from remote_display.locations import Location, LocationError, location_screens, scoped_values, screen_scopes
from services import traffic as road_traffic
from remote_display.playlist_store import PlaylistStore, PlaylistStoreError, registry_snapshot_path, store_path
from remote_display import registration
from remote_display.provisioning import (
    InvalidJoinCodeError,
    ProvisioningError,
    ProvisioningStore,
    client_env,
    provisioning_path,
)
from remote_display.rate_limit import RateLimiter
from remote_display.resource_stats import TrafficCounter, traffic_kind
from remote_display.screenshot_uploads import WIRE_VERSION as SCREENSHOT_UPLOAD_VERSION
from remote_display.server_rendering import CLOCK_SCREENS
from remote_display.screenshot_uploads import ScreenshotInbox, UploadRejected, parse_time, upload_dir
from remote_display.registry import (
    Assignment,
    AssignmentLookup,
    ClientRecord,
    ClientRegistry,
    ClientDisabledError,
    RegistryError,
    UnknownClientError,
    UnknownLeaseError,
)

MAX_REQUEST_BYTES = 64 * 1024
IMMUTABLE_MAX_AGE_SECONDS = 365 * 24 * 3600
MAINTENANCE_INTERVAL_SECONDS = 600
# A render worker still busy this long after the render timeout is replaced.
RENDER_KILL_GRACE_SECONDS = 5
# Routine status changes (last seen, current screen) reach the config UI's
# registry snapshot at most this often; registrations, lease changes and
# deliveries are written at once.  Clients sync every 30 s, so a throttled
# change is written by a later request well within a sync interval.
SNAPSHOT_MIN_INTERVAL_SECONDS = 20
_PROJECT_ROOT = Path(__file__).resolve().parent

WEB_LOGGER = logging.getLogger("desk_display.display_server")


@dataclass
class DisplayServerConfig:
    auth_token: str | None = None
    admin_token: str | None = None
    allow_unauthenticated: bool = False
    lease_seconds: int = 300
    sync_interval_seconds: int = 30
    static_clients: Mapping[str, str] = field(default_factory=dict)
    artifact_dir: Path = _PROJECT_ROOT / "cache" / "artifacts"
    artifact_retention_seconds: float = 24 * 3600
    artifact_max_bytes: int | None = None
    server_cache_max_bytes: int = 1536 << 20
    image_cache_max_bytes: int = 256 << 20
    render_workers: int = 2
    render_timeout_seconds: float = 30
    render_min_interval_seconds: float = 30
    public_url: str | None = None
    playlist_store_path: Path | None = None
    registry_snapshot_path: Path | None = None
    # "provisioned": each client registers with its own credential from the
    # provisioning store. "shared": every client presents auth_token.
    enrollment: str = "provisioned"
    clients_path: Path | None = None
    # Update/restart commands queued by the config UI's Display Clients page.
    commands_path: Path | None = None
    rate_limits: bool = True
    # Screenshots that clients on other networks upload for the collector;
    # None turns uploads off.
    screenshot_upload_dir: Path | None = None
    screenshot_upload_max_bytes: int = 2 * 1024 * 1024
    screenshot_upload_max_total_bytes: int = 64 * 1024 * 1024
    screenshot_upload_retention_seconds: float = 7 * 24 * 3600

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> DisplayServerConfig:
        settings = deployment_config.load_settings(Role.SERVER, env)
        return cls(
            auth_token=settings["DESK_DISPLAY_SERVER_AUTH_TOKEN"],
            admin_token=settings["DESK_DISPLAY_SERVER_ADMIN_TOKEN"],
            allow_unauthenticated=bool(settings["DESK_DISPLAY_SERVER_ALLOW_UNAUTHENTICATED"]),
            lease_seconds=settings["DESK_DISPLAY_CLIENT_LEASE_SECONDS"],
            static_clients=settings["DESK_DISPLAY_STATIC_CLIENTS"] or {},
            artifact_dir=Path(settings["DESK_DISPLAY_ARTIFACT_DIR"] or _PROJECT_ROOT / "cache" / "artifacts").expanduser(),
            artifact_retention_seconds=float(settings["DESK_DISPLAY_ARTIFACT_RETENTION_HOURS"]) * 3600,
            artifact_max_bytes=int(settings["DESK_DISPLAY_ARTIFACT_MAX_MB"]) * 1024 * 1024,
            server_cache_max_bytes=int(settings["DESK_DISPLAY_SERVER_CACHE_MAX_MB"]) << 20,
            image_cache_max_bytes=int(settings["DESK_DISPLAY_IMAGE_CACHE_MAX_MB"]) << 20,
            render_workers=settings["DESK_DISPLAY_RENDER_WORKERS"],
            render_timeout_seconds=float(settings["DESK_DISPLAY_RENDER_TIMEOUT_SECONDS"]),
            render_min_interval_seconds=float(settings["DESK_DISPLAY_RENDER_MIN_INTERVAL_SECONDS"]),
            public_url=settings["DESK_DISPLAY_SERVER_PUBLIC_URL"],
            playlist_store_path=store_path(env),
            registry_snapshot_path=registry_snapshot_path(env),
            enrollment=settings["DESK_DISPLAY_SERVER_ENROLLMENT"],
            clients_path=provisioning_path(env),
            commands_path=commands_path(env),
            rate_limits=bool(settings["DESK_DISPLAY_SERVER_RATE_LIMITS"]),
            screenshot_upload_dir=upload_dir(env) if settings["DESK_DISPLAY_SCREENSHOT_UPLOADS"] else None,
            screenshot_upload_max_bytes=int(settings["DESK_DISPLAY_SCREENSHOT_UPLOAD_MAX_KB"]) * 1024,
            screenshot_upload_max_total_bytes=int(settings["DESK_DISPLAY_SCREENSHOT_UPLOAD_MAX_MB"]) * 1024 * 1024,
            screenshot_upload_retention_seconds=float(settings["DESK_DISPLAY_SCREENSHOT_UPLOAD_RETENTION_DAYS"]) * 86400,
        )


def _token_matches(provided: str | None, expected: str | None) -> bool:
    """Constant-time comparison that tolerates non-ASCII header values."""

    return bool(provided) and bool(expected) and hmac.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    )


def _bearer() -> str | None:
    header = request.headers.get("Authorization", "")
    return header[7:].strip() or None if header.startswith("Bearer ") else None


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def _json_body(name: str | None = None) -> Any:
    if not request.is_json:
        raise ModelValidationError("", "expected an application/json body")
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ModelValidationError("", "body must be a JSON object")
    return payload if name is None else payload.get(name)


def _error(status: int, code: str, message: str, **details: Any):
    return jsonify({"error": code, "message": message, **details}), status


_VOLATILE_SNAPSHOT_FIELDS = frozenset({"last_seen", "lease_expires_at", "status", "resources"})


def _snapshot_signature(snapshot: Mapping[str, Any]) -> Any:
    """The parts of a registry snapshot whose change is written at once."""

    return sorted(
        (client_id, sorted((k, repr(v)) for k, v in entry.items() if k not in _VOLATILE_SNAPSHOT_FIELDS))
        for client_id, entry in (snapshot.get("clients") or {}).items()
    )


def _with_interaction_targets(demand: ClientDemand, capabilities: ClientCapabilities) -> ClientDemand:
    """Add the tiles a touch client can open from its interactive quads.

    Tapping a quad tile shows that screen full screen, so a touch client
    needs those artifacts ready: they become interaction-dependency demand
    (``touch_targets``), rendered and listed in the manifest like any other.
    """

    if not capabilities.has_touch:
        return demand
    from rendering.screen_classes import interaction_targets

    targets = interaction_targets(set(demand.required_screens) | set(demand.alternate_screens))
    if targets <= set(demand.touch_targets):
        return demand
    wire = demand.to_wire()
    wire["touch_targets"] = sorted(targets | set(demand.touch_targets))
    return ClientDemand.from_wire(wire, path="demand")


def create_app(
    config: DisplayServerConfig | None = None,
    *,
    assignments: AssignmentLookup | None = None,
    clock: Callable[[], float] = time.time,
    renderer: Renderer | None = None,
    revisions: RevisionSource | None = None,
    data_health: Callable[[], Mapping[str, Any]] | None = None,
    render_executor: Any = None,
    playlist_documents: Callable[[str, str], Mapping[str, Any] | None] | None = None,
    display_status: Callable[[], Mapping[str, Any]] | None = None,
    vertical_speed_adjustments: Callable[[str], float | None] | None = None,
    client_locations: Callable[[], Mapping[str, Location]] | None = None,
    scoped_revisions: ScopedRevisionSource | None = None,
    located_display_status: Callable[[Location], Mapping[str, Any]] | None = None,
    live_clock: Callable[[str, str, int | None], Any] | None = None,
) -> Flask:
    """Build the API.

    ``assignments`` looks up each client's assigned playlist and
    ``playlist_documents(playlist_id, revision)`` its document, which the
    config response carries for the client's playlist cache.  With a
    ``renderer`` and ``revisions`` source the app also owns a
    :class:`RenderCoordinator` (``app.extensions["desk_display_render_coordinator"]``)
    that the caller ticks or starts. ``display_status()`` returns the feed
    summary (:func:`remote_display.display_status.feed_summary`) each
    heartbeat response carries for the client's side displays.
    ``vertical_speed_adjustments(client_id)`` returns a display's own
    vertical scroll adjustment (set on the Clients page), which its manifest
    carries, or None to keep the global one.  ``client_locations()`` returns
    the displays that have their own weather location (also set on the
    Clients page); their weather and astronomy screens are rendered once per
    location with revisions from ``scoped_revisions``, and their heartbeat
    feed summary comes from ``located_display_status(location)``.
    ``live_clock(screen_id, profile_id, colors_seed)`` draws a clock face at
    the current time for clients that cannot draw it themselves.
    """

    config = config or DisplayServerConfig.from_env()
    if vertical_speed_adjustments is None and config.playlist_store_path is not None:
        vertical_speed_adjustments = PlaylistStore(config.playlist_store_path).vertical_speed_adjustment
    if client_locations is None and config.playlist_store_path is not None:
        client_locations = PlaylistStore(config.playlist_store_path).client_locations

    try:
        server_location = Location.parse(os.environ.get("WEATHER_LATITUDE"), os.environ.get("WEATHER_LONGITUDE"))
    except LocationError:
        server_location = None

    def location_of(client_id: str) -> Location | None:
        if client_locations is None:
            return None
        try:
            location = client_locations().get(client_id)
        except (OSError, ValueError, PlaylistStoreError):
            WEB_LOGGER.warning("Could not read display locations", exc_info=True)
            return None
        # A display set to the server's own location shares the server's renders.
        return None if location == server_location else location

    def client_screen_scopes(client_id: str, screens: Iterable[str]) -> Mapping[str, str]:
        screens = list(screens)
        # Hyper's traffic screen shows the outbound segments (services/traffic.py).
        return {**screen_scopes(location_of(client_id), screens), **road_traffic.screen_scopes(client_id, screens)}

    if assignments is None and config.playlist_store_path is not None:
        store = PlaylistStore(config.playlist_store_path)

        def assignments(client_id: str) -> Assignment | None:
            stored = store.assignment_for(client_id)
            if stored is None:
                return None
            return Assignment(stored.playlist_id, stored.playlist_revision, stored.screens, stored.alternates)

        if playlist_documents is None:
            def playlist_documents(playlist_id: str, playlist_revision: str) -> Mapping[str, Any] | None:
                playlist = store.snapshot()["playlists"].get(playlist_id)
                if playlist is None or playlist.get("revision") != playlist_revision:
                    return None
                return playlist["document"]

    registry = ClientRegistry(
        lease_seconds=config.lease_seconds,
        sync_interval_seconds=config.sync_interval_seconds,
        static_clients=dict(config.static_clients),
        assignments=assignments or (lambda _client_id: None),
        assignments_managed=assignments is not None,
        clock=clock,
    )
    artifacts = ArtifactStore(
        config.artifact_dir,
        grace_seconds=config.artifact_retention_seconds,
        max_bytes=config.artifact_max_bytes,
        clock=clock,
    )
    app = Flask(__name__)
    # Playlist documents are ordered: screen order is play order.
    app.json.sort_keys = False
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    app.extensions["desk_display_registry"] = registry
    app.extensions["desk_display_config"] = config
    app.extensions["desk_display_artifacts"] = artifacts
    provisioning = None if config.clients_path is None else ProvisioningStore(config.clients_path, clock=clock)
    app.extensions["desk_display_provisioning"] = provisioning
    commands = None if config.commands_path is None else CommandStore(config.commands_path, clock=clock)
    app.extensions["desk_display_client_commands"] = commands
    limiter = RateLimiter() if config.rate_limits else None
    inbox = None if config.screenshot_upload_dir is None else ScreenshotInbox(
        config.screenshot_upload_dir,
        max_image_bytes=config.screenshot_upload_max_bytes,
        max_total_bytes=config.screenshot_upload_max_total_bytes,
        retention_seconds=config.screenshot_upload_retention_seconds,
        clock=clock,
    )
    app.extensions["desk_display_screenshot_inbox"] = inbox
    app.extensions["desk_display_rate_limiter"] = limiter
    # Bytes each display sent and received, for the config UI's Stats page.
    traffic = TrafficCounter()
    app.extensions["desk_display_traffic"] = traffic
    provisioned = config.enrollment == "provisioned"
    coordinator = None
    if renderer is not None and revisions is not None:
        coordinator = RenderCoordinator(
            registry,
            artifacts,
            renderer,
            revisions,
            workers=config.render_workers,
            timeout_seconds=config.render_timeout_seconds,
            min_interval_seconds=config.render_min_interval_seconds,
            data_health=data_health,
            executor=render_executor,
            clock=clock,
            screen_scopes=client_screen_scopes if scoped_revisions is not None else None,
            scoped_revisions=scoped_revisions,
        )
    app.extensions["desk_display_render_coordinator"] = coordinator
    app.extensions["desk_display_location_of"] = location_of

    def maintenance() -> list[str]:
        """Release references of clients that went away, then collect garbage."""

        registry.expire()
        now = clock()
        keep = {r.client_id for r in registry.records() if r.lease_state(now) in {"active", "static"}}
        artifacts.prune_holders(keep)
        if inbox is not None:
            try:
                inbox.prune()
            except OSError as exc:
                WEB_LOGGER.warning("Could not prune uploaded client screenshots: %s", exc)
        return artifacts.collect_garbage()

    app.extensions["desk_display_maintenance"] = maintenance

    # ── Plumbing ───────────────────────────────────────────────────────────

    def _forget_removed() -> None:
        """Drop displays removed from the Clients page (or the admin API).

        A display that registered again after its removal (possible only with
        a shared server token) is a new registration and stays.
        """

        if provisioning is None:
            return
        try:
            removed = provisioning.removed()
        except (OSError, ValueError, ProvisioningError) as exc:
            WEB_LOGGER.warning("Could not read removed clients: %s", exc)
            return
        forgotten = False
        for client_id, removed_at in removed.items():
            record = registry.get(client_id)
            if record is not None and (record.registered_at or 0) <= removed_at and registry.forget(client_id):
                WEB_LOGGER.info("client %s was removed; dropped its lease and status", client_id)
                forgotten = True
        if forgotten:
            _publish_snapshot()

    @app.before_request
    def _start() -> None:
        g.started = time.perf_counter()
        registry.expire()
        _forget_removed()

    published: dict[str, Any] = {"at": None, "signature": None}

    def _publish_snapshot() -> None:
        """Write the registry snapshot, throttling writes of routine status alone.

        Every heartbeat changes ``last_seen``; writing (and fsyncing) the file
        for each one wears a Pi's SD card for no visible gain.
        """

        if config.registry_snapshot_path is None:
            return
        snapshot = registry.snapshot()
        signature = _snapshot_signature(snapshot)
        now = clock()
        last = published["at"]
        if (
            signature == published["signature"]
            and last is not None
            and 0 <= now - last < SNAPSHOT_MIN_INTERVAL_SECONDS
        ):
            return
        try:
            registry.write_snapshot(config.registry_snapshot_path, snapshot)
        except OSError as exc:
            WEB_LOGGER.warning("Could not write client registry snapshot: %s", exc)
            return
        published["at"], published["signature"] = now, signature

    _publish_snapshot()

    @app.after_request
    def _finish(response):
        if request.path.startswith("/api/v1/") and request.path != "/api/v1/health":
            traffic.record(g.get("client_id"), traffic_kind(request.endpoint),
                           request.content_length or 0, response.content_length or 0)
        # Registrations, heartbeats, deliveries and admin changes alter what the
        # config UI shows; health checks and asset downloads do not.
        if (
            response.status_code < 400
            and request.path.startswith("/api/v1/")
            and request.path != "/api/v1/health"
            and "/artifacts/" not in request.path
            and not request.path.endswith("/screenshots")
            and "/clock/" not in request.path
            and request.path != "/api/v1/admin/status"
        ):
            _publish_snapshot()
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers["X-Content-Type-Options"] = "nosniff"
        deployment_config.redact_response(response)
        # A client showing the nixie clock asks for it every second.
        quiet = "/clock/" in request.path and response.status_code < 400
        WEB_LOGGER.log(
            logging.DEBUG if quiet else logging.INFO,
            "%s %s -> %s %dms",
            request.method,
            request.path,
            response.status_code,
            int((time.perf_counter() - g.get("started", time.perf_counter())) * 1000),
        )
        return response

    @app.errorhandler(ModelValidationError)
    def _invalid(exc: ModelValidationError):
        if isinstance(exc, UnsupportedCapabilitiesError):
            return _error(409, "unsupported_capabilities", str(exc), field=exc.path)
        return _error(400, "invalid_payload", str(exc), field=exc.path)

    @app.errorhandler(IncompatibleClientError)
    def _incompatible(exc: IncompatibleClientError):
        return jsonify(exc.as_response()), 409

    @app.errorhandler(RegistryError)
    def _registry_error(exc: RegistryError):
        if exc.status == 401:
            _auth_failed()
        return jsonify(exc.as_response()), exc.status

    @app.errorhandler(ProvisioningError)
    def _provisioning_error(exc: ProvisioningError):
        return jsonify(exc.as_response()), exc.status

    class _RateLimited(Exception):
        def __init__(self, seconds: float) -> None:
            super().__init__("rate limited")
            self.seconds = seconds

    @app.errorhandler(_RateLimited)
    def _rate_limited(exc: _RateLimited):
        response, status = _error(429, "rate_limited", "too many requests; retry later",
                                  retry_after_seconds=max(1, int(exc.seconds + 0.999)))
        response.headers["Retry-After"] = str(max(1, int(exc.seconds + 0.999)))
        return response, status

    def _remote() -> str:
        return request.remote_addr or "unknown"

    def _limit(kind: str, key: str) -> None:
        if limiter is not None and (wait := limiter.hit(kind, key)) > 0:
            raise _RateLimited(wait)

    def _auth_failed() -> None:
        if limiter is not None:
            limiter.hit("auth_failure", _remote())

    def _check_auth_lockout() -> None:
        if limiter is not None and (wait := limiter.retry_after("auth_failure", _remote())) > 0:
            raise _RateLimited(wait)

    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException):
        code = (exc.name or "error").lower().replace(" ", "_")
        return _error(exc.code or 500, code, exc.description or exc.name)

    def _client_id(raw: str) -> str:
        try:
            return identifier(raw, "client_id")
        except ModelValidationError:
            raise UnknownLeaseError("unknown client or credential") from None

    _LIMIT_KIND = {"heartbeat": "heartbeat", "client_config": "manifest", "client_manifest": "manifest",
                   "client_artifact": "artifact", "client_screenshot": "screenshot", "client_clock": "clock"}

    def _client() -> ClientRecord:
        """Authenticate the per-client credential for the client ID in the URL.

        A lease issued under a provisioned credential that has since been
        rotated, revoked or disabled is refused at once.
        """

        _check_auth_lockout()
        client_id = _client_id(request.view_args["client_id"])
        credential = _bearer()
        if credential is None:
            raise UnknownLeaseError("missing client credential")
        record = registry.authenticate(client_id, credential)
        if provisioned and not config.allow_unauthenticated:
            current = None if provisioning is None else provisioning.current_credential_id(client_id)
            if current is None or record.enrollment_id != current:
                registry.end_lease(client_id)
                raise UnknownLeaseError("this client's credential was rotated, revoked or disabled")
        _limit(_LIMIT_KIND.get(request.endpoint or "", "manifest"), client_id)
        g.client_id = record.client_id
        return record

    def _require_admin():
        _check_auth_lockout()
        if not config.admin_token:
            return _error(403, "admin_disabled", "the admin API is disabled; set DESK_DISPLAY_SERVER_ADMIN_TOKEN")
        if not _token_matches(_bearer(), config.admin_token):
            _auth_failed()
            return _error(401, "unauthorized", "admin token required")
        return None

    def _assignment(client_id: str) -> Assignment | None:
        return registry.assignments(client_id)

    def _manifest(record: ClientRecord) -> dict[str, Any]:
        """Build *record*'s manifest and note which artifacts it references."""

        demand = registry.effective_demand(record)
        if demand is not None:
            requested = set(demand.required_screens) | set(demand.alternate_screens)
            interactive = set(demand.touch_targets)
        else:
            requested, interactive = set(), set()
        client_id = record.client_id
        configuration: dict[str, Any] = {
            "lease_seconds": registry.lease_seconds,
            "heartbeat_interval_seconds": registry.heartbeat_interval_seconds,
            "sync_interval_seconds": registry.sync_interval_seconds,
        }
        # Only when set, so displays without their own adjustment keep the
        # manifest revision they had.
        adjustment = None if vertical_speed_adjustments is None else vertical_speed_adjustments(client_id)
        if adjustment is not None:
            configuration["vertical_speed_adjustment"] = adjustment
        manifest = build_client_manifest(
            artifacts,
            client_id=client_id,
            display_profile=record.capabilities.display_profile,
            requested_screens=requested,
            interactive_screens=interactive,
            screen_scopes=client_screen_scopes(client_id, requested | interactive)
            if scoped_revisions is not None else None,
            assignment=_assignment_payload(client_id, delivered=True),
            configuration=configuration,
            artifact_url=lambda name: f"/api/v1/clients/{client_id}/artifacts/{name}",
            now=clock(),
        )
        artifacts.reference(client_id, referenced_hashes(manifest))
        return manifest

    def _lease(record: ClientRecord) -> dict[str, Any]:
        return {
            "lease_seconds": registry.lease_seconds,
            "lease_expires_at": _iso(record.lease_expires_at),
            "heartbeat_interval_seconds": registry.heartbeat_interval_seconds,
            "sync_interval_seconds": registry.sync_interval_seconds,
            # Heartbeats may carry a ``telemetry`` document of these versions.
            "client_telemetry_versions": [ClientTelemetry.WIRE_VERSION],
            # ... and a ``resources`` document (CPU, storage, data) for the Stats page.
            "client_resource_versions": [ClientResources.WIRE_VERSION],
            # Heartbeats may carry ``commands`` (results); see client_commands.
            **({"client_command_versions": [CLIENT_COMMAND_VERSION]} if commands is not None else {}),
            # Clock faces drawn at the current time (see client_clock).
            **({"live_clock_faces": sorted(CLOCK_SCREENS)} if live_clock is not None else {}),
            # Clients may PUT their latest screenshots (see client_screenshot).
            **({"client_screenshot_upload_versions": [SCREENSHOT_UPLOAD_VERSION],
                "screenshot_upload_max_bytes": inbox.max_image_bytes} if inbox is not None else {}),
        }

    def _assignment_payload(client_id: str, *, delivered: bool = False) -> dict[str, Any]:
        """Assignment fields for a response; ``delivered`` records the send."""

        assignment = _assignment(client_id)
        if delivered:
            registry.mark_delivered(client_id, None if assignment is None else assignment.playlist_revision)
        if assignment is None:
            return {"assignment_state": "unassigned", "assigned_playlist": None}
        return {
            "assignment_state": "assigned",
            "assigned_playlist": {
                "playlist_id": assignment.playlist_id,
                "playlist_revision": assignment.playlist_revision,
            },
        }

    # ── Public ─────────────────────────────────────────────────────────────

    @app.get("/api/v1/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/v1/join")
    def join():
        """Trade a one-time join code for the new display's setup script."""

        _check_auth_lockout()
        _limit("register", _remote())
        try:
            issued, ticket = _provisioning_store().redeem_join_ticket(request.form.get("code"))
        except InvalidJoinCodeError:
            _auth_failed()
            raise
        text, _warnings = client_env(issued, ticket.get("server_url") or config.public_url,
                                     allow_insecure_transport=bool(ticket.get("allow_insecure_transport")))
        registry.end_lease(issued.client_id)
        script = registration.join_script(text, issued.client_id, issued.display_profile,
                                          registration.repository_url())
        WEB_LOGGER.info("Join code redeemed for client %s from %s", issued.client_id, _remote())
        # Carries the client's credential: never cached.
        return script, 200, {"Content-Type": "text/x-shellscript; charset=utf-8", "Cache-Control": "no-store"}

    # ── Registration and client endpoints ──────────────────────────────────

    @app.post("/api/v1/register")
    def register():
        _check_auth_lockout()
        _limit("register", _remote())
        payload = _json_body()
        unknown = sorted(set(payload) - {"capabilities", "demand", "client_credential"})
        if unknown:
            raise ModelValidationError(unknown[0], "unknown field")
        raw_caps = payload.get("capabilities")
        if not isinstance(raw_caps, dict):
            raise ModelValidationError("capabilities", "is required")
        # Version negotiation first, so an old client gets the documented 409.
        versions = registration_response({
            "protocol_version": raw_caps.get("protocol_version"),
            "client_software_version": raw_caps.get("client_software_version") or "unknown",
            "client_id": raw_caps.get("client_id") or "unknown",
        })
        capabilities = ClientCapabilities.from_wire(raw_caps, path="capabilities")
        enrollment_id = None
        if config.allow_unauthenticated:
            pass
        elif provisioned:
            enrollment_id = None if provisioning is None else provisioning.verify(capabilities.client_id, _bearer())
            if enrollment_id is None:
                _auth_failed()
                state = None if provisioning is None else (provisioning.get(capabilities.client_id) or {}).get("state")
                if state == "disabled":
                    raise ClientDisabledError("this client has been disabled by an administrator")
                return _error(401, "unauthorized", "this client's provisioned credential is required")
            existing = registry.get(capabilities.client_id)
            if existing is not None and existing.enrollment_id not in (None, enrollment_id):
                registry.end_lease(capabilities.client_id)  # issued under a rotated credential
        elif not _token_matches(_bearer(), config.auth_token):
            _auth_failed()
            return _error(401, "unauthorized", "server token required")
        demand = None
        if payload.get("demand") is not None:
            demand = _with_interaction_targets(ClientDemand.from_wire(payload["demand"], path="demand"),
                                               capabilities)
        credential = payload.get("client_credential")
        if credential is not None and not isinstance(credential, str):
            raise ModelValidationError("client_credential", "must be a string")
        registration = registry.register(capabilities, demand, credential=credential,
                                         enrollment_id=enrollment_id, address=request.remote_addr)
        record = registration.record
        WEB_LOGGER.info(
            "client %s %s (profile %s)",
            record.client_id,
            "renewed its lease" if registration.renewed else "registered",
            record.capabilities.display_profile,
        )
        body = {
            **versions,
            "accepted_protocol_versions": sorted(SERVER_ACCEPTED_CLIENT_PROTOCOL_VERSIONS),
            "accepted_render_package_versions": sorted(CLIENT_SUPPORTED_RENDER_PACKAGE_SCHEMA_VERSIONS),
            "client_id": record.client_id,
            "static_client": record.static,
            "renewed": registration.renewed,
            **_assignment_payload(record.client_id, delivered=True),
            "manifest_revision": _manifest(record)["manifest_revision"],
            **_lease(record),
            "client_credential": registration.credential,
        }
        return jsonify(body), 200 if registration.renewed else 201

    @app.post("/api/v1/clients/<client_id>/heartbeat")
    def heartbeat(client_id: str):
        record = _client()
        payload = _json_body()
        unknown = sorted(set(payload) - {"status", "demand", "telemetry", "resources", "commands"})
        if unknown:
            raise ModelValidationError(unknown[0], "unknown field")
        status = ClientStatus.from_wire(payload.get("status"), path="status")
        demand = None
        if payload.get("demand") is not None:
            demand = _with_interaction_targets(ClientDemand.from_wire(payload["demand"], path="demand"),
                                               record.capabilities)
        telemetry = None
        if payload.get("telemetry") is not None:
            telemetry = ClientTelemetry.from_wire(payload["telemetry"], path="telemetry")
        resources = None
        if payload.get("resources") is not None:
            resources = ClientResources.from_wire(payload["resources"], path="resources")
        command_results = None
        if payload.get("commands") is not None:
            command_results = parse_heartbeat_commands(payload["commands"])
        record = registry.heartbeat(record.client_id, _bearer() or "", status, demand, telemetry,
                                    address=request.remote_addr, resources=resources)
        body = {
            "client_id": record.client_id,
            **_assignment_payload(record.client_id, delivered=True),
            "manifest_revision": _manifest(record)["manifest_revision"],
            **_lease(record),
        }
        if command_results is not None and commands is not None:
            # Only a client that speaks the command protocol is sent commands.
            try:
                commands.record_results(record.client_id, command_results)
                body["commands"] = commands.take_pending(record.client_id)
            except OSError as exc:
                WEB_LOGGER.warning("Client command store unavailable: %s", exc)
        location = location_of(record.client_id) if located_display_status is not None else None
        if display_status is not None or location is not None:
            try:
                summary = located_display_status(location) if location is not None else display_status()
            except Exception:  # noqa: BLE001 - the heartbeat must not fail over a summary
                WEB_LOGGER.debug("Display status summary failed", exc_info=True)
            else:
                if summary:
                    body["display_status"] = deployment_config.scrub_secrets(dict(summary))
        return jsonify(body)

    def _playlist_payload(assignment_payload: Mapping[str, Any]) -> dict[str, Any] | None:
        """The assigned playlist document, for the client's playlist cache."""

        assigned = assignment_payload.get("assigned_playlist")
        if not assigned or playlist_documents is None:
            return None
        document = playlist_documents(assigned["playlist_id"], assigned["playlist_revision"])
        if document is None:
            return None
        return {
            "playlist_id": assigned["playlist_id"],
            "playlist_revision": assigned["playlist_revision"],
            "playlist_schema_version": PLAYLIST_SCHEMA_VERSION,
            "document": document,
        }

    @app.get("/api/v1/clients/<client_id>/config")
    def client_config(client_id: str):
        record = _client()
        assignment = _assignment_payload(record.client_id, delivered=True)
        return jsonify({
            **deployment_config.scrub_secrets({
                "client_config_schema_version": CLIENT_CONFIG_SCHEMA_VERSION,
                "client_id": record.client_id,
                "display_profile": record.capabilities.display_profile,
                **assignment,
                **_lease(record),
            }),
            # Playlist documents hold only screen IDs and scheduling; they are
            # validated on save and never carry settings or credentials.
            "playlist": _playlist_payload(assignment),
        })

    @app.get("/api/v1/clients/<client_id>/manifest")
    def client_manifest(client_id: str):
        record = _client()
        manifest = _manifest(record)
        etag = manifest["manifest_revision"]
        if request.if_none_match.contains(etag):
            response = app.response_class(status=304)
        else:
            response = jsonify(manifest)
        response.set_etag(etag)
        response.headers["Cache-Control"] = "private, no-cache"
        return response

    @app.get("/api/v1/clients/<client_id>/artifacts/<name>")
    def client_artifact(client_id: str, name: str):
        """Serve an immutable artifact this client's manifests referenced."""

        record = _client()
        found = artifacts.open_object(name)
        sha = name.split(".", 1)[0]
        if found is None or sha not in artifacts.referenced_by(record.client_id):
            return _error(404, "not_found", "no such artifact")
        path, media_type = found
        if request.if_none_match.contains(sha):
            response = app.response_class(status=304)
            response.set_etag(sha)
        else:
            # conditional=True answers Range and If-Range, so an interrupted
            # download can resume from where it stopped.
            response = send_file(path, mimetype=media_type, conditional=True, etag=sha,
                                 max_age=IMMUTABLE_MAX_AGE_SECONDS, last_modified=None)
        response.headers["Cache-Control"] = f"private, max-age={IMMUTABLE_MAX_AGE_SECONDS}, immutable"
        return response

    @app.get("/api/v1/clients/<client_id>/clock/<screen_id>.png")
    def client_clock(client_id: str, screen_id: str):
        """The date or nixie face at the current time, for this client's profile.

        For clients that show images but cannot draw the time from a clock
        package. ``colors`` (an integer seed) keeps the date face's colours
        steady across one showing. Never cached.
        """

        record = _client()
        if live_clock is None or screen_id not in CLOCK_SCREENS:
            return _error(404, "not_found", "no such clock face")
        seed = None
        raw = request.args.get("colors")
        if raw is not None:
            if not raw.isdigit() or len(raw) > 10:
                raise ModelValidationError("colors", "must be a non-negative integer")
            seed = int(raw)
        try:
            image = live_clock(screen_id, record.capabilities.display_profile, seed)
        except Exception:  # noqa: BLE001 - the client falls back to the cached still
            WEB_LOGGER.warning("Live %s clock for %s failed", screen_id, record.client_id, exc_info=True)
            return _error(503, "clock_unavailable", "the clock face could not be drawn")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        response = app.response_class(buffer.getvalue(), mimetype="image/png")
        response.headers["Cache-Control"] = "no-store"
        return response

    # ── Admin ──────────────────────────────────────────────────────────────

    @app.put("/api/v1/clients/<client_id>/screenshots")
    def client_screenshot(client_id: str):
        """Keep a client's latest screenshot of one screen for the collector.

        ``?screen=<screen id>&captured_at=<ISO time>`` with an ``image/png``
        body.  Displays on another network upload here because
        ``scripts/collect_client_screenshots.py`` cannot reach them.
        """

        record = _client()
        if inbox is None:
            return _error(404, "not_found", "screenshot uploads are turned off on this server")
        if request.mimetype != "image/png":
            return _error(415, "unsupported_media_type", "expected an image/png body")
        length = request.content_length
        if length is None:
            return _error(411, "length_required", "Content-Length is required")
        if length > inbox.max_image_bytes:
            return _error(413, "too_large", f"screenshot exceeds {inbox.max_image_bytes} bytes")
        # Read directly: the app-wide request limit (MAX_REQUEST_BYTES) is
        # for JSON bodies and is far below a screenshot.
        data = request.environ["wsgi.input"].read(length)
        if len(data) != length:
            return _error(400, "invalid_payload", "the body ended early")
        try:
            entry = inbox.save(record.client_id, request.args.get("screen", ""), data,
                               captured_at=parse_time(request.args.get("captured_at")))
        except UploadRejected as exc:
            return jsonify(exc.as_response()), exc.status
        except OSError as exc:
            WEB_LOGGER.warning("Could not store a screenshot from %s: %s", record.client_id, exc)
            return _error(507, "storage_unavailable", "the server could not store the screenshot")
        return jsonify(entry), 201

    @app.get("/api/v1/admin/status")
    def admin_status():
        if (denied := _require_admin()) is not None:
            return denied
        now = clock()
        clients = []
        for record in registry.records():
            clients.append({
                "client_id": record.client_id,
                "static": record.static,
                "lease_state": record.lease_state(now),
                "registered_at": _iso(record.registered_at),
                "last_seen": _iso(record.last_seen),
                "lease_expires_at": _iso(record.lease_expires_at),
                "capabilities": record.capabilities.to_wire(),
                "demand": None if record.demand is None else record.demand.to_wire(),
                "status": None if record.status is None else record.status.to_wire(),
                "telemetry": None if record.telemetry is None else record.telemetry.to_wire(),
                "resources": None if record.resources is None else record.resources.to_wire(),
                "address": record.address,
                **_assignment_payload(record.client_id),
            })
        demand = [
            {
                "source": entry.source,
                "client_id": entry.client_id,
                "display_profile": entry.capabilities.display_profile,
                "screens": list(entry.demand.all_screens),
            }
            for entry in registry.demand_entries()
        ]
        return jsonify(deployment_config.scrub_secrets({
            "server_time": _iso(now),
            "lease_seconds": registry.lease_seconds,
            "clients": clients,
            "demand": demand,
            "artifacts": artifacts.stats(),
        }))

    @app.get("/api/v1/admin/render-status")
    def admin_render_status():
        if (denied := _require_admin()) is not None:
            return denied
        if coordinator is None:
            return _error(404, "rendering_disabled", "this server has no render coordinator")
        return jsonify(deployment_config.scrub_secrets(coordinator.status()))

    @app.put("/api/v1/admin/prerender/<name>")
    def admin_put_prerender(name: str):
        if (denied := _require_admin()) is not None:
            return denied
        payload = _json_body()
        unknown = sorted(set(payload) - {"display_profile", "screens"})
        if unknown:
            raise ModelValidationError(unknown[0], "unknown field")
        screens = payload.get("screens")
        if not isinstance(screens, list):
            raise ModelValidationError("screens", "must be a list")
        entry = registry.add_prerender(name, payload.get("display_profile"), screens)
        return jsonify({
            "name": name,
            "display_profile": entry.capabilities.display_profile,
            "screens": list(entry.demand.all_screens),
        })

    @app.delete("/api/v1/admin/prerender/<name>")
    def admin_delete_prerender(name: str):
        if (denied := _require_admin()) is not None:
            return denied
        if not registry.remove_prerender(name):
            return _error(404, "not_found", "no such pre-render entry")
        return "", 204

    def _provisioning_store() -> ProvisioningStore:
        if provisioning is None:
            raise ProvisioningError("no provisioning store is configured (DESK_DISPLAY_SERVER_CLIENTS_PATH)")
        return provisioning

    def _issued(issued, status: int):
        text, warnings = client_env(issued, config.public_url)
        response = jsonify({
            "client_id": issued.client_id,
            "display_profile": issued.display_profile,
            "client_credential": issued.credential,
            "client_env": text,
            "warnings": warnings,
            "note": "The credential is shown only now; store it on the client.",
        })
        # The one response allowed to carry a secret: never cached, never logged.
        response.headers["Cache-Control"] = "no-store"
        return response, status

    @app.get("/api/v1/admin/clients")
    def admin_list_clients():
        if (denied := _require_admin()) is not None:
            return denied
        return jsonify({"enrollment": config.enrollment, "clients": _provisioning_store().records()})

    @app.post("/api/v1/admin/clients")
    def admin_provision_client():
        if (denied := _require_admin()) is not None:
            return denied
        payload = _json_body()
        unknown = sorted(set(payload) - {"client_id", "display_profile", "playlist_id"})
        if unknown:
            raise ModelValidationError(unknown[0], "unknown field")
        playlist_id = payload.get("playlist_id")
        if playlist_id is not None and config.playlist_store_path is None:
            raise ModelValidationError("playlist_id", "this server has no playlist store")
        if playlist_id is not None and playlist_id not in PlaylistStore(config.playlist_store_path).snapshot()["playlists"]:
            raise ModelValidationError("playlist_id", "unknown playlist")
        issued = _provisioning_store().provision(payload.get("client_id"), payload.get("display_profile"),
                                                 actor="admin-api")
        if playlist_id is not None:
            PlaylistStore(config.playlist_store_path).assign(issued.client_id, playlist_id,
                                                             expected_playlist_id=None, actor="admin-api")
        return _issued(issued, 201)

    @app.post("/api/v1/admin/clients/<client_id>/<action>")
    def admin_client_action(client_id: str, action: str):
        if (denied := _require_admin()) is not None:
            return denied
        if action not in {"disable", "enable", "rotate", "revoke", "remove"}:
            return _error(404, "not_found", "unknown action")
        client_id = _client_id(client_id)
        if action == "remove":
            record = registry.get(client_id)
            if record is not None and record.static:
                return _error(409, "static_client", "remove this client from DESK_DISPLAY_STATIC_CLIENTS instead")
            _provisioning_store().remove(client_id, actor="admin-api")
            registry.end_lease(client_id)
            registry.forget(client_id)
            if config.playlist_store_path is not None:
                PlaylistStore(config.playlist_store_path).forget_client(client_id, actor="admin-api")
            if commands is not None:
                commands.forget(client_id)
            if inbox is not None:
                inbox.forget(client_id)
            return jsonify({"client_id": client_id, "removed": True})
        known = provisioning is not None and provisioning.get(client_id) is not None
        if action == "rotate":
            issued = _provisioning_store().rotate(client_id, actor="admin-api")
            registry.end_lease(client_id)
            return _issued(issued, 200)
        if action == "revoke":
            record = _provisioning_store().revoke(client_id, actor="admin-api")
            registry.end_lease(client_id)
            return jsonify({"client_id": client_id, "state": record["state"]})
        disabled = action == "disable"
        if known:
            provisioning.set_disabled(client_id, disabled, actor="admin-api")
        try:
            record = registry.set_disabled(client_id, disabled)
        except UnknownClientError:
            if not known:
                raise
            return jsonify({"client_id": client_id, "disabled": disabled})
        return jsonify({"client_id": record.client_id, "disabled": record.disabled})

    return app


def run_display_server() -> None:
    logging.basicConfig(
        level=deployment_config.resolve_log_level(),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    deployment_config.install_secret_log_redaction()
    if deployment_config.resolve_role() is not Role.SERVER:
        raise SystemExit("display_server.py requires DESK_DISPLAY_ROLE=server")
    deployment_config.startup_check("display server")
    settings = deployment_config.load_settings(Role.SERVER)
    from remote_display.server_rendering import LiveClock, ServerRendering
    from remote_display.display_status import feed_summary
    from services.server_feeds import ServerFeedService

    from paths import resolve_cache_file_path

    from screens.ncaa_fbs_scoreboard import download_missing_team_logos

    feeds = ServerFeedService(
        state_path=str(resolve_cache_file_path("DESK_DISPLAY_SERVER_FEED_STATE_PATH", "server_feed_state.json")),
        download_ncaa_fbs_logos=download_missing_team_logos,
    )
    from rendering.profile_process import ProfileProcessPool

    render_timeout = float(settings["DESK_DISPLAY_RENDER_TIMEOUT_SECONDS"])
    # Start-up (imports, fonts) gets several render timeouts. A render still
    # running shortly after the coordinator gives up on it is discarded
    # anyway, so its worker is replaced then, freeing the render slot rather
    # than holding it for minutes.
    # One process per render worker for each profile, so renders for the same
    # profile run as concurrently as the coordinator allows.
    workers = ProfileProcessPool(timeout_seconds=max(60.0, 4 * render_timeout),
                                 render_timeout_seconds=render_timeout + RENDER_KILL_GRACE_SECONDS,
                                 processes_per_profile=settings["DESK_DISPLAY_RENDER_WORKERS"])
    rendering = ServerRendering(feeds=feeds, profile_processes=workers)
    atexit.register(rendering.close)
    # Started on first use: only clients that cannot draw the time ask.
    live_clock = LiveClock(ProfileProcessPool(timeout_seconds=max(60.0, 4 * render_timeout),
                                              render_timeout_seconds=render_timeout + RENDER_KILL_GRACE_SECONDS,
                                              processes_per_profile=1))
    atexit.register(live_clock.close)
    server_config = DisplayServerConfig.from_env()
    # Before create_app, which rewrites the registry snapshot the seed reads.
    _apply_location_seed(server_config)
    app = create_app(
        server_config,
        renderer=rendering.render,
        revisions=rendering.revisions,
        scoped_revisions=rendering.scoped_revisions,
        data_health=rendering.health,
        display_status=lambda: feed_summary(feeds.data.snapshot().values),
        located_display_status=lambda location: feed_summary(
            scoped_values(feeds.data.snapshot().values, location.scope)),
        live_clock=live_clock.render,
    )
    _start_maintenance(_with_cache_budgets(app.extensions["desk_display_maintenance"], server_config))
    _start_stats(app, server_config)
    registry = app.extensions["desk_display_registry"]
    global_screens, located_screens = demand_by_location(registry, app.extensions["desk_display_location_of"])
    feeds.start(global_screens, demanded_locations=located_screens)
    app.extensions["desk_display_render_coordinator"].start()
    host, port = settings["DESK_DISPLAY_SERVER_HOST"], settings["DESK_DISPLAY_SERVER_PORT"]
    cert, key = settings["DESK_DISPLAY_SERVER_TLS_CERT"], settings["DESK_DISPLAY_SERVER_TLS_KEY"]
    WEB_LOGGER.info("Display server listening on %s:%s", host, port)
    if cert and key:
        from werkzeug.serving import run_simple

        run_simple(host, port, app, ssl_context=(cert, key), threaded=True)
        return
    from waitress import serve

    serve(app, host=host, port=port, threads=8)


def _start_stats(app: Flask, config: DisplayServerConfig) -> None:
    """Sample CPU by purpose, traffic and storage for the config UI's Stats page."""

    from remote_display import resource_stats

    if not resource_stats.stats_enabled():
        return
    registry = app.extensions["desk_display_registry"]
    persist = resource_stats.history_path()
    sampler = resource_stats.StatsSampler(
        role="server",
        traffic=app.extensions["desk_display_traffic"],
        clients=lambda: registry.snapshot()["clients"],
        storage={
            "Artifact store": (config.artifact_dir, config.artifact_max_bytes),
            "Server caches (cache/)": (_PROJECT_ROOT / "cache", config.server_cache_max_bytes),
            "Image caches (images/cache/)": (_PROJECT_ROOT / "images" / "cache", config.image_cache_max_bytes),
            **({"Uploaded client screenshots": (config.screenshot_upload_dir,
                                                config.screenshot_upload_max_total_bytes)}
               if config.screenshot_upload_dir is not None else {}),
        },
        persisted=resource_stats.load_persisted(persist),
    )
    publisher = resource_stats.StatsPublisher(sampler, live_path=resource_stats.stats_path(), persist_path=persist,
                                              reset_path=resource_stats.reset_request_path())
    publisher.start()
    atexit.register(publisher.stop)
    _exit_cleanly_on_sigterm()


def _exit_cleanly_on_sigterm() -> None:
    """Turn systemd's SIGTERM into a normal exit, so atexit handlers run.

    Python's default SIGTERM action ends the process at once, skipping atexit:
    the Stats history saved on exit (and the render workers' shutdown) would
    never happen on ``systemctl restart``.
    """

    import signal

    if threading.current_thread() is not threading.main_thread():
        return
    if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))


def _apply_location_seed(config: DisplayServerConfig) -> None:
    """Give the known displays their locations once (remote_display/location_seed.py)."""

    from remote_display import location_seed
    from remote_display.registry import read_snapshot

    if config.playlist_store_path is None:
        return
    try:
        store = PlaylistStore(config.playlist_store_path)
        provisioning = None if config.clients_path is None else ProvisioningStore(config.clients_path)
        snapshot = read_snapshot(config.registry_snapshot_path) if config.registry_snapshot_path else {}
        clients = location_seed.known_clients(
            store, snapshot,
            provisioned=[r["client_id"] for r in provisioning.records()] if provisioning is not None else (),
            static=config.static_clients,
        )
        changed = location_seed.apply(store, clients)
    except (OSError, ValueError, PlaylistStoreError, ModelValidationError):
        WEB_LOGGER.warning("Could not apply the display locations", exc_info=True)
        return
    if changed:
        WEB_LOGGER.info("Set weather locations for %s", ", ".join(changed))


def demand_by_location(
    registry: ClientRegistry, location_of: Callable[[str], Location | None]
) -> tuple[Callable[[], set[str]], Callable[[], dict[Location, set[str]]]]:
    """Feed demand split by place: the server's own location, and each display's.

    A display with its own location demands its weather screens for that
    location only, so the server's weather is not fetched for it alone.
    """

    def split() -> tuple[set[str], dict[Location, set[str]]]:
        located = location_screens()
        own: set[str] = set()
        places: dict[Location, set[str]] = {}
        for entry in registry.demand_entries():
            screens = set(entry.demand.all_screens)
            location = location_of(entry.client_id)
            if location is None:
                own |= screens
                continue
            own |= screens - located
            places.setdefault(location, set()).update(screens & located)
        return own, places

    return (lambda: split()[0]), (lambda: split()[1])


def _with_cache_budgets(maintenance: Callable[[], list[str]],
                        config: DisplayServerConfig) -> Callable[[], list[str]]:
    """Also keep images/cache/ within its limit, and warn when cache/ is over its own."""

    from remote_display.cache_budget import BudgetWarning, prune_to_budget

    server_caches = BudgetWarning(_PROJECT_ROOT / "cache", config.server_cache_max_bytes,
                                  what="Server caches")

    def run() -> list[str]:
        removed = maintenance()
        try:
            prune_to_budget(_PROJECT_ROOT / "images" / "cache", config.image_cache_max_bytes)
            server_caches.check()
        except Exception:  # pragma: no cover - logged and retried
            WEB_LOGGER.exception("Cache budget check failed")
        return removed

    return run


def _start_maintenance(maintenance: Callable[[], list[str]]) -> threading.Thread:
    """Collect unreferenced artifacts periodically in the background."""

    def loop() -> None:
        while True:
            try:
                maintenance()
            except Exception:  # pragma: no cover - logged and retried
                WEB_LOGGER.exception("Artifact maintenance failed")
            time.sleep(MAINTENANCE_INTERVAL_SECONDS)

    thread = threading.Thread(target=loop, name="artifact-maintenance", daemon=True)
    thread.start()
    return thread


def _load_dotenv() -> None:
    """Load ``.env`` beside this file without overriding the environment."""

    if os.environ.get("CONFIG_LOAD_DOTENV", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    path = _PROJECT_ROOT / ".env"
    if path.is_file():
        for key, value in deployment_config.parse_env_file(path).items():
            os.environ.setdefault(key, value)


if __name__ == "__main__":  # pragma: no cover
    _load_dotenv()
    run_display_server()
