#!/usr/bin/env python3
"""Desk Display render server API (``/api/v1``).

A separate service from the screenshot Feed server (``feed_server.py``),
whose endpoints and semantics are unchanged.  Run it with
``DESK_DISPLAY_ROLE=server python3 display_server.py``.

Authentication
    ``POST /api/v1/register`` needs ``Authorization: Bearer <server token>``
    (``DESK_DISPLAY_SERVER_AUTH_TOKEN``).  A successful registration returns a
    per-client ``client_credential``; every ``/api/v1/clients/<id>/...``
    endpoint needs ``Authorization: Bearer <client credential>`` for *that*
    client ID, so one client can never read another's configuration, status
    or manifest.  ``/api/v1/admin/...`` needs ``DESK_DISPLAY_SERVER_ADMIN_TOKEN``
    and is disabled when it is unset.  ``/api/v1/health`` is public and says
    only whether the service is up.

Endpoints
    ``POST /api/v1/register``                      register or renew a lease
    ``POST /api/v1/clients/<id>/heartbeat``        report status, renew lease
    ``GET  /api/v1/clients/<id>/config``           client configuration
    ``GET  /api/v1/clients/<id>/manifest``         current manifest (ETag)
    ``GET  /api/v1/clients/<id>/artifacts/<sha256>.<ext>``
                                                   immutable artifact (ETag, Range)
    ``GET  /api/v1/health``                        liveness
    ``GET  /api/v1/admin/status``                  clients, leases and demand
    ``PUT|DELETE /api/v1/admin/prerender/<name>``  explicit pre-render demand
    ``POST /api/v1/admin/clients/<id>/disable|enable``
    ``GET  /api/v1/admin/render-status``           queue, render and data health
"""
from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
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
from remote_display.manifest import build_client_manifest, referenced_hashes
from remote_display.render_coordinator import RenderCoordinator, Renderer, RevisionSource
from remote_display.models import (
    ClientCapabilities,
    ClientDemand,
    ClientStatus,
    ModelValidationError,
    UnsupportedCapabilitiesError,
    identifier,
)
from remote_display.playlist_store import PlaylistStore, registry_snapshot_path, store_path
from remote_display.registry import (
    Assignment,
    AssignmentLookup,
    ClientRecord,
    ClientRegistry,
    RegistryError,
    UnknownLeaseError,
)

MAX_REQUEST_BYTES = 64 * 1024
IMMUTABLE_MAX_AGE_SECONDS = 365 * 24 * 3600
MAINTENANCE_INTERVAL_SECONDS = 600
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
    render_workers: int = 2
    render_timeout_seconds: float = 30
    render_min_interval_seconds: float = 30
    public_url: str | None = None
    playlist_store_path: Path | None = None
    registry_snapshot_path: Path | None = None

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
            render_workers=settings["DESK_DISPLAY_RENDER_WORKERS"],
            render_timeout_seconds=float(settings["DESK_DISPLAY_RENDER_TIMEOUT_SECONDS"]),
            render_min_interval_seconds=float(settings["DESK_DISPLAY_RENDER_MIN_INTERVAL_SECONDS"]),
            public_url=settings["DESK_DISPLAY_SERVER_PUBLIC_URL"],
            playlist_store_path=store_path(env),
            registry_snapshot_path=registry_snapshot_path(env),
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
) -> Flask:
    """Build the API.

    ``assignments`` looks up each client's assigned playlist and
    ``playlist_documents(playlist_id, revision)`` its document, which the
    config response carries for the client's playlist cache.  With a
    ``renderer`` and ``revisions`` source the app also owns a
    :class:`RenderCoordinator` (``app.extensions["desk_display_render_coordinator"]``)
    that the caller ticks or starts.
    """

    config = config or DisplayServerConfig.from_env()
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
        )
    app.extensions["desk_display_render_coordinator"] = coordinator

    def maintenance() -> list[str]:
        """Release references of clients that went away, then collect garbage."""

        registry.expire()
        now = clock()
        keep = {r.client_id for r in registry.records() if r.lease_state(now) in {"active", "static"}}
        artifacts.prune_holders(keep)
        return artifacts.collect_garbage()

    app.extensions["desk_display_maintenance"] = maintenance

    # ── Plumbing ───────────────────────────────────────────────────────────

    @app.before_request
    def _start() -> None:
        g.started = time.perf_counter()
        registry.expire()

    def _publish_snapshot() -> None:
        if config.registry_snapshot_path is None:
            return
        try:
            registry.write_snapshot(config.registry_snapshot_path)
        except OSError as exc:
            WEB_LOGGER.warning("Could not write client registry snapshot: %s", exc)

    _publish_snapshot()

    @app.after_request
    def _finish(response):
        # Registrations, heartbeats, deliveries and admin changes alter what the
        # config UI shows; health checks and asset downloads do not.
        if (
            response.status_code < 400
            and request.path.startswith("/api/v1/")
            and request.path != "/api/v1/health"
            and "/artifacts/" not in request.path
            and request.path != "/api/v1/admin/status"
        ):
            _publish_snapshot()
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers["X-Content-Type-Options"] = "nosniff"
        deployment_config.redact_response(response)
        WEB_LOGGER.info(
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
        return jsonify(exc.as_response()), exc.status

    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException):
        code = (exc.name or "error").lower().replace(" ", "_")
        return _error(exc.code or 500, code, exc.description or exc.name)

    def _client_id(raw: str) -> str:
        try:
            return identifier(raw, "client_id")
        except ModelValidationError:
            raise UnknownLeaseError("unknown client or credential") from None

    def _client() -> ClientRecord:
        """Authenticate the per-client credential for the client ID in the URL."""

        client_id = _client_id(request.view_args["client_id"])
        credential = _bearer()
        if credential is None:
            raise UnknownLeaseError("missing client credential")
        return registry.authenticate(client_id, credential)

    def _require_admin():
        if not config.admin_token:
            return _error(403, "admin_disabled", "the admin API is disabled; set DESK_DISPLAY_SERVER_ADMIN_TOKEN")
        if not _token_matches(_bearer(), config.admin_token):
            return _error(401, "unauthorized", "admin token required")
        return None

    def _assignment(client_id: str) -> Assignment | None:
        return registry.assignments(client_id)

    def _manifest(record: ClientRecord) -> dict[str, Any]:
        """Build *record*'s manifest and note which artifacts it references."""

        demand = record.demand
        assignment = _assignment(record.client_id)
        if demand is not None:
            requested = set(demand.required_screens) | set(demand.alternate_screens)
            interactive = set(demand.touch_targets)
        elif assignment is not None:
            requested, interactive = set(assignment.screens) | set(assignment.alternates), set()
        else:
            requested, interactive = set(), set()
        client_id = record.client_id
        manifest = build_client_manifest(
            artifacts,
            client_id=client_id,
            display_profile=record.capabilities.display_profile,
            requested_screens=requested,
            interactive_screens=interactive,
            assignment=_assignment_payload(client_id, delivered=True),
            configuration={
                "lease_seconds": registry.lease_seconds,
                "heartbeat_interval_seconds": registry.heartbeat_interval_seconds,
                "sync_interval_seconds": registry.sync_interval_seconds,
            },
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

    # ── Registration and client endpoints ──────────────────────────────────

    @app.post("/api/v1/register")
    def register():
        if not config.allow_unauthenticated and not _token_matches(_bearer(), config.auth_token):
            return _error(401, "unauthorized", "server token required")
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
        demand = None
        if payload.get("demand") is not None:
            demand = _with_interaction_targets(ClientDemand.from_wire(payload["demand"], path="demand"),
                                               capabilities)
        credential = payload.get("client_credential")
        if credential is not None and not isinstance(credential, str):
            raise ModelValidationError("client_credential", "must be a string")
        registration = registry.register(capabilities, demand, credential=credential)
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
        unknown = sorted(set(payload) - {"status", "demand"})
        if unknown:
            raise ModelValidationError(unknown[0], "unknown field")
        status = ClientStatus.from_wire(payload.get("status"), path="status")
        demand = None
        if payload.get("demand") is not None:
            demand = _with_interaction_targets(ClientDemand.from_wire(payload["demand"], path="demand"),
                                               record.capabilities)
        record = registry.heartbeat(record.client_id, _bearer() or "", status, demand)
        return jsonify({
            "client_id": record.client_id,
            **_assignment_payload(record.client_id, delivered=True),
            "manifest_revision": _manifest(record)["manifest_revision"],
            **_lease(record),
        })

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

    # ── Admin ──────────────────────────────────────────────────────────────

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

    @app.post("/api/v1/admin/clients/<client_id>/<action>")
    def admin_client_action(client_id: str, action: str):
        if (denied := _require_admin()) is not None:
            return denied
        if action not in {"disable", "enable"}:
            return _error(404, "not_found", "unknown action")
        record = registry.set_disabled(client_id, action == "disable")
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
    from remote_display.server_rendering import ServerRendering
    from services.server_feeds import ServerFeedService

    from paths import resolve_cache_file_path

    feeds = ServerFeedService(state_path=str(
        resolve_cache_file_path("DESK_DISPLAY_SERVER_FEED_STATE_PATH", "server_feed_state.json")))
    rendering = ServerRendering(feeds=feeds)
    app = create_app(
        DisplayServerConfig.from_env(),
        renderer=rendering.render,
        revisions=rendering.revisions,
        data_health=rendering.health,
    )
    _start_maintenance(app.extensions["desk_display_maintenance"])
    registry = app.extensions["desk_display_registry"]
    feeds.start(lambda: {s for entry in registry.demand_entries() for s in entry.demand.all_screens})
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
