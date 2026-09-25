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
    ``GET  /api/v1/clients/<id>/manifest``         current manifest
    ``GET  /api/v1/clients/<id>/assets/<path>``    rendered artifact
    ``GET  /api/v1/health``                        liveness
    ``GET  /api/v1/admin/status``                  clients, leases and demand
    ``PUT|DELETE /api/v1/admin/prerender/<name>``  explicit pre-render demand
    ``POST /api/v1/admin/clients/<id>/disable|enable``
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
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
    build_manifest,
    registration_response,
)
from protocol_versions import CLIENT_CONFIG_SCHEMA_VERSION
from remote_display.models import (
    ClientCapabilities,
    ClientDemand,
    ClientStatus,
    ModelValidationError,
    UnsupportedCapabilitiesError,
    identifier,
)
from remote_display.registry import (
    Assignment,
    AssignmentLookup,
    ClientRecord,
    ClientRegistry,
    RegistryError,
    UnknownLeaseError,
)

MAX_REQUEST_BYTES = 64 * 1024
_ASSET_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_ASSET_DEPTH = 4
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
    public_url: str | None = None

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
            public_url=settings["DESK_DISPLAY_SERVER_PUBLIC_URL"],
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


def create_app(
    config: DisplayServerConfig | None = None,
    *,
    assignments: AssignmentLookup | None = None,
    clock: Callable[[], float] = time.time,
) -> Flask:
    """Build the API.  ``assignments`` looks up each client's assigned playlist."""

    config = config or DisplayServerConfig.from_env()
    registry = ClientRegistry(
        lease_seconds=config.lease_seconds,
        sync_interval_seconds=config.sync_interval_seconds,
        static_clients=dict(config.static_clients),
        assignments=assignments or (lambda _client_id: None),
        clock=clock,
    )
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    app.extensions["desk_display_registry"] = registry
    app.extensions["desk_display_config"] = config

    # ── Plumbing ───────────────────────────────────────────────────────────

    @app.before_request
    def _start() -> None:
        g.started = time.perf_counter()
        registry.expire()

    @app.after_request
    def _finish(response):
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

    def _manifest_revision(record: ClientRecord) -> str:
        assignment = _assignment(record.client_id)
        basis = {
            "profile": record.capabilities.display_profile,
            "assignment": None if assignment is None else [assignment.playlist_id, assignment.playlist_revision],
            "screens": [] if record.demand is None else list(record.demand.all_screens),
        }
        digest = hashlib.sha256(json.dumps(basis, sort_keys=True).encode("utf-8")).hexdigest()
        return f"m-{digest[:20]}"

    def _lease(record: ClientRecord) -> dict[str, Any]:
        return {
            "lease_seconds": registry.lease_seconds,
            "lease_expires_at": _iso(record.lease_expires_at),
            "heartbeat_interval_seconds": registry.heartbeat_interval_seconds,
            "sync_interval_seconds": registry.sync_interval_seconds,
        }

    def _assignment_payload(client_id: str) -> dict[str, Any]:
        assignment = _assignment(client_id)
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
            demand = ClientDemand.from_wire(payload["demand"], path="demand")
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
            **_assignment_payload(record.client_id),
            "manifest_revision": _manifest_revision(record),
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
            demand = ClientDemand.from_wire(payload["demand"], path="demand")
        record = registry.heartbeat(record.client_id, _bearer() or "", status, demand)
        return jsonify({
            "client_id": record.client_id,
            **_assignment_payload(record.client_id),
            "manifest_revision": _manifest_revision(record),
            **_lease(record),
        })

    @app.get("/api/v1/clients/<client_id>/config")
    def client_config(client_id: str):
        record = _client()
        return jsonify(deployment_config.scrub_secrets({
            "client_config_schema_version": CLIENT_CONFIG_SCHEMA_VERSION,
            "client_id": record.client_id,
            "display_profile": record.capabilities.display_profile,
            **_assignment_payload(record.client_id),
            **_lease(record),
        }))

    @app.get("/api/v1/clients/<client_id>/manifest")
    def client_manifest(client_id: str):
        record = _client()
        manifest = build_manifest(
            client_id=record.client_id,
            manifest_revision=_manifest_revision(record),
            display_profile=record.capabilities.display_profile,
            logical_width=record.capabilities.logical_width,
            logical_height=record.capabilities.logical_height,
            **_assignment_payload(record.client_id),
            requested_screens=[] if record.demand is None else list(record.demand.all_screens),
            artifacts=[],
            cache_complete=False,
        )
        return jsonify(manifest)

    @app.get("/api/v1/clients/<client_id>/assets/<path:asset>")
    def client_asset(client_id: str, asset: str):
        _client()
        segments = asset.split("/")
        if len(segments) > _MAX_ASSET_DEPTH or not all(_ASSET_SEGMENT_RE.match(s) and ".." not in s for s in segments):
            return _error(404, "not_found", "no such asset")
        root = config.artifact_dir.resolve()
        path = root.joinpath(*segments).resolve()
        if root not in path.parents or not path.is_file():
            return _error(404, "not_found", "no such asset")
        response = send_file(path, conditional=True, etag=True, max_age=0)
        response.headers["Cache-Control"] = "private, no-cache"
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
        }))

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
    app = create_app(DisplayServerConfig.from_env())
    host, port = settings["DESK_DISPLAY_SERVER_HOST"], settings["DESK_DISPLAY_SERVER_PORT"]
    cert, key = settings["DESK_DISPLAY_SERVER_TLS_CERT"], settings["DESK_DISPLAY_SERVER_TLS_KEY"]
    WEB_LOGGER.info("Display server listening on %s:%s", host, port)
    if cert and key:
        from werkzeug.serving import run_simple

        run_simple(host, port, app, ssl_context=(cert, key), threaded=True)
        return
    from waitress import serve

    serve(app, host=host, port=port, threads=8)


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
