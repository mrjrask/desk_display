"""Configuration UI pages and APIs for server-managed per-client playlists.

Registered on the config UI app by :func:`register`.  The pages are split
from the rotation editor to keep both usable:

``/playlists``  playlist library: create, clone, rename, edit, reorder,
                validate, import, export, guarded delete, the clients using
                each playlist, and a demand preview with capability warnings.
``/clients``    client registry: identity, state, profile, capabilities,
                rotation, version, current screen, cache age, heartbeat,
                assignment, and saved / delivered / acknowledged revisions;
                provisioning, credential rotation, revocation and
                disable/enable (a new credential is shown once, never again);
                update (git pull) and restart buttons that queue a command
                the client collects on its next heartbeat and reports back.
``/clients/add`` guided "Add a display" wizard: checks the server is
                reachable on the LAN, names the display and picks its profile
                and playlist, shows a one-paste install command, and watches
                the new display come online.

All state lives in :class:`remote_display.playlist_store.PlaylistStore`; the
client registry comes from the snapshot the render server publishes.  Every
mutating request must be a JSON request carrying ``X-Requested-With:
desk-display`` (a CSRF guard browsers will not add cross-site), and
responses pass through the config UI's secret redaction.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from flask import Blueprint, jsonify, render_template, request

import deployment_config
from deployment_config import Role
from display_profiles import PROFILE_PRESETS
from remote_display import registration
from remote_display.client_commands import ACTIONS, CommandError, CommandStore, commands_path
from remote_display.models import ModelValidationError, identifier
from remote_display.playlist_store import (
    MAX_NAME_LENGTH,
    PlaylistStore,
    PlaylistStoreError,
    PlaylistValidationError,
    age_seconds,
    document_screens,
    registry_snapshot_path,
    store_path,
    validate_document,
)
from remote_display.provisioning import (
    ProvisioningError,
    ProvisioningStore,
    client_env,
    provisioning_path,
)
from remote_display.registry import read_snapshot

CSRF_HEADER = "X-Requested-With"
CSRF_VALUE = "desk-display"
AUDIT_LIMIT = 100


def _iso(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def client_state(entry: Mapping[str, Any], heartbeat_interval: float, now: float) -> str:
    """online, stale, expired, disabled, or never_connected."""

    if entry.get("disabled"):
        return "disabled"
    expires = age_seconds(entry.get("lease_expires_at"), now)
    if entry.get("lease_expires_at") is None:
        return "never_connected" if entry.get("static") else "expired"
    if expires is not None and expires > 0:
        return "expired"
    seen = age_seconds(entry.get("last_seen"), now)
    if seen is not None and seen > heartbeat_interval * 1.5:
        return "stale"
    return "online"


TELEMETRY_FIELDS = (
    "heartbeat_rtt_ms", "manifest_fetch_ms", "last_sync_duration_ms", "download_count",
    "download_bytes", "download_ms", "displayed_content_age_seconds", "consecutive_failures",
)


def _telemetry_row(telemetry: Any) -> dict[str, Any] | None:
    """The delivery timings a client last reported, without the wire envelope."""

    if not isinstance(telemetry, Mapping):
        return None
    return {key: telemetry.get(key) for key in TELEMETRY_FIELDS}


def revision_state(saved: str | None, delivered: str | None, acknowledged: str | None) -> str:
    if saved is None:
        return "unassigned"
    if delivered != saved:
        return "pending_delivery"
    if acknowledged != saved:
        return "pending_acknowledgment"
    return "in_sync"


def capability_warnings(document: Mapping[str, Any], client: Mapping[str, Any]) -> list[dict[str, str]]:
    """Warnings for showing *document* on the client described by *client*."""

    required, alternates = document_screens(document)
    screens = set(required) | set(alternates)
    caps = client.get("capabilities") or {}
    profile_id = caps.get("display_profile")
    profile = PROFILE_PRESETS.get(profile_id or "")
    warnings: list[dict[str, str]] = []

    def warn(code: str, message: str, severity: str = "warning") -> None:
        warnings.append({"code": code, "severity": severity, "message": message})

    if profile is None:
        warn("unknown_profile", f"client reports unknown display profile {profile_id!r}", "error")
        return warnings
    state = client.get("state")
    if state in {"disabled", "expired"}:
        warn("client_inactive", f"client is {state}; it will not receive this playlist until it reconnects")
    # Phase 10b: the same deterministic fallbacks the client applies.
    from remote_display.fallbacks import plan

    grouped: dict[tuple[str, str, str], list[str]] = {}
    for screen, fallback in plan(screens, supports_animation=bool(caps.get("supports_animation")),
                                 has_touch=bool(caps.get("has_touch")), color_mode=profile.color_mode).items():
        for note in fallback.notes:
            grouped.setdefault(note, []).append(screen)
    for (code, severity, message), affected in sorted(grouped.items()):
        warn(code, f"{message}: " + ", ".join(sorted(affected)), severity)
    if min(profile.width, profile.height) < 200:
        from rendering.screen_classes import CLASSIFICATIONS, COMPOSITE, INTERACTIVE_FOCUS

        quads = sorted(s for s in screens if getattr(CLASSIFICATIONS.get(s), "kind", None)
                       in (COMPOSITE, INTERACTIVE_FOCUS))
        if quads:
            warn("small_display", f"{profile.width}x{profile.height} tiles are very small for: " + ", ".join(quads))
    versions = caps.get("render_package_versions") or []
    if versions and 1 not in versions:
        warn("package_version", "client supports no render package version this server produces", "error")
    return warnings


def register(
    app,
    *,
    actor: Callable[[], str],
    active_document: Callable[[], dict[str, Any]],
    env: Mapping[str, str] | None = None,
    clock: Callable[[], float] | None = None,
) -> Blueprint:
    """Add the playlist library and client registry to *app*."""

    import time

    now = clock or time.time
    app.extensions["desk_display_playlist_store"] = PlaylistStore(store_path(env))
    app.extensions["desk_display_registry_snapshot"] = registry_snapshot_path(env)
    app.extensions["desk_display_provisioning"] = ProvisioningStore(provisioning_path(env))
    app.extensions["desk_display_client_commands"] = CommandStore(commands_path(env))
    blueprint = Blueprint("remote_playlists", __name__)

    def _store() -> PlaylistStore:
        # Looked up per request so tests and tools can point the UI elsewhere.
        return app.extensions["desk_display_playlist_store"]

    def _snapshot_file():
        return app.extensions["desk_display_registry_snapshot"]

    def _provisioning() -> ProvisioningStore:
        return app.extensions["desk_display_provisioning"]

    def _commands() -> CommandStore:
        return app.extensions["desk_display_client_commands"]

    def server_settings() -> dict[str, Any]:
        source = os.environ if env is None else env
        try:
            return deployment_config.load_settings(Role.SERVER, source)
        except ValueError:
            return {"DESK_DISPLAY_SERVER_PUBLIC_URL": (source.get("DESK_DISPLAY_SERVER_PUBLIC_URL") or "").strip()}

    def browser_host() -> str | None:
        return urlsplit("//" + request.host).hostname if request.host else None

    def static_clients() -> dict[str, str]:
        source = os.environ if env is None else env
        try:
            return deployment_config.load_settings(Role.SERVER, source)["DESK_DISPLAY_STATIC_CLIENTS"] or {}
        except ValueError:
            return {}

    @blueprint.before_request
    def _csrf_guard():
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            if request.headers.get(CSRF_HEADER) != CSRF_VALUE:
                return jsonify({"error": "csrf_check_failed", "message": f"missing {CSRF_HEADER} header"}), 403
        return None

    @blueprint.errorhandler(PlaylistStoreError)
    def _store_error(exc: PlaylistStoreError):
        return jsonify(exc.as_response()), exc.status

    @blueprint.errorhandler(ProvisioningError)
    def _provisioning_error(exc: ProvisioningError):
        return jsonify(exc.as_response()), exc.status

    @blueprint.errorhandler(CommandError)
    def _command_error(exc: CommandError):
        return jsonify(exc.as_response()), exc.status

    @blueprint.errorhandler(registration.ServerUrlError)
    def _server_url_error(exc: registration.ServerUrlError):
        return jsonify({"error": "invalid_request", "message": str(exc), "field": "server_url"}), 400

    @blueprint.errorhandler(ModelValidationError)
    def _model_error(exc: ModelValidationError):
        return jsonify({"error": "invalid_request", "message": str(exc), "field": exc.path}), 400

    def body() -> dict[str, Any]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise PlaylistValidationError("expected a JSON object body")
        return payload

    def summarize(playlist: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
        required, alternates = document_screens(playlist["document"])
        return {
            "id": playlist["id"],
            "name": playlist["name"],
            "revision": playlist["revision"],
            "created_at": playlist.get("created_at"),
            "updated_at": playlist.get("updated_at"),
            "updated_by": playlist.get("updated_by"),
            "clients": _store().clients_using(playlist["id"], data),
            "screen_count": len(required),
            "alternate_count": len(alternates),
            "sequence_length": len(playlist["document"].get("sequence") or []),
        }

    def client_rows() -> list[dict[str, Any]]:
        data = _store().snapshot()
        snapshot = read_snapshot(_snapshot_file())
        heartbeat = float(snapshot.get("heartbeat_interval_seconds") or 100)
        entries: dict[str, dict[str, Any]] = {k: dict(v) for k, v in snapshot["clients"].items()}
        for client_id, profile in static_clients().items():
            preset = PROFILE_PRESETS.get(profile)
            entries.setdefault(client_id, {
                "client_id": client_id,
                "static": True,
                "lease_expires_at": None,
                "capabilities": {
                    "display_profile": profile,
                    "logical_width": preset.width if preset else None,
                    "logical_height": preset.height if preset else None,
                },
            })
        provisioned = {record["client_id"]: record for record in _provisioning().records()}
        for client_id, record in provisioned.items():
            entries.setdefault(client_id, {"client_id": client_id, "static": False, "lease_expires_at": None,
                                           "unknown": True,
                                           "capabilities": {"display_profile": record["display_profile"]}})
        for client_id in list(data["assignments"]) + list(data["clients"]):
            entries.setdefault(client_id, {"client_id": client_id, "static": False, "lease_expires_at": None, "unknown": True})
        current = now()
        rows = []
        for client_id, entry in sorted(entries.items()):
            caps = entry.get("capabilities") or {}
            status = entry.get("status") or {}
            assignment = data["assignments"].get(client_id)
            playlist = data["playlists"].get(assignment["playlist_id"]) if assignment else None
            saved = playlist["revision"] if playlist else None
            delivered = entry.get("delivered_playlist_revision")
            acknowledged = (status.get("accepted_revisions") or {}).get("playlist_revision")
            state = "unknown" if entry.get("unknown") else client_state(entry, heartbeat, current)
            credential = provisioned.get(client_id)
            if credential is not None and entry.get("unknown"):
                state = "never_connected"
            if credential is not None and credential["state"] != "active":
                state = credential["state"]  # disabled or revoked wins over the last lease
            row = {
                "client_id": client_id,
                "friendly_name": (data["clients"].get(client_id) or {}).get("friendly_name"),
                "kind": "static" if entry.get("static") else "dynamic",
                "state": state,
                "display_profile": caps.get("display_profile"),
                "dimensions": (
                    f"{caps['logical_width']}x{caps['logical_height']}"
                    if caps.get("logical_width") and caps.get("logical_height") else None
                ),
                "capabilities": {
                    key: caps.get(key)
                    for key in ("image_formats", "color_modes", "render_package_versions",
                                "supports_animation", "has_touch", "buttons", "hardware")
                    if key in caps
                },
                "physical_rotation": status.get("physical_rotation"),
                "software_version": caps.get("client_software_version"),
                "current_screen": status.get("current_screen"),
                "playback_state": status.get("playback_state"),
                "cache_age_seconds": status.get("cache_age_seconds"),
                "last_sync_age_seconds": status.get("last_sync_age_seconds"),
                # Client-measured delivery timings (None from clients that predate them).
                "telemetry": _telemetry_row(entry.get("telemetry")),
                "last_heartbeat": entry.get("last_seen"),
                "lease_expires_at": entry.get("lease_expires_at"),
                "assignment": None if playlist is None else {
                    "playlist_id": playlist["id"],
                    "playlist_name": playlist["name"],
                    "assigned_at": assignment.get("assigned_at"),
                    "assigned_by": assignment.get("assigned_by"),
                },
                "saved_revision": saved,
                "delivered_revision": delivered,
                "acknowledged_revision": acknowledged,
                "revision_state": revision_state(saved, delivered, acknowledged),
                "recent_errors": status.get("recent_errors") or [],
                # The latest update and restart command of each kind, newest first.
                "commands": _latest_commands(client_id),
                "credential": None if credential is None else {
                    "state": credential["state"],
                    "created_at": credential.get("created_at"),
                    "updated_at": credential.get("updated_at"),
                },
            }
            row["warnings"] = capability_warnings(playlist["document"], {**entry, "state": state}) if playlist and caps.get("display_profile") else []
            rows.append(row)
        return rows

    def _latest_commands(client_id: str) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for command in _commands().for_client(client_id):
            latest.setdefault(command["action"], command)
        return [
            {**command, "label": ACTIONS[command["action"]],
             "requested_at": _iso(command.get("requested_at")),
             "finished_at": _iso(command.get("finished_at"))}
            for command in sorted(latest.values(), key=lambda c: c.get("requested_at") or 0, reverse=True)
        ]

    def respond(payload: Any, status: int = 200):
        return jsonify(deployment_config.scrub_secrets(payload)), status

    # ── Pages ──────────────────────────────────────────────────────────────

    @blueprint.get("/playlists")
    def playlists_page():
        return render_template("playlists.html", csrf_header=CSRF_HEADER, csrf_value=CSRF_VALUE)

    @blueprint.get("/clients")
    def clients_page():
        return render_template("clients.html", csrf_header=CSRF_HEADER, csrf_value=CSRF_VALUE)

    @blueprint.get("/clients/add")
    def add_client_page():
        return render_template("client_wizard.html", csrf_header=CSRF_HEADER, csrf_value=CSRF_VALUE)

    # ── Playlist library API ───────────────────────────────────────────────

    @blueprint.get("/api/playlists")
    def list_playlists():
        data = _store().snapshot()
        items = sorted((summarize(p, data) for p in data["playlists"].values()), key=lambda p: p["name"].lower())
        return respond({"store_revision": data["store_revision"], "playlists": items})

    @blueprint.post("/api/playlists")
    def create_playlist():
        payload = body()
        document = payload.get("document")
        if document is None:
            document = active_document()
        playlist = _store().create(payload.get("name"), document, actor=actor())
        return respond(playlist, 201)

    @blueprint.post("/api/playlists/validate")
    def validate_playlist():
        payload = body()
        try:
            document = validate_document(payload.get("document"))
        except PlaylistValidationError as exc:
            return respond({"valid": False, "error": str(exc), "field": exc.details.get("field")})
        required, alternates = document_screens(document)
        return respond({"valid": True, "required_screens": list(required), "alternate_screens": list(alternates)})

    @blueprint.post("/api/playlists/import")
    def import_playlist():
        payload = body()
        playlist = _store().import_playlist(payload.get("export"), actor=actor(), name=payload.get("name"))
        return respond(playlist, 201)

    @blueprint.get("/api/playlists/audit")
    def audit_log():
        return respond({"entries": list(reversed(_store().snapshot()["audit"][-AUDIT_LIMIT:]))})

    @blueprint.get("/api/playlists/<playlist_id>")
    def get_playlist(playlist_id: str):
        data = _store().snapshot()
        playlist = _store().get(playlist_id)
        return respond({**playlist, "clients": _store().clients_using(playlist_id, data)})

    @blueprint.put("/api/playlists/<playlist_id>")
    def update_playlist(playlist_id: str):
        payload = body()
        playlist = _store().update(playlist_id, payload.get("document"),
                                expected_revision=payload.get("expected_revision"), actor=actor())
        return respond(playlist)

    @blueprint.post("/api/playlists/<playlist_id>/rename")
    def rename_playlist(playlist_id: str):
        payload = body()
        return respond(_store().rename(playlist_id, payload.get("name"),
                                    expected_revision=payload.get("expected_revision"), actor=actor()))

    @blueprint.post("/api/playlists/<playlist_id>/reorder")
    def reorder_playlist(playlist_id: str):
        payload = body()
        return respond(_store().reorder(playlist_id, payload.get("order"),
                                     expected_revision=payload.get("expected_revision"), actor=actor()))

    @blueprint.post("/api/playlists/<playlist_id>/clone")
    def clone_playlist(playlist_id: str):
        payload = body()
        return respond(_store().clone(playlist_id, payload.get("name"), actor=actor()), 201)

    @blueprint.delete("/api/playlists/<playlist_id>")
    def delete_playlist(playlist_id: str):
        _store().delete(playlist_id, expected_revision=request.args.get("expected_revision"), actor=actor())
        return "", 204

    @blueprint.get("/api/playlists/<playlist_id>/export")
    def export_playlist(playlist_id: str):
        export = _store().export(playlist_id)
        response = jsonify(export)
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in export["name"])[:60] or "playlist"
        response.headers["Content-Disposition"] = f'attachment; filename="{safe_name}.playlist.json"'
        return response

    @blueprint.get("/api/playlists/<playlist_id>/preview")
    def preview_playlist(playlist_id: str):
        playlist = _store().get(playlist_id)
        required, alternates = document_screens(playlist["document"])
        rows = {row["client_id"]: row for row in client_rows()}
        target_ids = request.args.getlist("client") or _store().clients_using(playlist_id)
        snapshot = read_snapshot(_snapshot_file())["clients"]
        clients = []
        for client_id in target_ids:
            row = rows.get(client_id)
            if row is None:
                clients.append({"client_id": client_id, "warnings": [
                    {"code": "unknown_client", "severity": "error", "message": "client is not known to the server"}
                ]})
                continue
            entry = {**(snapshot.get(client_id) or {}), "state": row["state"]}
            if not entry.get("capabilities"):
                entry["capabilities"] = {"display_profile": row["display_profile"]}
            clients.append({
                "client_id": client_id,
                "display_profile": row["display_profile"],
                "state": row["state"],
                "warnings": capability_warnings(playlist["document"], entry),
            })
        profiles = sorted({c.get("display_profile") for c in clients if c.get("display_profile")})
        return respond({
            "playlist_id": playlist_id,
            "revision": playlist["revision"],
            "required_screens": list(required),
            "alternate_screens": list(alternates),
            "render_demand": [{"display_profile": p, "screens": len(required) + len(alternates)} for p in profiles],
            "clients": clients,
        })

    # ── Client registry API ────────────────────────────────────────────────

    @blueprint.get("/api/clients")
    def list_clients():
        data = _store().snapshot()
        return respond({
            "generated_at": _iso_now(),
            "clients": client_rows(),
            "playlists": [{"id": p["id"], "name": p["name"], "revision": p["revision"]}
                          for p in sorted(data["playlists"].values(), key=lambda p: p["name"].lower())],
        })

    @blueprint.put("/api/clients/<client_id>/assignment")
    def assign_client(client_id: str):
        payload = body()
        if "expected_playlist_id" not in payload:
            raise PlaylistValidationError("expected_playlist_id is required (null when unassigned)",
                                          field="expected_playlist_id")
        identifier(client_id, "client_id")
        assignment = _store().assign(client_id, payload.get("playlist_id"),
                                  expected_playlist_id=payload.get("expected_playlist_id"), actor=actor())
        return respond({"client_id": client_id, "assignment": assignment})

    @blueprint.post("/api/clients/<client_id>/fork")
    def fork_client_playlist(client_id: str):
        payload = body()
        clone = _store().fork_for_client(client_id, expected_playlist_id=payload.get("expected_playlist_id"),
                                      actor=actor(), name=payload.get("name"))
        return respond(clone, 201)

    def issued_payload(issued, server_url: str | None = None, *,
                       allow_insecure_transport: bool | None = None) -> dict[str, Any]:
        url = server_url or registration.suggested_server_url(server_settings(), browser_host())
        if allow_insecure_transport is None:  # rotation: keep a plain-HTTP LAN client connecting
            parts = urlsplit(url)
            allow_insecure_transport = parts.scheme == "http" and not registration.is_loopback(parts.hostname or "")
        text, warnings = client_env(issued, url, allow_insecure_transport=allow_insecure_transport)
        return {"client_id": issued.client_id, "display_profile": issued.display_profile,
                "client_env": text, "warnings": warnings,
                "credentials_filename": registration.credentials_filename(issued.client_id),
                "install_command": registration.install_command(
                    text, issued.client_id, issued.display_profile, registration.repository_url()),
                "note": "Copy this now: the credential is never shown again."}

    def join_payload(client_id: str, server_url: str | None, insecure: bool | None) -> dict[str, Any]:
        url = server_url or registration.suggested_server_url(server_settings(), browser_host())
        parts = urlsplit(url)
        plain_http = parts.scheme == "http" and not registration.is_loopback(parts.hostname or "")
        insecure = plain_http if insecure is None else insecure
        code, expires = _provisioning().create_join_ticket(
            client_id, server_url=url, allow_insecure_transport=insecure, actor=actor())
        return {"client_id": client_id, "server_url": url,
                "join_command": registration.join_command(url, code),
                "join_expires_at": datetime.fromtimestamp(expires, timezone.utc).isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "join_expires_in_seconds": max(0, round(expires - now())),
                "allow_insecure_transport": insecure,
                "warnings": [] if insecure or not plain_http else [
                    f"{url} is plain HTTP and the display was not allowed to use it; it will refuse to connect."]}

    @blueprint.post("/api/clients/<client_id>/join")
    def new_join_code(client_id: str):
        payload = body()
        server_url = payload.get("server_url")
        if server_url is not None:
            server_url = registration.normalize_server_url(server_url)
        insecure = payload.get("allow_insecure_transport")
        response, status = respond(join_payload(identifier(client_id, "client_id"), server_url,
                                                None if insecure is None else insecure is True))
        response.headers["Cache-Control"] = "no-store"
        return response, status

    @blueprint.get("/api/clients/registration")
    def registration_status():
        data = _store().snapshot()
        known = {row["client_id"] for row in client_rows()}
        return respond({
            **registration.readiness(server_settings(), browser_host=browser_host()),
            "profiles": registration.profile_choices(),
            "playlists": [{"id": p["id"], "name": p["name"]}
                          for p in sorted(data["playlists"].values(), key=lambda p: p["name"].lower())],
            "client_ids": sorted(known),
        })

    @blueprint.post("/api/clients/provision")
    def provision_client():
        payload = body()
        playlist_id = payload.get("playlist_id") or None
        if playlist_id is not None and playlist_id not in _store().snapshot()["playlists"]:
            raise PlaylistValidationError("unknown playlist", field="playlist_id")
        server_url = payload.get("server_url")
        if server_url is not None:
            server_url = registration.normalize_server_url(server_url)
        insecure = payload.get("allow_insecure_transport")
        if insecure is not None:
            insecure = insecure is True
        friendly_name = payload.get("friendly_name")
        if friendly_name is not None:
            if not isinstance(friendly_name, str) or len(" ".join(friendly_name.split())) > MAX_NAME_LENGTH:
                raise PlaylistValidationError(f"friendly_name must be at most {MAX_NAME_LENGTH} characters",
                                              field="friendly_name")
            friendly_name = " ".join(friendly_name.split()) or None
        issued = _provisioning().provision(payload.get("client_id"), payload.get("display_profile"),
                                           actor=actor())
        if playlist_id is not None:
            _store().assign(issued.client_id, playlist_id, expected_playlist_id=None, actor=actor())
        if friendly_name:
            _store().set_friendly_name(issued.client_id, friendly_name, actor=actor())
        if payload.get("join") is True:
            # The credential just issued is never shown: the display gets its
            # own when it redeems the join code.
            response, status = respond({**join_payload(issued.client_id, server_url, insecure),
                                        "display_profile": issued.display_profile}, 201)
            response.headers["Cache-Control"] = "no-store"
            return response, status
        response, status = respond(issued_payload(
            issued, server_url, allow_insecure_transport=insecure), 201)
        response.headers["Cache-Control"] = "no-store"
        return response, status

    @blueprint.post("/api/clients/<client_id>/credential/<action>")
    def credential_action(client_id: str, action: str):
        store = _provisioning()
        if action == "rotate":
            response, status = respond(issued_payload(store.rotate(client_id, actor=actor())))
            response.headers["Cache-Control"] = "no-store"
            return response, status
        if action == "revoke":
            record = store.revoke(client_id, actor=actor())
        elif action in {"disable", "enable"}:
            record = store.set_disabled(client_id, action == "disable", actor=actor())
        else:
            return jsonify({"error": "not_found", "message": "unknown action"}), 404
        return respond({"client_id": record["client_id"], "state": record["state"]})

    @blueprint.post("/api/clients/<client_id>/commands")
    def queue_client_command(client_id: str):
        payload = body()
        known = {row["client_id"] for row in client_rows()}
        client_id = identifier(client_id, "client_id")
        if client_id not in known:
            return jsonify({"error": "unknown_client", "message": "no client with this ID"}), 404
        command = _commands().queue(client_id, payload.get("action"), actor=actor())
        return respond({"client_id": client_id, "command": {**command, "label": ACTIONS[command["action"]]}}, 202)

    @blueprint.put("/api/clients/<client_id>/name")
    def name_client(client_id: str):
        payload = body()
        _store().set_friendly_name(client_id, payload.get("friendly_name"), actor=actor())
        return respond({"client_id": client_id, "friendly_name": payload.get("friendly_name") or None})

    app.register_blueprint(blueprint)
    return blueprint


__all__ = ["capability_warnings", "client_state", "register", "revision_state"]
