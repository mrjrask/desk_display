"""Configuration UI pages and APIs for server-managed per-client playlists.

Registered on the config UI app by :func:`register`.  The pages are split
from the rotation editor to keep both usable:

``/playlists``  playlist library: create, clone, rename, edit, reorder,
                validate, import, export, guarded delete, the clients using
                each playlist, and a demand preview with capability warnings.
``/clients``    client registry: identity, state, profile, capabilities,
                rotation, version, current screen, cache age, heartbeat,
                assignment, and saved / delivered / acknowledged revisions.

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

from flask import Blueprint, jsonify, render_template, request

import deployment_config
from deployment_config import Role
from display_profiles import PROFILE_PRESETS
from remote_display.models import ModelValidationError, identifier
from remote_display.playlist_store import (
    PlaylistStore,
    PlaylistStoreError,
    PlaylistValidationError,
    age_seconds,
    document_screens,
    registry_snapshot_path,
    store_path,
    validate_document,
)
from remote_display.registry import read_snapshot

CSRF_HEADER = "X-Requested-With"
CSRF_VALUE = "desk-display"
AUDIT_LIMIT = 100

# Screens that scroll or animate on the display.  Phase 10 replaces this
# heuristic with an explicit remote-mode classification for every screen.
ANIMATED_SCREENS = frozenset({
    "news headlines",
    "news headlines 2",
    "weather radar",
    "MLB Scoreboard",
    "MLB Scoreboard v2",
    "quad",
    "weather quad",
    "cubs schedule quad",
    "sox schedule quad",
})
TOUCH_EXPANDABLE_SCREENS = frozenset({"quad", "weather quad", "cubs schedule quad", "sox schedule quad"})
COLOR_DEPENDENT_SCREENS = frozenset({"weather radar", "air quality"})


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
    if not caps.get("supports_animation"):
        animated = sorted(screens & ANIMATED_SCREENS)
        if animated:
            warn("no_animation", "client cannot animate; these screens fall back to still images: " + ", ".join(animated))
    if not caps.get("has_touch"):
        expandable = sorted(screens & TOUCH_EXPANDABLE_SCREENS)
        if expandable:
            warn("no_touch", "client has no touch input, so quad tiles cannot expand: " + ", ".join(expandable), "info")
    if profile.color_mode == "1":
        mono = sorted(screens & COLOR_DEPENDENT_SCREENS)
        if mono:
            warn("monochrome", f"{profile.profile_id} is 1-bit monochrome; colour-coded detail is lost on: " + ", ".join(mono))
    if min(profile.width, profile.height) < 200:
        quads = sorted(screens & TOUCH_EXPANDABLE_SCREENS)
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
    blueprint = Blueprint("remote_playlists", __name__)

    def _store() -> PlaylistStore:
        # Looked up per request so tests and tools can point the UI elsewhere.
        return app.extensions["desk_display_playlist_store"]

    def _snapshot_file():
        return app.extensions["desk_display_registry_snapshot"]

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
            }
            row["warnings"] = capability_warnings(playlist["document"], {**entry, "state": state}) if playlist and caps.get("display_profile") else []
            rows.append(row)
        return rows

    def respond(payload: Any, status: int = 200):
        return jsonify(deployment_config.scrub_secrets(payload)), status

    # ── Pages ──────────────────────────────────────────────────────────────

    @blueprint.get("/playlists")
    def playlists_page():
        return render_template("playlists.html", csrf_header=CSRF_HEADER, csrf_value=CSRF_VALUE)

    @blueprint.get("/clients")
    def clients_page():
        return render_template("clients.html", csrf_header=CSRF_HEADER, csrf_value=CSRF_VALUE)

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

    @blueprint.put("/api/clients/<client_id>/name")
    def name_client(client_id: str):
        payload = body()
        _store().set_friendly_name(client_id, payload.get("friendly_name"), actor=actor())
        return respond({"client_id": client_id, "friendly_name": payload.get("friendly_name") or None})

    app.register_blueprint(blueprint)
    return blueprint


__all__ = ["capability_warnings", "client_state", "register", "revision_state"]
