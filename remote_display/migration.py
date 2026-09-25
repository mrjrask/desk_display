"""Migrate a standalone installation's screen configuration to the server.

The active standalone rotation (``paths.resolve_screens_config_paths()``:
``screens_config.local.json`` when present, else ``screens_config.json``, or
a small/large default bundle when neither exists) becomes one shared playlist
in the server's :class:`~remote_display.playlist_store.PlaylistStore`, with
its screens, playlists, sequence, frequencies, extra seconds and alternates
unchanged. Styles and quad layouts are carried in the migration bundle and
can be installed as the server's rendering configuration.

Guarantees
    * **Preview first**: :func:`plan` changes nothing and lists every
      semantic change (renamed legacy IDs, dropped retired or unknown
      screens, unsupported keys, remote-only limits, assignment conflicts).
    * **Non-destructive**: original files are never modified; everything
      written (the bundle, installed style/layout files) is written to a
      temporary file and renamed into place, and replaced files are kept in
      the bundle.
    * **Repeatable**: the playlist is found by its content revision, so a
      rerun (or a rerun after an interrupted one) reuses it instead of
      creating a duplicate, and existing assignments are never changed
      silently.
    * **Reversible**: :func:`rollback` undoes what a bundle records.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from remote_display.playlist_store import (
    ConflictError,
    PlaylistStore,
    PlaylistStoreError,
    document_revision,
    validate_document,
)
from rendering.screen_classes import CLASSIFICATIONS, UNSUPPORTED
from screens_catalog import LEGACY_RETIRED_SCREEN_IDS, SCREEN_IDS, canonical_screen_id

BUNDLE_FORMAT = "desk-display-standalone-migration"
BUNDLE_SCHEMA_VERSION = 1
DOCUMENT_KEYS = ("screens", "playlists", "sequence", "scroll")
_ACTIVE = frozenset(SCREEN_IDS)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_DIR = _PROJECT_ROOT / ".runtime" / "server" / "migrations"
DEFAULT_BUNDLES = {
    "large": _PROJECT_ROOT / "default_screens_large.json",
    "small": _PROJECT_ROOT / "default_screens_small.json",
}


class MigrationError(Exception):
    """The source cannot be migrated as it is; nothing was changed."""


@dataclass(frozen=True)
class Change:
    kind: str  # renamed, removed, duplicate, unsupported, remote_limit, assignment, style, layout
    severity: str  # info, warning, error
    message: str
    screen: str | None = None


@dataclass
class Source:
    kind: str  # "local", "default" or "bundled-<size>"
    config_path: str
    config: dict[str, Any]
    raw: dict[str, str]  # file name -> original text
    style: dict[str, Any] = field(default_factory=dict)
    layouts: dict[str, Any] = field(default_factory=dict)
    style_path: str | None = None
    layouts_path: str | None = None


@dataclass
class Plan:
    source: Source
    name: str
    document: dict[str, Any]
    revision: str
    style: dict[str, Any]
    layouts: dict[str, Any]
    changes: list[Change]
    existing_playlist_id: str | None
    assignments: dict[str, dict[str, Any]]  # client -> {"action", "current"}

    @property
    def blocked(self) -> bool:
        return any(change.severity == "error" for change in self.changes)

    def report(self) -> dict[str, Any]:
        return {
            "source": {"kind": self.source.kind, "config_path": self.source.config_path},
            "name": self.name,
            "revision": self.revision,
            "existing_playlist_id": self.existing_playlist_id,
            "screens": list(self.document["screens"]),
            "assignments": self.assignments,
            "changes": [asdict(change) for change in self.changes],
        }


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MigrationError(f"cannot read {path}: {exc}") from None
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise MigrationError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise MigrationError(f"{path} must contain a JSON object")
    return data, text


# ── Reading the standalone configuration ──────────────────────────────────


def load_source(*, defaults: str | None = None, config_path: Path | None = None,
                style_path: Path | None = None, layouts_path: Path | None = None) -> Source:
    """Read the active standalone configuration (never modifying it).

    *defaults* (``small`` or ``large``) is used only when no rotation config
    exists, as a fresh install would.
    """

    from paths import resolve_layouts_config_path, resolve_screens_config_paths, resolve_style_config_path

    if config_path is None:
        paths = resolve_screens_config_paths()
        config_path = paths.active_path
        kind = "local" if paths.active_path == paths.local_override_path else "default"
    else:
        kind = "local"
    style_path = style_path or resolve_style_config_path()
    layouts_path = layouts_path or resolve_layouts_config_path()
    raw: dict[str, str] = {}
    style: dict[str, Any] = {}
    if config_path.exists():
        config, raw["config"] = _read_json(config_path)
    elif defaults is not None:
        if defaults not in DEFAULT_BUNDLES:
            raise MigrationError(f"unknown defaults {defaults!r}; choose small or large")
        bundle, raw["config"] = _read_json(DEFAULT_BUNDLES[defaults])
        config = bundle.get("config") if isinstance(bundle.get("config"), dict) else bundle
        style = bundle.get("style") if isinstance(bundle.get("style"), dict) else {}
        kind, config_path = f"bundled-{defaults}", DEFAULT_BUNDLES[defaults]
    else:
        raise MigrationError(f"no screen configuration at {config_path}; pass defaults='small' or 'large'")
    if style_path.exists():
        style, raw["style"] = _read_json(style_path)
    layouts: dict[str, Any] = {}
    if layouts_path.exists():
        layouts, raw["layouts"] = _read_json(layouts_path)
    return Source(kind, str(config_path), config, raw, style, layouts, str(style_path), str(layouts_path))


# ── Normalizing ────────────────────────────────────────────────────────────


def _screen(raw: Any, where: str, changes: list[Change]) -> str | None:
    if not isinstance(raw, str):
        changes.append(Change("removed", "warning", f"{where}: {raw!r} is not a screen ID; dropped"))
        return None
    canonical = canonical_screen_id(raw)
    if canonical in LEGACY_RETIRED_SCREEN_IDS:
        changes.append(Change("removed", "warning", f"{where}: {raw!r} was retired; dropped", raw))
        return None
    if canonical not in _ACTIVE:
        changes.append(Change("removed", "warning", f"{where}: unknown screen {raw!r}; dropped", raw))
        return None
    if canonical != raw:
        changes.append(Change("renamed", "info", f"{where}: legacy ID {raw!r} is now {canonical!r}", canonical))
    return canonical


def normalize(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[Change]]:
    """A valid playlist document from a standalone config, and what changed."""

    changes: list[Change] = []
    if not isinstance(config.get("screens"), dict):
        raise MigrationError("the configuration has no 'screens' object")
    for key in sorted(set(config) - set(DOCUMENT_KEYS)):
        changes.append(Change("unsupported", "warning", f"top-level key {key!r} has no server equivalent; dropped"))

    screens: dict[str, Any] = {}
    for raw_id, spec in config["screens"].items():
        sid = _screen(raw_id, "screens", changes)
        if sid is None:
            continue
        if sid in screens:
            changes.append(Change("duplicate", "warning",
                                  f"screens: {raw_id!r} duplicates {sid!r}; kept the first entry", sid))
            continue
        spec = copy.deepcopy(spec)
        if isinstance(spec, dict) and isinstance(spec.get("alt"), dict):
            alt = spec["alt"]
            targets = alt.get("screen")
            listed = [targets] if isinstance(targets, str) else list(targets or [])
            kept = [t for t in (_screen(t, f"alternate of {sid!r}", changes) for t in listed) if t]
            if not kept:
                del spec["alt"]
                changes.append(Change("removed", "warning", f"{sid!r} lost its alternate; it now always shows itself",
                                      sid))
            else:
                alt["screen"] = kept[0] if isinstance(targets, str) else kept
        screens[sid] = spec

    playlists: dict[str, Any] = {}
    for key, playlist in (config.get("playlists") or {}).items():
        if not isinstance(playlist, dict) or not isinstance(playlist.get("steps"), list):
            changes.append(Change("removed", "warning", f"playlist {key!r} has no steps list; dropped"))
            continue
        playlist = copy.deepcopy(playlist)
        steps = []
        for step in playlist["steps"]:
            if isinstance(step, dict) and "screen" in step:
                sid = _screen(step["screen"], f"playlist {playlist.get('label') or key!r}", changes)
                if sid is None:
                    continue
                step = {**step, "screen": sid}
            steps.append(step)
        playlist["steps"] = steps
        playlists[key] = playlist

    document: dict[str, Any] = {"screens": screens}
    if "playlists" in config:
        document["playlists"] = playlists
    if "sequence" in config:
        sequence = []
        for entry in config.get("sequence") or []:
            if isinstance(entry, dict) and "playlist" in entry and entry["playlist"] not in playlists:
                changes.append(Change("removed", "warning",
                                      f"sequence step for missing playlist {entry['playlist']!r}; dropped"))
                continue
            sequence.append(copy.deepcopy(entry))
        document["sequence"] = sequence
    if "scroll" in config:
        document["scroll"] = copy.deepcopy(config["scroll"])

    for sid in screens:
        entry = CLASSIFICATIONS.get(sid)
        if entry is not None and entry.kind == UNSUPPORTED:
            changes.append(Change("remote_limit", "warning",
                                  f"{sid!r} {entry.note}; remote clients skip it", sid))
    try:
        document = validate_document(document)
    except PlaylistStoreError as exc:
        raise MigrationError(f"the migrated schedule is not valid: {exc}") from None
    return document, changes


def normalize_screen_map(data: Mapping[str, Any], what: str) -> tuple[dict[str, Any], list[Change]]:
    """Canonical screen keys for a style or layouts file (``{"screens": {...}}``)."""

    changes: list[Change] = []
    result = copy.deepcopy(dict(data))
    entries = data.get("screens")
    if isinstance(entries, dict):
        mapped: dict[str, Any] = {}
        for raw_id, value in entries.items():
            canonical = canonical_screen_id(raw_id) if isinstance(raw_id, str) else raw_id
            if canonical in _ACTIVE and canonical != raw_id:
                changes.append(Change(what, "info", f"{what}: legacy ID {raw_id!r} is now {canonical!r}", canonical))
                key = canonical
            else:
                # Unknown keys are kept as they are: the renderer ignores them, and
                # dropping them would lose settings for nothing.
                key = raw_id
            if key not in mapped:
                mapped[key] = copy.deepcopy(value)
        result["screens"] = mapped
    return result, changes


# ── Planning ───────────────────────────────────────────────────────────────


def plan(source: Source, store: PlaylistStore, *, clients: Iterable[str] = (), name: str | None = None,
         client_capabilities: Mapping[str, Mapping[str, Any]] | None = None) -> Plan:
    """Everything a migration would do, without doing it."""

    document, changes = normalize(source.config)
    style, style_changes = normalize_screen_map(source.style, "style")
    layouts, layout_changes = normalize_screen_map(source.layouts, "layout")
    changes += style_changes + layout_changes
    revision = document_revision(document)
    data = store.snapshot()
    existing = next((p["id"] for p in sorted(data["playlists"].values(), key=lambda p: p["id"])
                     if p["revision"] == revision), None)
    target = existing or "(new playlist)"
    assignments: dict[str, dict[str, Any]] = {}
    for client in clients:
        current = (data["assignments"].get(client) or {}).get("playlist_id")
        if current is None:
            action = "assign"
        elif current == existing:
            action = "unchanged"
        else:
            action = "conflict"
            changes.append(Change("assignment", "warning",
                                  f"client {client!r} already plays {current!r}; not reassigned to {target} "
                                  "unless forced"))
        assignments[client] = {"action": action, "current": current}
    if client_capabilities:
        from remote_playlists_ui import capability_warnings

        for client, capabilities in client_capabilities.items():
            for warning in capability_warnings(document, {"capabilities": capabilities}):
                changes.append(Change("remote_limit", warning["severity"] if warning["severity"] != "error"
                                      else "warning", f"client {client!r}: {warning['message']}"))
    if existing:
        changes.append(Change("assignment", "info", f"this configuration is already on the server as {existing}; "
                                                   "it will be reused, not duplicated"))
    label = name or f"Standalone rotation ({source.kind})"
    return Plan(source, label, document, revision, style, layouts, changes, existing, assignments)


# ── Applying and rolling back ──────────────────────────────────────────────


def _write_atomic(path: Path, text: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        if mode is not None:
            os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def bundle(plan_: Plan, *, created_at: float | None = None) -> dict[str, Any]:
    """The versioned, self-contained record of a migration (also a backup)."""

    return {
        "format": BUNDLE_FORMAT,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at": created_at if created_at is not None else time.time(),
        "report": plan_.report(),
        "document": plan_.document,
        "style": plan_.style,
        "layouts": plan_.layouts,
        "originals": {name: {"sha256": _sha(text), "text": text} for name, text in plan_.source.raw.items()},
        "applied": None,
    }


def export_bundle(plan_: Plan, path: Path) -> Path:
    from deployment_config import scrub_secrets

    _write_atomic(path, json.dumps(scrub_secrets(bundle(plan_)), indent=2), mode=0o600)
    return path


def apply(plan_: Plan, store: PlaylistStore, *, actor: str = "migration", bundle_dir: Path = DEFAULT_BUNDLE_DIR,
          force_assign: bool = False, install_style: bool = False) -> dict[str, Any]:
    """Create (or reuse) the playlist, assign clients, optionally install style/layouts.

    The bundle is written before anything changes and updated afterwards,
    so an interrupted run leaves a record and a rerun finishes the job.
    """

    if plan_.blocked:
        raise MigrationError("the preview has errors; nothing was changed")
    record = bundle(plan_)
    path = Path(bundle_dir) / f"standalone-{plan_.revision}-{int(record['created_at'])}.json"
    export = lambda: _write_atomic(path, json.dumps(record, indent=2), mode=0o600)  # noqa: E731
    export()

    playlist_id = plan_.existing_playlist_id
    data = store.snapshot()
    playlist_id = playlist_id or next((p["id"] for p in data["playlists"].values()
                                       if p["revision"] == plan_.revision), None)
    created = playlist_id is None
    if created:
        playlist_id = store.create(plan_.name, plan_.document, actor=actor, action="migrate")["id"]
    assigned: dict[str, str | None] = {}
    skipped: list[str] = []
    for client, step in plan_.assignments.items():
        current = (store.snapshot()["assignments"].get(client) or {}).get("playlist_id")
        if current == playlist_id:
            continue
        if current is not None and not force_assign:
            skipped.append(client)
            continue
        try:
            store.assign(client, playlist_id, expected_playlist_id=current, actor=actor)
        except ConflictError:
            skipped.append(client)
            continue
        assigned[client] = current
    installed: dict[str, Any] = {}
    if install_style:
        for key, target, content in (("style", plan_.source.style_path, plan_.style),
                                     ("layouts", plan_.source.layouts_path, plan_.layouts)):
            if not target or not content:
                continue
            target_path = Path(target)
            previous = target_path.read_text(encoding="utf-8") if target_path.exists() else None
            text = json.dumps(content, indent=2) + "\n"
            if previous is not None and json.loads(previous) == content:
                continue
            record["applied"] = {"playlist_id": playlist_id, "created": created, "assigned": assigned,
                                 "installed": {**installed, key: {"path": target, "previous": previous}}}
            export()  # record the backup before replacing the file
            _write_atomic(target_path, text)
            installed[key] = {"path": target, "previous": previous}
    record["applied"] = {"playlist_id": playlist_id, "created": created, "assigned": assigned,
                         "skipped": skipped, "installed": installed}
    export()
    return {"bundle": str(path), **record["applied"]}


def rollback(bundle_path: Path, store: PlaylistStore, *, actor: str = "migration") -> dict[str, Any]:
    """Undo what an applied bundle records, leaving later changes alone."""

    record, _text = _read_json(Path(bundle_path))
    if record.get("format") != BUNDLE_FORMAT:
        raise MigrationError(f"{bundle_path} is not a migration bundle")
    applied = record.get("applied") or {}
    playlist_id = applied.get("playlist_id")
    restored: list[str] = []
    for client, previous in (applied.get("assigned") or {}).items():
        current = (store.snapshot()["assignments"].get(client) or {}).get("playlist_id")
        if current != playlist_id:
            continue  # changed since; leave it
        store.assign(client, previous, expected_playlist_id=current, actor=actor)
        restored.append(client)
    deleted = False
    if applied.get("created") and playlist_id:
        playlist = store.snapshot()["playlists"].get(playlist_id)
        if playlist is not None and not store.clients_using(playlist_id):
            store.delete(playlist_id, expected_revision=playlist["revision"], actor=actor)
            deleted = True
    files: list[str] = []
    for item in (applied.get("installed") or {}).values():
        target = Path(item["path"])
        if item.get("previous") is None:
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
        else:
            _write_atomic(target, item["previous"])
        files.append(str(target))
    record["rolled_back"] = {"at": time.time(), "assignments": restored, "deleted_playlist": deleted,
                             "files": files}
    _write_atomic(Path(bundle_path), json.dumps(record, indent=2), mode=0o600)
    return record["rolled_back"]


__all__ = [
    "BUNDLE_FORMAT",
    "Change",
    "MigrationError",
    "Plan",
    "Source",
    "apply",
    "bundle",
    "export_bundle",
    "load_source",
    "normalize",
    "plan",
    "rollback",
]
