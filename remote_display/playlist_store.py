"""Server-owned playlist library and per-client assignments.

The render server is the source of truth for which playlist each remote
display plays.  This module stores that state in one versioned JSON document
(server configuration, never ``.env``)::

    {
      "schema_version": 1,
      "store_revision": 12,
      "playlists": {"pl-…": {"id", "name", "revision", "document", ...}},
      "assignments": {"<client id>": {"playlist_id", "assigned_at", "assigned_by"}},
      "clients": {"<client id>": {"friendly_name"}},
      "audit": [{"at", "actor", "action", "target", "detail"}, ...]
    }

A playlist ``document`` has exactly the shape of ``screens_config.json``
(``screens``, ``playlists``, ``sequence``, optional ``scroll``), so frequency,
extra duration, sequence, alternate-screen and scheduling semantics are the
ones :func:`schedule.build_scheduler` already implements, and every document
must build a scheduler before it is saved.

Guarantees
    * **Stable IDs**: playlists get a random ``pl-…`` ID that never changes,
      including on rename.
    * **Content revisions**: ``revision`` is a hash of the canonical document,
      so identical content always has the same revision.
    * **Optimistic concurrency**: every change names the revision (or
      assignment) it was based on and fails with :class:`ConflictError` if
      someone else changed it first.  Nothing is overwritten silently.
    * **Atomic writes**: changes are written to a temporary file, fsynced and
      renamed over the store under a process lock and an advisory file lock.
    * **Audit**: every change appends an entry naming the actor.
    * **Exactly one assignment per client**; many clients may share a
      playlist, and a playlist in use cannot be deleted.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import secrets
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deployment_config import scrub_secrets
from remote_display.models import ModelValidationError, identifier, screen_id

try:  # pragma: no cover - fcntl is unavailable on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

STORE_SCHEMA_VERSION = 1
EXPORT_FORMAT = "desk-display-playlist"
EXPORT_SCHEMA_VERSION = 1
MAX_DOCUMENT_BYTES = 256 * 1024
MAX_NAME_LENGTH = 80
MAX_AUDIT_ENTRIES = 500
_DOCUMENT_KEYS = {"screens", "playlists", "sequence", "scroll"}
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE_PATH = _PROJECT_ROOT / ".runtime" / "server" / "playlists.json"
DEFAULT_REGISTRY_PATH = _PROJECT_ROOT / ".runtime" / "server" / "clients.json"


def store_path(env: Mapping[str, str] | None = None) -> Path:
    source = os.environ if env is None else env
    raw = (source.get("DESK_DISPLAY_PLAYLIST_STORE_PATH") or "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_STORE_PATH


def registry_snapshot_path(env: Mapping[str, str] | None = None) -> Path:
    source = os.environ if env is None else env
    raw = (source.get("DESK_DISPLAY_CLIENT_REGISTRY_PATH") or "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_REGISTRY_PATH


class PlaylistStoreError(Exception):
    status = 400
    code = "invalid_request"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details

    def as_response(self) -> dict[str, Any]:
        return {"error": self.code, "message": str(self), **self.details}


class PlaylistValidationError(PlaylistStoreError):
    code = "invalid_playlist"


class NotFoundError(PlaylistStoreError):
    status = 404
    code = "not_found"


class ConflictError(PlaylistStoreError):
    """The caller's view is stale: someone else changed the object first."""

    status = 409
    code = "revision_conflict"


class InUseError(PlaylistStoreError):
    status = 409
    code = "playlist_in_use"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def document_revision(document: Mapping[str, Any]) -> str:
    """Content revision: identical documents always share a revision."""

    return "r-" + hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()[:20]


def _name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlaylistValidationError("name must be a non-empty string", field="name")
    name = " ".join(value.split())
    if len(name) > MAX_NAME_LENGTH:
        raise PlaylistValidationError(f"name must be at most {MAX_NAME_LENGTH} characters", field="name")
    return name


def _screen(value: Any, field: str) -> str:
    try:
        return screen_id(value, field)
    except ModelValidationError as exc:
        raise PlaylistValidationError(str(exc), field=exc.path) from None


def _is_enabled(spec: Any) -> bool:
    if isinstance(spec, bool):
        return spec
    if isinstance(spec, (int, float)):
        return spec > 0
    if isinstance(spec, dict):
        frequency = spec.get("frequency", 1)
        return isinstance(frequency, (int, float)) and not isinstance(frequency, bool) and frequency > 0
    return False


def validate_document(document: Any) -> dict[str, Any]:
    """Return a canonicalized copy of a playlist document, or raise.

    Every screen reference (entries, alternates and playlist steps) must be an
    active ID in :mod:`screens_catalog`, legacy IDs are canonicalized, and the
    document must build a :class:`schedule.ScreenScheduler`.
    """

    if not isinstance(document, dict):
        raise PlaylistValidationError("document must be a JSON object", field="document")
    unknown = sorted(set(document) - _DOCUMENT_KEYS)
    if unknown:
        raise PlaylistValidationError(f"unknown document key {unknown[0]!r}", field=f"document.{unknown[0]}")
    if len(canonical_json(document).encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise PlaylistValidationError(f"document exceeds {MAX_DOCUMENT_BYTES} bytes", field="document")
    screens = document.get("screens")
    if not isinstance(screens, dict):
        raise PlaylistValidationError("document.screens must be an object", field="document.screens")

    result = copy.deepcopy(document)
    canonical_screens: dict[str, Any] = {}
    for raw_id, spec in screens.items():
        sid = _screen(raw_id, f"document.screens[{raw_id!r}]")
        if sid in canonical_screens:
            raise PlaylistValidationError(f"screen {sid!r} is listed twice", field="document.screens")
        spec = copy.deepcopy(spec)
        if isinstance(spec, dict) and isinstance(spec.get("alt"), dict):
            alt = spec["alt"]
            raw_alt = alt.get("screen")
            field = f"document.screens[{sid!r}].alt.screen"
            if isinstance(raw_alt, str):
                alt["screen"] = _screen(raw_alt, field)
            elif isinstance(raw_alt, list):
                alt["screen"] = [_screen(item, field) for item in raw_alt]
            else:
                raise PlaylistValidationError("alternate screen must be an ID or a list of IDs", field=field)
        canonical_screens[sid] = spec
    result["screens"] = canonical_screens

    playlists = result.get("playlists")
    if playlists is not None:
        if not isinstance(playlists, dict):
            raise PlaylistValidationError("document.playlists must be an object", field="document.playlists")
        for playlist_key, playlist in playlists.items():
            steps = playlist.get("steps") if isinstance(playlist, dict) else None
            if not isinstance(steps, list):
                raise PlaylistValidationError(
                    "each playlist needs a steps list", field=f"document.playlists[{playlist_key!r}]"
                )
            for index, step in enumerate(steps):
                if isinstance(step, dict) and "screen" in step:
                    step["screen"] = _screen(step["screen"], f"document.playlists[{playlist_key!r}].steps[{index}]")
    sequence = result.get("sequence")
    if sequence is not None and not isinstance(sequence, list):
        raise PlaylistValidationError("document.sequence must be a list", field="document.sequence")

    from schedule import build_scheduler

    try:
        build_scheduler(result)
    except Exception as exc:  # noqa: BLE001 - surface the scheduler's own message
        raise PlaylistValidationError(f"schedule is not valid: {exc}", field="document") from None
    return result


def document_screens(document: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(required, alternates)`` screen IDs a document can show."""

    required: set[str] = set()
    alternates: set[str] = set()
    for sid, spec in (document.get("screens") or {}).items():
        if _is_enabled(spec):
            required.add(sid)
        if isinstance(spec, dict) and isinstance(spec.get("alt"), dict):
            alt = spec["alt"].get("screen")
            alternates.update([alt] if isinstance(alt, str) else list(alt or []))
    for playlist in (document.get("playlists") or {}).values():
        for step in playlist.get("steps", []) if isinstance(playlist, dict) else []:
            if isinstance(step, dict) and isinstance(step.get("screen"), str):
                required.add(step["screen"])
    return tuple(sorted(required)), tuple(sorted(alternates - required))


@dataclass(frozen=True)
class StoredAssignment:
    client_id: str
    playlist_id: str
    playlist_revision: str
    screens: tuple[str, ...]
    alternates: tuple[str, ...]


class PlaylistStore:
    """File-backed playlist library.  Safe to share between processes."""

    def __init__(self, path: str | os.PathLike[str], *, clock=_now) -> None:
        self.path = Path(path).expanduser()
        self._clock = clock
        self._lock = threading.RLock()
        self._cache: tuple[float, int, dict[str, Any]] | None = None

    # ── Persistence ────────────────────────────────────────────────────────

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "schema_version": STORE_SCHEMA_VERSION,
            "store_revision": 0,
            "playlists": {},
            "assignments": {},
            "clients": {},
            "audit": [],
        }

    def _read(self) -> dict[str, Any]:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return self._empty()
        if self._cache and self._cache[:2] == (stat.st_mtime, stat.st_size):
            return copy.deepcopy(self._cache[2])
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != STORE_SCHEMA_VERSION:
            raise PlaylistStoreError(f"unsupported playlist store schema in {self.path}")
        for key, default in self._empty().items():
            data.setdefault(key, default)
        self._cache = (stat.st_mtime, stat.st_size, data)
        return copy.deepcopy(data)

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[dict[str, Any]]:
        """Read-modify-write under both locks; commit atomically on success."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with self._lock, open(lock_path, "a+") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                self._cache = None
                data = self._read()
                yield data
                data["store_revision"] = int(data.get("store_revision", 0)) + 1
                data["audit"] = data["audit"][-MAX_AUDIT_ENTRIES:]
                self._write(data)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _write(self, data: dict[str, Any]) -> None:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_name)
            raise
        with contextlib.suppress(OSError):
            dir_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        self._cache = None

    def _audit(self, data: dict[str, Any], actor: str, action: str, target: str, **detail: Any) -> None:
        data["audit"].append({
            "at": self._clock(),
            "actor": actor or "unknown",
            "action": action,
            "target": target,
            "detail": scrub_secrets(detail),
        })

    # ── Reads ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._read()

    def get(self, playlist_id: str) -> dict[str, Any]:
        playlist = self.snapshot()["playlists"].get(playlist_id)
        if playlist is None:
            raise NotFoundError("no such playlist", playlist_id=playlist_id)
        return playlist

    def clients_using(self, playlist_id: str, data: Mapping[str, Any] | None = None) -> list[str]:
        data = data or self.snapshot()
        return sorted(cid for cid, a in data["assignments"].items() if a.get("playlist_id") == playlist_id)

    def assignment_for(self, client_id: str) -> StoredAssignment | None:
        """Lookup used by the render server (``registry.AssignmentLookup``)."""

        data = self.snapshot()
        assignment = data["assignments"].get(client_id)
        playlist = data["playlists"].get(assignment["playlist_id"]) if assignment else None
        if playlist is None:
            return None
        required, alternates = document_screens(playlist["document"])
        return StoredAssignment(client_id, playlist["id"], playlist["revision"], required, alternates)

    # ── Playlist changes ───────────────────────────────────────────────────

    def create(self, name: str, document: Any, *, actor: str, action: str = "create") -> dict[str, Any]:
        name = _name(name)
        document = validate_document(document)
        now = self._clock()
        with self._transaction() as data:
            playlist_id = f"pl-{secrets.token_hex(8)}"
            playlist = {
                "id": playlist_id,
                "name": name,
                "revision": document_revision(document),
                "document": document,
                "created_at": now,
                "updated_at": now,
                "updated_by": actor,
            }
            data["playlists"][playlist_id] = playlist
            self._audit(data, actor, action, playlist_id, name=name, revision=playlist["revision"])
        return copy.deepcopy(playlist)

    def clone(self, playlist_id: str, name: str | None = None, *, actor: str) -> dict[str, Any]:
        source = self.get(playlist_id)
        return self.create(name or f"{source['name']} (copy)", source["document"], actor=actor, action="clone")

    def _require_revision(self, playlist: Mapping[str, Any], expected: Any) -> None:
        if expected != playlist["revision"]:
            raise ConflictError(
                "this playlist changed since you loaded it; reload and reapply your edit",
                playlist_id=playlist["id"],
                current_revision=playlist["revision"],
                expected_revision=expected,
            )

    def update(self, playlist_id: str, document: Any, *, expected_revision: Any, actor: str) -> dict[str, Any]:
        document = validate_document(document)
        with self._transaction() as data:
            playlist = data["playlists"].get(playlist_id)
            if playlist is None:
                raise NotFoundError("no such playlist", playlist_id=playlist_id)
            self._require_revision(playlist, expected_revision)
            previous = playlist["revision"]
            playlist["document"] = document
            playlist["revision"] = document_revision(document)
            playlist["updated_at"] = self._clock()
            playlist["updated_by"] = actor
            self._audit(data, actor, "edit", playlist_id, from_revision=previous, to_revision=playlist["revision"])
            return copy.deepcopy(playlist)

    def reorder(self, playlist_id: str, order: Any, *, expected_revision: Any, actor: str) -> dict[str, Any]:
        """Reorder the document's sequence by a permutation of its indices."""

        playlist = self.get(playlist_id)
        sequence = list(playlist["document"].get("sequence") or [])
        if not isinstance(order, list) or sorted(order) != list(range(len(sequence))) or any(
            type(i) is not int for i in order
        ):
            raise PlaylistValidationError(
                f"order must be a permutation of 0..{len(sequence) - 1}", field="order"
            )
        document = copy.deepcopy(playlist["document"])
        document["sequence"] = [sequence[i] for i in order]
        return self.update(playlist_id, document, expected_revision=expected_revision, actor=actor)

    def rename(self, playlist_id: str, name: str, *, expected_revision: Any, actor: str) -> dict[str, Any]:
        name = _name(name)
        with self._transaction() as data:
            playlist = data["playlists"].get(playlist_id)
            if playlist is None:
                raise NotFoundError("no such playlist", playlist_id=playlist_id)
            self._require_revision(playlist, expected_revision)
            old = playlist["name"]
            playlist["name"] = name
            playlist["updated_at"] = self._clock()
            playlist["updated_by"] = actor
            self._audit(data, actor, "rename", playlist_id, old_name=old, new_name=name)
            return copy.deepcopy(playlist)

    def delete(self, playlist_id: str, *, expected_revision: Any, actor: str) -> None:
        with self._transaction() as data:
            playlist = data["playlists"].get(playlist_id)
            if playlist is None:
                raise NotFoundError("no such playlist", playlist_id=playlist_id)
            self._require_revision(playlist, expected_revision)
            users = self.clients_using(playlist_id, data)
            if users:
                raise InUseError(
                    "reassign these clients before deleting the playlist", playlist_id=playlist_id, clients=users
                )
            del data["playlists"][playlist_id]
            self._audit(data, actor, "delete", playlist_id, name=playlist["name"], revision=playlist["revision"])

    # ── Assignments and client names ───────────────────────────────────────

    def assign(self, client_id: str, playlist_id: str | None, *, expected_playlist_id: Any, actor: str) -> dict[str, Any] | None:
        """Set (or clear, with ``None``) the client's single active playlist.

        ``expected_playlist_id`` is the assignment the caller saw (``None`` for
        unassigned); a mismatch raises :class:`ConflictError`.
        """

        client_id = _client(client_id)
        with self._transaction() as data:
            current = data["assignments"].get(client_id)
            current_id = current["playlist_id"] if current else None
            if expected_playlist_id != current_id:
                raise ConflictError(
                    "this client's assignment changed since you loaded it",
                    client_id=client_id,
                    current_playlist_id=current_id,
                    expected_playlist_id=expected_playlist_id,
                )
            if playlist_id is None:
                data["assignments"].pop(client_id, None)
                self._audit(data, actor, "unassign", client_id, playlist_id=current_id)
                return None
            if playlist_id not in data["playlists"]:
                raise NotFoundError("no such playlist", playlist_id=playlist_id)
            assignment = {"playlist_id": playlist_id, "assigned_at": self._clock(), "assigned_by": actor}
            data["assignments"][client_id] = assignment
            self._audit(data, actor, "assign", client_id, playlist_id=playlist_id, previous=current_id)
            return copy.deepcopy(assignment)

    def fork_for_client(self, client_id: str, *, expected_playlist_id: Any, actor: str, name: str | None = None) -> dict[str, Any]:
        """Clone the client's shared playlist and assign the copy to it alone."""

        client_id = _client(client_id)
        if not expected_playlist_id:
            raise PlaylistValidationError("the client has no playlist to clone", field="client_id")
        clone = self.clone(expected_playlist_id, name or f"{self.get(expected_playlist_id)['name']} ({client_id})", actor=actor)
        try:
            self.assign(client_id, clone["id"], expected_playlist_id=expected_playlist_id, actor=actor)
        except PlaylistStoreError:
            with contextlib.suppress(PlaylistStoreError):
                self.delete(clone["id"], expected_revision=clone["revision"], actor=actor)
            raise
        return clone

    def set_friendly_name(self, client_id: str, name: str | None, *, actor: str) -> None:
        client_id = _client(client_id)
        with self._transaction() as data:
            if name:
                data["clients"].setdefault(client_id, {})["friendly_name"] = _name(name)
            else:
                data["clients"].pop(client_id, None)
            self._audit(data, actor, "rename_client", client_id, friendly_name=name or None)

    # ── Import and export ──────────────────────────────────────────────────

    def export(self, playlist_id: str) -> dict[str, Any]:
        playlist = self.get(playlist_id)
        return scrub_secrets({
            "format": EXPORT_FORMAT,
            "schema_version": EXPORT_SCHEMA_VERSION,
            "name": playlist["name"],
            "revision": playlist["revision"],
            "document": playlist["document"],
        })

    def import_playlist(self, payload: Any, *, actor: str, name: str | None = None) -> dict[str, Any]:
        """Create a new playlist (with a new ID) from an export."""

        if not isinstance(payload, dict) or payload.get("format") != EXPORT_FORMAT:
            raise PlaylistValidationError(f"expected a {EXPORT_FORMAT} export", field="format")
        if payload.get("schema_version") != EXPORT_SCHEMA_VERSION:
            raise PlaylistValidationError("unsupported export schema version", field="schema_version")
        unknown = sorted(set(payload) - {"format", "schema_version", "name", "revision", "document"})
        if unknown:
            raise PlaylistValidationError(f"unknown export key {unknown[0]!r}", field=unknown[0])
        return self.create(name or payload.get("name") or "Imported playlist", scrub_secrets(payload.get("document")),
                           actor=actor, action="import")


def _client(value: Any) -> str:
    try:
        return identifier(value, "client_id")
    except ModelValidationError as exc:
        raise PlaylistValidationError(str(exc), field="client_id") from None


def epoch_to_iso(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def age_seconds(iso: str | None, now: float | None = None) -> float | None:
    if not iso:
        return None
    try:
        then = datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
    return max(0.0, (now if now is not None else time.time()) - then)
