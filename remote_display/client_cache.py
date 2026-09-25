"""Revisioned client-side cache of the server-assigned playlist.

A display client keeps its assigned playlist on disk so it can boot and play
without the server.  Layout under the cache root::

    playlist/current.json    the active, validated playlist (read-only locally)
    playlist/previous.json   the playlist it replaced
    playback.json            local playback state, kept separately
    override.json            an optional local emergency override
    staging/                 in-progress writes

Rules
    * **Validate before activation**: a new playlist must match its declared
      schema and content revision, reference only known screens, build a
      scheduler, and have every artifact it needs usable locally.  Anything
      else is rejected and the current playlist keeps playing.
    * **Atomic**: the new playlist is written to ``staging/``, fsynced and
      renamed into place; the old one becomes ``previous.json``.  A crash at
      any point leaves a complete current or previous playlist.
    * **Offline boot**: :meth:`ClientCache.load` returns the newest valid
      cached playlist, falling back to the previous copy if the current one
      is missing or corrupt.
    * **Acknowledgment**: :meth:`ClientCache.accepted_revisions` is what the
      next heartbeat reports, so the server sees which revision the client
      actually switched to.
    * **Separate playback state**: position, current screen, hold, touch-focus
      return target and history live in ``playback.json`` and are reconciled
      against each new playlist, never stored inside server configuration.
    * **Visible override**: a local emergency override is a separate file and
      is always reported as such; it never modifies the cached server playlist.
"""
from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from protocol_versions import PLAYLIST_SCHEMA_VERSION
from remote_display.models import AcceptedRevisions, ModelValidationError, identifier, revision
from remote_display.playlist_store import (
    PlaylistValidationError,
    _is_enabled,
    document_revision,
    document_screens,
    validate_document,
)

LOGGER = logging.getLogger("desk_display.client_cache")

CACHE_SCHEMA_VERSION = 1
SUPPORTED_PLAYLIST_SCHEMA_VERSIONS = frozenset({PLAYLIST_SCHEMA_VERSION})
MAX_HISTORY = 50


class PlaylistRejected(Exception):
    """A new playlist was not activated; the current one keeps playing."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class CachedPlaylist:
    """A validated playlist.  ``document`` is read-only."""

    playlist_id: str
    playlist_revision: str
    schema_version: int
    document: Mapping[str, Any]
    manifest_revision: str | None = None
    source: str = "server"  # "server" or "override"

    @property
    def screens(self) -> tuple[str, ...]:
        return playback_order(self.document)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "playlist_id": self.playlist_id,
            "playlist_revision": self.playlist_revision,
            "schema_version": self.schema_version,
            "document": _thaw(self.document),
            "manifest_revision": self.manifest_revision,
        }


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


@dataclass
class PlaybackState:
    """Local playback position; owned by the client, never by the server."""

    playlist_id: str | None = None
    playlist_revision: str | None = None
    sequence_index: int = 0
    step_index: int = 0
    current_screen: str | None = None
    hold: Mapping[str, Any] | None = None
    focus_return: str | None = None
    history: list[str] = field(default_factory=list)

    def remember(self, screen: str) -> None:
        self.history.append(screen)
        del self.history[:-MAX_HISTORY]


def playback_order(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Screens in the order a playlist plays them, without duplicates.

    ``sequence`` entries come first in their order; the remaining enabled
    screens follow by ID, since the server stores documents with sorted keys
    and object order carries no meaning on the wire.
    """

    order: list[str] = []
    playlists = document.get("playlists") or {}
    for entry in document.get("sequence") or ():
        if not isinstance(entry, Mapping):
            continue
        if entry.get("playlist") in playlists:
            for step in playlists[entry["playlist"]].get("steps") or ():
                if isinstance(step, Mapping) and step.get("screen"):
                    order.append(step["screen"])
        elif entry.get("screen"):
            order.append(entry["screen"])
    for screen, spec in sorted((document.get("screens") or {}).items()):
        if _is_enabled(_thaw(spec)):
            order.append(screen)
    seen: set[str] = set()
    return tuple(s for s in order if not (s in seen or seen.add(s)))


def validate_playlist(payload: Mapping[str, Any]) -> CachedPlaylist:
    """Validate a server playlist payload; raise :class:`PlaylistRejected`."""

    if not isinstance(payload, Mapping):
        raise PlaylistRejected("invalid_playlist", "playlist payload must be an object")
    try:
        playlist_id = identifier(payload.get("playlist_id"), "playlist_id")
        playlist_revision = revision(payload.get("playlist_revision"), "playlist_revision")
    except ModelValidationError as exc:
        raise PlaylistRejected("invalid_playlist", str(exc)) from None
    schema_version = payload.get("playlist_schema_version", payload.get("schema_version"))
    if type(schema_version) is not int or schema_version not in SUPPORTED_PLAYLIST_SCHEMA_VERSIONS:
        raise PlaylistRejected("unsupported_schema", f"playlist schema {schema_version!r} is not supported")
    try:
        document = validate_document(copy.deepcopy(payload.get("document")))
    except PlaylistValidationError as exc:
        raise PlaylistRejected("invalid_playlist", str(exc)) from None
    if document_revision(document) != playlist_revision:
        raise PlaylistRejected("revision_mismatch", "playlist content does not match its declared revision")
    if not playback_order(document):
        raise PlaylistRejected("empty_playlist", "playlist has no screens to play")
    manifest_revision = payload.get("manifest_revision")
    return CachedPlaylist(
        playlist_id=playlist_id,
        playlist_revision=playlist_revision,
        schema_version=schema_version,
        document=_freeze(document),
        manifest_revision=manifest_revision if isinstance(manifest_revision, str) else None,
    )


def required_screens(playlist: CachedPlaylist) -> set[str]:
    required, _alternates = document_screens(_thaw(playlist.document))
    return set(required)


class ClientCache:
    """On-disk playlist cache and playback state for one display client."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser()
        self._lock = threading.RLock()

    # ── Paths and atomic IO ────────────────────────────────────────────────

    @property
    def _current(self) -> Path:
        return self.root / "playlist" / "current.json"

    @property
    def _previous(self) -> Path:
        return self.root / "playlist" / "previous.json"

    @property
    def _playback(self) -> Path:
        return self.root / "playback.json"

    @property
    def _override(self) -> Path:
        return self.root / "override.json"

    def _write(self, path: Path, data: Mapping[str, Any]) -> None:
        staging = self.root / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = staging / f"{path.name}.{secrets.token_hex(6)}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(data, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise
        with contextlib.suppress(OSError):
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            LOGGER.warning("Ignoring unreadable cache file %s: %s", path, exc)
            return None
        return data if isinstance(data, dict) else None

    def _load_file(self, path: Path) -> CachedPlaylist | None:
        data = self._read(path)
        if data is None:
            return None
        try:
            return validate_playlist({**data, "playlist_schema_version": data.get("schema_version")})
        except PlaylistRejected as exc:
            LOGGER.warning("Ignoring invalid cached playlist %s: %s", path, exc.message)
            return None

    # ── Playlist ───────────────────────────────────────────────────────────

    def load(self) -> CachedPlaylist | None:
        """The playlist to play at boot: override, current, else previous."""

        override = self.override()
        if override is not None:
            return override
        return self._load_file(self._current) or self._load_file(self._previous)

    def server_playlist(self) -> CachedPlaylist | None:
        """The newest valid cached server playlist, ignoring any override."""

        return self._load_file(self._current) or self._load_file(self._previous)

    def previous(self) -> CachedPlaylist | None:
        return self._load_file(self._previous)

    def offer(
        self,
        payload: Mapping[str, Any],
        *,
        artifact_usable: Callable[[str], bool],
        manifest_revision: str | None = None,
    ) -> CachedPlaylist:
        """Validate and atomically activate a new server playlist.

        ``artifact_usable(screen)`` reports whether a screen's artifacts are
        downloaded and verified locally.  Every required screen must be
        usable before the playlist is activated.  Raises
        :class:`PlaylistRejected` and leaves the current playlist in place
        otherwise.
        """

        playlist = validate_playlist(payload)
        if manifest_revision is not None:
            playlist = CachedPlaylist(**{**playlist.__dict__, "manifest_revision": manifest_revision})
        missing = sorted(s for s in required_screens(playlist) if not artifact_usable(s))
        if missing:
            raise PlaylistRejected("artifacts_unavailable", "artifacts not yet usable: " + ", ".join(missing))
        with self._lock:
            current = self._load_file(self._current)
            if current is not None and current.playlist_revision == playlist.playlist_revision \
                    and current.playlist_id == playlist.playlist_id \
                    and current.manifest_revision == playlist.manifest_revision:
                return current
            if current is not None:
                self._write(self._previous, current.to_dict())
            self._write(self._current, playlist.to_dict())
        LOGGER.info("Activated playlist %s revision %s", playlist.playlist_id, playlist.playlist_revision)
        return playlist

    def accepted_revisions(self) -> AcceptedRevisions:
        """What the next heartbeat reports as accepted."""

        playlist = self.server_playlist()
        if playlist is None:
            return AcceptedRevisions()
        return AcceptedRevisions(
            playlist_revision=playlist.playlist_revision,
            manifest_revision=playlist.manifest_revision,
        )

    # ── Emergency override ─────────────────────────────────────────────────

    def set_override(self, document: Mapping[str, Any], *, reason: str) -> CachedPlaylist:
        """Play a local playlist instead of the server's until cleared."""

        validated = validate_document(copy.deepcopy(document))
        self._write(self._override, {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "reason": str(reason)[:200],
            "document": validated,
        })
        return self.override()

    def clear_override(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._override.unlink()

    def override(self) -> CachedPlaylist | None:
        data = self._read(self._override)
        if data is None:
            return None
        try:
            document = validate_document(data.get("document"))
        except PlaylistValidationError as exc:
            LOGGER.warning("Ignoring invalid local override: %s", exc)
            return None
        return CachedPlaylist(
            playlist_id="local-override",
            playlist_revision=document_revision(document),
            schema_version=PLAYLIST_SCHEMA_VERSION,
            document=_freeze(document),
            source="override",
        )

    # ── Playback state ─────────────────────────────────────────────────────

    def load_playback(self) -> PlaybackState:
        data = self._read(self._playback) or {}
        state = PlaybackState()
        for name in ("playlist_id", "playlist_revision", "current_screen", "focus_return"):
            if isinstance(data.get(name), str):
                setattr(state, name, data[name])
        for name in ("sequence_index", "step_index"):
            if type(data.get(name)) is int and data[name] >= 0:
                setattr(state, name, data[name])
        if isinstance(data.get("hold"), dict):
            state.hold = data["hold"]
        if isinstance(data.get("history"), list):
            state.history = [s for s in data["history"] if isinstance(s, str)][-MAX_HISTORY:]
        return state

    def save_playback(self, state: PlaybackState) -> None:
        data = asdict(state)
        data["hold"] = None if state.hold is None else dict(state.hold)
        data["cache_schema_version"] = CACHE_SCHEMA_VERSION
        self._write(self._playback, data)


def reconcile_playback(state: PlaybackState, playlist: CachedPlaylist) -> PlaybackState:
    """Carry playback state onto *playlist*.

    The current screen is kept when the new playlist still plays it;
    otherwise playback moves deterministically to the new playlist's first
    screen.  History, hold and focus-return entries for screens the playlist
    no longer has are dropped.
    """

    order = playlist.screens
    valid = set(order)
    same_playlist = state.playlist_id == playlist.playlist_id and state.playlist_revision == playlist.playlist_revision
    result = PlaybackState(
        playlist_id=playlist.playlist_id,
        playlist_revision=playlist.playlist_revision,
        sequence_index=state.sequence_index if same_playlist else 0,
        step_index=state.step_index if same_playlist else 0,
        history=[s for s in state.history if s in valid][-MAX_HISTORY:],
        focus_return=state.focus_return if state.focus_return in valid else None,
    )
    if state.current_screen in valid:
        result.current_screen = state.current_screen
        hold_screen = (state.hold or {}).get("screen")
        result.hold = state.hold if hold_screen in valid else None
    else:
        result.current_screen = order[0]
        result.sequence_index = result.step_index = 0
    return result


__all__ = [
    "CachedPlaylist",
    "ClientCache",
    "PlaybackState",
    "PlaylistRejected",
    "playback_order",
    "reconcile_playback",
    "required_screens",
    "validate_playlist",
]

