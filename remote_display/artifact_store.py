"""Immutable, atomically published render artifacts.

The render server writes every artifact here and clients download them by
content address, so a published URL never changes content.  Layout under the
store root::

    objects/<sha[:2]>/<sha256>.<ext>   immutable artifact bytes
    lineages/<lineage>.json            current and previous good output of one
                                       (screen, profile, client scope)
    references.json                    which manifests reference which objects
    staging/                           in-progress publications
    .lock                              advisory lock for writers

Guarantees
    * **Atomic publication**: bytes are written to ``staging/``, fsynced,
      validated (media type, dimensions, colour mode, checksum and render key
      schema) and only then renamed into ``objects/``.  A lineage record is
      replaced atomically afterwards, so no reader ever sees a partial object
      or a record pointing at one.  An interrupted publication leaves only a
      staging file or an unreferenced object, both of which are cleaned up.
    * **Last known good**: invalid output and render failures never replace
      good output.  The previous object stays current and :meth:`resolve`
      reports it as ``fallback`` with the failure, or as ``stale`` once its
      refresh deadline passes or a newer render is pending.
    * **Retention**: each lineage keeps its current object plus a few previous
      ones.  Objects referenced by a lineage or by any client's current or
      previous manifest are never deleted; other objects are deleted only
      after the grace period, measured from when they stopped being
      referenced.
    * **Concurrent readers**: objects are never rewritten, and deleting one
      that a reader already opened does not disturb that reader.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import re
import secrets
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from remote_display.models import ModelValidationError, RenderKey, identifier, screen_id

try:  # pragma: no cover - fcntl is unavailable on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

LOGGER = logging.getLogger("desk_display.artifact_store")

STORE_SCHEMA_VERSION = 1
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
DEFAULT_PREVIOUS_REVISIONS = 3
DEFAULT_REFRESH_SECONDS = 300
STAGING_MAX_AGE_SECONDS = 3600

# Media types the store accepts, and the extension of their immutable URL.
MEDIA_TYPES: Mapping[str, str] = {
    "image/png": "png",
    "application/vnd.desk-display.render-package+json": "json",
}
ARTIFACT_TYPES: Mapping[str, str] = {
    "image/png": "static_image",
    "application/vnd.desk-display.render-package+json": "render_package",
}
_OBJECT_NAME_RE = re.compile(r"^([0-9a-f]{64})\.([a-z]{2,5})$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactStoreError(Exception):
    """Base class for artifact store failures."""


class InvalidArtifactError(ArtifactStoreError):
    """Rendered output failed validation and was not published."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def lineage_id(screen: str, profile: str, client_scope: str | None = None) -> str:
    """Stable identity of successive renders of one screen for one profile."""

    basis = json.dumps([screen_id(screen, "screen_id"), profile, client_scope], separators=(",", ":"))
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


def object_name(sha256: str, media_type: str) -> str:
    return f"{sha256}.{MEDIA_TYPES[media_type]}"


@dataclass(frozen=True)
class ArtifactRecord:
    """Metadata of one published, immutable artifact."""

    sha256: str
    length: int
    media_type: str
    width: int
    height: int
    color_mode: str
    render_key: Mapping[str, Any]
    render_key_digest: str
    generated_at: float
    refresh_deadline: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return object_name(self.sha256, self.media_type)

    @property
    def artifact_type(self) -> str:
        return ARTIFACT_TYPES[self.media_type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "length": self.length,
            "media_type": self.media_type,
            "width": self.width,
            "height": self.height,
            "color_mode": self.color_mode,
            "render_key": dict(self.render_key),
            "render_key_digest": self.render_key_digest,
            "generated_at": self.generated_at,
            "refresh_deadline": self.refresh_deadline,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactRecord:
        return cls(
            sha256=str(data["sha256"]),
            length=int(data["length"]),
            media_type=str(data["media_type"]),
            width=int(data["width"]),
            height=int(data["height"]),
            color_mode=str(data["color_mode"]),
            render_key=dict(data["render_key"]),
            render_key_digest=str(data["render_key_digest"]),
            generated_at=float(data["generated_at"]),
            refresh_deadline=float(data["refresh_deadline"]),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class ResolvedArtifact:
    """What a manifest should list for one lineage right now.

    ``state`` is ``fresh`` (the latest render succeeded and is within its
    refresh deadline), ``stale`` (the output is past its deadline or a newer
    render is pending), ``fallback`` (the latest render failed; the last
    known good output is served) or ``missing`` (nothing good exists yet).
    """

    state: str
    record: ArtifactRecord | None
    failure: Mapping[str, Any] | None = None

    @property
    def stale(self) -> bool:
        return self.state != "fresh"


class ArtifactStore:
    """Content-addressed artifact storage with last-known-good fallback."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        grace_seconds: float = 24 * 3600,
        previous_revisions: int = DEFAULT_PREVIOUS_REVISIONS,
        max_bytes: int | None = None,
        clock=time.time,
    ) -> None:
        self.root = Path(root).expanduser()
        self.grace_seconds = float(grace_seconds)
        self.previous_revisions = max(0, int(previous_revisions))
        self.max_bytes = max_bytes
        self._clock = clock
        self._lock = threading.RLock()

    # ── Paths and locking ──────────────────────────────────────────────────

    @property
    def _objects(self) -> Path:
        return self.root / "objects"

    @property
    def _lineages(self) -> Path:
        return self.root / "lineages"

    @property
    def _staging(self) -> Path:
        return self.root / "staging"

    @property
    def _references(self) -> Path:
        return self.root / "references.json"

    def _object_path(self, name: str) -> Path:
        return self._objects / name[:2] / name

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock, open(self.root / ".lock", "a+") as handle:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _write_json(self, path: Path, data: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._staging.mkdir(parents=True, exist_ok=True)
        tmp = self._staging / f"{path.name}.{secrets.token_hex(6)}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(data, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise
        _fsync_dir(path.parent)

    @staticmethod
    def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except (OSError, ValueError) as exc:
            LOGGER.warning("Ignoring unreadable artifact store file %s: %s", path, exc)
            return default
        return data if isinstance(data, dict) else default

    def _lineage(self, lineage: str) -> dict[str, Any]:
        return self._read_json(
            self._lineages / f"{lineage}.json",
            {"schema_version": STORE_SCHEMA_VERSION, "current": None, "previous": [], "failure": None, "pending": None},
        )

    # ── Publication ────────────────────────────────────────────────────────

    def publish(
        self,
        key: RenderKey,
        data: bytes,
        *,
        media_type: str = "image/png",
        expected_sha256: str | None = None,
        refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
        metadata: Mapping[str, Any] | None = None,
        package: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Validate *data* for *key* and publish it as the lineage's current output.

        *package* is an optional render package for the same key; it is
        validated, stored as its own immutable object first, and referenced
        from the record's ``metadata["package"]``, so it is published and
        retained together with the still image.

        Raises :class:`InvalidArtifactError` (and records the failure, keeping
        the previous good output) when the output is unusable.
        """

        try:
            return self._publish(key, data, media_type, expected_sha256, refresh_seconds, metadata or {}, package)
        except InvalidArtifactError as exc:
            self.record_failure(key, exc.code, exc.message)
            raise

    def publish_image(self, key: RenderKey, image: Any, **kwargs: Any) -> ArtifactRecord:
        """Encode a Pillow image as PNG and publish it."""

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return self.publish(key, buffer.getvalue(), media_type="image/png", **kwargs)

    def _publish(
        self,
        key: RenderKey,
        data: bytes,
        media_type: str,
        expected_sha256: str | None,
        refresh_seconds: float,
        metadata: Mapping[str, Any],
        package: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        if not isinstance(key, RenderKey):
            raise InvalidArtifactError("invalid_render_key", "render key must be a RenderKey")
        try:
            key = RenderKey.from_wire(key.to_wire())
        except ModelValidationError as exc:
            raise InvalidArtifactError("invalid_render_key", str(exc)) from None
        if media_type not in MEDIA_TYPES:
            raise InvalidArtifactError("unsupported_media_type", f"unsupported media type {media_type!r}")
        if not isinstance(data, bytes | bytearray) or not data:
            raise InvalidArtifactError("empty_output", "render produced no output")
        if len(data) > MAX_ARTIFACT_BYTES:
            raise InvalidArtifactError("too_large", f"output exceeds {MAX_ARTIFACT_BYTES} bytes")
        if metadata.get("error_image"):
            raise InvalidArtifactError("error_image", "renderer produced an error placeholder")
        if not isinstance(refresh_seconds, int | float) or refresh_seconds <= 0:
            raise InvalidArtifactError("invalid_metadata", "refresh_seconds must be positive")
        try:
            json.dumps(dict(metadata))
        except (TypeError, ValueError):
            raise InvalidArtifactError("invalid_metadata", "metadata must be JSON-serializable") from None

        if package is not None:
            metadata = {**metadata, "package": self._store_package(key, package)}
        self._staging.mkdir(parents=True, exist_ok=True)
        tmp = self._staging / f"{secrets.token_hex(8)}.part"
        try:
            with open(tmp, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            # Verify what reached the disk, not the buffer we meant to write.
            written = tmp.read_bytes()
            sha256 = hashlib.sha256(written).hexdigest()
            if expected_sha256 is not None and sha256 != expected_sha256:
                raise InvalidArtifactError("checksum_mismatch", "output does not match its declared checksum")
            width, height, mode = _inspect(written, media_type, key)
            name = object_name(sha256, media_type)
            now = self._clock()
            record = ArtifactRecord(
                sha256=sha256,
                length=len(written),
                media_type=media_type,
                width=width,
                height=height,
                color_mode=mode,
                render_key=key.to_wire(),
                render_key_digest=key.digest,
                generated_at=now,
                refresh_deadline=now + float(refresh_seconds),
                metadata=dict(metadata),
            )
            with self._locked():
                target = self._object_path(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    os.unlink(tmp)  # identical content is already published
                else:
                    os.replace(tmp, target)
                    _fsync_dir(target.parent)
                lineage = lineage_id(key.screen_id, key.render_profile, key.client_scope)
                slot = self._lineage(lineage)
                current = slot.get("current")
                previous = [p for p in slot.get("previous", []) if p.get("sha256") != sha256]
                if current and current.get("sha256") != sha256:
                    previous.insert(0, current)
                evicted = previous[self.previous_revisions:]
                slot.update({
                    "schema_version": STORE_SCHEMA_VERSION,
                    "screen_id": key.screen_id,
                    "render_profile": key.render_profile,
                    "client_scope": key.client_scope,
                    "current": record.to_dict(),
                    "previous": previous[: self.previous_revisions],
                    "failure": None,
                    "pending": None,
                })
                self._write_json(self._lineages / f"{lineage}.json", slot)
                if evicted:
                    self._release_hashes({h for e in evicted for h in record_hashes(e)}, now)
            return record
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)

    def _store_package(self, key: RenderKey, package: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and store *key*'s render package; return its reference."""

        from remote_display.render_package import (
            MAX_PACKAGE_BYTES,
            RENDER_PACKAGE_MEDIA_TYPE,
            PackageError,
            package_bytes,
            validate_package,
        )

        try:
            document = validate_package(package, key=key)
        except PackageError as exc:
            raise InvalidArtifactError("invalid_package", exc.message) from None
        data = package_bytes(document)
        if len(data) > MAX_PACKAGE_BYTES:
            raise InvalidArtifactError("too_large", f"render package exceeds {MAX_PACKAGE_BYTES} bytes")
        sha256 = hashlib.sha256(data).hexdigest()
        self._staging.mkdir(parents=True, exist_ok=True)
        tmp = self._staging / f"{secrets.token_hex(8)}.part"
        try:
            with open(tmp, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            with self._locked():
                target = self._object_path(object_name(sha256, RENDER_PACKAGE_MEDIA_TYPE))
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    os.replace(tmp, target)
                    _fsync_dir(target.parent)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
        return {
            "sha256": sha256,
            "length": len(data),
            "media_type": RENDER_PACKAGE_MEDIA_TYPE,
            "kind": document["kind"],
            "classification": document["classification"],
            "render_package_schema_version": document["render_package_schema_version"],
        }

    def record_failure(self, key: RenderKey, code: str, message: str) -> None:
        """Remember that rendering *key* failed; the previous output stays current."""

        lineage = lineage_id(key.screen_id, key.render_profile, key.client_scope)
        with self._locked():
            slot = self._lineage(lineage)
            failure = slot.get("failure") or {}
            slot.update({
                "screen_id": key.screen_id,
                "render_profile": key.render_profile,
                "client_scope": key.client_scope,
                "failure": {
                    "code": str(code)[:64],
                    "message": str(message)[:300],
                    "at": self._clock(),
                    "render_key_digest": key.digest,
                    "consecutive": int(failure.get("consecutive", 0)) + 1,
                },
                "pending": None,
            })
            self._write_json(self._lineages / f"{lineage}.json", slot)
        LOGGER.warning("Render of %s for %s failed (%s): %s", key.screen_id, key.render_profile, code, message)

    def mark_pending(self, key: RenderKey) -> None:
        """Note that a newer render for this lineage has started."""

        lineage = lineage_id(key.screen_id, key.render_profile, key.client_scope)
        with self._locked():
            slot = self._lineage(lineage)
            current = slot.get("current") or {}
            if current.get("render_key_digest") == key.digest:
                return
            slot.update({
                "screen_id": key.screen_id,
                "render_profile": key.render_profile,
                "client_scope": key.client_scope,
                "pending": {"render_key_digest": key.digest, "at": self._clock()},
            })
            self._write_json(self._lineages / f"{lineage}.json", slot)

    # ── Reading ────────────────────────────────────────────────────────────

    def resolve(self, screen: str, profile: str, client_scope: str | None = None) -> ResolvedArtifact:
        """Return the output a manifest should list for this lineage now."""

        slot = self._lineage(lineage_id(screen, profile, client_scope))
        current = slot.get("current")
        failure = slot.get("failure")
        record = ArtifactRecord.from_dict(current) if current else None
        if record is not None and not self._object_path(record.name).is_file():
            LOGGER.error("Artifact %s for %s is missing from the store", record.name, screen)
            record = None
        public_failure = None if not failure else {
            "code": failure.get("code"),
            "message": failure.get("message"),
            "at": _iso(failure.get("at")),
            "consecutive": failure.get("consecutive", 1),
        }
        if record is None:
            return ResolvedArtifact("missing", None, public_failure)
        if failure:
            return ResolvedArtifact("fallback", record, public_failure)
        if slot.get("pending") or self._clock() >= record.refresh_deadline:
            return ResolvedArtifact("stale", record, None)
        return ResolvedArtifact("fresh", record, None)

    def previous(self, screen: str, profile: str, client_scope: str | None = None) -> list[ArtifactRecord]:
        """Previous good revisions of a lineage, newest first."""

        slot = self._lineage(lineage_id(screen, profile, client_scope))
        return [ArtifactRecord.from_dict(p) for p in slot.get("previous", [])]

    def open_object(self, name: str) -> tuple[Path, str] | None:
        """Return ``(path, media_type)`` for an immutable object name, if retained."""

        match = _OBJECT_NAME_RE.match(name or "")
        if not match:
            return None
        media_type = next((m for m, ext in MEDIA_TYPES.items() if ext == match.group(2)), None)
        if media_type is None:
            return None
        path = self._object_path(name)
        return (path, media_type) if path.is_file() else None

    # ── References and garbage collection ──────────────────────────────────

    def reference(self, holder: str, hashes: Iterable[str]) -> None:
        """Record the objects *holder*'s latest manifest lists.

        The holder's previous manifest stays referenced too, so a client that
        is still downloading it is not cut off.
        """

        holder = identifier(holder, "holder")
        wanted = sorted({h for h in hashes if _SHA_RE.match(h)})
        with self._locked():
            refs = self._read_refs()
            entry = refs["holders"].get(holder) or {"current": [], "previous": []}
            if entry["current"] == wanted:
                return
            now = self._clock()
            dropped = set(entry["previous"]) - set(wanted) - set(entry["current"])
            refs["holders"][holder] = {"current": wanted, "previous": entry["current"], "at": now}
            for sha in wanted:
                refs["released"].pop(sha, None)
            self._mark_released(refs, dropped, now)
            self._write_json(self._references, refs)

    def release(self, holder: str) -> None:
        """Forget a holder (for example a client whose lease expired)."""

        with self._locked():
            refs = self._read_refs()
            entry = refs["holders"].pop(holder, None)
            if entry is None:
                return
            self._mark_released(refs, set(entry["current"]) | set(entry["previous"]), self._clock())
            self._write_json(self._references, refs)

    def prune_holders(self, keep: Iterable[str]) -> list[str]:
        """Release every holder not in *keep*; return the released holders."""

        keep = set(keep)
        with self._locked():
            refs = self._read_refs()
            gone = sorted(set(refs["holders"]) - keep)
            if not gone:
                return []
            now = self._clock()
            for holder in gone:
                entry = refs["holders"].pop(holder)
                self._mark_released(refs, set(entry["current"]) | set(entry["previous"]), now)
            self._write_json(self._references, refs)
        return gone

    def referenced_by(self, holder: str) -> set[str]:
        entry = self._read_refs()["holders"].get(holder) or {}
        return set(entry.get("current", [])) | set(entry.get("previous", []))

    def collect_garbage(self) -> list[str]:
        """Delete unreferenced objects whose grace period has passed."""

        removed: list[str] = []
        with self._locked():
            now = self._clock()
            protected = self._protected()
            refs = self._read_refs()
            released = refs["released"]
            for path in self._iter_objects():
                sha = path.name.split(".", 1)[0]
                if sha in protected:
                    continue
                try:
                    since = float(released.get(sha) or path.stat().st_mtime)
                except OSError:
                    continue
                if now - since < self.grace_seconds:
                    continue
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
                    removed.append(path.name)
                released.pop(sha, None)
            for sha in list(released):
                if not any(self._objects.glob(f"{sha[:2]}/{sha}.*")):
                    released.pop(sha, None)
            self._write_json(self._references, refs)
            self._clean_staging(now)
            total = sum(p.stat().st_size for p in self._iter_objects())
        if self.max_bytes is not None and total > self.max_bytes:
            LOGGER.warning(
                "Artifact store holds %d bytes, over its %d byte budget; everything left is "
                "referenced or inside the %ds grace period", total, self.max_bytes, int(self.grace_seconds),
            )
        if removed:
            LOGGER.info("Artifact store removed %d unreferenced objects", len(removed))
        return removed

    def stats(self) -> dict[str, Any]:
        objects = list(self._iter_objects())
        return {
            "objects": len(objects),
            "bytes": sum(p.stat().st_size for p in objects if p.exists()),
            "lineages": len(list(self._lineages.glob("*.json"))) if self._lineages.is_dir() else 0,
        }

    def _protected(self) -> set[str]:
        protected: set[str] = set()
        if self._lineages.is_dir():
            for path in self._lineages.glob("*.json"):
                slot = self._read_json(path, {})
                for entry in [slot.get("current")] + list(slot.get("previous") or []):
                    if entry:
                        protected.update(record_hashes(entry))
        for entry in self._read_refs()["holders"].values():
            protected.update(entry.get("current", []))
            protected.update(entry.get("previous", []))
        return protected

    def _read_refs(self) -> dict[str, Any]:
        refs = self._read_json(self._references, {})
        refs.setdefault("schema_version", STORE_SCHEMA_VERSION)
        if not isinstance(refs.get("holders"), dict):
            refs["holders"] = {}
        if not isinstance(refs.get("released"), dict):
            refs["released"] = {}
        return refs

    def _release_hashes(self, hashes: set[str], now: float) -> None:
        refs = self._read_refs()
        self._mark_released(refs, hashes, now)
        self._write_json(self._references, refs)

    @staticmethod
    def _mark_released(refs: dict[str, Any], hashes: set[str], now: float) -> None:
        for sha in hashes:
            refs["released"][sha] = now

    def _iter_objects(self) -> Iterator[Path]:
        if not self._objects.is_dir():
            return
        for path in self._objects.glob("*/*"):
            if _OBJECT_NAME_RE.match(path.name):
                yield path

    def _clean_staging(self, now: float) -> None:
        if not self._staging.is_dir():
            return
        for path in self._staging.iterdir():
            with contextlib.suppress(OSError):
                if now - path.stat().st_mtime > STAGING_MAX_AGE_SECONDS:
                    path.unlink()


def record_hashes(record: Mapping[str, Any] | ArtifactRecord) -> set[str]:
    """Objects a record keeps alive: its still image and any render package."""

    if isinstance(record, ArtifactRecord):
        record = record.to_dict()
    hashes = {str(record["sha256"])} if record.get("sha256") else set()
    package = (record.get("metadata") or {}).get("package")
    if isinstance(package, Mapping) and _SHA_RE.match(str(package.get("sha256") or "")):
        hashes.add(package["sha256"])
    return hashes


def _inspect(data: bytes, media_type: str, key: RenderKey) -> tuple[int, int, str]:
    """Validate output against its render key and return ``(width, height, mode)``."""

    if media_type == "image/png":
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(io.BytesIO(data)) as probe:
                if probe.format != "PNG":
                    raise InvalidArtifactError("invalid_media", f"output is {probe.format}, not PNG")
                probe.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                width, height, mode = image.width, image.height, image.mode
        except InvalidArtifactError:
            raise
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise InvalidArtifactError("invalid_media", f"output is not a valid PNG: {exc}") from None
        if (width, height) != (key.width, key.height):
            raise InvalidArtifactError(
                "wrong_dimensions", f"output is {width}x{height}, expected {key.width}x{key.height}"
            )
        if mode != key.color_mode:
            raise InvalidArtifactError("wrong_color_mode", f"output is {mode}, expected {key.color_mode}")
        return width, height, mode
    from remote_display.render_package import PackageError, validate_package

    try:
        validate_package(data, key=key)
    except PackageError as exc:
        code = "invalid_media" if exc.code == "invalid_json" else "invalid_schema"
        raise InvalidArtifactError(code, f"invalid render package: {exc.message}") from None
    return key.width, key.height, key.color_mode


def _fsync_dir(path: Path) -> None:
    with contextlib.suppress(OSError):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


__all__ = [
    "ARTIFACT_TYPES",
    "ArtifactRecord",
    "ArtifactStore",
    "ArtifactStoreError",
    "InvalidArtifactError",
    "MEDIA_TYPES",
    "ResolvedArtifact",
    "lineage_id",
    "object_name",
    "record_hashes",
]
