"""Server synchronization for the thin display client.

:class:`ClientSync` talks to the render server in the background while the
display keeps playing whatever is already cached:

1. register (or renew) with the server and keep the per-client credential;
2. fetch the client configuration and its assigned playlist document;
3. send a heartbeat with bounded status and the playlist's screen demand;
4. fetch the manifest (``If-None-Match``) and download only artifacts that
   are not cached yet, validating each one before it is published;
5. activate the new playlist and manifest together once every required
   screen is usable locally.

Nothing here blocks playback: the display loop only reads
:meth:`ClientSync.active`, which returns the last activated content.
Failures back off exponentially with jitter.

:class:`ArtifactCache` stores verified artifacts by content hash and bounds
its size without ever evicting what the active or recent manifests use.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import random
import secrets
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from protocol import client_supports_manifest
from remote_display.client_cache import (
    CachedPlaylist,
    ClientCache,
    PlaylistRejected,
    validate_playlist,
)
from remote_display.models import (
    ClientCapabilities,
    ClientDemand,
    ClientStatus,
    ErrorSummary,
    ModelValidationError,
    PackageCapabilities,
)
from remote_display.playlist_store import document_screens

LOGGER = logging.getLogger("desk_display.client_sync")

STATIC_IMAGE = "static_image"
SUPPORTED_ARTIFACT_TYPES = frozenset({STATIC_IMAGE})
MEDIA_EXTENSIONS = {"image/png": "png"}
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
# Decoded pixels are bounded separately from the download, so a small,
# highly compressed image cannot expand into an oversized bitmap.
MAX_DECODED_BYTES = 64 * 1024 * 1024
MAX_RECENT_ERRORS = 8
REQUEST_TIMEOUT_SECONDS = 15


# ── Transport ────────────────────────────────────────────────────────────────


class TransportError(Exception):
    """The server could not be reached."""


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None

    def header(self, name: str) -> str | None:
        lowered = name.lower()
        for key, value in self.headers.items():
            if key.lower() == lowered:
                return value
        return None


class TooLargeError(TransportError):
    """A response exceeded the caller's byte limit."""


class RequestsTransport:
    """HTTP(S) transport over ``requests`` with bounded downloads."""

    def __init__(self, base_url: str, *, verify: bool | str = True, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        import requests

        self.base_url = base_url.rstrip("/")
        self.verify = verify
        self.timeout = timeout
        self._session = requests.Session()
        self._requests = requests

    def __call__(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Any = None,
        max_bytes: int = MAX_ARTIFACT_BYTES,
    ) -> Response:
        try:
            with self._session.request(
                method,
                self.base_url + path,
                headers=dict(headers or {}),
                json=json_body,
                timeout=self.timeout,
                verify=self.verify,
                stream=True,
                allow_redirects=False,
            ) as response:
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_content(64 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise TooLargeError(f"response exceeds {max_bytes} bytes")
                    chunks.append(chunk)
                return Response(response.status_code, b"".join(chunks), dict(response.headers))
        except self._requests.RequestException as exc:
            raise TransportError(f"{type(exc).__name__}: {exc}") from None


# ── Errors ───────────────────────────────────────────────────────────────────


class SyncError(Exception):
    """One sync pass failed; the client keeps playing its cached content."""

    def __init__(self, code: str, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retry_after = retry_after


class InvalidArtifact(SyncError):
    pass


class ErrorLog:
    """Recent error classes, counted and bounded for heartbeat status."""

    def __init__(self, clock: Callable[[], float], limit: int = MAX_RECENT_ERRORS) -> None:
        self._clock = clock
        self._limit = limit
        self._entries: dict[str, list[Any]] = {}
        self._lock = threading.Lock()

    def record(self, code: str, message: str) -> None:
        with self._lock:
            entry = self._entries.pop(code, [message, 0, 0.0])
            entry[0], entry[1], entry[2] = message, entry[1] + 1, self._clock()
            self._entries[code] = entry
            while len(self._entries) > self._limit:
                self._entries.pop(next(iter(self._entries)))

    def summaries(self) -> tuple[ErrorSummary, ...]:
        now = self._clock()
        with self._lock:
            items = list(self._entries.items())
        result = []
        for code, (message, count, seen) in items:
            with contextlib.suppress(ModelValidationError):
                result.append(ErrorSummary(code=code, message=message, count=count,
                                           last_seen_age_seconds=max(0.0, round(now - seen, 1))))
        return tuple(result)


# ── Backoff ──────────────────────────────────────────────────────────────────


class Backoff:
    """Exponential backoff with full jitter, reset on success."""

    def __init__(self, base: float = 2.0, maximum: float = 300.0, rng: random.Random | None = None) -> None:
        self.base = base
        self.maximum = maximum
        self.failures = 0
        self._rng = rng or random.Random()

    def success(self) -> None:
        self.failures = 0

    def failure(self) -> float:
        self.failures += 1
        ceiling = min(self.maximum, self.base * (2 ** min(self.failures - 1, 16)))
        return self._rng.uniform(ceiling / 2, ceiling)


# ── Artifact cache ───────────────────────────────────────────────────────────


def _atomic_write(path: Path, data: bytes, staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = staging / f"{path.name}.{secrets.token_hex(6)}.tmp"
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def validate_artifact(data: bytes, entry: Mapping[str, Any], profile: Mapping[str, Any]) -> None:
    """Raise :class:`InvalidArtifact` unless *data* is exactly what *entry* promises."""

    if entry.get("artifact_type") not in SUPPORTED_ARTIFACT_TYPES:
        raise InvalidArtifact("unsupported_artifact", f"artifact type {entry.get('artifact_type')!r} is not supported")
    if entry.get("media_type") not in MEDIA_EXTENSIONS:
        raise InvalidArtifact("unsupported_media_type", f"media type {entry.get('media_type')!r} is not supported")
    if len(data) != entry.get("length"):
        raise InvalidArtifact("length_mismatch", f"downloaded {len(data)} bytes, expected {entry.get('length')}")
    if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
        raise InvalidArtifact("checksum_mismatch", "artifact checksum does not match the manifest")
    from PIL import Image, UnidentifiedImageError

    expected = (entry.get("width"), entry.get("height"), entry.get("color_mode"))
    if expected != (profile.get("logical_width"), profile.get("logical_height"), profile.get("color_mode")):
        raise InvalidArtifact("wrong_dimensions", "artifact does not match this display's profile")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format != "PNG":
                raise InvalidArtifact("invalid_media", f"artifact is {probe.format}, not PNG")
            bands = len(probe.getbands())
            if probe.width * probe.height * max(1, bands) > MAX_DECODED_BYTES:
                raise InvalidArtifact("too_large", "artifact decodes to more than the allowed size")
            if (probe.width, probe.height, probe.mode) != expected:
                raise InvalidArtifact("wrong_dimensions", "artifact dimensions or color mode do not match the manifest")
            probe.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except InvalidArtifact:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidArtifact("invalid_media", f"artifact is not a valid PNG: {exc}") from None


class ArtifactCache:
    """Verified artifacts stored by content hash, bounded in size.

    Eviction removes the least recently used objects that no protected
    manifest references; objects the active or retained manifests use are
    never removed, even when that leaves the cache over its bound.
    """

    def __init__(self, root: str | os.PathLike[str], *, max_bytes: int, keep_manifests: int = 3) -> None:
        self.root = Path(root).expanduser()
        self.max_bytes = max(0, int(max_bytes))
        self.keep_manifests = max(1, int(keep_manifests))
        self._objects = self.root / "artifacts"
        self._manifests = self.root / "manifests"
        self._staging = self.root / "staging"
        self._lock = threading.RLock()

    # Objects

    @staticmethod
    def object_name(entry: Mapping[str, Any]) -> str:
        sha = str(entry.get("sha256") or "")
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise InvalidArtifact("invalid_manifest", "artifact entry has no valid sha256")
        return f"{sha}.{MEDIA_EXTENSIONS.get(str(entry.get('media_type')), 'bin')}"

    def path_for(self, entry: Mapping[str, Any]) -> Path:
        return self._objects / self.object_name(entry)

    def has(self, entry: Mapping[str, Any]) -> bool:
        try:
            return self.path_for(entry).stat().st_size == entry.get("length")
        except (OSError, SyncError):
            return False

    def store(self, entry: Mapping[str, Any], data: bytes, profile: Mapping[str, Any]) -> Path:
        validate_artifact(data, entry, profile)
        path = self.path_for(entry)
        with self._lock:
            _atomic_write(path, data, self._staging)
        return path

    def read(self, entry: Mapping[str, Any]) -> bytes | None:
        """Return verified bytes, or ``None`` (and drop the file) if corrupt."""

        try:
            path = self.path_for(entry)
            data = path.read_bytes()
        except (OSError, SyncError):
            return None
        if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            LOGGER.warning("Dropping corrupt cached artifact %s", path.name)
            with contextlib.suppress(OSError):
                path.unlink()
            return None
        with contextlib.suppress(OSError):
            os.utime(path)
        return data

    # Manifests (index 0 is active; higher indexes are older)

    def _manifest_path(self, index: int) -> Path:
        return self._manifests / f"{index}.json"

    def manifests(self) -> list[dict[str, Any]]:
        result = []
        for index in range(self.keep_manifests):
            try:
                data = json.loads(self._manifest_path(index).read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except (OSError, ValueError) as exc:
                LOGGER.warning("Ignoring unreadable cached manifest %s: %s", index, exc)
                continue
            if isinstance(data, dict) and isinstance(data.get("manifest_revision"), str):
                result.append(data)
        return result

    def manifest(self, revision: str | None = None) -> dict[str, Any] | None:
        manifests = self.manifests()
        if revision is not None:
            for manifest in manifests:
                if manifest.get("manifest_revision") == revision:
                    return manifest
        return manifests[0] if manifests else None

    def activate_manifest(self, manifest: Mapping[str, Any]) -> None:
        with self._lock:
            current = self.manifests()
            if current and current[0].get("manifest_revision") == manifest.get("manifest_revision"):
                return
            retained = [m for m in current if m.get("manifest_revision") != manifest.get("manifest_revision")]
            ordered = [dict(manifest), *retained][: self.keep_manifests]
            # Write oldest first so a crash never leaves index 0 missing its
            # predecessor; index 0 (the new active manifest) goes last.
            for index in reversed(range(len(ordered))):
                _atomic_write(self._manifest_path(index), json.dumps(ordered[index], sort_keys=True).encode(),
                              self._staging)

    # Eviction

    def protected(self, extra: Iterable[Mapping[str, Any]] = ()) -> set[str]:
        names: set[str] = set()
        for manifest in [*self.manifests(), *extra]:
            for entry in manifest.get("artifacts") or ():
                with contextlib.suppress(SyncError):
                    names.add(self.object_name(entry))
        return names

    def size(self) -> int:
        return sum(p.stat().st_size for p in self._iter_objects())

    def _iter_objects(self) -> Iterable[Path]:
        if self._objects.is_dir():
            yield from (p for p in self._objects.iterdir() if p.is_file())

    def evict(self, extra_protected: Iterable[Mapping[str, Any]] = ()) -> list[str]:
        """Remove unprotected objects, least recently used first, until within bound."""

        with self._lock:
            protected = self.protected(extra_protected)
            objects = []
            total = 0
            for path in self._iter_objects():
                with contextlib.suppress(OSError):
                    stat = path.stat()
                    objects.append((stat.st_mtime, path, stat.st_size))
                    total += stat.st_size
            removed = []
            for _mtime, path, size in sorted(objects):
                if total <= self.max_bytes:
                    break
                if path.name in protected:
                    continue
                with contextlib.suppress(OSError):
                    path.unlink()
                    total -= size
                    removed.append(path.name)
            if total > self.max_bytes:
                LOGGER.warning("Artifact cache is %d bytes over its bound; everything left is in use",
                               total - self.max_bytes)
            if self._staging.is_dir():
                for path in self._staging.iterdir():
                    with contextlib.suppress(OSError):
                        if time.time() - path.stat().st_mtime > 3600:
                            path.unlink()
            return removed


# ── Synchronizer ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ActiveContent:
    """What the display plays: a playlist and the manifest activated with it."""

    playlist: CachedPlaylist | None
    manifest: Mapping[str, Any] | None

    def entry(self, screen: str) -> Mapping[str, Any] | None:
        if not self.manifest:
            return None
        for entry in self.manifest.get("artifacts") or ():
            if entry.get("screen_id") == screen and entry.get("sha256"):
                return entry
        return None

    @property
    def revision(self) -> tuple[str | None, str | None]:
        return (
            None if self.playlist is None else self.playlist.playlist_revision,
            None if self.manifest is None else self.manifest.get("manifest_revision"),
        )


@dataclass
class PlaybackReport:
    """Filled in by the display loop; sent with each heartbeat."""

    playback_state: str = "starting"
    current_screen: str | None = None
    physical_rotation: int = 0


class ClientSync:
    """Keep the local cache in step with the server, never blocking playback."""

    def __init__(
        self,
        capabilities: ClientCapabilities,
        transport: Callable[..., Response],
        cache: ClientCache,
        artifacts: ArtifactCache,
        *,
        enrollment_token: str | None = None,
        sync_interval_seconds: int = 30,
        credential_path: str | os.PathLike[str] | None = None,
        report: Callable[[], PlaybackReport] | None = None,
        clock: Callable[[], float] = time.time,
        rng: random.Random | None = None,
    ) -> None:
        self.capabilities = capabilities
        self.transport = transport
        self.cache = cache
        self.artifacts = artifacts
        self.sync_interval_seconds = sync_interval_seconds
        self._enrollment_token = enrollment_token
        self._credential_path = Path(credential_path) if credential_path else artifacts.root / "client_credential"
        self.report = report or PlaybackReport
        self._clock = clock
        self.backoff = Backoff(rng=rng)
        self.errors = ErrorLog(clock)
        self._lock = threading.RLock()
        self._credential: str | None = self._load_credential()
        # The last manifest fetched (activated or not), for If-None-Match.
        self._fetched: dict[str, Any] | None = None
        self._last_sync: float | None = None
        self._active = self._load_active()
        self._stop = threading.Event()
        self.connected = False

    # Credential (the per-client lease credential, never the enrollment token)

    def _load_credential(self) -> str | None:
        with contextlib.suppress(OSError):
            value = self._credential_path.read_text(encoding="utf-8").strip()
            return value or None
        return None

    def _save_credential(self, value: str | None) -> None:
        self._credential = value
        if value is None:
            with contextlib.suppress(FileNotFoundError):
                self._credential_path.unlink()
            return
        self._credential_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._credential_path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
        os.replace(tmp, self._credential_path)

    # Active content

    def _load_active(self) -> ActiveContent:
        playlist = self.cache.load()
        manifest = self.artifacts.manifest(None if playlist is None else playlist.manifest_revision)
        return ActiveContent(playlist, manifest)

    def active(self) -> ActiveContent:
        with self._lock:
            return self._active

    @property
    def client_id(self) -> str:
        return self.capabilities.client_id

    def last_sync_age(self) -> float | None:
        return None if self._last_sync is None else max(0.0, self._clock() - self._last_sync)

    # HTTP helpers

    def _request(self, method: str, path: str, *, auth: str | None, json_body: Any = None,
                 headers: Mapping[str, str] | None = None, max_bytes: int = 1024 * 1024) -> Response:
        request_headers = dict(headers or {})
        if auth:
            request_headers["Authorization"] = f"Bearer {auth}"
        try:
            return self.transport(method, path, headers=request_headers, json_body=json_body, max_bytes=max_bytes)
        except TooLargeError as exc:
            raise SyncError("too_large", str(exc)) from None
        except TransportError as exc:
            self.connected = False
            raise SyncError("server_unreachable", str(exc)) from None

    def _client_request(self, method: str, path: str, **kwargs: Any) -> Response:
        """A request with the lease credential; re-registers once on 401."""

        if self._credential is None:
            self._register()
        response = self._request(method, path, auth=self._credential, **kwargs)
        if response.status == 401:
            LOGGER.info("Client credential rejected; registering again")
            self._save_credential(None)
            self._register()
            response = self._request(method, path, auth=self._credential, **kwargs)
        return response

    @staticmethod
    def _fail(response: Response, what: str) -> SyncError:
        body = response.json() if isinstance(response.json(), dict) else {}
        retry = body.get("retry_after_seconds")
        return SyncError(
            str(body.get("error") or f"http_{response.status}"),
            f"{what} failed with HTTP {response.status}",
            retry_after=float(retry) if isinstance(retry, int | float) else None,
        )

    # Protocol steps

    def _demand(self, playlist: CachedPlaylist | None) -> dict[str, Any] | None:
        if playlist is None:
            return None
        required, alternates = document_screens(json.loads(json.dumps(playlist.to_dict()["document"])))
        caps = self.capabilities
        return ClientDemand(
            client_id=caps.client_id,
            playlist_revision=playlist.playlist_revision,
            required_screens=required,
            alternate_screens=alternates,
            package_capabilities=PackageCapabilities(
                render_package_versions=caps.render_package_versions,
                image_formats=caps.image_formats,
                supports_animation=caps.supports_animation,
            ),
            sync_interval_seconds=max(5, int(self.sync_interval_seconds)),
        ).to_wire()

    def _register(self) -> None:
        body: dict[str, Any] = {"capabilities": self.capabilities.to_wire()}
        demand = self._demand(self.active().playlist)
        if demand is not None:
            body["demand"] = demand
        if self._credential:
            body["client_credential"] = self._credential
        response = self._request("POST", "/api/v1/register", auth=self._enrollment_token, json_body=body)
        if response.status not in (200, 201):
            raise self._fail(response, "registration")
        payload = response.json() or {}
        credential = payload.get("client_credential")
        if not isinstance(credential, str) or not credential:
            raise SyncError("invalid_response", "registration returned no client credential")
        self._save_credential(credential)
        self.connected = True

    def cache_age(self) -> float | None:
        """Seconds since the active manifest was generated by the server."""

        manifest = self.active().manifest
        generated_at = manifest.get("generated_at") if manifest else None
        if not isinstance(generated_at, str):
            return None
        try:
            generated = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        except ValueError:
            return None
        return max(0.0, round(self._clock() - generated.timestamp(), 1))

    def status(self) -> dict[str, Any]:
        report = self.report()
        active = self.active()
        cache_age = self.cache_age()
        age = self.last_sync_age()
        return ClientStatus(
            client_id=self.client_id,
            playback_state=report.playback_state,
            accepted_revisions=self.cache.accepted_revisions(),
            current_screen=report.current_screen,
            current_playlist=None if active.playlist is None else active.playlist.playlist_id,
            last_sync_age_seconds=None if age is None else round(age, 1),
            cache_age_seconds=cache_age,
            physical_rotation=report.physical_rotation,
            recent_errors=self.errors.summaries(),
        ).to_wire()

    def sync_once(self) -> ActiveContent:
        """Run one full sync pass; raise :class:`SyncError` on failure."""

        prefix = f"/api/v1/clients/{self.client_id}"
        response = self._client_request("GET", f"{prefix}/config")
        if response.status != 200:
            raise self._fail(response, "configuration")
        config = response.json() or {}
        offered = None
        if config.get("playlist"):
            try:
                offered = validate_playlist(config["playlist"])
            except PlaylistRejected as exc:
                raise SyncError(exc.code, f"server playlist rejected: {exc.message}") from None
        target = offered or self.active().playlist

        heartbeat = {"status": self.status()}
        demand = self._demand(target)
        if demand is not None:
            heartbeat["demand"] = demand
        response = self._client_request("POST", f"{prefix}/heartbeat", json_body=heartbeat)
        if response.status != 200:
            raise self._fail(response, "heartbeat")

        fetched = self._fetched
        headers = {"If-None-Match": f'"{fetched["manifest_revision"]}"'} if fetched else {}
        response = self._client_request("GET", f"{prefix}/manifest", headers=headers)
        if response.status == 304 and fetched is not None:
            manifest = dict(fetched)
        elif response.status == 200:
            manifest = response.json()
            if not isinstance(manifest, dict) or not isinstance(manifest.get("manifest_revision"), str):
                raise SyncError("invalid_manifest", "manifest response is not a manifest")
        else:
            raise self._fail(response, "manifest")
        if not client_supports_manifest(manifest):
            raise SyncError("incompatible_manifest", "manifest uses an unsupported schema version")
        if manifest.get("client_id") != self.client_id:
            raise SyncError("invalid_manifest", "manifest is for another client")

        self._download(manifest)
        self._last_sync = self._clock()
        self.connected = True
        if offered is None and config.get("assignment_state") == "unassigned":
            LOGGER.info("No playlist is assigned; keeping the cached playlist")
        if target is not None:
            self._activate(target, manifest)
        self._fetched = manifest
        self.artifacts.evict()
        return self.active()

    def _download(self, manifest: Mapping[str, Any]) -> None:
        profile = manifest
        for entry in manifest.get("artifacts") or ():
            if not entry.get("sha256") or entry.get("artifact_type") not in SUPPORTED_ARTIFACT_TYPES:
                continue
            if self.artifacts.has(entry):
                continue
            url = str(entry.get("url") or "")
            if urlsplit(url).netloc or not url.startswith(f"/api/v1/clients/{self.client_id}/artifacts/"):
                self.errors.record("invalid_manifest", "artifact URL is outside this client's API")
                continue
            length = entry.get("length")
            if type(length) is not int or not 0 < length <= MAX_ARTIFACT_BYTES:
                self.errors.record("too_large", f"artifact for {entry.get('screen_id')} is too large")
                continue
            try:
                response = self._client_request("GET", url, max_bytes=length)
                if response.status != 200:
                    raise self._fail(response, "artifact download")
                self.artifacts.store(entry, response.body, profile)
            except SyncError as exc:
                if exc.code == "server_unreachable":
                    raise
                self.errors.record(exc.code, f"{entry.get('screen_id')}: {exc.message}")
                LOGGER.warning("Artifact for %s rejected: %s", entry.get("screen_id"), exc.message)

    def _usable(self, manifest: Mapping[str, Any]) -> Callable[[str], bool]:
        content = ActiveContent(None, manifest)

        def usable(screen: str) -> bool:
            entry = content.entry(screen)
            return entry is not None and entry.get("artifact_type") in SUPPORTED_ARTIFACT_TYPES \
                and self.artifacts.has(entry)

        return usable

    def _activate(self, playlist: CachedPlaylist, manifest: Mapping[str, Any]) -> None:
        payload = {**playlist.to_dict(), "playlist_schema_version": playlist.schema_version}
        try:
            activated = self.cache.offer(payload, artifact_usable=self._usable(manifest),
                                         manifest_revision=manifest["manifest_revision"])
        except PlaylistRejected as exc:
            self.errors.record(exc.code, exc.message)
            LOGGER.info("Not activating yet: %s", exc.message)
            return
        self.artifacts.activate_manifest(manifest)
        with self._lock:
            override = self.cache.override()
            self._active = ActiveContent(override or activated, manifest)

    # Background loop

    def run(self) -> None:
        """Sync until :meth:`stop`; failures back off with jitter."""

        while not self._stop.is_set():
            delay = self.step()
            self._stop.wait(delay)

    def step(self) -> float:
        """One pass; return seconds until the next one."""

        try:
            self.sync_once()
        except SyncError as exc:
            self.errors.record(exc.code, exc.message)
            delay = self.backoff.failure()
            if exc.retry_after:
                delay = max(delay, exc.retry_after)
            LOGGER.warning("Sync failed (%s); retrying in %.0fs", exc.message, delay)
            return delay
        except Exception as exc:
            self.errors.record("sync_error", type(exc).__name__)
            LOGGER.exception("Unexpected sync failure")
            return self.backoff.failure()
        self.backoff.success()
        return float(self.sync_interval_seconds)

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run, name="client-sync", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()


__all__ = [
    "ActiveContent",
    "ArtifactCache",
    "Backoff",
    "ClientSync",
    "PlaybackReport",
    "RequestsTransport",
    "Response",
    "SyncError",
    "TransportError",
    "validate_artifact",
]
