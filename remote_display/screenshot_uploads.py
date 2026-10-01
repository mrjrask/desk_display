"""Screenshots display clients upload to the server.

``scripts/collect_client_screenshots.py`` fetches each display's screenshots
from its own config UI, which only works for displays on the server's network.
A display on another network (behind a VPN or another router) can still reach
the server, so it pushes its latest screenshots there instead:

* :class:`UploadQueue` (client) remembers, per screen, the newest screenshot
  file the panel saved and hands out the ones due for upload, at most one per
  screen every few minutes.  ``ClientSync`` sends them over its existing
  authenticated connection (``PUT /api/v1/clients/<id>/screenshots``) after a
  successful heartbeat, once the server advertises
  ``client_screenshot_upload_versions``.
* :class:`ScreenshotInbox` (server) keeps only the latest image per screen per
  client under ``.runtime/server/client_screenshots/<client id>/`` with an
  ``index.json``, bounded by a per-image size cap, a total size cap and a
  retention period.  The config UI lists and serves them to the collector
  (``/api/clients/<id>/uploaded-screenshots``).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("desk_display.screenshot_uploads")

WIRE_VERSION = 1
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPLOAD_DIR = _PROJECT_ROOT / ".runtime" / "server" / "client_screenshots"
INDEX_NAME = "index.json"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_SCREEN_ID_LENGTH = 128
MAX_DIMENSION = 4096
DEFAULT_MAX_IMAGE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 64 * 1024 * 1024
DEFAULT_RETENTION_SECONDS = 7 * 24 * 3600
# Client side: a screen is uploaded at most this often, and at most this many
# screenshots go up after each heartbeat, so a pass stays short.
DEFAULT_UPLOAD_INTERVAL_SECONDS = 600
MIN_UPLOAD_INTERVAL_SECONDS = 60
MAX_UPLOADS_PER_PASS = 10
_FILE_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}-[0-9a-f]{10}\.png$")
_CLIENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class UploadRejected(Exception):
    status = 400
    code = "invalid_screenshot"

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        if status is not None:
            self.status = status
        if code is not None:
            self.code = code

    def as_response(self) -> dict[str, Any]:
        return {"error": self.code, "message": str(self)}


def upload_dir(env: Mapping[str, str] | None = None) -> Path:
    """Where the server keeps uploaded client screenshots."""

    source = os.environ if env is None else env
    raw = (source.get("DESK_DISPLAY_SCREENSHOT_UPLOAD_DIR") or "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_UPLOAD_DIR


def screen_id(value: Any) -> str:
    """Validate a screen ID from an upload (screen IDs may contain spaces)."""

    if not isinstance(value, str) or not value.strip():
        raise UploadRejected("screen is required")
    if len(value) > MAX_SCREEN_ID_LENGTH or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise UploadRejected("screen is not a valid screen ID")
    return value


def file_name(screen: str) -> str:
    """A safe, stable file name for *screen* (distinct IDs never collide)."""

    prefix = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in screen.strip())[:80].strip("_")
    digest = hashlib.sha256(screen.encode("utf-8")).hexdigest()[:10]
    return f"{prefix or 'screen'}-{digest}.png"


def _iso(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: Any) -> float | None:
    """An ISO-8601 timestamp as epoch seconds, or None when absent or invalid."""

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def validate_png(data: bytes) -> tuple[int, int]:
    """Check *data* is a PNG of sane dimensions; return ``(width, height)``."""

    if not data.startswith(PNG_SIGNATURE):
        raise UploadRejected("only PNG screenshots are accepted", status=415, code="unsupported_media_type")
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width > MAX_DIMENSION or height > MAX_DIMENSION:
                raise UploadRejected(f"screenshot is larger than {MAX_DIMENSION}x{MAX_DIMENSION}")
            image.verify()
    except UploadRejected:
        raise
    except Exception as exc:  # noqa: BLE001 - any decoder error means a bad image
        raise UploadRejected(f"not a readable PNG ({type(exc).__name__})") from None
    return width, height


def _write_atomically(target: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


# ── Server ───────────────────────────────────────────────────────────────────


class ScreenshotInbox:
    """The latest uploaded screenshot of each screen, per client, on disk."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        retention_seconds: float = DEFAULT_RETENTION_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root)
        self.max_image_bytes = int(max_image_bytes)
        self.max_total_bytes = int(max_total_bytes)
        self.retention_seconds = float(retention_seconds)
        self._clock = clock
        self._lock = threading.Lock()

    def _client_dir(self, client_id: str) -> Path:
        if not isinstance(client_id, str) or not _CLIENT_RE.match(client_id):
            raise UploadRejected("invalid client id")
        return self.root / client_id

    @staticmethod
    def _read_index(folder: Path) -> dict[str, dict[str, Any]]:
        try:
            payload = json.loads((folder / INDEX_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        screens = payload.get("screens") if isinstance(payload, dict) else None
        if not isinstance(screens, dict):
            return {}
        return {
            key: value for key, value in screens.items()
            if isinstance(key, str) and isinstance(value, dict)
            and isinstance(value.get("file"), str) and _FILE_RE.match(value["file"])
        }

    @staticmethod
    def _write_index(folder: Path, screens: Mapping[str, Any]) -> None:
        data = json.dumps({"version": WIRE_VERSION, "screens": screens}, indent=1).encode("utf-8")
        _write_atomically(folder / INDEX_NAME, data + b"\n")

    def save(self, client_id: str, screen: str, data: bytes, *, captured_at: float | None = None) -> dict[str, Any]:
        """Store *data* as *client_id*'s latest screenshot of *screen*."""

        screen = screen_id(screen)
        if len(data) > self.max_image_bytes:
            raise UploadRejected(f"screenshot exceeds {self.max_image_bytes} bytes", status=413, code="too_large")
        width, height = validate_png(data)
        folder = self._client_dir(client_id)
        now = self._clock()
        if captured_at is None or captured_at > now + 300:
            captured_at = now
        name = file_name(screen)
        with self._lock:
            folder.mkdir(parents=True, exist_ok=True)
            _write_atomically(folder / name, data)
            screens = self._read_index(folder)
            # Re-inserted at the end: the order uploads arrive in roughly
            # follows the display's playback order.
            screens.pop(screen, None)
            entry = {"file": name, "captured_at": _iso(captured_at), "received_at": _iso(now),
                     "bytes": len(data), "width": width, "height": height}
            screens[screen] = entry
            self._write_index(folder, screens)
            self._prune_locked()
        return {"screen": screen, **entry}

    def clients(self) -> list[str]:
        try:
            return sorted(p.name for p in self.root.iterdir() if p.is_dir() and _CLIENT_RE.match(p.name))
        except OSError:
            return []

    def screens(self, client_id: str) -> list[dict[str, Any]]:
        """The stored screenshots of *client_id*, in upload order."""

        folder = self._client_dir(client_id)
        return [{"screen": screen, **entry} for screen, entry in self._read_index(folder).items()
                if (folder / entry["file"]).is_file()]

    def image_path(self, client_id: str, name: str) -> Path | None:
        if not isinstance(name, str) or not _FILE_RE.match(name):
            return None
        path = self._client_dir(client_id) / name
        return path if path.is_file() else None

    def prune(self) -> list[str]:
        """Drop expired screenshots, stray files and anything over the size cap."""

        with self._lock:
            return self._prune_locked()

    def _prune_locked(self) -> list[str]:
        removed: list[str] = []
        now = self._clock()
        cutoff = now - self.retention_seconds
        entries: list[tuple[float, str, str, int]] = []
        for client_id in self.clients():
            folder = self.root / client_id
            screens = self._read_index(folder)
            kept: dict[str, dict[str, Any]] = {}
            for screen, entry in screens.items():
                received = parse_time(entry.get("received_at")) or 0.0
                path = folder / entry["file"]
                if received < cutoff or not path.is_file():
                    with contextlib.suppress(OSError):
                        path.unlink()
                    removed.append(f"{client_id}/{screen}")
                    continue
                kept[screen] = entry
                entries.append((received, client_id, screen, int(entry.get("bytes") or 0)))
            referenced = {entry["file"] for entry in kept.values()} | {INDEX_NAME}
            for path in folder.iterdir():
                stray_tmp = path.name.endswith(".tmp") and now - _mtime(path) > 3600
                if path.is_file() and path.name not in referenced and (not path.name.endswith(".tmp") or stray_tmp):
                    with contextlib.suppress(OSError):
                        path.unlink()
            if kept != screens:
                self._update_or_remove(folder, kept)
        total = sum(size for *_, size in entries)
        if total > self.max_total_bytes:
            for received, client_id, screen, size in sorted(entries):
                if total <= self.max_total_bytes:
                    break
                folder = self.root / client_id
                screens = self._read_index(folder)
                entry = screens.pop(screen, None)
                if entry is not None:
                    with contextlib.suppress(OSError):
                        (folder / entry["file"]).unlink()
                    self._update_or_remove(folder, screens)
                    removed.append(f"{client_id}/{screen}")
                total -= size
        return removed

    def _update_or_remove(self, folder: Path, screens: Mapping[str, Any]) -> None:
        if screens:
            self._write_index(folder, screens)
        else:
            shutil.rmtree(folder, ignore_errors=True)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# ── Client ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PendingUpload:
    screen: str
    path: Path
    captured_at: float


class UploadQueue:
    """Which saved screenshots a client should upload next.

    Holds only file paths (the panel's ``current/<screen>.png``), never
    images, so a long playlist costs no memory.  A screen is due when it has
    not been uploaded for ``interval_seconds``.
    """

    def __init__(self, interval_seconds: float = DEFAULT_UPLOAD_INTERVAL_SECONDS, *,
                 clock: Callable[[], float] = time.time) -> None:
        self.interval_seconds = max(float(MIN_UPLOAD_INTERVAL_SECONDS), float(interval_seconds))
        self._clock = clock
        self._lock = threading.Lock()
        self._latest: dict[str, PendingUpload] = {}
        self._uploaded: dict[str, float] = {}

    def offer(self, screen: str, path: str | os.PathLike[str], captured_at: float | None = None) -> None:
        """Note that *path* now holds *screen*'s newest screenshot."""

        with self._lock:
            self._latest[screen] = PendingUpload(screen, Path(path),
                                                 self._clock() if captured_at is None else captured_at)

    def due(self, limit: int = MAX_UPLOADS_PER_PASS, *, min_interval: float = 0.0) -> list[PendingUpload]:
        """Up to *limit* screens not uploaded recently, least recently uploaded first."""

        interval = max(self.interval_seconds, float(min_interval or 0))
        now = self._clock()
        with self._lock:
            ready = [item for screen, item in self._latest.items()
                     if now - self._uploaded.get(screen, float("-inf")) >= interval
                     and item.captured_at > self._uploaded.get(screen, float("-inf"))]
            ready.sort(key=lambda item: self._uploaded.get(item.screen, float("-inf")))
            return ready[: max(0, limit)]

    def mark_uploaded(self, item: PendingUpload) -> None:
        with self._lock:
            self._uploaded[item.screen] = self._clock()

    def postpone(self, item: PendingUpload) -> None:
        """Skip *item* for one interval (a permanent rejection or a vanished file)."""

        self.mark_uploaded(item)


__all__ = [
    "DEFAULT_UPLOAD_DIR",
    "PendingUpload",
    "ScreenshotInbox",
    "UploadQueue",
    "UploadRejected",
    "WIRE_VERSION",
    "file_name",
    "parse_time",
    "upload_dir",
    "validate_png",
]
