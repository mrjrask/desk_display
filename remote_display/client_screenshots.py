"""Screenshots and the display heartbeat for a display client's panel.

The config UI's Screenshots and Feed pages read the same files ``main.py``
writes when standalone: ``<SCREENSHOT_DIR>/<screen>/<screen>_<utc>.png``
history, ``<SCREENSHOT_DIR>/current/<screen>.png`` for the latest frame and
``current/display_status.json`` for the heartbeat. A client (including the
local panel of a combined install) writes them here for each screen it
presents, so those pages work in every mode.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

LOGGER = logging.getLogger("desk_display.client.screenshots")

MAX_SCREENSHOTS_PER_SCREEN = 5
_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def sanitize_directory_name(name: str) -> str:
    """Filesystem-friendly directory name, keeping spaces (as ``main.py``)."""

    safe = name.strip().replace("/", "-").replace("\\", "-")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in (" ", "-", "_"))
    return safe or "Screens"


def sanitize_filename_prefix(name: str) -> str:
    """Filesystem-friendly filename prefix (as ``main.py``)."""

    safe = name.strip().replace("/", "-").replace("\\", "-").replace(" ", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in ("_", "-"))
    return safe or "screen"


def _replace_atomically(target: Path, write: Any) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


class ClientScreenshots:
    """Writes presented frames and the heartbeat where the config UI reads them."""

    def __init__(
        self,
        screenshot_dir: Path,
        *,
        profile_id: str,
        width: int,
        height: int,
        enabled: bool = True,
        max_per_screen: int = MAX_SCREENSHOTS_PER_SCREEN,
        now: Any = None,
    ) -> None:
        self.screenshot_dir = Path(screenshot_dir)
        self.current_dir = self.screenshot_dir / "current"
        self.enabled = enabled
        self.max_per_screen = max(1, int(max_per_screen))
        self.display = {"profile_id": profile_id, "width": width, "height": height}
        self._now = now or (lambda: datetime.datetime.now(datetime.timezone.utc))
        self.loop_iteration = 0
        self.play_counts: dict[str, int] = {}

    @classmethod
    def from_settings(cls, settings: dict[str, Any], profile: Any) -> ClientScreenshots:
        # Resolved without creating anything: an unusable screenshot or archive
        # path must not stop the panel starting, so failures surface in record().
        from paths import screenshot_dir_path

        enabled = settings.get("ENABLE_SCREENSHOTS", True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() not in {"0", "false", "no", "off", ""}
        return cls(screenshot_dir_path(), profile_id=profile.profile_id,
                   width=profile.width, height=profile.height, enabled=bool(enabled))

    def record(self, screen_id: str, image: Image.Image) -> None:
        """Save *image* as *screen_id*'s latest frame and update the heartbeat.

        Never raises: a full disk or bad permissions must not stop playback.
        """

        self.loop_iteration += 1
        self.play_counts[screen_id] = self.play_counts.get(screen_id, 0) + 1
        now = self._now()
        if self.enabled:
            try:
                self._save(screen_id, image, now)
            except Exception as exc:  # noqa: BLE001 - playback must continue
                LOGGER.warning("Could not save screenshot for %s: %s", screen_id, exc)
        try:
            self._write_status(screen_id, image, now)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Could not update display heartbeat: %s", exc)

    def _save(self, screen_id: str, image: Image.Image, now: datetime.datetime) -> None:
        prefix = sanitize_filename_prefix(screen_id)
        folder = self.screenshot_dir / sanitize_directory_name(screen_id)
        folder.mkdir(parents=True, exist_ok=True)
        # Filenames are operational artifacts and use UTC, as main.py does.
        image.save(folder / f"{prefix}_{now.astimezone(datetime.timezone.utc):%Y%m%d_%H%M%S}.png")
        self._prune(folder)
        self.current_dir.mkdir(parents=True, exist_ok=True)
        _replace_atomically(self.current_dir / f"{prefix}.png", lambda fh: image.save(fh, format="PNG"))
        # A client has no ticker payload; a sidecar left by main.py would make the
        # Feed page keep animating old headlines instead of showing this frame.
        (self.current_dir / f"{prefix}.ticker.json").unlink(missing_ok=True)

    def _prune(self, folder: Path) -> None:
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
        for path in files[: max(0, len(files) - self.max_per_screen)]:
            try:
                path.unlink()
            except OSError as exc:
                LOGGER.warning("Failed to prune screenshot %s: %s", path, exc)

    def _write_status(self, screen_id: str, image: Image.Image, now: datetime.datetime) -> None:
        payload = {
            "screen_id": screen_id,
            "loop_iteration": self.loop_iteration,
            "rendered_at": now.isoformat(),
            "image_digest": hashlib.sha256(image.tobytes()).hexdigest()[:12],
            "frame_id": None,
            "display": dict(self.display),
            "screen_play_counts": dict(self.play_counts),
        }
        self.current_dir.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        _replace_atomically(self.current_dir / "display_status.json", lambda fh: fh.write(data))
