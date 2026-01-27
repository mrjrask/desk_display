from __future__ import annotations

"""Shared helpers for locating writable storage directories."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

APP_DIR_NAME = "desk_display_display_hat_mini"
_SHARED_HINT_PATH = Path(__file__).resolve().parent / ".data_dir_hint"


@dataclass(frozen=True)
class StoragePaths:
    """Resolved filesystem locations for runtime storage."""

    screenshot_dir: Path
    current_screenshot_dir: Path
    archive_base: Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_env_path(name: str, base_dir: Path) -> Optional[Path]:
    raw = os.environ.get(name)
    if not raw:
        return None
    resolved = Path(raw).expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved


def resolve_storage_paths(*, logger: Optional[object] = None) -> StoragePaths:
    """Return filesystem paths for screenshots and archives.

    Screenshots write to ``<project_root>/screenshots`` and archives live in
    ``<project_root>/screenshot_archive`` by default. Set ``SCREENSHOT_DIR`` or
    ``SCREENSHOT_ARCHIVE_BASE`` in the environment (including via ``.env``) to
    override the output locations. A ``current`` folder mirrors the latest
    capture for each screen.
    """

    base_dir = _project_root()
    screenshot_dir = _resolve_env_path("SCREENSHOT_DIR", base_dir) or (base_dir / "screenshots")
    archive_base = _resolve_env_path("SCREENSHOT_ARCHIVE_BASE", base_dir) or (
        base_dir / "screenshot_archive"
    )

    current_screenshot_dir = screenshot_dir / "current"

    screenshot_dir.mkdir(parents=True, exist_ok=True)
    current_screenshot_dir.mkdir(parents=True, exist_ok=True)
    archive_base.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info("Using screenshot directory %s", screenshot_dir)
        logger.info("Using current screenshot directory %s", current_screenshot_dir)
        logger.info("Using screenshot archive base %s", archive_base)

    return StoragePaths(
        screenshot_dir=screenshot_dir,
        current_screenshot_dir=current_screenshot_dir,
        archive_base=archive_base,
    )
