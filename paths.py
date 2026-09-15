"""Shared helpers for locating project runtime/config paths.

Centralizing path resolution here keeps environment variable precedence and
fallback behavior consistent across CLI code, background services, and the web
configuration UI.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

APP_DIR_NAME = "desk_display_display_hat_mini"
_SHARED_HINT_PATH = Path(__file__).resolve().parent / ".data_dir_hint"
_LEGACY_ROOT_HISTORY_FILES = frozenset(
    {"pressure_history.json", "weather_metric_history.json"}
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoragePaths:
    """Resolved filesystem locations for runtime storage."""

    screenshot_dir: Path
    current_screenshot_dir: Path
    archive_base: Path


@dataclass(frozen=True)
class ScreensConfigPaths:
    """Resolved screen config paths with local override semantics."""

    default_path: Path
    local_override_path: Path
    active_path: Path


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


def resolve_screens_config_paths() -> ScreensConfigPaths:
    """Resolve default/local screen config paths and active path selection.

    Precedence:
    1. ``SCREENS_CONFIG_PATH`` and ``SCREENS_CONFIG_LOCAL_PATH`` env overrides.
    2. Project-root defaults: ``screens_config.json`` and
       ``screens_config.local.json``.
    3. Active path prefers local override only when that file exists.
    """

    base_dir = _project_root()
    default_path = _resolve_env_path("SCREENS_CONFIG_PATH", base_dir) or (
        base_dir / "screens_config.json"
    )
    local_override_path = _resolve_env_path("SCREENS_CONFIG_LOCAL_PATH", base_dir) or (
        base_dir / "screens_config.local.json"
    )
    active_path = local_override_path if local_override_path.exists() else default_path
    return ScreensConfigPaths(
        default_path=default_path,
        local_override_path=local_override_path,
        active_path=active_path,
    )


def resolve_style_config_path() -> Path:
    """Resolve the style config path from env or project default."""

    base_dir = _project_root()
    return _resolve_env_path("SCREENS_STYLE_PATH", base_dir) or (base_dir / "screens_style.json")


def resolve_layouts_config_path() -> Path:
    """Resolve the layouts config path from env or project default."""

    base_dir = _project_root()
    return _resolve_env_path("SCREENS_LAYOUTS_PATH", base_dir) or (
        base_dir / "screens_layouts.json"
    )


def resolve_news_feeds_config_path() -> Path:
    """Resolve the news headlines feed config path from env or project default.

    This is the file operators edit to add, remove, or re-point the RSS/Atom
    feeds shown on the "news headlines" screen. See ``news_feeds.json`` at the
    project root for the default topic list.
    """

    base_dir = _project_root()
    return _resolve_env_path("NEWS_FEEDS_CONFIG_PATH", base_dir) or (base_dir / "news_feeds.json")


def resolve_news_feeds_config_path_2() -> Path:
    """Resolve the second news headlines feed config path from env or project default.

    This is the file operators edit to add, remove, or re-point the RSS/Atom
    feeds shown on the "news headlines 2" screen. See ``news_feeds_2.json`` at
    the project root for the default topic list.
    """

    base_dir = _project_root()
    return _resolve_env_path("NEWS_FEEDS_CONFIG_PATH_2", base_dir) or (
        base_dir / "news_feeds_2.json"
    )


def resolve_cache_file_path(env_var: str, filename: str) -> Path:
    """Resolve a runtime cache/history JSON file path.

    Precedence:
    1. ``env_var`` override (absolute, or relative to the project root).
    2. Project-root default: ``cache/<filename>``.

    The two weather history files formerly stored at the project root are
    migrated to ``cache/`` when their default paths are resolved. Explicit
    environment overrides are returned unchanged and never trigger migration.
    Other callers create the parent directory themselves on write.
    """

    base_dir = _project_root()
    override = _resolve_env_path(env_var, base_dir)
    if override is not None:
        return override

    canonical_path = base_dir / "cache" / filename
    if filename in _LEGACY_ROOT_HISTORY_FILES:
        _migrate_legacy_root_history(base_dir / filename, canonical_path)
    return canonical_path


def _migrate_legacy_root_history(legacy_path: Path, canonical_path: Path) -> None:
    """Move a legacy root history file without replacing canonical data."""

    if not legacy_path.exists():
        return
    if canonical_path.exists():
        logger.warning(
            "Legacy history file %s was not migrated because canonical history "
            "file %s already exists; preserving both files.",
            legacy_path,
            canonical_path,
        )
        return

    try:
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        # A hard link publishes the complete legacy file atomically and, unlike
        # rename/shutil.move on POSIX, fails rather than replacing a destination
        # created by another process after the exists() check above. Both paths
        # are within the project filesystem; unlinking the old name completes
        # the move while retaining the same file contents and metadata.
        os.link(legacy_path, canonical_path)
        legacy_path.unlink()
        logger.info("Migrated legacy history file %s to %s.", legacy_path, canonical_path)
    except FileExistsError:
        logger.warning(
            "Legacy history file %s was not migrated because canonical history "
            "file %s already exists; preserving both files.",
            legacy_path,
            canonical_path,
        )
    except OSError as exc:
        logger.warning(
            "Could not migrate legacy history file %s to %s: %s",
            legacy_path,
            canonical_path,
            exc,
        )


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
