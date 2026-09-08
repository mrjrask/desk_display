"""Shared state for temporarily looping one display screen.

The configuration UI and renderer run as separate processes.  A tiny JSON
control file lets the UI request diagnostic playback without changing the
saved rotation configuration or restarting either service.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from paths import resolve_storage_paths
from screens_catalog import SCREEN_IDS, canonical_screen_id


def control_path() -> Path:
    override = os.environ.get("DESK_DISPLAY_DIAGNOSTIC_CONTROL_PATH")
    if override:
        return Path(os.path.expandvars(override)).expanduser()
    return resolve_storage_paths().current_screenshot_dir / "diagnostic_playback.json"


def normalize_screen_id(value: object) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    requested = canonical_screen_id(value.strip())
    by_casefold = {screen_id.casefold(): screen_id for screen_id in SCREEN_IDS}
    return by_casefold.get(requested.casefold())


def load_diagnostic_screen() -> Optional[str]:
    try:
        payload = json.loads(control_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return normalize_screen_id(payload.get("screen_id"))


def save_diagnostic_screen(screen_id: Optional[str]) -> Optional[str]:
    """Persist a diagnostic screen, or clear diagnostic mode with ``None``."""

    normalized = normalize_screen_id(screen_id)
    if screen_id is not None and normalized is None:
        raise ValueError(f"Unknown screen: {screen_id}")

    path = control_path()
    if normalized is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f"{path.name}.",
            suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
            json.dump({"screen_id": normalized}, handle)
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return normalized
