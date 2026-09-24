"""Configuration storage with versioning, rollback, and pruning."""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

DEFAULT_RETENTION = 25


class ConfigStore:
    """Persist the active configuration and maintain a version history."""

    def __init__(
        self,
        config_path: str,
        *,
        db_path: Optional[str] = None,
        archive_dir: Optional[str] = None,
        retention: int = DEFAULT_RETENTION,
        initialize: bool = True,
    ) -> None:
        self.config_path = Path(config_path)
        self.db_path = (
            Path(db_path) if db_path else self.config_path.with_suffix(".history.sqlite3")
        )
        self.archive_dir = (
            Path(archive_dir) if archive_dir else self.config_path.parent / "config_versions"
        )
        self.retention = max(1, retention)
        self._save_lock = threading.Lock()
        if initialize:
            self._ensure_database()

    # ------------------------------------------------------------------
    # Public API
    def load(self) -> dict[str, Any]:
        try:
            with self.config_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")
        return data

    def save(
        self,
        config: dict[str, Any],
        *,
        actor: str = "system",
        summary: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        with self._save_lock:
            self._ensure_database()
            current = self.load()
            summary = summary or summarise_diff(current, config)
            metadata = metadata or {}
            metadata.setdefault("actor", actor)
            try:
                previous_bytes = self.config_path.read_bytes()
            except FileNotFoundError:
                previous_bytes = None

            self._write_config(config)
            try:
                version_id = self._record_version(
                    config,
                    actor=actor,
                    summary=summary,
                    metadata=metadata,
                )
            except Exception:
                if previous_bytes is None:
                    self.config_path.unlink(missing_ok=True)
                else:
                    self._write_bytes(previous_bytes)
                raise

            with contextlib.suppress(OSError):
                self._prune_history()
            return version_id

    def list_versions(self, limit: int = 20) -> list[dict[str, Any]]:
        self._ensure_database()
        query = """
            SELECT id, created_at, actor, summary
            FROM config_versions
            ORDER BY id DESC
            LIMIT ?
        """
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (max(1, limit),)).fetchall()
        return [dict(row) for row in rows]

    def latest_version_id(self) -> Optional[int]:
        self._ensure_database()
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn, conn:
            row = conn.execute("SELECT id FROM config_versions ORDER BY id DESC LIMIT 1").fetchone()
        return int(row[0]) if row else None

    def load_version(self, version_id: int) -> dict[str, Any]:
        self._ensure_database()
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT config_json FROM config_versions WHERE id = ?",
                (version_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown version id {version_id}")
        payload = json.loads(row["config_json"])
        if not isinstance(payload, dict):
            raise ValueError("Stored configuration is not a JSON object")
        return payload

    def rollback(self, version_id: int, *, actor: str = "system") -> dict[str, Any]:
        config = self.load_version(version_id)
        summary = f"Rollback to version {version_id}"
        self.save(config, actor=actor, summary=summary, metadata={"rollback_from": version_id})
        return config

    # ------------------------------------------------------------------
    # Internal helpers
    def _ensure_database(self) -> None:
        os.makedirs(self.db_path.parent, exist_ok=True)
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS config_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        config_json TEXT NOT NULL,
                        metadata_json TEXT
                    )
                    """
            )
            conn.commit()
        os.makedirs(self.archive_dir, exist_ok=True)

    def _write_config(self, config: dict[str, Any]) -> None:
        payload = json.dumps(config, indent=2, sort_keys=False) + "\n"
        self._write_bytes(payload.encode("utf-8"))

    def _write_bytes(self, payload: bytes) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.config_path.parent,
                prefix=f".{self.config_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as staged:
                staged_path = Path(staged.name)
                staged.write(payload)
                staged.flush()
                os.fsync(staged.fileno())
            os.replace(staged_path, self.config_path)
        finally:
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)

    def _record_version(
        self,
        config: dict[str, Any],
        *,
        actor: str,
        summary: str,
        metadata: dict[str, Any],
    ) -> int:
        payload = json.dumps(config, indent=2, sort_keys=False)
        metadata_json = json.dumps(metadata, sort_keys=True)
        created_at = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

        archive_path: Optional[Path] = None
        try:
            with contextlib.closing(sqlite3.connect(self.db_path)) as conn, conn:
                cursor = conn.execute(
                    """INSERT INTO config_versions
                        (created_at, actor, summary, config_json, metadata_json)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                    (created_at, actor, summary, payload, metadata_json),
                )
                version_id = cursor.lastrowid
                archive_path = self.archive_dir / f"{version_id:06d}.json"
                with archive_path.open("w", encoding="utf-8") as fh:
                    fh.write(payload)
                conn.commit()
        except Exception:
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)
            raise

        return int(version_id)

    def _prune_history(self) -> None:
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id FROM config_versions ORDER BY id DESC LIMIT -1 OFFSET ?",
                (self.retention,),
            ).fetchall()
            stale_ids = [row["id"] for row in rows]
            if stale_ids:
                conn.executemany(
                    "DELETE FROM config_versions WHERE id = ?", [(vid,) for vid in stale_ids]
                )
                conn.commit()

        for archive_file in sorted(self.archive_dir.glob("*.json"))[: -self.retention]:
            with contextlib.suppress(OSError):
                archive_file.unlink()


def summarise_diff(old: dict[str, Any], new: dict[str, Any]) -> str:
    """Generate a human-readable summary of configuration changes."""

    def _normalise_screens(config: dict[str, Any]) -> dict[str, Any]:
        screens = config.get("screens")
        if isinstance(screens, dict):
            return screens
        return {}

    old_screens = _normalise_screens(old)
    new_screens = _normalise_screens(new)

    added: list[str] = []
    removed: list[str] = []
    changed: list[str] = []

    for key in sorted(set(old_screens) | set(new_screens)):
        if key not in old_screens:
            added.append(key)
        elif key not in new_screens:
            removed.append(key)
        elif old_screens.get(key) != new_screens.get(key):
            changed.append(key)

    parts: list[str] = []
    if added:
        parts.append("Added screens: " + ", ".join(added))
    if changed:
        parts.append("Updated screens: " + ", ".join(changed))
    if removed:
        parts.append("Removed screens: " + ", ".join(removed))

    return "; ".join(parts) if parts else "Configuration saved"
