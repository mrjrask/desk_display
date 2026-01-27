"""Configuration storage with versioning, rollback, and pruning."""
from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    ) -> None:
        self.config_path = Path(config_path)
        self.db_path = Path(db_path) if db_path else self.config_path.with_suffix(".history.sqlite3")
        self.archive_dir = Path(archive_dir) if archive_dir else self.config_path.parent / "config_versions"
        self.retention = max(1, retention)
        self._ensure_database()

    # ------------------------------------------------------------------
    # Public API
    def load(self) -> Dict[str, Any]:
        try:
            with self.config_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError
            return data
        except Exception:
            return {}

    def save(
        self,
        config: Dict[str, Any],
        *,
        actor: str = "system",
        summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        current = self.load()
        summary = summary or summarise_diff(current, config)
        metadata = metadata or {}
        metadata.setdefault("actor", actor)

        self._write_config(config)
        version_id = self._record_version(config, actor=actor, summary=summary, metadata=metadata)
        self._prune_history()
        return version_id

    def list_versions(self, limit: int = 20) -> List[Dict[str, Any]]:
        query = """
            SELECT id, created_at, actor, summary
            FROM config_versions
            ORDER BY id DESC
            LIMIT ?
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (max(1, limit),)).fetchall()
        return [dict(row) for row in rows]

    def latest_version_id(self) -> Optional[int]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT id FROM config_versions ORDER BY id DESC LIMIT 1").fetchone()
        return int(row[0]) if row else None

    def load_version(self, version_id: int) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
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

    def rollback(self, version_id: int, *, actor: str = "system") -> Dict[str, Any]:
        config = self.load_version(version_id)
        summary = f"Rollback to version {version_id}"
        self.save(config, actor=actor, summary=summary, metadata={"rollback_from": version_id})
        return config

    # ------------------------------------------------------------------
    # Internal helpers
    def _ensure_database(self) -> None:
        os.makedirs(self.db_path.parent, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
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

    def _write_config(self, config: Dict[str, Any]) -> None:
        tmp_path = self.config_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=False)
            fh.write("\n")
        tmp_path.replace(self.config_path)

    def _record_version(
        self,
        config: Dict[str, Any],
        *,
        actor: str,
        summary: str,
        metadata: Dict[str, Any],
    ) -> int:
        payload = json.dumps(config, indent=2, sort_keys=False)
        metadata_json = json.dumps(metadata, sort_keys=True)
        created_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO config_versions (created_at, actor, summary, config_json, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (created_at, actor, summary, payload, metadata_json),
            )
            version_id = cursor.lastrowid
            conn.commit()

        archive_path = self.archive_dir / f"{version_id:06d}.json"
        with archive_path.open("w", encoding="utf-8") as fh:
            fh.write(payload)

        return int(version_id)

    def _prune_history(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id FROM config_versions ORDER BY id DESC LIMIT -1 OFFSET ?",
                (self.retention,),
            ).fetchall()
            stale_ids = [row["id"] for row in rows]
            if stale_ids:
                conn.executemany("DELETE FROM config_versions WHERE id = ?", [(vid,) for vid in stale_ids])
                conn.commit()

        for archive_file in sorted(self.archive_dir.glob("*.json"))[:-self.retention]:
            try:
                archive_file.unlink()
            except OSError:
                pass


def summarise_diff(old: Dict[str, Any], new: Dict[str, Any]) -> str:
    """Generate a human-readable summary of configuration changes."""

    def _normalise_screens(config: Dict[str, Any]) -> Dict[str, Any]:
        screens = config.get("screens")
        if isinstance(screens, dict):
            return screens
        return {}

    old_screens = _normalise_screens(old)
    new_screens = _normalise_screens(new)

    added: List[str] = []
    removed: List[str] = []
    changed: List[str] = []

    for key in sorted(set(old_screens) | set(new_screens)):
        if key not in old_screens:
            added.append(key)
        elif key not in new_screens:
            removed.append(key)
        elif old_screens.get(key) != new_screens.get(key):
            changed.append(key)

    parts: List[str] = []
    if added:
        parts.append("Added screens: " + ", ".join(added))
    if changed:
        parts.append("Updated screens: " + ", ".join(changed))
    if removed:
        parts.append("Removed screens: " + ", ".join(removed))

    return "; ".join(parts) if parts else "Configuration saved"
