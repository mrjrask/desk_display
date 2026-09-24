"""Helpers for structural migration of schedule configurations to schema v2.

Migration is structural and preserves numeric ``every`` values by copying them
into playlist rule steps.  It does not produce the scheduler's top-level
``screens`` mapping, and the scheduler does not interpret playlist rule steps,
so migrated legacy data does not gain the runtime frequency, startup hydration,
or alternate semantics that ``schedule.build_scheduler`` applies to ``screens``.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

LEGACY_VERSION = 1
TARGET_VERSION = 2


class MigrationError(RuntimeError):
    """Raised when a configuration cannot be migrated."""


@dataclass
class MigrationResult:
    config: dict[str, Any]
    migrated: bool


def migrate_config(data: dict[str, Any], *, source: str | None = None) -> MigrationResult:
    """Return schema v2 data without changing existing numeric rule values.

    The migration is structural only: legacy ``every`` values are kept inside
    playlist rule steps, not converted into the scheduler's ``screens`` format.
    """

    if not isinstance(data, dict):
        raise MigrationError("Configuration must be a JSON object")

    if data.get("version") == TARGET_VERSION and "playlists" in data:
        return MigrationResult(dict(data), False)

    if "playlists" in data and "sequence" in data:
        migrated = data.get("version") != TARGET_VERSION
        result = dict(data)
        result["version"] = TARGET_VERSION
        existing_metadata = data.get("metadata")
        metadata = dict(existing_metadata) if isinstance(existing_metadata, dict) else {}
        metadata.setdefault("migrated_from", data.get("version", LEGACY_VERSION))
        result["metadata"] = metadata
        return MigrationResult(result, migrated)

    sequence = data.get("sequence")
    if not isinstance(sequence, list):
        raise MigrationError("Legacy configurations must provide a sequence array")

    playlist_steps = [legacy_item_to_step(entry) for entry in sequence]

    config_v2 = {
        "version": TARGET_VERSION,
        "catalog": {"presets": {}},
        "metadata": {
            "migrated_from": data.get("version", LEGACY_VERSION),
            "source": source or "inline",
        },
        "playlists": {
            "main": {
                "label": "Migrated sequence",
                "steps": playlist_steps,
            }
        },
        "sequence": [{"playlist": "main"}],
    }

    # Note: this migration is structural only and does not produce the
    # scheduler's top-level "screens" mapping (see module docstring), so
    # config_v2 cannot be validated with schedule.build_scheduler here.

    return MigrationResult(config_v2, True)


def legacy_item_to_step(entry: Any) -> dict[str, Any]:
    """Convert a legacy entry while preserving every ``every`` frequency value."""

    if isinstance(entry, str):
        return {"screen": entry}

    if not isinstance(entry, dict):
        raise MigrationError(f"Unsupported legacy entry: {entry!r}")

    if "screen" in entry and len(entry) == 1:
        return legacy_item_to_step(entry["screen"])

    if "variants" in entry:
        options = entry["variants"]
        if not isinstance(options, list) or not options:
            raise MigrationError("variants entries must be a non-empty list")
        if any(not isinstance(opt, str) for opt in options):
            raise MigrationError("variants entries must be screen identifiers")
        return {"rule": {"type": "variants", "options": list(options)}}

    if "cycle" in entry:
        children = entry["cycle"]
        if not isinstance(children, list) or not children:
            raise MigrationError("cycle entries must be non-empty lists")
        return {
            "rule": {
                "type": "cycle",
                "items": [legacy_item_to_step(child) for child in children],
            }
        }

    if "every" in entry:
        try:
            frequency = int(entry.get("every"))
        except (TypeError, ValueError) as exc:
            raise MigrationError("every rule requires an integer frequency") from exc
        if frequency <= 0:
            raise MigrationError("every frequency must be greater than zero")
        child = entry.get("screen") or entry.get("item")
        if child is None:
            raise MigrationError("every rule requires a child entry")
        return {
            "rule": {
                "type": "every",
                "frequency": frequency,
                "item": legacy_item_to_step(child),
            }
        }

    raise MigrationError(f"Unsupported legacy entry: {entry!r}")


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise MigrationError("Configuration must be a JSON object")
    return data


def write_json(path: str, data: dict[str, Any]) -> None:
    # Screen key order is significant (it drives rotation order), so
    # preserve insertion order instead of alphabetizing keys.
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")


def _cmd_migrate(args: argparse.Namespace) -> int:
    input_path = args.input
    output_path = args.output or input_path
    if not os.path.exists(input_path):
        raise MigrationError(f"Input file '{input_path}' does not exist")

    payload = load_json(input_path)
    result = migrate_config(payload, source=input_path)
    write_json(output_path, result.config)
    if result.migrated:
        print(f"Migrated configuration → {output_path}")
    else:
        print("Configuration already uses schema v2")
    return 0


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Schedule configuration migrations")
    sub = parser.add_subparsers(dest="command", required=True)

    migrate_parser = sub.add_parser("migrate", help="migrate a configuration file to schema v2")
    migrate_parser.add_argument(
        "--input", required=True, help="Path to the existing configuration JSON"
    )
    migrate_parser.add_argument(
        "--output", help="Optional output path; defaults to in-place overwrite"
    )
    migrate_parser.set_defaults(func=_cmd_migrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
