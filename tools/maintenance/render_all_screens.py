#!/usr/bin/env python3
"""Backward-compatible wrapper for rendering all screens.

This shim intentionally mirrors ``tools.maintenance.render_screens`` so users
running the legacy entrypoint still pick up current rendering behavior (such
as refreshed scoreboard payload hydration and interactive CLI prompts.
"""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_project_root_env_file() -> None:
    candidate_paths = (
        Path.home() / "desk_display" / ".env",
        PROJECT_ROOT / ".env",
    )

    env_path = next((path for path in candidate_paths if path.is_file()), None)
    if env_path is None:
        return

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.lower().startswith("export "):
            key = key[7:].strip()
        value = value.strip()
        if not key:
            continue
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        os.environ.setdefault(key, value)


os.environ.setdefault("CONFIG_LOAD_DOTENV", "1")
_load_project_root_env_file()

from tools.maintenance import render_screens as _impl

_LEGACY_EXCLUDED_EXPORTS = {"check_github_updates", "check_apt_updates"}

for _name, _value in _impl.__dict__.items():
    if _name in {"__name__", "__file__", "__package__", "__loader__", "__spec__"}:
        continue
    if _name in _LEGACY_EXCLUDED_EXPORTS:
        continue
    globals()[_name] = _value

# Keep explicit aliases for the most-used integration points. This makes it
# clear that the wrapper tracks the underlying implementation and helps static
# tooling/readers that look for direct attributes on this module.
main = _impl.main
build_cache = _impl.build_cache


if __name__ == "__main__":
    raise SystemExit(main())
