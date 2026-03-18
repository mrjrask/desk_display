#!/usr/bin/env python3
"""Backward-compatible wrapper for rendering all screens."""

from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_project_root_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
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

for _name, _value in _impl.__dict__.items():
    if _name in {"__name__", "__file__", "__package__", "__loader__", "__spec__"}:
        continue
    globals()[_name] = _value


if __name__ == "__main__":
    raise SystemExit(main())
