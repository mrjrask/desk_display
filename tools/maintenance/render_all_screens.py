#!/usr/bin/env python3
"""Backward-compatible wrapper for rendering all screens."""

from tools.maintenance import render_screens as _impl

for _name, _value in _impl.__dict__.items():
    if _name in {"__name__", "__file__", "__package__", "__loader__", "__spec__"}:
        continue
    globals()[_name] = _value


if __name__ == "__main__":
    raise SystemExit(main())
