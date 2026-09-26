#!/usr/bin/env python3
"""Shared helper: re-exec a directly-run script under this project's venv.

Several scripts import packages (Pillow, requests, Flask, or this repo's own
``config``/``data_fetch``, which pull in Pillow/pytz) that only live inside
the project's virtualenv, not in the system Python. Call
``reexec_with_project_venv()`` as the very first thing a script does -- before
any of those imports -- so `python3 scripts/whatever.py` works even when the
caller forgot to activate ``venv``/``.venv`` first.

Only call this from a script's own ``if __name__ == "__main__":`` guard (or
unconditionally in a script that is never imported as a library). Calling it
unconditionally at import time is unsafe for a module that other code may
import directly: ``os.execv`` replaces the whole process image, which would
tear down an importing test runner instead of just re-launching this script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def reexec_with_project_venv() -> None:
    for candidate in (PROJECT_ROOT / ".venv", PROJECT_ROOT / "venv"):
        python_path = (
            candidate / "Scripts" / "python.exe"
            if os.name == "nt"
            else candidate / "bin" / "python"
        )
        if not python_path.exists():
            continue

        # Compare sys.prefix against this specific candidate rather than
        # just "are we in some venv": a caller running under an unrelated
        # venv (one without this project's dependencies installed) should
        # still be re-executed into the project's own venv.
        if Path(sys.prefix).resolve() == candidate.resolve():
            return  # already running under this venv

        os.execv(str(python_path), [str(python_path), *sys.argv])
        return
