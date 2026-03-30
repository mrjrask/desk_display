"""Platform adapter package.

This package name intentionally matches the project architecture vocabulary.
To avoid breaking imports of Python's stdlib ``platform`` module, we proxy all
stdlib symbols from ``platform.py`` into this namespace.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sysconfig

_stdlib_platform_path = pathlib.Path(sysconfig.get_path("stdlib")) / "platform.py"
_stdlib_spec = importlib.util.spec_from_file_location("_stdlib_platform", _stdlib_platform_path)
if _stdlib_spec and _stdlib_spec.loader:  # pragma: no branch - defensive
    _stdlib_module = importlib.util.module_from_spec(_stdlib_spec)
    _stdlib_spec.loader.exec_module(_stdlib_module)
    for _name in dir(_stdlib_module):
        if _name.startswith("__") and _name not in {"__all__", "__doc__"}:
            continue
        globals()[_name] = getattr(_stdlib_module, _name)
    __all__ = list(getattr(_stdlib_module, "__all__", []))
