import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_update_screen_config_import_does_not_require_rendering_dependencies():
    code = """
import builtins
import runpy

original_import = builtins.__import__

def reject_rendering_dependencies(name, *args, **kwargs):
    if name == "config" or name == "PIL" or name.startswith("PIL."):
        raise ModuleNotFoundError(f"blocked rendering dependency: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = reject_rendering_dependencies
runpy.run_path("scripts/update_screen_config.py", run_name="update_screen_config_test")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
