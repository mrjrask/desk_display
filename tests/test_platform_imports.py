import platform
import sysconfig
from pathlib import Path


def test_platform_import_resolves_to_stdlib():
    platform_file = Path(platform.__file__).resolve()
    stdlib_dir = Path(sysconfig.get_path("stdlib")).resolve()
    repo_root = Path(__file__).resolve().parents[1]

    assert platform_file.is_relative_to(stdlib_dir)
    assert not platform_file.is_relative_to(repo_root)
