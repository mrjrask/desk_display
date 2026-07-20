import importlib.util
import sys
from pathlib import Path


def _load_script_module(name: str, filename: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


test_all = _load_script_module("desk_display_test_all_script", "test_all.py")


def test_discover_standalone_scripts_excludes_aggregate_runner():
    discovered = {path.name for path in test_all._discover_standalone_scripts()}

    assert "test_api_connections.py" in discovered
    assert "test_all.py" not in discovered


def test_build_commands_runs_pytest_before_standalone_scripts():
    commands = test_all._build_commands(["-q"])

    assert commands[0].name == "pytest suite"
    assert commands[0].command[-1] == "-q"
    assert any("scripts/test_api_connections.py" in command.command for command in commands[1:])


def test_main_forwards_dash_prefixed_pytest_args_without_separator(monkeypatch, capsys):
    captured_pytest_args = None

    def fake_build_commands(pytest_args):
        nonlocal captured_pytest_args
        captured_pytest_args = pytest_args
        return []

    monkeypatch.setattr(test_all, "_build_commands", fake_build_commands)

    assert test_all.main(["--list", "-q", "-k", "weather"]) == 0
    assert captured_pytest_args == ["-q", "-k", "weather"]
    assert capsys.readouterr().out == ""


def test_main_accepts_optional_separator_before_pytest_args(monkeypatch):
    captured_pytest_args = None

    def fake_build_commands(pytest_args):
        nonlocal captured_pytest_args
        captured_pytest_args = pytest_args
        return []

    monkeypatch.setattr(test_all, "_build_commands", fake_build_commands)

    assert test_all.main(["--list", "--", "-q"]) == 0
    assert captured_pytest_args == ["-q"]
