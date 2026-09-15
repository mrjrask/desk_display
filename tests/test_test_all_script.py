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
lint_cleanup_report = _load_script_module(
    "desk_display_lint_cleanup_report", "lint_cleanup_report.py"
)
lint_baseline = _load_script_module("desk_display_lint_baseline", "check_lint_baseline.py")


def test_discover_standalone_scripts_excludes_aggregate_runner():
    discovered = {path.name for path in test_all._discover_standalone_scripts()}

    assert "test_api_connections.py" in discovered
    assert "test_all.py" not in discovered


def test_build_commands_runs_only_hardware_independent_quality_checks():
    commands = test_all._build_commands(["-q"])

    assert [command.name for command in commands] == [
        "Ruff static checks",
        "Ruff suppression baseline",
        "pytest suite",
    ]
    assert commands[0].command[2:5] == ("ruff", "check", ".")
    assert commands[0].command[-4:] == ("--select", "F", "--ignore", "F401,F841")
    assert commands[1].command == (
        sys.executable,
        "scripts/check_lint_baseline.py",
    )
    assert commands[2].command[-1] == "-q"
    assert not any("scripts/test_api_connections.py" in command.command for command in commands)


def test_diagnostic_commands_include_standalone_scripts():
    commands = test_all._build_diagnostic_commands()

    assert any("scripts/test_api_connections.py" in command.command for command in commands)


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


def test_lint_cleanup_option_adds_report_only_ruff_command():
    commands = test_all._build_commands([])
    commands.append(test_all._build_lint_cleanup_command())

    lint_command = commands[-1]
    assert lint_command.name == "staged Ruff cleanup report"
    assert lint_command.command[-1] == "scripts/lint_cleanup_report.py"


def test_lint_cleanup_report_groups_findings_by_module_and_rule():
    grouped = lint_cleanup_report._group_findings(
        [
            {"filename": "/repo/main.py", "code": "RUF001"},
            {"filename": "/repo/main.py", "code": "RUF001"},
            {"filename": "/repo/utils.py", "code": "UP006"},
        ]
    )

    assert grouped["main.py"] == {"RUF001": 2}
    assert grouped["utils.py"] == {"UP006": 1}


def test_lint_baseline_requires_config_and_baseline_to_match(monkeypatch):
    monkeypatch.setattr(lint_baseline, "BASELINE", {"main.py": {"RUF001", "SIM102"}})

    assert lint_baseline.find_baseline_drift({"main.py": ["RUF001", "SIM102"]}) == []
    assert lint_baseline.find_baseline_drift({"main.py": ["RUF001", "SIM102", "B018"]}) == [
        "main.py: added B018"
    ]
    assert lint_baseline.find_baseline_drift({"main.py": ["RUF001"]}) == [
        "main.py: baseline still contains removed SIM102"
    ]


def test_lint_baseline_python_310_fallback_parses_per_file_ignores(tmp_path, monkeypatch):
    config_path = tmp_path / "pyproject.toml"
    config_path.write_text(
        """
[tool.ruff.lint.per-file-ignores]
"main.py" = ["RUF001", "SIM102"]
"utils.py" = ["UP006"]

[tool.ruff.lint.isort]
combine-as-imports = true
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(lint_baseline, "tomllib", None)

    assert lint_baseline.load_per_file_ignores(config_path) == {
        "main.py": ["RUF001", "SIM102"],
        "utils.py": ["UP006"],
    }
