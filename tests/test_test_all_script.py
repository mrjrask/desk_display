from scripts import test_all


def test_discover_standalone_scripts_excludes_aggregate_runner():
    discovered = {path.name for path in test_all._discover_standalone_scripts()}

    assert "test_api_connections.py" in discovered
    assert "test_all.py" not in discovered


def test_build_commands_runs_pytest_before_standalone_scripts():
    commands = test_all._build_commands(["-q"])

    assert commands[0].name == "pytest suite"
    assert commands[0].command[-1] == "-q"
    assert any("scripts/test_api_connections.py" in command.command for command in commands[1:])
