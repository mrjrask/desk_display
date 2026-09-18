import json
import os
import subprocess
import sys

import data_fetch


def _run_python(source, *, extra_env=None):
    env = os.environ.copy()
    env["CONFIG_LOAD_DOTENV"] = "0"
    env.pop("AHL_API_KEY", None)
    env.pop("AHL_SCHEDULE_ICS_URL", None)
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_ahl_source_defaults_are_empty():
    result = _run_python(
        "import json, config; "
        "print(json.dumps([config.AHL_API_KEY, config.AHL_SCHEDULE_ICS_URL]))"
    )

    assert json.loads(result.stdout.strip()) == ["", ""]


def test_ahl_environment_values_override_empty_defaults():
    result = _run_python(
        "import json, config; "
        "print(json.dumps([config.AHL_API_KEY, config.AHL_SCHEDULE_ICS_URL]))",
        extra_env={
            "AHL_API_KEY": "local-test-key",
            "AHL_SCHEDULE_ICS_URL": "https://calendar.example.test/team.ics",
        },
    )

    assert json.loads(result.stdout.strip()) == [
        "local-test-key",
        "https://calendar.example.test/team.ics",
    ]


def test_application_imports_without_optional_ahl_credentials():
    _run_python("import main")


def test_missing_api_key_skips_hockeytech_request(monkeypatch, caplog):
    monkeypatch.setattr(data_fetch, "AHL_API_KEY", "")
    monkeypatch.setattr(
        data_fetch._session,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected request")),
    )

    assert data_fetch._ahl_request("schedule") is None
    assert "Set AHL_API_KEY in the environment" in caplog.text


def test_missing_ics_url_returns_empty_schedule(monkeypatch, caplog):
    monkeypatch.setattr(data_fetch, "AHL_SCHEDULE_ICS_URL", "")
    monkeypatch.setattr(
        data_fetch._session,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected request")),
    )

    assert data_fetch._fetch_wolves_ics_games() == []
    assert "Set AHL_SCHEDULE_ICS_URL in the environment" in caplog.text
