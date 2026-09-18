import logging
import os
import subprocess
import sys

import pytest

from env_config import env_float, env_int, non_negative_env_float, non_negative_env_int


@pytest.mark.parametrize(
    ("parser", "raw", "expected"),
    [
        (env_int, " 42 ", 42),
        (env_float, " 2.5 ", 2.5),
    ],
)
def test_numeric_env_accepts_valid_values(monkeypatch, parser, raw, expected):
    monkeypatch.setenv("TEST_NUMBER", raw)

    assert parser("TEST_NUMBER", 7) == expected


@pytest.mark.parametrize("raw", ["", "  ", "not-a-number"])
def test_numeric_env_rejects_empty_and_malformed_values(monkeypatch, caplog, raw):
    monkeypatch.setenv("TEST_NUMBER", raw)

    with caplog.at_level(logging.WARNING):
        assert env_int("TEST_NUMBER", 7) == 7

    assert "TEST_NUMBER" in caplog.text
    assert repr(raw) in caplog.text
    assert "using default 7" in caplog.text


def test_numeric_env_uses_default_without_warning_when_unset(monkeypatch, caplog):
    monkeypatch.delenv("TEST_NUMBER", raising=False)

    assert env_float("TEST_NUMBER", 1.25) == 1.25
    assert not caplog.records


@pytest.mark.parametrize(
    ("parser", "default"),
    [(non_negative_env_int, 7), (non_negative_env_float, 2.5)],
)
def test_non_negative_env_rejects_negative_values(monkeypatch, caplog, parser, default):
    monkeypatch.setenv("TEST_NUMBER", "-1")

    with caplog.at_level(logging.WARNING):
        assert parser("TEST_NUMBER", default) == default

    assert "must be at least 0" in caplog.text


@pytest.mark.parametrize("parser", [non_negative_env_int, non_negative_env_float])
def test_non_negative_env_permits_meaningful_zero(monkeypatch, parser):
    monkeypatch.setenv("TEST_NUMBER", "0")

    assert parser("TEST_NUMBER", 9) == 0


def test_malformed_numeric_environment_does_not_break_import():
    variable = "FEED_MAX_UPLOAD_BYTES"
    environment = os.environ.copy()
    environment[variable] = "definitely-not-numeric"
    # Keep this regression test independent of optional web/image dependencies.
    bootstrap = """
import sys
import types

class Flask:
    def __init__(self, *args, **kwargs):
        self.config = {}
    def __getattr__(self, name):
        return lambda *args, **kwargs: lambda function: function

flask = types.ModuleType("flask")
flask.Flask = Flask
flask.g = types.SimpleNamespace()
flask.request = types.SimpleNamespace()
for name in ("abort", "jsonify", "render_template", "send_from_directory"):
    setattr(flask, name, lambda *args, **kwargs: None)
sys.modules["flask"] = flask

pillow = types.ModuleType("PIL")
pillow.Image = types.SimpleNamespace()
pillow.UnidentifiedImageError = ValueError
sys.modules["PIL"] = pillow

import feed_server
"""

    result = subprocess.run(
        [sys.executable, "-c", bootstrap],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert variable in result.stderr
