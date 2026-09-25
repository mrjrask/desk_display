"""Phase 17b: converting an existing .env into a server or client configuration."""

from __future__ import annotations

import importlib.util
import os
import stat
from pathlib import Path

import pytest

import deployment_config as dc
import env_conversion

ROOT = Path(__file__).resolve().parents[1]

OWM = "owm-" + "k" * 28
ADMIN = "admin-" + "a" * 40
UI_PASSWORD = "ui-password-" + "p" * 12
CLIENT_TOKEN = "ddc_" + "c" * 40
UNKNOWN_SECRET = "legacy-secret-" + "s" * 20

STANDALONE = f"""# My desk display
# Weather
OWM_API_KEY={OWM}
WEATHER_LATITUDE=41.9
WEATHER_LONGITUDE=-87.6

# Display
DESK_DISPLAY_PROFILE=hyperpixel4_square
DISPLAY_ROTATION=90
DISPLAY_ROTATION=180
export SCREEN_UI_PASSWORD="{UI_PASSWORD}"
SCREEN_CONFIG_HOST=127.0.0.1
DESK_DISPLAY_SERVER_ADMIN_TOKEN={ADMIN}
MY_OLD_API_KEY={UNKNOWN_SECRET}
SOMETHING_ELSE=kept on servers
# OWM_API_KEY_WIFFY=old-disabled-key-000000
"""


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "convert_env_cli", ROOT / "scripts" / "convert_env.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def env_file(tmp_path):
    path = tmp_path / ".env"
    path.write_text(STANDALONE, encoding="utf-8")
    os.chmod(path, 0o640)
    return path


@pytest.fixture
def cli():
    return load_cli()


def names(conversion, action):
    return [c.name for c in conversion.changes if c.action == action]


def backups(path: Path) -> list[Path]:
    return sorted(path.parent.glob(path.name + ".bak-*"))


def assert_no_secrets(text: str) -> None:
    for secret in (
        OWM,
        ADMIN,
        UI_PASSWORD,
        CLIENT_TOKEN,
        UNKNOWN_SECRET,
        "old-disabled-key-000000",
    ):
        assert secret not in text


# ── Conversion ─────────────────────────────────────────────────────────────


def test_server_conversion_keeps_content_and_drops_hardware(env_file):
    conversion = env_conversion.convert(env_file, dc.Role.SERVER)
    env = conversion.env
    assert env[dc.ROLE_ENV] == "server" and env["OWM_API_KEY"] == OWM
    assert env["SCREEN_UI_PASSWORD"] == UI_PASSWORD and env["SOMETHING_ELSE"] == "kept on servers"
    assert "DISPLAY_ROTATION" in names(conversion, "removed") and "DISPLAY_ROTATION" not in env
    assert "DESK_DISPLAY_PROFILE" not in env
    assert "DESK_DISPLAY_SERVER_PORT" in names(conversion, "added")
    assert conversion.text.startswith("# My desk display\n# Weather\n")  # comments survive in place
    assert env_conversion.ADDED_HEADER.format(role="server") in conversion.text
    assert conversion.report.ok
    # The written result passes the very check the server runs at startup.
    dc.startup_check("server", conversion.env)


def test_client_conversion_removes_every_credential(env_file):
    conversion = env_conversion.convert(
        env_file,
        dc.Role.CLIENT,
        {
            "DESK_DISPLAY_SERVER_URL": "https://render.lan:8765",
            "DESK_DISPLAY_CLIENT_ID": "office",
            "DESK_DISPLAY_CLIENT_TOKEN": CLIENT_TOKEN,
        },
    )
    env = conversion.env
    for name in (
        "OWM_API_KEY",
        "SCREEN_UI_PASSWORD",
        "DESK_DISPLAY_SERVER_ADMIN_TOKEN",
        "MY_OLD_API_KEY",
        "SOMETHING_ELSE",
        "WEATHER_LATITUDE",
    ):
        assert name not in env and name in names(conversion, "removed")
    assert "OWM_API_KEY_WIFFY" not in conversion.text  # even the commented-out credential
    assert env["DESK_DISPLAY_PROFILE"] == "hyperpixel4_square" and env["DISPLAY_ROTATION"] == "180"
    assert env["DESK_DISPLAY_CLIENT_TOKEN"] == CLIENT_TOKEN and env[dc.ROLE_ENV] == "client"
    assert conversion.report.ok, conversion.render_report()
    dc.startup_check("client", conversion.env)


def test_missing_client_values_become_placeholders_and_block_the_write(env_file, cli, capsys):
    conversion = env_conversion.convert(env_file, dc.Role.CLIENT)
    assert "# Required: a stable, unique ID for this display." in conversion.text
    assert "DESK_DISPLAY_CLIENT_ID=\n" in conversion.text and not conversion.report.ok
    before = env_file.read_text()
    assert cli.main(["--role", "client", "--env-file", str(env_file)]) == 1
    assert "Not written" in capsys.readouterr().out
    assert env_file.read_text() == before and not backups(env_file)
    assert cli.main(["--role", "client", "--env-file", str(env_file), "--allow-invalid"]) == 1
    assert "DESK_DISPLAY_CLIENT_ID=" in env_file.read_text() and backups(env_file)


def test_duplicate_lines_keep_the_effective_last_value(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "DESK_DISPLAY_SERVER_PORT=1\n# note\nDESK_DISPLAY_SERVER_PORT=2\n", encoding="utf-8"
    )
    conversion = env_conversion.convert(path, dc.Role.SERVER)
    assert (
        conversion.text.count("DESK_DISPLAY_SERVER_PORT=") == 1
        and conversion.env["DESK_DISPLAY_SERVER_PORT"] == "2"
    )
    assert [c.reason for c in conversion.changes if c.name == "DESK_DISPLAY_SERVER_PORT"] == [
        "duplicate; a later line sets it"
    ]
    assert "# note" in conversion.text


def test_unknown_lines_are_kept_on_servers_and_dropped_from_clients_by_default(env_file):
    assert "SOMETHING_ELSE" in env_conversion.convert(env_file, dc.Role.SERVER).env
    assert "SOMETHING_ELSE" not in env_conversion.convert(env_file, dc.Role.CLIENT).env
    kept = env_conversion.convert(env_file, dc.Role.CLIENT, keep_unknown=True)
    assert "SOMETHING_ELSE" in kept.env
    stripped = env_conversion.convert(env_file, dc.Role.SERVER, keep_unknown=False)
    assert "SOMETHING_ELSE" not in stripped.env


def test_supplied_values_must_belong_to_the_role(env_file):
    with pytest.raises(env_conversion.ConversionError):
        env_conversion.convert(env_file, dc.Role.CLIENT, {"OWM_API_KEY": OWM})
    with pytest.raises(env_conversion.ConversionError):
        env_conversion.convert(env_file, dc.Role.STANDALONE)


def test_values_with_spaces_round_trip(tmp_path):
    path = tmp_path / ".env"
    path.write_text("DESK_DISPLAY_PROFILE=hyperpixel4_square\n", encoding="utf-8")
    conversion = env_conversion.convert(
        path,
        dc.Role.CLIENT,
        {
            "DESK_DISPLAY_CLIENT_ID": "office",
            "DESK_DISPLAY_SERVER_URL": "https://r.lan",
            "DESK_DISPLAY_CLIENT_TOKEN": 'a b#"c' + "x" * 10,
        },
    )
    assert conversion.env["DESK_DISPLAY_CLIENT_TOKEN"] == 'a b#"c' + "x" * 10


# ── Secrets ────────────────────────────────────────────────────────────────


def test_reports_and_diffs_never_contain_secret_values(env_file):
    for role, supplied in (
        (dc.Role.SERVER, {}),
        (dc.Role.CLIENT, {"DESK_DISPLAY_CLIENT_TOKEN": CLIENT_TOKEN}),
    ):
        conversion = env_conversion.convert(env_file, role, supplied)
        for text in (conversion.render_diff(), conversion.render_report()):
            assert_no_secrets(text)
        assert dc.REDACTED in conversion.render_diff()


def test_cli_output_never_contains_secret_values(env_file, cli, capsys, tmp_path):
    credentials = tmp_path / "office.env.client"
    credentials.write_text(
        f"DESK_DISPLAY_SERVER_URL=https://render.lan:8765\nDESK_DISPLAY_CLIENT_ID=office\n"
        f"DESK_DISPLAY_CLIENT_TOKEN={CLIENT_TOKEN}\n",
        encoding="utf-8",
    )
    for extra in (
        ["--role", "server", "--dry-run"],
        ["--role", "client", "--dry-run", "--credentials", str(credentials)],
        ["--role", "client", "--credentials", str(credentials)],
    ):
        cli.main(["--env-file", str(env_file), *extra])
        captured = capsys.readouterr()
        assert_no_secrets(captured.out + captured.err)
    assert dc.parse_env_file(env_file)["DESK_DISPLAY_CLIENT_TOKEN"] == CLIENT_TOKEN


def test_token_file_and_refusal_of_client_options_for_a_server(env_file, cli, tmp_path, capsys):
    token = tmp_path / "token.txt"
    token.write_text(CLIENT_TOKEN + "\n", encoding="utf-8")
    args = [
        "--env-file",
        str(env_file),
        "--role",
        "client",
        "--server-url",
        "https://render.lan:8765",
        "--client-id",
        "office",
        "--token-file",
        str(token),
    ]
    assert cli.main(args) == 0
    assert dc.parse_env_file(env_file)["DESK_DISPLAY_CLIENT_TOKEN"] == CLIENT_TOKEN
    assert cli.main(["--env-file", str(env_file), "--role", "server", "--client-id", "x"]) == 2
    assert_no_secrets(capsys.readouterr().out)


# ── Writing ────────────────────────────────────────────────────────────────


def test_dry_run_writes_nothing(env_file, cli, capsys):
    before = env_file.read_text()
    assert cli.main(["--role", "server", "--env-file", str(env_file), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "-DISPLAY_ROTATION=90" in out and "Dry run" in out
    assert env_file.read_text() == before and not backups(env_file)


def test_write_backs_up_and_a_rerun_is_a_no_op(env_file, cli, capsys):
    original = env_file.read_text()
    assert cli.main(["--role", "server", "--env-file", str(env_file)]) == 0
    [backup] = backups(env_file)
    assert backup.read_text() == original and stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o640  # the original's mode is kept
    converted = env_file.read_text()
    capsys.readouterr()
    assert cli.main(["--role", "server", "--env-file", str(env_file)]) == 0
    assert "nothing to change" in capsys.readouterr().out
    assert env_file.read_text() == converted and len(backups(env_file)) == 1
    assert not env_conversion.convert(env_file, dc.Role.SERVER).changes


def test_client_rerun_is_a_no_op(env_file):
    supplied = {
        "DESK_DISPLAY_SERVER_URL": "https://render.lan:8765",
        "DESK_DISPLAY_CLIENT_ID": "office",
        "DESK_DISPLAY_CLIENT_TOKEN": CLIENT_TOKEN,
    }
    env_conversion.write(env_conversion.convert(env_file, dc.Role.CLIENT, supplied))
    again = env_conversion.convert(env_file, dc.Role.CLIENT, supplied)
    assert not again.changed and not again.changes
    assert env_conversion.write(again) is None


def test_interrupted_write_leaves_the_original_intact(env_file, monkeypatch):
    original = env_file.read_text()
    conversion = env_conversion.convert(env_file, dc.Role.SERVER)

    def power_cut(src, dst):
        raise OSError("power cut")

    monkeypatch.setattr(env_conversion.os, "replace", power_cut)
    with pytest.raises(OSError):
        env_conversion.write(conversion)
    assert env_file.read_text() == original
    assert not [p for p in env_file.parent.iterdir() if p.name.endswith(".tmp")]
    assert backups(env_file)[0].read_text() == original
    monkeypatch.undo()
    env_conversion.write(
        env_conversion.convert(env_file, dc.Role.SERVER)
    )  # a rerun finishes the job
    assert dc.parse_env_file(env_file)[dc.ROLE_ENV] == "server"


def test_a_file_edited_after_reading_is_not_overwritten(env_file):
    conversion = env_conversion.convert(env_file, dc.Role.SERVER)
    env_file.write_text(STANDALONE + "LATE=1\n", encoding="utf-8")
    with pytest.raises(env_conversion.ConversionError):
        env_conversion.write(conversion)
    assert env_file.read_text().endswith("LATE=1\n")


def test_the_repository_examples_convert_to_valid_configurations(tmp_path):
    for source in (".env.example", ".env.server.example"):
        path = tmp_path / source
        path.write_text((ROOT / source).read_text(), encoding="utf-8")
        assert env_conversion.convert(path, dc.Role.SERVER).report.ok
    path = tmp_path / ".env.client.example"
    path.write_text((ROOT / ".env.client.example").read_text(), encoding="utf-8")
    assert not env_conversion.convert(path, dc.Role.CLIENT).changes
