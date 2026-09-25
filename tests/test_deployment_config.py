"""Tests for role-specific configuration, examples, validation and secret exclusion."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

import deployment_config as dc
from deployment_config import Role

ROOT = Path(__file__).resolve().parents[1]
SERVER_TOKEN = "s3rver-token-" + "x" * 32
CLIENT_OK = {
    "DESK_DISPLAY_ROLE": "client",
    "DESK_DISPLAY_CLIENT_ID": "office-display",
    "DESK_DISPLAY_SERVER_URL": "https://server.lan:8765",
    "DESK_DISPLAY_CLIENT_TOKEN": SERVER_TOKEN,
    "DESK_DISPLAY_PROFILE": "hyperpixel4",
}
SERVER_OK = {
    "DESK_DISPLAY_ROLE": "server",
    "DESK_DISPLAY_SERVER_AUTH_TOKEN": SERVER_TOKEN,
    "SCREEN_CONFIG_HOST": "127.0.0.1",
}


def _errors(report: dc.ValidationReport) -> dict[str | None, str]:
    return {issue.name: issue.message for issue in report.errors}


def _warnings(report: dc.ValidationReport) -> dict[str | None, str]:
    return {issue.name: issue.message for issue in report.warnings}


# ── Examples ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "filename, role",
    [
        (".env.server.example", Role.SERVER),
        (".env.client.example", Role.CLIENT),
        (".env.example", Role.STANDALONE),
    ],
)
def test_every_example_value_parses(filename, role):
    values = dc.parse_env_file(ROOT / filename)
    assert values["DESK_DISPLAY_ROLE"] == role.value
    for name, raw in values.items():
        setting = dc.SETTINGS_BY_NAME.get(name)
        assert setting is not None, f"{filename}: unknown setting {name}"
        assert role in setting.roles, f"{filename}: {name} does not apply to {role.value}"
        dc.parse_value(setting, raw)
    dc.load_settings(role, values)


@pytest.mark.parametrize("role", [Role.SERVER, Role.CLIENT])
def test_committed_examples_match_catalog(role):
    committed = (ROOT / f".env.{role.value}.example").read_text(encoding="utf-8")
    assert committed == dc.render_example(role), (
        f"regenerate with: python3 -m deployment_config example --role {role.value}"
    )


@pytest.mark.parametrize("role", [Role.SERVER, Role.CLIENT])
def test_examples_are_thorough(role):
    values = dc.parse_env_file(ROOT / f".env.{role.value}.example")
    assert set(values) == {s.name for s in dc.settings_for_role(role)}


def test_server_example_covers_requested_sections():
    text = (ROOT / ".env.server.example").read_text(encoding="utf-8")
    for heading in (
        "Server bind address and client authentication",
        "Location and content timezone",
        "Weather providers",
        "Sports and teams",
        "News",
        "ADS-B",
        "Maps and travel",
        "Screen configuration, styles, and layouts",
        "Refresh intervals",
        "Render workers",
        "Artifact storage",
        "Client leases and static clients",
        "Configuration UI",
        "Logs",
        "Diagnostics",
    ):
        assert f"# {heading}" in text, heading


def test_client_example_covers_requested_sections():
    text = (ROOT / ".env.client.example").read_text(encoding="utf-8")
    for heading in (
        "Client identity",
        "Server URL and client authentication",
        "Display profile",
        "Output driver",
        "Physical rotation",
        "Framebuffer and panel hardware",
        "Local artifact cache",
        "Synchronization and heartbeat",
        "Backlight and dark hours",
        "Buttons",
        "Touch and keyboard",
        "Offline startup",
        "Logs",
        "Diagnostics",
    ):
        assert f"# {heading}" in text, heading


def test_client_needs_no_upstream_keys_or_provider_urls():
    values = dc.parse_env_file(ROOT / ".env.client.example")
    for name in values:
        setting = dc.SETTINGS_BY_NAME[name]
        assert not setting.provider, name
        if setting.secret:
            assert name in {"DESK_DISPLAY_CLIENT_TOKEN", "FEED_UPLOAD_TOKEN"}, name
    client_settings = dc.settings_for_role(Role.CLIENT)
    assert not [s.name for s in client_settings if s.provider]


def test_client_example_validates_once_token_is_filled():
    env = dc.parse_env_file(ROOT / ".env.client.example")
    assert set(_errors(dc.validate(Role.CLIENT, env))) == {"DESK_DISPLAY_CLIENT_TOKEN"}
    env["DESK_DISPLAY_CLIENT_TOKEN"] = SERVER_TOKEN
    report = dc.validate(Role.CLIENT, env)
    assert report.ok, report.errors
    assert not report.warnings


def test_server_example_validates_with_provisioned_clients():
    env = dc.parse_env_file(ROOT / ".env.server.example")
    assert dc.validate(Role.SERVER, env).ok  # per-client credentials need no shared token
    env["DESK_DISPLAY_SERVER_ENROLLMENT"] = "shared"
    assert set(_errors(dc.validate(Role.SERVER, env))) == {"DESK_DISPLAY_SERVER_AUTH_TOKEN"}
    env["DESK_DISPLAY_SERVER_AUTH_TOKEN"] = SERVER_TOKEN
    assert dc.validate(Role.SERVER, env).ok


def test_legacy_example_stays_documented_and_valid():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "legacy standalone" in text
    assert ".env.server.example" in text and ".env.client.example" in text
    assert dc.validate(Role.STANDALONE, dc.parse_env_file(ROOT / ".env.example")).ok


def test_configuration_doc_reference_is_current():
    doc = (ROOT / "CONFIGURATION.md").read_text(encoding="utf-8")
    assert dc.render_settings_reference() in doc
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "CONFIGURATION.md" in readme


def test_hot_reload_documents_are_documented():
    doc = (ROOT / "CONFIGURATION.md").read_text(encoding="utf-8")
    assert "requires a restart" in doc
    for name in dc.HOT_RELOAD_DOCUMENTS:
        assert name in dc.SETTINGS_BY_NAME
        assert f"`{name}`" in doc


def test_catalog_covers_every_variable_the_code_reads():
    import ast

    names: set[str] = set()
    skip_dirs = {"tests", "vendor", ".git", "node_modules", "venv", ".venv"}
    for path in ROOT.rglob("*.py"):
        if skip_dirs.intersection(path.relative_to(ROOT).parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            attr = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            owner = ast.unparse(func.value) if isinstance(func, ast.Attribute) else ""
            is_env_read = attr == "getenv" or (attr == "get" and owner.endswith("environ"))
            if is_env_read and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names.add(first.value)
    ignored = {"HOME", "PATH", "USER", "LANG", "PYTHONPATH", "PWD", "SHELL", "TERM",
               "HOSTNAME", "LOGNAME", "SUDO_USER", "PYTEST_CURRENT_TEST", "CI",
               "GITHUB_ACTIONS", "VIRTUAL_ENV", "XDG_SESSION_TYPE", "XDG_CONFIG_HOME",
               "XDG_CACHE_HOME", "XDG_DATA_HOME", "APPDATA", "LOCALAPPDATA", "TMPDIR"}
    missing = sorted(n for n in names - set(dc.SETTINGS_BY_NAME) - ignored if n.isupper() and not n.startswith("GITHUB_"))
    assert not missing, f"add these settings to deployment_config.SETTINGS: {missing}"


# ── Parsing and defaults ────────────────────────────────────────────────────


def test_role_resolution():
    assert dc.resolve_role({}) is Role.STANDALONE
    assert dc.resolve_role({"DESK_DISPLAY_ROLE": " Client "}) is Role.CLIENT
    with pytest.raises(ValueError, match="not a role"):
        dc.resolve_role({"DESK_DISPLAY_ROLE": "kiosk"})


def test_defaults_apply_when_unset():
    server = dc.load_settings(Role.SERVER, {})
    assert server["DESK_DISPLAY_SERVER_HOST"] == "127.0.0.1"
    assert server["DESK_DISPLAY_SERVER_PORT"] == 8765
    assert server["DESK_DISPLAY_RENDER_WORKERS"] == 2
    assert server["DESK_DISPLAY_CLIENT_LEASE_SECONDS"] == 300
    assert server["DESK_DISPLAY_CONTENT_TIMEZONE"] == "America/Chicago"
    assert server["DESK_DISPLAY_SERVER_AUTH_TOKEN"] is None
    assert "DESK_DISPLAY_CLIENT_ID" not in server

    client = dc.load_settings(Role.CLIENT, {})
    assert client["DESK_DISPLAY_OUTPUT"] == "auto"
    assert client["DISPLAY_ROTATION"] == 0
    assert client["DESK_DISPLAY_TLS_VERIFY"] is True
    assert client["DESK_DISPLAY_OFFLINE_START"] is True
    assert client["DESK_DISPLAY_SYNC_INTERVAL_SECONDS"] == 30
    assert client["DESK_DISPLAY_HEARTBEAT_INTERVAL_SECONDS"] == 60
    assert "OWM_API_KEY" not in client
    assert "SCREEN_UI_PASSWORD" not in client


@pytest.mark.parametrize(
    "name, raw, expected",
    [
        ("DISPLAY_ROTATION", "270", 270),
        ("DISPLAY_ROTATION", "1", 90),
        ("DESK_DISPLAY_OUTPUT", "hat", "displayhatmini"),
        ("DESK_DISPLAY_OUTPUT", "KMS", "kernel"),
        ("DESK_DISPLAY_PROFILE", "HyperPixel4", "hyperpixel4"),
        ("DESK_DISPLAY_TLS_VERIFY", "off", False),
        ("WAVESHARE_OLED_TEMP_ADDR", "0x3C", 0x3C),
        ("INSIDE_I2C_BUSES", "1, 2,,10", ["1", "2", "10"]),
        ("DESK_DISPLAY_LOG_LEVEL", "debug", "DEBUG"),
        ("DESK_DISPLAY_STATIC_CLIENTS", "den:hdmi_1080p, desk:display_hat_mini",
         {"den": "hdmi_1080p", "desk": "display_hat_mini"}),
        ("NHL_BREAK_WINDOWS_JSON", '[{"start": "2026-02-01"}]', [{"start": "2026-02-01"}]),
        ("DESK_DISPLAY_SERVER_PORT", "", None),
    ],
)
def test_parse_value(name, raw, expected):
    assert dc.parse_value(dc.SETTINGS_BY_NAME[name], raw) == expected


def test_parse_env_file_handles_quotes_comments_and_export(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# comment\nexport A=1\nB='two words'\nC=3 # trailing\nD=\"#kept\"\nnot a line\n",
        encoding="utf-8",
    )
    assert dc.parse_env_file(path) == {"A": "1", "B": "two words", "C": "3", "D": "#kept"}


def test_resolve_log_level():
    assert dc.resolve_log_level({"DESK_DISPLAY_LOG_LEVEL": "warning"}) == logging.WARNING
    assert dc.resolve_log_level({"DESK_DISPLAY_LOG_LEVEL": "loud"}) == logging.INFO
    assert dc.resolve_log_level({}) == logging.INFO


# ── Validation errors ───────────────────────────────────────────────────────


def test_valid_client_and_server():
    assert dc.validate(Role.CLIENT, CLIENT_OK).ok
    assert dc.validate(Role.SERVER, SERVER_OK).ok


def test_client_rejects_misplaced_server_variables():
    env = {**CLIENT_OK, "OWM_API_KEY": "abc123456789", "DESK_DISPLAY_RENDER_WORKERS": "4",
           "SCREEN_UI_PASSWORD": "hunter22", "WEATHER_LATITUDE": ""}
    errors = _errors(dc.validate(Role.CLIENT, env))
    assert "provider credential does not belong on a client" in errors["OWM_API_KEY"]
    assert "server setting does not belong on a client" in errors["DESK_DISPLAY_RENDER_WORKERS"]
    assert "SCREEN_UI_PASSWORD" in errors
    # An empty misplaced value configures nothing, so it is not flagged.
    assert "WEATHER_LATITUDE" not in errors


def test_client_requires_identity_and_server():
    errors = _errors(dc.validate(Role.CLIENT, {"DESK_DISPLAY_ROLE": "client"}))
    assert set(errors) == {
        "DESK_DISPLAY_CLIENT_ID",
        "DESK_DISPLAY_SERVER_URL",
        "DESK_DISPLAY_CLIENT_TOKEN",
        "DESK_DISPLAY_PROFILE",
    }


def test_client_rejects_malformed_identity():
    errors = _errors(dc.validate(Role.CLIENT, {**CLIENT_OK, "DESK_DISPLAY_CLIENT_ID": "bad id!"}))
    assert set(errors) == {"DESK_DISPLAY_CLIENT_ID"}


@pytest.mark.parametrize("role", [Role.CLIENT, Role.STANDALONE])
def test_unknown_profile_and_invalid_rotation(role):
    env = {**CLIENT_OK, "DESK_DISPLAY_PROFILE": "crt_tv", "DISPLAY_ROTATION": "45"}
    if role is Role.STANDALONE:
        env = {k: v for k, v in env.items() if not k.startswith(("DESK_DISPLAY_CLIENT", "DESK_DISPLAY_SERVER"))}
        env["DESK_DISPLAY_ROLE"] = "standalone"
    errors = _errors(dc.validate(role, env))
    assert "unknown display profile" in errors["DESK_DISPLAY_PROFILE"]
    assert "90, 180 or 270" in errors["DISPLAY_ROTATION"]


def test_unknown_output_driver():
    errors = _errors(dc.validate(Role.CLIENT, {**CLIENT_OK, "DESK_DISPLAY_OUTPUT": "vga"}))
    assert "unknown output driver" in errors["DESK_DISPLAY_OUTPUT"]


@pytest.mark.parametrize(
    "output, profile, ok",
    [
        ("displayhatmini", "display_hat_mini", True),
        ("displayhatmini", "hyperpixel4", False),
        ("minipitft", "adafruit_minipitft_114", True),
        ("minipitft", "display_hat_mini", False),
        ("framebuffer", "display_hat_mini", False),
        ("kernel", "adafruit_minipitft_114", False),
        ("framebuffer", "hyperpixel4", True),
        ("window", "display_hat_mini", True),
        ("headless", "hdmi_1080p", True),
        ("auto", "display_hat_mini", True),
    ],
)
def test_driver_profile_compatibility(output, profile, ok):
    env = {**CLIENT_OK, "DESK_DISPLAY_OUTPUT": output, "DESK_DISPLAY_PROFILE": profile}
    report = dc.validate(Role.CLIENT, env)
    assert report.ok is ok, report.errors


def test_client_insecure_transport():
    http = {**CLIENT_OK, "DESK_DISPLAY_SERVER_URL": "http://server.lan:8765"}
    assert "insecure" in _errors(dc.validate(Role.CLIENT, http))["DESK_DISPLAY_SERVER_URL"]
    allowed = dc.validate(Role.CLIENT, {**http, "DESK_DISPLAY_ALLOW_INSECURE_TRANSPORT": "1"})
    assert allowed.ok and "DESK_DISPLAY_SERVER_URL" in _warnings(allowed)
    assert dc.validate(Role.CLIENT, {**CLIENT_OK, "DESK_DISPLAY_SERVER_URL": "http://127.0.0.1:8765"}).ok
    no_verify = {**CLIENT_OK, "DESK_DISPLAY_TLS_VERIFY": "0"}
    assert "DESK_DISPLAY_TLS_VERIFY" in _errors(dc.validate(Role.CLIENT, no_verify))
    ftp = {**CLIENT_OK, "DESK_DISPLAY_SERVER_URL": "webcal://server.lan"}
    assert "DESK_DISPLAY_SERVER_URL" in _errors(dc.validate(Role.CLIENT, ftp))


def test_client_missing_ca_bundle(tmp_path):
    env = {**CLIENT_OK, "DESK_DISPLAY_SERVER_CA_BUNDLE": str(tmp_path / "missing.pem")}
    assert "file not found" in _errors(dc.validate(Role.CLIENT, env))["DESK_DISPLAY_SERVER_CA_BUNDLE"]


def test_server_authentication_rules():
    local = {"DESK_DISPLAY_ROLE": "server", "SCREEN_CONFIG_HOST": "127.0.0.1"}
    assert dc.validate(Role.SERVER, local).ok  # provisioned by default: no shared token
    shared = {**local, "DESK_DISPLAY_SERVER_ENROLLMENT": "shared"}
    missing = _errors(dc.validate(Role.SERVER, shared))
    assert "required" in missing["DESK_DISPLAY_SERVER_AUTH_TOKEN"]

    short = _errors(dc.validate(Role.SERVER, {**SERVER_OK, **shared, "DESK_DISPLAY_SERVER_AUTH_TOKEN": "short"}))
    assert "insecure" in short["DESK_DISPLAY_SERVER_AUTH_TOKEN"]

    ignored = dc.validate(Role.SERVER, SERVER_OK)
    assert ignored.ok and any(w.name == "DESK_DISPLAY_SERVER_AUTH_TOKEN" for w in ignored.warnings)

    loopback_open = {"DESK_DISPLAY_SERVER_ALLOW_UNAUTHENTICATED": "1", "SCREEN_CONFIG_HOST": "127.0.0.1"}
    assert dc.validate(Role.SERVER, loopback_open).ok

    public_open = _errors(dc.validate(
        Role.SERVER, {**loopback_open, "DESK_DISPLAY_SERVER_HOST": "0.0.0.0"}
    ))
    assert "insecure" in public_open["DESK_DISPLAY_SERVER_ALLOW_UNAUTHENTICATED"]


def test_server_admin_token_rules():
    short = _errors(dc.validate(Role.SERVER, {**SERVER_OK, "DESK_DISPLAY_SERVER_ADMIN_TOKEN": "short"}))
    assert "insecure" in short["DESK_DISPLAY_SERVER_ADMIN_TOKEN"]
    same = _errors(dc.validate(Role.SERVER, {**SERVER_OK, "DESK_DISPLAY_SERVER_ADMIN_TOKEN": SERVER_TOKEN}))
    assert "must differ" in same["DESK_DISPLAY_SERVER_ADMIN_TOKEN"]
    assert dc.validate(Role.SERVER, {**SERVER_OK, "DESK_DISPLAY_SERVER_ADMIN_TOKEN": "a" * 40}).ok
    assert dc.SETTINGS_BY_NAME["DESK_DISPLAY_SERVER_ADMIN_TOKEN"].secret


def test_server_tls_pairing_and_public_bind_warning(tmp_path):
    cert = tmp_path / "cert.pem"
    cert.write_text("cert", encoding="utf-8")
    errors = _errors(dc.validate(Role.SERVER, {**SERVER_OK, "DESK_DISPLAY_SERVER_TLS_CERT": str(cert)}))
    assert "together" in errors["DESK_DISPLAY_SERVER_TLS_KEY"]

    public = dc.validate(Role.SERVER, {**SERVER_OK, "DESK_DISPLAY_SERVER_HOST": "0.0.0.0"})
    assert public.ok and "without TLS" in _warnings(public)["DESK_DISPLAY_SERVER_HOST"]


def test_config_ui_auth_rules():
    errors = _errors(dc.validate(Role.SERVER, {**SERVER_OK, "SCREEN_AUTH_ENABLED": "1"}))
    assert "requires a password" in errors["SCREEN_UI_PASSWORD"]
    warnings = _warnings(dc.validate(Role.SERVER, {**SERVER_OK, "SCREEN_CONFIG_HOST": "0.0.0.0"}))
    assert "without a password" in warnings["SCREEN_UI_PASSWORD"]


def test_content_validation():
    env = {**SERVER_OK, "WEATHER_LATITUDE": "41.9", "DESK_DISPLAY_CONTENT_TIMEZONE": "Mars/Base",
           "AIR_QUALITY_LATITUDE": "123", "AIR_QUALITY_LONGITUDE": "-87"}
    errors = _errors(dc.validate(Role.SERVER, env))
    assert "set both" in errors["WEATHER_LONGITUDE"]
    assert "timezone" in errors["DESK_DISPLAY_CONTENT_TIMEZONE"]
    assert "between -90 and 90" in errors["AIR_QUALITY_LATITUDE"]


def test_invalid_numbers_and_booleans():
    env = {**SERVER_OK, "DESK_DISPLAY_RENDER_WORKERS": "0", "DESK_DISPLAY_SERVER_PORT": "http",
           "DESK_DISPLAY_SERVER_ALLOW_UNAUTHENTICATED": "maybe"}
    errors = _errors(dc.validate(Role.SERVER, env))
    assert "at least 1" in errors["DESK_DISPLAY_RENDER_WORKERS"]
    assert "whole number" in errors["DESK_DISPLAY_SERVER_PORT"]
    assert "1/0" in errors["DESK_DISPLAY_SERVER_ALLOW_UNAUTHENTICATED"]


def test_static_clients_validation():
    errors = _errors(dc.validate(Role.SERVER, {**SERVER_OK, "DESK_DISPLAY_STATIC_CLIENTS": "den:crt"}))
    assert "unknown profile" in errors["DESK_DISPLAY_STATIC_CLIENTS"]


def test_standalone_warns_on_role_exclusive_settings_and_unknown_names():
    env = {"DESK_DISPLAY_CLIENT_ID": "desk", "DESK_DISPLAY_RENDER_WORKERS": "4", "NOT_A_SETTING": "1",
           "SCREEN_CONFIG_HOST": "127.0.0.1"}
    report = dc.validate(Role.STANDALONE, env)
    assert report.ok
    warnings = _warnings(report)
    assert "ignored in the standalone role" in warnings["DESK_DISPLAY_CLIENT_ID"]
    assert "DESK_DISPLAY_RENDER_WORKERS" in warnings
    assert "not a Desk Display setting" in warnings["NOT_A_SETTING"]
    assert "NOT_A_SETTING" not in _warnings(dc.validate(Role.STANDALONE, env, check_unknown=False))


def test_startup_check_fails_closed_for_split_roles(caplog):
    with pytest.raises(dc.ConfigurationError, match="DESK_DISPLAY_CLIENT_ID"):
        dc.startup_check("client", {"DESK_DISPLAY_ROLE": "client"})
    with pytest.raises(dc.ConfigurationError, match="not a role"):
        dc.startup_check("display", {"DESK_DISPLAY_ROLE": "kiosk"})
    assert dc.startup_check("client", CLIENT_OK).ok


def test_startup_check_only_logs_for_standalone(caplog):
    with caplog.at_level(logging.WARNING, logger="desk_display.config"):
        report = dc.startup_check("display", {"DISPLAY_ROTATION": "45"})
    assert not report.ok
    assert "DISPLAY_ROTATION" in caplog.text


def test_cli_check(tmp_path, capsys):
    path = tmp_path / ".env"
    path.write_text("DESK_DISPLAY_ROLE=client\nOWM_API_KEY=abc\n", encoding="utf-8")
    assert dc._cli(["check", "--env-file", str(path)]) == 1
    out = capsys.readouterr().out
    assert "error: OWM_API_KEY" in out
    good = tmp_path / "good.env"
    good.write_text("\n".join(f"{k}={v}" for k, v in CLIENT_OK.items()), encoding="utf-8")
    assert dc._cli(["check", "--env-file", str(good)]) == 0


# ── Secret exclusion ────────────────────────────────────────────────────────

SECRET_ENV = {
    "OWM_API_KEY": "owm-secret-value-123",
    "DESK_DISPLAY_SERVER_AUTH_TOKEN": SERVER_TOKEN,
    "AHL_SCHEDULE_ICS_URL": "https://stanza.example/private-calendar-abc",
    "SCREEN_UI_PASSWORD": "short",
}


def test_every_credential_is_marked_secret():
    for name in ("OWM_API_KEY", "AIRNOW_API_KEY", "AHL_API_KEY", "WEATHERKIT_PRIVATE_KEY",
                 "DESK_DISPLAY_SERVER_AUTH_TOKEN", "DESK_DISPLAY_CLIENT_TOKEN",
                 "SCREEN_UI_PASSWORD", "SCREEN_SESSION_SECRET", "FEED_UPLOAD_TOKEN"):
        assert dc.SETTINGS_BY_NAME[name].secret, name


def test_scrub_secrets_drops_keys_and_redacts_values():
    payload = {
        "screens": [{"id": "weather", "url": "https://api.example/?appid=owm-secret-value-123"}],
        "owm_api_key": "anything",
        "nested": {"Desk_Display_Server_Auth_Token": "x", "ok": ("a", SERVER_TOKEN)},
        "note": "calendar https://stanza.example/private-calendar-abc",
        "short": "short",
    }
    scrubbed = dc.scrub_secrets(payload, SECRET_ENV)
    assert dc.find_secrets(scrubbed, SECRET_ENV) == []
    assert "owm_api_key" not in scrubbed
    assert scrubbed["screens"][0]["url"].endswith("appid=[redacted]")
    assert scrubbed["nested"] == {"ok": ["a", "[redacted]"]}
    assert scrubbed["note"] == "calendar [redacted]"
    # Values under 8 characters are never used for text redaction.
    assert scrubbed["short"] == "short"
    assert set(dc.find_secrets(payload, SECRET_ENV)) == {
        "screens[0].url", "owm_api_key", "nested.Desk_Display_Server_Auth_Token",
        "nested.ok[1]", "note",
    }


def test_manifest_never_carries_server_credentials(monkeypatch):
    import protocol

    for name, value in SECRET_ENV.items():
        monkeypatch.setenv(name, value)
    manifest = protocol.build_manifest(
        client_id="desk",
        OWM_API_KEY="owm-secret-value-123",
        items=[{"source": "https://api.example/?key=owm-secret-value-123"}],
    )
    assert dc.find_secrets(manifest, SECRET_ENV) == []
    assert manifest["client_id"] == "desk"
    assert protocol.client_supports_manifest(manifest)

    registration = protocol.registration_response(
        protocol.build_registration("desk", "1.0") if hasattr(protocol, "build_registration")
        else {"client_id": "desk", "client_software_version": "1.0",
              "protocol_version": protocol.NETWORK_PROTOCOL_VERSION}
    )
    assert dc.find_secrets(registration, SECRET_ENV) == []


def test_log_redaction(monkeypatch, caplog):
    monkeypatch.setenv("OWM_API_KEY", "owm-secret-value-123")
    monkeypatch.setattr(dc, "_ORIGINAL_RECORD_FACTORY", None)
    original = logging.getLogRecordFactory()
    try:
        dc.install_secret_log_redaction()
        with caplog.at_level(logging.INFO):
            logging.getLogger("test").info("GET %s", "https://api.example/?appid=owm-secret-value-123")
        assert "owm-secret-value-123" not in caplog.text
        assert "appid=[redacted]" in caplog.text
    finally:
        logging.setLogRecordFactory(original)


def test_config_ui_and_feed_server_responses_are_redacted(monkeypatch):
    flask = pytest.importorskip("flask")
    monkeypatch.setenv("OWM_API_KEY", "owm-secret-value-123")

    app = flask.Flask(__name__)
    app.after_request(dc.redact_response)

    @app.route("/json")
    def _json():
        return flask.jsonify({"log": "fetch appid=owm-secret-value-123 failed"})

    @app.route("/png")
    def _png():
        return flask.Response(b"owm-secret-value-123", mimetype="image/png")

    client = app.test_client()
    assert b"owm-secret-value-123" not in client.get("/json").data
    assert b"[redacted]" in client.get("/json").data
    assert client.get("/png").data == b"owm-secret-value-123"

    import config_ui
    import feed_server

    for module in (config_ui, feed_server):
        hooks = module.app.after_request_funcs.get(None, [])
        assert any(getattr(hook, "__name__", "") == "_redact_secrets" for hook in hooks), module


def test_feed_server_scrubs_uploaded_client_status(monkeypatch, tmp_path):
    import feed_server

    monkeypatch.setenv("OWM_API_KEY", "owm-secret-value-123")
    monkeypatch.setattr(feed_server, "FEED_UPLOAD_TOKEN", "upload-token-123456")
    monkeypatch.setattr(feed_server, "_source_current_dir", lambda source: tmp_path / source)
    client = feed_server.app.test_client()
    response = client.post(
        "/api/feed/desk/status",
        json={"screen_id": "weather", "error": "appid=owm-secret-value-123", "owm_api_key": "x"},
        headers={"Authorization": "Bearer upload-token-123456"},
    )
    assert response.status_code == 200, response.data
    stored = (tmp_path / "desk" / "display_status.json").read_text(encoding="utf-8")
    assert "owm-secret-value-123" not in stored and "owm_api_key" not in stored
    assert "weather" in stored
