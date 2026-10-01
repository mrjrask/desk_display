"""The Stats page API: server-published stats, local fallback, per-display rows."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from flask import Flask

import stats_ui
from remote_display import resource_stats as rs

ROOT = Path(__file__).resolve().parents[1]
NOW = 1_800_000_000.0


class StubSampler:
    def __init__(self):
        self.calls = 0

    def sample(self):
        self.calls += 1
        return {"schema_version": 1, "role": "local", "generated_at": NOW, "cpu": {}, "traffic": {}}


@pytest.fixture
def env(tmp_path):
    return {
        "DESK_DISPLAY_SERVER_STATS_PATH": str(tmp_path / "stats.json"),
        "DESK_DISPLAY_CLIENT_REGISTRY_PATH": str(tmp_path / "clients.json"),
        "DESK_DISPLAY_PLAYLIST_STORE_PATH": str(tmp_path / "playlists.json"),
    }


def make_app(env, sampler=None, now=NOW):
    app = Flask(__name__, template_folder=str(ROOT / "templates"))
    app.jinja_env.globals.update(machine_hostname="square", screenshots_only=False)
    stats_ui.register(app, env=env, clock=lambda: now, local_sampler=sampler or StubSampler())
    return app.test_client()


def published(**overrides):
    return {
        "schema_version": 1, "role": "server", "generated_at": NOW - 5,
        "cpu": {"project_percent": 12.0, "by_purpose": {"rendering": 10.0, "feeds": 2.0}},
        "traffic": {
            "totals": {"office": {"bytes_out": 5000, "bytes_in": 100, "requests": 4, "kinds": {}},
                       "_unauthenticated": {"bytes_out": 10, "bytes_in": 10, "requests": 1, "kinds": {}},
                       "gone": {"bytes_out": 1, "bytes_in": 1, "requests": 1, "kinds": {}}},
            "session": {"office": {"bytes_out": 3000}},
            "rates": {"office": {"out_bps": 25.0, "in_bps": 1.0}},
        },
        **overrides,
    }


def test_server_stats_and_display_rows(env, tmp_path):
    rs.write_json_atomic(Path(env["DESK_DISPLAY_SERVER_STATS_PATH"]), published())
    Path(env["DESK_DISPLAY_CLIENT_REGISTRY_PATH"]).write_text(json.dumps({
        "schema_version": 1, "heartbeat_interval_seconds": 30,
        "clients": {"office": {
            "client_id": "office", "last_seen": "2027-01-15T08:00:00Z", "address": "10.0.0.7",
            "capabilities": {"display_profile": "hyperpixel4_square"},
            "resources": {"type": "client_resources", "version": 1, "process_cpu_percent": 4.5, "cache_bytes": 99},
            "telemetry": {"download_bytes": 2048, "heartbeat_rtt_ms": 20.0},
        }},
    }))
    sampler = StubSampler()
    body = make_app(env, sampler).get("/api/stats").get_json()
    assert body["source"] == "server" and body["server_stale_seconds"] is None
    assert body["stats"]["cpu"]["project_percent"] == 12.0
    assert sampler.calls == 0  # a fresh server document needs no local sample
    rows = {row["client_id"]: row for row in body["clients"]}
    office = rows["office"]
    assert office["resources"]["process_cpu_percent"] == 4.5
    assert "type" not in office["resources"]
    assert office["traffic_total"]["bytes_out"] == 5000
    assert office["traffic_session"]["bytes_out"] == 3000
    assert office["traffic_rate"]["out_bps"] == 25.0
    assert office["last_sync_download_bytes"] == 2048
    assert office["display_profile"] == "hyperpixel4_square"
    # A display the server once served but no longer knows keeps its traffic row.
    assert rows["gone"]["resources"] is None and rows["gone"]["state"] == "unknown"
    assert "_unauthenticated" not in rows


def test_stale_or_missing_server_stats_fall_back_to_a_local_sample(env):
    sampler = StubSampler()
    client = make_app(env, sampler)
    body = client.get("/api/stats").get_json()
    assert body["source"] == "local" and body["server_stale_seconds"] is None
    assert body["stats"]["role"] == "local"
    client.get("/api/stats")
    assert sampler.calls == 1  # reused within the minimum interval

    rs.write_json_atomic(Path(env["DESK_DISPLAY_SERVER_STATS_PATH"]),
                         published(generated_at=NOW - rs.STALE_AFTER_SECONDS - 60))
    body = make_app(env, StubSampler()).get("/api/stats").get_json()
    assert body["source"] == "local"
    assert body["server_stale_seconds"] == round(rs.STALE_AFTER_SECONDS + 60)


def test_stats_page_renders_with_nav(env):
    page = make_app(env).get("/stats")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert 'href="/stats" aria-current="page"' in html
    assert "CPU by purpose" in html and "/api/stats" in html


def test_every_config_page_links_the_stats_page():
    for name in ("clients.html", "playlists.html", "screen_config.html", "screenshots.html", "client_wizard.html"):
        assert 'href="/stats"' in (ROOT / "templates" / name).read_text(encoding="utf-8"), name


def test_client_rows_mark_stale_and_disabled_displays():
    snapshot = {"heartbeat_interval_seconds": 30, "clients": {
        "a": {"last_seen": "2027-01-15T07:00:00Z"},
        "b": {"last_seen": "2027-01-15T08:00:00Z", "disabled": True},
    }}
    now = 1_800_000_000.0  # 2027-01-15T08:00:00Z
    rows = {row["client_id"]: row for row in stats_ui.client_stats_rows(snapshot, {"a": {"friendly_name": "Den"}}, {}, now)}
    assert rows["a"]["state"] == "stale" and rows["a"]["friendly_name"] == "Den"
    assert rows["b"]["state"] == "disabled"
