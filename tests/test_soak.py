"""Phase 20a: the soak monitor, release gates and rollback triggers for Phase 20b."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

import soak

ADMIN = "admin-token-" + "a" * 32
TOKEN = "server-token-" + "s" * 32


# ── Sampling a live server ──────────────────────────────────────────────────


@pytest.fixture
def live_server(tmp_path):
    pytest.importorskip("flask")
    from concurrent.futures import Future

    from PIL import Image
    from werkzeug.serving import make_server

    import display_client
    import display_server
    from display_profiles import PROFILE_PRESETS
    from remote_display.models import ScreenRevisions
    from remote_display.render_coordinator import RenderOutput

    class Inline:
        def submit(self, fn):
            future = Future()
            future.set_result(fn())
            return future

    def renderer(key):
        preset = PROFILE_PRESETS[key.render_profile]
        return RenderOutput(image=Image.new(preset.color_mode, (preset.width, preset.height), 7))

    config = display_server.DisplayServerConfig(enrollment="shared", auth_token=TOKEN, admin_token=ADMIN,
                                                artifact_dir=tmp_path / "artifacts")
    app = display_server.create_app(config, renderer=renderer,
                                    revisions=lambda screens: {s: ScreenRevisions("s", "d", "r") for s in screens},
                                    render_executor=Inline())
    caps = display_client.capabilities_for("office", PROFILE_PRESETS["hyperpixel4"]).to_wire()
    response = app.test_client().post("/api/v1/register", json={"capabilities": caps},
                                      headers={"Authorization": f"Bearer {TOKEN}"})
    credential = response.get_json()["client_credential"]
    server = make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", credential
    server.shutdown()


def test_a_sample_records_the_server_view_without_secrets(live_server, tmp_path):
    url, credential = live_server
    (tmp_path / "cache" / "client").mkdir(parents=True)
    (tmp_path / "cache" / "client" / "a.png").write_bytes(b"x" * 2048)
    sample = soak.collect_sample(project_dir=tmp_path, server_url=url, admin_token=ADMIN, now=100)
    assert sample["v"] == soak.SAMPLE_VERSION and sample["t"] == 100
    server = sample["server"]
    assert server["reachable"] is True
    assert server["clients"]["office"]["lease_state"] == "active"
    assert set(server["render"]) >= {"renders", "failures", "failing", "not_fresh", "queue"}
    assert server["playlists"] == {"office": "unassigned"}
    assert sample["disk"]["cache/client"] == 0.0 and 0 < sample["disk"]["free_percent"] <= 100
    text = json.dumps(sample)
    assert ADMIN not in text and TOKEN not in text and credential not in text


def test_a_sample_without_the_admin_token_records_the_refusal(live_server, tmp_path):
    url, _ = live_server
    sample = soak.collect_sample(project_dir=tmp_path, server_url=url, admin_token="wrong", now=1)
    assert sample["server"] == {"reachable": False, "error": "401"}
    unreachable = soak.collect_sample(project_dir=tmp_path, server_url="http://127.0.0.1:9", admin_token=ADMIN,
                                      timeout=1)
    assert unreachable["server"]["reachable"] is False


def test_processes_are_found_by_what_they_run(tmp_path):
    def fake(pid, argv, rss_pages, utime=150, stime=50):
        entry = tmp_path / str(pid)
        (entry / "fd").mkdir(parents=True)
        for n in range(3):
            (entry / "fd" / str(n)).touch()
        (entry / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
        fields = ["S"] + ["0"] * 10 + [str(utime), str(stime)] + ["0"] * 30
        (entry / "stat").write_text(f"{pid} (python3) " + " ".join(fields))
        (entry / "statm").write_text(f"1000 {rss_pages} 0 0 0 0 0")

    fake(11, ["/opt/venv/bin/python", "display_server.py"], 25600)
    fake(12, ["/opt/venv/bin/python", "display_client.py"], 12800)
    fake(13, ["/usr/bin/python3", "soak.py", "sample"], 999)
    fake(14, ["bash", "display_server.sh"], 1)
    (tmp_path / "self").mkdir()
    found = soak._processes(tmp_path)
    assert set(found) == {"server", "client"}
    assert found["server"]["pid"] == 11 and found["server"]["fds"] == 3
    assert found["server"]["rss_mb"] == pytest.approx(25600 * os.sysconf("SC_PAGE_SIZE") / 2**20, abs=0.1)
    assert found["server"]["cpu_seconds"] > 0


def test_the_sampler_reads_the_admin_token_from_the_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("DESK_DISPLAY_SERVER_ADMIN_TOKEN", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"FOO=1\nexport DESK_DISPLAY_SERVER_ADMIN_TOKEN='{ADMIN}'\n")
    assert soak._env_file_value(env, "DESK_DISPLAY_SERVER_ADMIN_TOKEN") == ADMIN
    assert soak._cli(["sample", "--out", str(tmp_path / "s.jsonl"), "--server-url", "http://127.0.0.1:9",
                      "--env-file", str(tmp_path / "missing")]) == 2
    assert soak._cli(["sample", "--once", "--out", str(tmp_path / "s.jsonl"), "--project-dir", str(tmp_path)]) == 0
    (line,) = (tmp_path / "s.jsonl").read_text().splitlines()
    assert "server" not in json.loads(line)


# ── Gates and triggers ──────────────────────────────────────────────────────


def run(hours=48.0, interval=60, clients=("office", "den"), **tweaks):
    """A healthy soak log, then *tweaks* per sample index: fn(i, sample)."""

    samples = []
    count = int(hours * 3600 / interval) + 1
    renders = 0
    for i in range(count):
        renders += 1
        sample = {
            "v": 1, "t": 1_000_000.0 + i * interval, "host": "server-pi",
            "server": {
                "reachable": True,
                "clients": {c: {"lease_state": "active", "playback_state": "playing", "current_screen": "date",
                                "last_sync_age_seconds": 20.0, "errors": 0} for c in clients},
                "render": {"renders": renders, "failures": 0, "failing": [], "not_fresh": 0, "queue": 0},
                "playlists": {c: "in_sync" for c in clients},
                "artifacts": {"objects": 10, "bytes": 1000},
            },
            "processes": {"server": {"pid": 100, "rss_mb": 150.0, "cpu_seconds": i, "fds": 30}},
            "disk": {"free_percent": 60.0, "cache/artifacts": 20.0, "cache/client": None},
        }
        for tweak in tweaks.values():
            tweak(i, sample)
        samples.append(sample)
    return samples


def failed(results):
    return {r.name for r in results if not r.passed}


def test_a_healthy_soak_is_a_go():
    results = soak.evaluate(run(), soak.Gates(), expected_clients=["office", "den"])
    assert soak.verdict(results) == "GO", soak.report(results)
    text = soak.report(results)
    assert text.startswith("Verdict: GO") and "Rollback triggers" in text


def test_a_short_or_patchy_soak_is_not_enough():
    assert failed(soak.evaluate(run(hours=6), soak.Gates())) == {"duration"}
    patchy = [s for i, s in enumerate(run()) if not 100 <= i < 110]
    assert failed(soak.evaluate(patchy, soak.Gates())) == {"monitoring"}
    assert soak.verdict(soak.evaluate([], soak.Gates())) == "NO-GO"


def test_a_missing_client_fails_its_gates():
    results = soak.evaluate(run(clients=("office",)), soak.Gates(), expected_clients=["office", "shelf"])
    assert {"shelf connected", "shelf playing", "shelf sync age"} <= failed(results)
    assert soak.verdict(results) == "ROLLBACK"  # a blank panel for the whole run


def test_a_blank_panel_for_too_long_triggers_rollback():
    def blank(i, sample):
        if 1000 <= i < 1012:  # 12 minutes on the diagnostic screen
            sample["server"]["clients"]["den"]["current_screen"] = None
            sample["server"]["clients"]["den"]["playback_state"] = "error"
    results = soak.evaluate(run(blank=blank), soak.Gates())
    assert failed(results) == {"den blank or disconnected"}
    assert soak.verdict(results) == "ROLLBACK"

    def brief(i, sample):
        if 1000 <= i < 1005:
            sample["server"]["clients"]["den"]["current_screen"] = None
    assert soak.verdict(soak.evaluate(run(brief=brief), soak.Gates())) == "GO"


def test_render_failures_count_across_server_restarts():
    def flaky(i, sample):
        since = i if i < 1500 else i - 1500  # the server restarted: its counters start again
        sample["server"]["render"]["renders"] = since + 1
        sample["server"]["render"]["failures"] = since // 20
    results = soak.evaluate(run(flaky=flaky), soak.Gates())
    rate = next(r for r in results if r.name == "render failures")
    # 2880 successful renders and 143 failures, with neither counter going backwards.
    assert not rate.passed and rate.detail.startswith("143 of 3023 renders failed (4.73%)"), rate.detail

    def stuck(i, sample):
        if 200 <= i < 220:
            sample["server"]["render"]["failing"] = ["date@hyperpixel4"]
    assert failed(soak.evaluate(run(stuck=stuck), soak.Gates())) == {"sustained render failure"}


def test_memory_growth_restarts_and_disk_are_judged_per_host():
    def leak(i, sample):
        sample["processes"]["server"]["rss_mb"] = 150.0 + i * 0.05  # about 144 MB over two days
    assert failed(soak.evaluate(run(leak=leak), soak.Gates())) == {"server-pi server memory"}

    def runaway(i, sample):
        sample["processes"]["server"]["rss_mb"] = 150.0 + i * 0.1
    assert soak.verdict(soak.evaluate(run(runaway=runaway), soak.Gates())) == "ROLLBACK"

    def restarted(i, sample):
        sample["processes"]["server"]["pid"] = 100 if i < 50 else 200
    assert failed(soak.evaluate(run(restarted=restarted), soak.Gates())) == {"server-pi server restarts"}

    def full(i, sample):
        sample["disk"]["free_percent"] = 4.0 if i > 2000 else 60.0
        sample["disk"]["cache/client"] = 300.0
    results = soak.evaluate(run(full=full), soak.Gates())
    assert failed(results) == {"server-pi free disk", "server-pi disk exhaustion", "server-pi cache/client size"}


def test_stuck_playlist_delivery_triggers_rollback():
    def stuck(i, sample):
        if 10 <= i < 25:
            sample["server"]["playlists"]["office"] = "pending_acknowledgment"
    assert failed(soak.evaluate(run(stuck=stuck), soak.Gates())) == {"office playlist delivery"}


def test_the_gates_command_reads_logs_and_exits_with_the_verdict(tmp_path, capsys):
    good = tmp_path / "good.jsonl"
    good.write_text("\n".join(json.dumps(s) for s in run()) + "\n")
    assert soak._cli(["gates", str(good), "--clients", "office,den"]) == 0
    assert capsys.readouterr().out.startswith("Verdict: GO")

    short = tmp_path / "gates.json"
    short.write_text(json.dumps({"min_hours": 100}))
    assert soak._cli(["gates", str(good), "--gates", str(short), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "NO-GO"

    short.write_text(json.dumps({"min_hourz": 1}))
    assert soak._cli(["gates", str(good), "--gates", str(short)]) == 2
    assert "unknown gate settings: min_hourz" in capsys.readouterr().err

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"v": 9, "t": 1}\n')
    assert soak._cli(["gates", str(bad)]) == 2


def test_the_documented_gates_match_the_defaults():
    doc = (Path(__file__).resolve().parents[1] / "docs" / "soak-and-release.md").read_text(encoding="utf-8")
    for name, value in soak.asdict(soak.Gates()).items():
        assert f"`{name}`" in doc, f"docs/soak-and-release.md does not document {name}"
        if not isinstance(value, dict):
            assert f"| `{name}` | {value:g} |" in doc, f"{name} default is not {value:g} in the doc"
