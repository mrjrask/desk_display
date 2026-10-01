"""Stats sampling: /proc parsing, CPU attribution by purpose, traffic, storage, history."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from remote_display import resource_stats as rs
from remote_display.models import ClientResources, ModelValidationError

ROOT = rs.PROJECT_ROOT


def stat_line(pid, comm, *, ppid=1, utime=0, stime=0, start=1000, rss_pages=100):
    # pid (comm) state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt
    # utime stime cutime cstime priority nice threads itrealvalue starttime vsize rss
    fields = ["S", ppid, 0, 0, 0, 0, 0, 0, 0, 0, 0, utime, stime, 0, 0, 20, 0, 1, 0, start, 0, rss_pages]
    return f"{pid} ({comm}) " + " ".join(str(f) for f in fields) + "\n"


# ── /proc reader ─────────────────────────────────────────────────────────────


def test_proc_reader_parses_a_fake_proc_tree(tmp_path):
    proc = tmp_path / "proc"
    (proc / "42" / "task" / "43").mkdir(parents=True)
    (proc / "42" / "stat").write_text(stat_line(42, "python3 (x) y", ppid=7, utime=250, stime=50, rss_pages=10))
    (proc / "42" / "task" / "43" / "stat").write_text(stat_line(43, "waitress-0", utime=100))
    (proc / "42" / "cmdline").write_bytes(b"/usr/bin/python3\0/opt/dd/display_server.py\0")
    (proc / "stat").write_text("cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 1 2 3 4\n")
    (proc / "meminfo").write_text("MemTotal:  1000 kB\nMemFree: 10 kB\nMemAvailable:  600 kB\n")
    (proc / "loadavg").write_text("0.42 0.30 0.20 1/100 999\n")
    (proc / "net").mkdir()
    (proc / "net" / "dev").write_text(
        "Inter-|   Receive  |  Transmit\n face |bytes packets|bytes\n"
        "    lo: 999 1 0 0 0 0 0 0 999 1 0 0 0 0 0 0\n"
        "  eth0: 1000 5 0 0 0 0 0 0 300 2 0 0 0 0 0 0\n"
        " wlan0: 24 1 0 0 0 0 0 0 6 1 0 0 0 0 0 0\n")
    thermal = tmp_path / "sys" / "class" / "thermal" / "thermal_zone0"
    thermal.mkdir(parents=True)
    (thermal / "temp").write_text("48312\n")
    reader = rs.ProcReader(proc, tmp_path / "sys")
    reader.ticks, reader.page = 100, 4096

    process = reader.process(42)
    assert process.comm == "python3 (x) y"  # a ')' in the name does not shift the fields
    assert (process.ppid, process.cpu_seconds, process.rss_bytes) == (7, 3.0, 40960)
    assert reader.thread(42, 43).comm == "waitress-0"
    assert reader.cmdline(42) == ["/usr/bin/python3", "/opt/dd/display_server.py"]
    assert reader.pids() == [42]
    assert reader.system_cpu() == (150.0, 1000.0)  # idle + iowait are not busy
    assert reader.memory() == (1024000, 614400)
    assert reader.load_1m() == 0.42
    assert reader.network() == {"eth0": (1000, 300), "wlan0": (24, 6)}  # loopback excluded
    assert reader.temperature_c() == 48.3
    assert reader.process(99) is None


def test_missing_proc_files_read_as_nothing(tmp_path):
    reader = rs.ProcReader(tmp_path / "none", tmp_path / "none")
    assert not reader.available()
    assert reader.pids() == [] and reader.network() == {} and reader.system_cpu() is None
    assert reader.memory() == (None, None) and reader.temperature_c() is None


# ── Classification ───────────────────────────────────────────────────────────


def test_project_processes_are_recognised_only_inside_the_project():
    server = rs.classify_process(["python3", str(ROOT / "display_server.py")], None)
    assert server.kind == "server"
    relative = rs.classify_process(["/venv/bin/python", "config_ui.py"], ROOT)
    assert relative.purpose == "config_ui"
    worker = rs.classify_process(["python3", "-m", "rendering.profile_process", "hyperpixel4_square"], ROOT)
    assert (worker.purpose, worker.profile) == ("rendering", "hyperpixel4_square")
    assert "hyperpixel4_square" in worker.label
    # Another project's main.py, or a worker started elsewhere, is not ours.
    assert rs.classify_process(["python3", "/home/pi/other/main.py"], None) is None
    assert rs.classify_process(["python3", "main.py"], Path("/home/pi/other")) is None
    assert rs.classify_process(["python3", "-m", "rendering.profile_process"], Path("/tmp")) is None
    assert rs.classify_process(["bash", "-c", "sleep 1"], ROOT) is None
    assert rs.classify_process([], ROOT) is None


@pytest.mark.parametrize("name, purpose", [
    ("waitress-3", "http"), ("MainThread", "http"), ("render-coordinator", "render_coordination"),
    ("render_1", "render_coordination"), ("server-feeds", "feeds"), ("feed-fetch_0", "feeds"),
    ("artifact-maintenance", "maintenance"), ("stats-sampler", "stats"), ("Thread-7 (kill)", "server_other"),
])
def test_thread_purposes(name, purpose):
    assert rs.thread_purpose(name) == purpose


def test_every_purpose_is_described():
    for key in ("rendering", "feeds", "http", "short_lived", "config_ui", "client"):
        assert rs.PURPOSES[key].label and rs.PURPOSES[key].description


# ── Sampler ──────────────────────────────────────────────────────────────────


class FakeReader:
    """A scriptable stand-in for ProcReader."""

    def __init__(self):
        self.procs = {}       # pid -> [argv, cwd, ProcStat]
        self.threads = {}     # tid -> ProcStat (of the sampling process)
        self.system = (0.0, 0.0)
        self.net = {"eth0": (0, 0)}

    def add(self, pid, argv, cpu, *, cwd=ROOT, start=1, rss=1 << 20):
        self.procs[pid] = [argv, cwd, rs.ProcStat(pid, "python3", 1, cpu, start, rss)]

    def set_cpu(self, pid, cpu):
        argv, cwd, stat = self.procs[pid]
        self.procs[pid][2] = rs.ProcStat(pid, stat.comm, stat.ppid, cpu, stat.start_ticks, stat.rss_bytes)

    def pids(self):
        return sorted(self.procs)

    def process(self, pid):
        return self.procs[pid][2] if pid in self.procs else None

    def thread(self, pid, tid):
        return self.threads.get(tid)

    def cmdline(self, pid):
        return self.procs[pid][0] if pid in self.procs else []

    def cwd(self, pid):
        return self.procs[pid][1] if pid in self.procs else None

    def system_cpu(self):
        return self.system

    def cpu_count(self):
        return 4

    def memory(self):
        return 4 << 30, 3 << 30

    def load_1m(self):
        return 0.5

    def uptime(self):
        return 1000.0

    def network(self):
        return self.net

    def temperature_c(self):
        return 50.0


class Times:
    def __init__(self):
        self.mono = 100.0
        self.wall = 1_800_000_000.0

    def advance(self, seconds):
        self.mono += seconds
        self.wall += seconds


def thread(tid, cpu, start=5):
    return rs.ProcStat(tid, "t", 1, cpu, start, 0)


def build(tmp_path, times, reader, *, traffic=None, persisted=None, clients=None):
    names = {11: "waitress-0", 12: "server-feeds", 13: "render-coordinator"}
    artifacts = tmp_path / "artifacts"
    (artifacts / "objects").mkdir(parents=True, exist_ok=True)
    (artifacts / "objects" / "a.png").write_bytes(b"x" * 300)
    return rs.StatsSampler(
        reader=reader, clock=lambda: times.wall, monotonic=lambda: times.mono, pid=100,
        thread_names=lambda: dict(names), traffic=traffic, clients=clients,
        storage={"Artifact store": (artifacts, 1000), "Missing": (tmp_path / "nope", None)},
        root=ROOT, persisted=persisted,
    )


def test_cpu_is_attributed_to_processes_threads_and_short_lived_work(tmp_path):
    times, reader = Times(), FakeReader()
    reader.add(100, ["python3", str(ROOT / "display_server.py")], 10.0)
    reader.add(200, ["python3", "-m", "rendering.profile_process", "hdmi_1080p"], 50.0)
    reader.add(300, ["python3", "config_ui.py"], 5.0)
    reader.add(400, ["python3", "/elsewhere/main.py"], 99.0, cwd=Path("/elsewhere"))  # not ours
    reader.threads = {11: thread(11, 1.0), 12: thread(12, 4.0), 13: thread(13, 2.0)}
    reader.system = (100.0, 1000.0)
    sampler = build(tmp_path, times, reader)

    first = sampler.sample()
    assert first["cpu"]["project_percent"] is None  # no rates from a single sample
    assert first["history"]["recent"] == []

    times.advance(10)
    reader.set_cpu(100, 13.0)   # +3 s over 10 s = 30% of a core
    reader.set_cpu(200, 55.0)   # +5 s = 50%
    reader.set_cpu(300, 5.5)    # +0.5 s = 5%
    # Live threads account for 2.5 s of the server's 3 s; 0.5 s ran in threads that exited.
    reader.threads = {11: thread(11, 1.5), 12: thread(12, 5.5), 13: thread(13, 2.5)}
    reader.system = (300.0, 2000.0)  # 200 busy of 1000 → 20%
    document = sampler.sample()

    by_purpose = document["cpu"]["by_purpose"]
    assert by_purpose == {"config_ui": 5.0, "feeds": 15.0, "http": 5.0, "render_coordination": 5.0,
                          "rendering": 50.0, "short_lived": 5.0}
    assert document["cpu"]["project_percent"] == 85.0
    assert document["system"]["cpu_percent"] == 20.0
    labels = [p["label"] for p in document["cpu"]["processes"]]
    assert labels == ["Render worker · hdmi_1080p", "Render server", "Config web UI"]
    assert document["cpu"]["threads"][0] == {"name": "server-feeds", "purpose": "feeds", "cpu_percent": 15.0}
    assert len(document["history"]["recent"]) == 1


def test_a_replaced_worker_starts_a_new_rate(tmp_path):
    times, reader = Times(), FakeReader()
    reader.add(200, ["python3", "-m", "rendering.profile_process", "p"], 50.0, start=1)
    sampler = build(tmp_path, times, reader)
    sampler.own_threads = False
    sampler.sample()
    times.advance(10)
    reader.add(200, ["python3", "-m", "rendering.profile_process", "p"], 1.0, start=2)  # pid reused
    document = sampler.sample()
    assert document["cpu"]["processes"][0]["cpu_percent"] is None  # never a negative or bogus rate


def test_network_traffic_storage_and_clients(tmp_path):
    times, reader = Times(), FakeReader()
    reader.add(100, ["python3", str(ROOT / "display_server.py")], 1.0)
    traffic = rs.TrafficCounter()
    clients = {"office": {"resources": {"process_cpu_percent": 4.0, "system_cpu_percent": 12.0}}}
    sampler = build(tmp_path, times, reader, traffic=traffic, clients=lambda: clients,
                    persisted={"since": 1_700_000_000.0,
                               "traffic": {"office": {"bytes_in": 5, "bytes_out": 1000, "requests": 2,
                                                      "kinds": {"artifact": {"bytes_out": 1000, "requests": 2}}}}})
    traffic.record("office", "artifact", 0, 2000)
    sampler.sample()
    times.advance(10)
    reader.net = {"eth0": (10_000, 4_000)}
    traffic.record("office", "artifact", 10, 5000)
    traffic.record(None, "register", 300, 100)
    document = sampler.sample()

    assert document["network"]["rx_bps"] == 1000.0 and document["network"]["tx_bps"] == 400.0
    assert document["traffic"]["rates"]["office"] == {"out_bps": 500.0, "in_bps": 1.0}
    assert document["traffic"]["session"]["office"]["bytes_out"] == 7000
    totals = document["traffic"]["totals"]
    assert totals["office"]["bytes_out"] == 8000 and totals["office"]["requests"] == 4
    assert totals["office"]["kinds"]["artifact"] == {"bytes_out": 8000, "requests": 4}
    assert totals["_unauthenticated"]["kinds"]["register"]["requests"] == 1
    assert document["totals_since"] == 1_700_000_000.0
    point = document["history"]["recent"][-1]
    assert point["clients"]["office"] == {"out_bps": 500.0, "in_bps": 1.0, "process_cpu_percent": 4.0,
                                          "system_cpu_percent": 12.0}
    store = document["storage"]["directories"]["Artifact store"]
    assert (store["bytes"], store["files"], store["limit_bytes"]) == (300, 1, 1000)
    assert document["storage"]["directories"]["Missing"]["bytes"] is None
    assert document["storage"]["disk"]["total_bytes"] > 0
    json.dumps(document)  # the published document is plain JSON


def test_history_averages_into_five_minute_buckets_and_persists(tmp_path):
    times, reader = Times(), FakeReader()
    times.wall = 1_800_000_000.0 - (1_800_000_000.0 % 300)
    reader.add(100, ["python3", str(ROOT / "display_server.py")], 0.0)
    sampler = build(tmp_path, times, reader)
    sampler.own_threads = False
    cpu = 0.0
    sampler.sample()
    for step in range(31):  # 10 s samples across a 5-minute boundary
        times.advance(10)
        cpu += 1.0 if step < 15 else 3.0  # 10% then 30% of a core
        reader.set_cpu(100, cpu)
        sampler.sample()
    assert len(sampler.long) == 1
    bucket = sampler.long[0]
    assert bucket["t"] == times.wall - (times.wall % 300) - 300
    assert bucket["cpu"]["server_other"] == pytest.approx((15 * 10 + 14 * 30) / 29, abs=0.01)

    state = sampler.persisted_state()
    restored = build(tmp_path, times, reader, persisted=json.loads(json.dumps(state)))
    assert list(restored.long) == list(sampler.long)


def test_publisher_writes_live_and_persisted_files(tmp_path):
    times, reader = Times(), FakeReader()
    reader.add(100, ["python3", str(ROOT / "display_server.py")], 0.0)
    traffic = rs.TrafficCounter()
    sampler = build(tmp_path, times, reader, traffic=traffic)
    live, persisted = tmp_path / "shm" / "stats.json", tmp_path / "runtime" / "history.json"
    publisher = rs.StatsPublisher(sampler, live_path=live, persist_path=persisted)
    publisher.step()
    assert rs.read_stats(live)["role"] == "server"
    traffic.record("office", "artifact", 0, 42)
    publisher.stop()
    saved = rs.load_persisted(persisted)
    assert saved["traffic"]["office"]["bytes_out"] == 42
    assert rs.read_stats(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{not json")
    assert rs.read_stats(tmp_path / "bad.json") is None
    assert rs.load_persisted(tmp_path / "bad.json") == {}


def test_stats_paths_and_switch():
    assert rs.stats_path({"DESK_DISPLAY_SERVER_STATS_PATH": "/x/live.json"}) == Path("/x/live.json")
    assert rs.history_path({"DESK_DISPLAY_SERVER_STATS_HISTORY_PATH": "/x/h.json"}) == Path("/x/h.json")
    default = rs.stats_path({})
    assert default.name in {"server_stats.json", "stats.json"}
    assert rs.stats_enabled({}) and not rs.stats_enabled({"DESK_DISPLAY_STATS_ENABLED": "0"})


def test_directory_size_counts_nested_files_without_following_links(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "one").write_bytes(b"1" * 10)
    (tmp_path / "a" / "b" / "two").write_bytes(b"2" * 5)
    (tmp_path / "link").symlink_to(tmp_path / "a")
    assert rs.directory_size(tmp_path) == {"bytes": 15, "files": 2}
    assert rs.directory_size(tmp_path / "absent") is None
    assert rs.disk_usage(tmp_path / "not" / "yet")["free_bytes"] > 0


# ── Client side ──────────────────────────────────────────────────────────────


def test_client_resources_validate_and_round_trip(tmp_path):
    times, reader = Times(), FakeReader()
    reader.add(100, ["python3", "display_client.py"], 2.0)
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "x.png").write_bytes(b"x" * 64)
    sampler = rs.ClientResourceSampler(cache_root=tmp_path, cache_limit_bytes=256 << 20, reader=reader,
                                       monotonic=lambda: times.mono, pid=100)
    sampler.sample()
    times.advance(10)
    reader.set_cpu(100, 3.0)
    reader.system = (100.0, 1000.0)
    values = sampler.sample(bytes_received=5000, bytes_sent=700)
    assert values["process_cpu_percent"] == 10.0
    assert values["cache_bytes"] == 64 and values["cache_limit_bytes"] == 256 << 20
    assert (values["bytes_received"], values["bytes_sent"]) == (5000, 700)
    wire = ClientResources(**values).to_wire()
    assert ClientResources.from_wire(json.loads(json.dumps(wire))).to_wire() == wire
    # Every field is optional: an empty report is valid.
    assert ClientResources.from_wire({"type": "client_resources", "version": 1}).cache_bytes is None
    with pytest.raises(ModelValidationError):
        ClientResources(cache_bytes=-1)
    with pytest.raises(ModelValidationError):
        ClientResources(process_cpu_percent=float("nan"))
