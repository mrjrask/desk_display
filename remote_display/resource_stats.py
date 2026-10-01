"""Resource statistics for the Stats page: CPU by purpose, data, storage.

Everything is read from ``/proc`` and ``/sys`` with the standard library, so
sampling costs a few small file reads and needs no new dependency.  Off Linux
the readers return nothing and the page shows what remains (disk usage).

The render server runs a :class:`StatsSampler` thread.  Each sample records:

* **CPU by purpose.**  Every project process on the machine is found by its
  command line (render server, per-profile render workers, config UI, local
  panel client, standalone display, feed server).  Inside the render server
  the sampler also reads each thread's CPU time and groups the threads by
  name: HTTP serving, render scheduling, feed refresh, artifact cleanup.
  Threads that start and exit between two samples (feed download pools) are
  counted as the process total minus its live threads.  Percentages are of
  one CPU core, like ``top``; the machine total is out of ``100 × cores``.
* **Data transferred.**  The server counts request and response bytes per
  display client and per endpoint kind (:class:`TrafficCounter`), and the
  machine's network interfaces are read from ``/proc/net/dev``.
* **Storage.**  Free space on the project's filesystem, and the sizes of the
  artifact store and caches against their limits.

The live document is written to :func:`stats_path` (tmpfs when available, so
frequent writes never touch the SD card) for the config UI to read.  Running
totals and a 24-hour history are written to :func:`history_path` every few
minutes, so they survive restarts.

Clients report their own figures in a heartbeat ``resources`` document
(:class:`remote_display.models.ClientResources`), built by
:class:`ClientResourceSampler`.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("desk_display.resource_stats")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_INTERVAL_SECONDS = 10.0
# The live file is rewritten every few samples; it lives on tmpfs when it can.
PUBLISH_EVERY_SAMPLES = 3
# Directory walks (artifact store, caches) are slower; refresh them rarely.
STORAGE_INTERVAL_SECONDS = 300.0
RECENT_POINTS = 360          # 1 hour at 10 s
LONG_BUCKET_SECONDS = 300.0  # 5-minute averages ...
LONG_POINTS = 288            # ... for 24 hours
PERSIST_INTERVAL_SECONDS = 600.0
STALE_AFTER_SECONDS = 120.0
SCHEMA_VERSION = 1

_TRUTHY_OFF = {"0", "false", "no", "off"}


# ── Paths ────────────────────────────────────────────────────────────────────


def _project_tag(root: Path = PROJECT_ROOT) -> str:
    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:10]


def stats_path(env: Mapping[str, str] | None = None) -> Path:
    """Where the render server publishes its live stats for the config UI.

    ``DESK_DISPLAY_SERVER_STATS_PATH`` wins; otherwise a per-user, per-checkout
    file on ``/dev/shm`` (memory, not the SD card), or ``.runtime/server``.
    """

    source = os.environ if env is None else env
    raw = (source.get("DESK_DISPLAY_SERVER_STATS_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser()
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK) and hasattr(os, "getuid"):
        return shm / f"desk_display-{os.getuid()}-{_project_tag()}" / "server_stats.json"
    return PROJECT_ROOT / ".runtime" / "server" / "stats.json"


def history_path(env: Mapping[str, str] | None = None) -> Path:
    """Persisted running totals and the 24-hour history (written every 10 minutes)."""

    source = os.environ if env is None else env
    raw = (source.get("DESK_DISPLAY_SERVER_STATS_HISTORY_PATH") or "").strip()
    return Path(raw).expanduser() if raw else PROJECT_ROOT / ".runtime" / "server" / "stats_history.json"


def stats_enabled(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return (source.get("DESK_DISPLAY_STATS_ENABLED") or "1").strip().lower() not in _TRUTHY_OFF


def write_json_atomic(path: Path, document: Mapping[str, Any], *, durable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, separators=(",", ":"))
            if durable:
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def read_stats(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The published stats document, or ``None`` when missing or unreadable."""

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        return None
    return data


# ── /proc readers ────────────────────────────────────────────────────────────


def _clock_ticks() -> int:
    try:
        return int(os.sysconf("SC_CLK_TCK"))
    except (AttributeError, ValueError, OSError):  # pragma: no cover - non-POSIX
        return 100


def _page_size() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):  # pragma: no cover - non-POSIX
        return 4096


@dataclass(frozen=True)
class ProcStat:
    """The parts of ``/proc/<pid>/stat`` the sampler uses."""

    pid: int
    comm: str
    ppid: int
    cpu_seconds: float
    start_ticks: int
    rss_bytes: int


class ProcReader:
    """Reads process, CPU, memory, network and thermal figures.

    *proc_root* and *sys_root* exist so tests can point at a fake tree.
    """

    def __init__(self, proc_root: str | os.PathLike[str] = "/proc",
                 sys_root: str | os.PathLike[str] = "/sys") -> None:
        self.proc = Path(proc_root)
        self.sys = Path(sys_root)
        self.ticks = _clock_ticks()
        self.page = _page_size()

    def _read(self, path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def available(self) -> bool:
        return (self.proc / "stat").is_file()

    def pids(self) -> list[int]:
        try:
            return sorted(int(entry.name) for entry in os.scandir(self.proc) if entry.name.isdigit())
        except OSError:
            return []

    def parse_stat(self, pid: int, text: str | None) -> ProcStat | None:
        if not text:
            return None
        open_paren, close_paren = text.find("("), text.rfind(")")
        if open_paren < 0 or close_paren < 0:
            return None
        rest = text[close_paren + 2:].split()
        try:
            # Fields after "pid (comm) ": state ppid ... utime(14) stime(15)
            # ... starttime(22) vsize(23) rss(24), counted from 1 overall.
            utime, stime = int(rest[11]), int(rest[12])
            return ProcStat(
                pid=pid,
                comm=text[open_paren + 1:close_paren],
                ppid=int(rest[1]),
                cpu_seconds=(utime + stime) / self.ticks,
                start_ticks=int(rest[19]),
                rss_bytes=int(rest[21]) * self.page,
            )
        except (IndexError, ValueError):
            return None

    def process(self, pid: int) -> ProcStat | None:
        return self.parse_stat(pid, self._read(self.proc / str(pid) / "stat"))

    def thread(self, pid: int, tid: int) -> ProcStat | None:
        return self.parse_stat(tid, self._read(self.proc / str(pid) / "task" / str(tid) / "stat"))

    def cmdline(self, pid: int) -> list[str]:
        try:
            raw = (self.proc / str(pid) / "cmdline").read_bytes()
        except OSError:
            return []
        return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]

    def cwd(self, pid: int) -> Path | None:
        try:
            return Path(os.readlink(self.proc / str(pid) / "cwd"))
        except OSError:
            return None

    def system_cpu(self) -> tuple[float, float] | None:
        """``(busy, total)`` jiffies across every core since boot."""

        text = self._read(self.proc / "stat")
        if not text:
            return None
        for line in text.splitlines():
            if line.startswith("cpu "):
                values = [int(v) for v in line.split()[1:] if v.isdigit()]
                if len(values) < 4:
                    return None
                idle = values[3] + (values[4] if len(values) > 4 else 0)
                # guest time is already inside user/nice.
                total = sum(values[:8])
                return float(total - idle), float(total)
        return None

    def cpu_count(self) -> int:
        return os.cpu_count() or 1

    def memory(self) -> tuple[int | None, int | None]:
        text = self._read(self.proc / "meminfo") or ""
        values: dict[str, int] = {}
        for line in text.splitlines():
            name, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[name] = int(parts[0]) * 1024
        return values.get("MemTotal"), values.get("MemAvailable")

    def load_1m(self) -> float | None:
        text = self._read(self.proc / "loadavg")
        try:
            return float(text.split()[0]) if text else None
        except (IndexError, ValueError):
            return None

    def uptime(self) -> float | None:
        text = self._read(self.proc / "uptime")
        try:
            return float(text.split()[0]) if text else None
        except (IndexError, ValueError):
            return None

    def network(self) -> dict[str, tuple[int, int]]:
        """``{interface: (rx_bytes, tx_bytes)}`` for every interface but loopback."""

        text = self._read(self.proc / "net" / "dev") or ""
        result: dict[str, tuple[int, int]] = {}
        for line in text.splitlines()[2:]:
            name, _, rest = line.partition(":")
            name = name.strip()
            fields = rest.split()
            if not name or name == "lo" or len(fields) < 9:
                continue
            with contextlib.suppress(ValueError):
                result[name] = (int(fields[0]), int(fields[8]))
        return result

    def temperature_c(self) -> float | None:
        text = self._read(self.sys / "class" / "thermal" / "thermal_zone0" / "temp")
        try:
            return round(int(text.strip()) / 1000.0, 1) if text else None
        except ValueError:
            return None


def disk_usage(path: str | os.PathLike[str]) -> dict[str, int] | None:
    """Total / used / free bytes of the filesystem holding *path*."""

    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        usage = shutil.disk_usage(target)
    except OSError:
        return None
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def directory_size(path: str | os.PathLike[str], *, max_entries: int = 200_000) -> dict[str, int] | None:
    """``{"bytes", "files"}`` under *path* (no symlinks followed), or ``None`` if absent."""

    root = Path(path)
    if not root.is_dir():
        return None
    total = files = seen = 0
    stack = [root]
    while stack and seen < max_entries:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    seen += 1
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                            files += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return {"bytes": total, "files": files}


# ── Classification ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Purpose:
    key: str
    label: str
    description: str


PURPOSES: dict[str, Purpose] = {p.key: p for p in (
    Purpose("rendering", "Rendering screens",
            "Render worker processes drawing each screen for each display profile."),
    Purpose("render_coordination", "Render scheduling",
            "Deciding what to render next, publishing artifacts and render packages."),
    Purpose("feeds", "Feed refresh",
            "Fetching and parsing sports, weather, news and other data feeds."),
    Purpose("http", "Serving displays",
            "The display API: heartbeats, manifests and artifact downloads."),
    Purpose("maintenance", "Artifact cleanup", "Deleting unreferenced artifacts."),
    Purpose("short_lived", "Short-lived threads",
            "Threads that started and finished between samples, mostly feed download pools."),
    Purpose("stats", "Stats sampling", "This sampler."),
    Purpose("server_other", "Other server work", "Any other render-server thread."),
    Purpose("config_ui", "Config web UI", "These pages."),
    Purpose("client", "Local panel client", "The display client driving this Pi's own panel."),
    Purpose("standalone", "Standalone display", "main.py, the all-in-one display."),
    Purpose("feed_server", "Screenshot feed server", "feed_server.py."),
)}

# (script basename, process kind, purpose)
_SCRIPTS = {
    "display_server.py": ("server", "Render server", "server_other"),
    "display_client.py": ("client", "Display client", "client"),
    "config_ui.py": ("config_ui", "Config web UI", "config_ui"),
    "main.py": ("standalone", "Standalone display", "standalone"),
    "feed_server.py": ("feed_server", "Screenshot feed server", "feed_server"),
}
RENDER_WORKER_MODULE = "rendering.profile_process"

# Thread name prefixes inside the render server, first match wins.
_THREAD_PURPOSES = (
    ("waitress", "http"),
    ("MainThread", "http"),
    ("render", "render_coordination"),
    ("server-feeds", "feeds"),
    ("feed-fetch", "feeds"),
    ("stock-quote", "feeds"),
    ("artifact-maintenance", "maintenance"),
    ("stats-sampler", "stats"),
)


def thread_purpose(name: str) -> str:
    for prefix, purpose in _THREAD_PURPOSES:
        if name.startswith(prefix):
            return purpose
    return "server_other"


@dataclass(frozen=True)
class ProjectProcess:
    kind: str
    label: str
    purpose: str
    profile: str | None = None


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def classify_process(argv: list[str], cwd: Path | None, root: Path = PROJECT_ROOT) -> ProjectProcess | None:
    """Which project process *argv* is, or ``None`` for anything else.

    A script must live in *root* (an absolute path there, or a relative path
    from a working directory there), so another project's ``main.py`` is not
    mistaken for the standalone display.
    """

    if not argv:
        return None
    for index, arg in enumerate(argv[1:4], start=1):
        if arg == "-m" and index + 1 < len(argv) and argv[index + 1] == RENDER_WORKER_MODULE:
            if cwd is None or not _within(cwd, root):
                return None
            profile = argv[index + 2] if index + 2 < len(argv) else None
            return ProjectProcess("render_worker", f"Render worker · {profile or 'unknown'}",
                                  "rendering", profile)
        name = os.path.basename(arg)
        if name in _SCRIPTS:
            path = Path(arg)
            located = path if path.is_absolute() else (cwd / path if cwd is not None else None)
            if located is None or not _within(located, root):
                return None
            kind, label, purpose = _SCRIPTS[name]
            return ProjectProcess(kind, label, purpose)
        if not arg.startswith("-"):
            break
    return None


# ── Server traffic ───────────────────────────────────────────────────────────


TRAFFIC_KINDS = ("artifact", "manifest", "config", "heartbeat", "register", "other")


def traffic_kind(endpoint: str | None) -> str:
    return {
        "client_artifact": "artifact",
        "client_manifest": "manifest",
        "client_config": "config",
        "heartbeat": "heartbeat",
        "register": "register",
        "join": "register",
    }.get(endpoint or "", "other")


class TrafficCounter:
    """Bytes the display API received and sent, per client and kind. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: dict[str, dict[str, Any]] = {}

    def record(self, client_id: str | None, kind: str, bytes_in: int, bytes_out: int) -> None:
        key = client_id or "_unauthenticated"
        with self._lock:
            entry = self._clients.setdefault(key, {"bytes_in": 0, "bytes_out": 0, "requests": 0, "kinds": {}})
            entry["bytes_in"] += max(0, int(bytes_in))
            entry["bytes_out"] += max(0, int(bytes_out))
            entry["requests"] += 1
            kinds = entry["kinds"].setdefault(kind, {"bytes_out": 0, "requests": 0})
            kinds["bytes_out"] += max(0, int(bytes_out))
            kinds["requests"] += 1

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return json.loads(json.dumps(self._clients))


def _add_traffic(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base or {}))
    for client_id, values in (extra or {}).items():
        entry = result.setdefault(client_id, {"bytes_in": 0, "bytes_out": 0, "requests": 0, "kinds": {}})
        for name in ("bytes_in", "bytes_out", "requests"):
            entry[name] = int(entry.get(name) or 0) + int(values.get(name) or 0)
        for kind, numbers in (values.get("kinds") or {}).items():
            target = entry.setdefault("kinds", {}).setdefault(kind, {"bytes_out": 0, "requests": 0})
            for name in ("bytes_out", "requests"):
                target[name] = int(target.get(name) or 0) + int(numbers.get(name) or 0)
    return result


# ── Sampler ──────────────────────────────────────────────────────────────────


def _round(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(value, digits)


class _Rates:
    """Turns cumulative counters into per-second rates between samples."""

    def __init__(self) -> None:
        self._last: dict[Any, tuple[float, float]] = {}

    def rate(self, key: Any, value: float, now: float) -> float | None:
        previous = self._last.get(key)
        self._last[key] = (value, now)
        if previous is None:
            return None
        elapsed = now - previous[1]
        if elapsed <= 0:
            return None
        return max(0.0, value - previous[0]) / elapsed

    def forget_except(self, keys: Iterable[Any]) -> None:
        keep = set(keys)
        for key in list(self._last):
            if key not in keep:
                del self._last[key]


class StatsSampler:
    """Samples the machine and builds the Stats page document.

    *own_threads* breaks this process's CPU down by thread (the render
    server).  *traffic* and *clients* are the server's per-client counters
    and a callable returning registry records (with ``resources``).
    *storage* maps labels to ``(path, limit_bytes or None)`` directories whose
    sizes are shown against their limits.
    """

    def __init__(
        self,
        *,
        role: str = "server",
        own_threads: bool = True,
        traffic: TrafficCounter | None = None,
        clients: Callable[[], Mapping[str, Mapping[str, Any]]] | None = None,
        storage: Mapping[str, tuple[str | os.PathLike[str], int | None]] | None = None,
        reader: ProcReader | None = None,
        root: Path = PROJECT_ROOT,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        thread_names: Callable[[], Mapping[int, str]] | None = None,
        pid: int | None = None,
        persisted: Mapping[str, Any] | None = None,
    ) -> None:
        self.role = role
        self.own_threads = own_threads
        self.traffic = traffic
        self.clients = clients
        self.storage = dict(storage or {})
        self.reader = reader or ProcReader()
        self.root = root
        self.clock = clock
        self.monotonic = monotonic
        self.thread_names = thread_names or _python_thread_names
        self.pid = os.getpid() if pid is None else pid
        self.started_at = clock()
        self._cpu = _Rates()
        self._threads = _Rates()
        self._net = _Rates()
        self._traffic_rates = _Rates()
        self._system: tuple[float, float] | None = None
        self._lock = threading.Lock()
        self.recent: deque[dict[str, Any]] = deque(maxlen=RECENT_POINTS)
        self.long: deque[dict[str, Any]] = deque(maxlen=LONG_POINTS)
        self._bucket: list[dict[str, Any]] = []
        self._bucket_start: float | None = None
        self._storage_cache: dict[str, Any] = {}
        self._storage_at: float | None = None
        self._latest: dict[str, Any] | None = None
        persisted = persisted or {}
        self._traffic_base: dict[str, Any] = dict(persisted.get("traffic") or {})
        self._totals_since: str | None = persisted.get("since")
        for point in persisted.get("long") or []:
            if isinstance(point, dict):
                self.long.append(point)

    # Processes and threads

    def _project_processes(self) -> list[tuple[ProcStat, ProjectProcess]]:
        found = []
        for pid in self.reader.pids():
            info = None
            if pid == self.pid:
                stat = self.reader.process(pid)
                info = ProjectProcess("server", "Render server", "server_other") if self.role == "server" \
                    else classify_process(self.reader.cmdline(pid), self.reader.cwd(pid), self.root)
                if info is None:
                    info = ProjectProcess("self", "This process", "config_ui")
            else:
                argv = self.reader.cmdline(pid)
                if not argv:
                    continue
                info = classify_process(argv, self.reader.cwd(pid), self.root)
                if info is None:
                    continue
                stat = self.reader.process(pid)
            if stat is not None:
                found.append((stat, info))
        return found

    def _own_thread_breakdown(self, now: float, process_rate: float | None) -> tuple[dict[str, float], list[dict[str, Any]]]:
        by_purpose: dict[str, float] = {}
        rows: list[dict[str, Any]] = []
        names = self.thread_names()
        live_total = 0.0
        keys = []
        for tid, name in names.items():
            stat = self.reader.thread(self.pid, tid)
            if stat is None:
                continue
            key = (tid, stat.start_ticks)
            keys.append(key)
            rate = self._threads.rate(key, stat.cpu_seconds, now)
            if rate is None:
                continue
            percent = rate * 100.0
            live_total += percent
            purpose = thread_purpose(name)
            by_purpose[purpose] = by_purpose.get(purpose, 0.0) + percent
            rows.append({"name": name, "purpose": purpose, "cpu_percent": round(percent, 2)})
        self._threads.forget_except(keys)
        if process_rate is not None and rows:
            remainder = process_rate * 100.0 - live_total
            if remainder > 0.05:
                by_purpose["short_lived"] = by_purpose.get("short_lived", 0.0) + remainder
        rows.sort(key=lambda row: -row["cpu_percent"])
        return by_purpose, rows

    # Storage

    def _storage(self, now: float) -> dict[str, Any]:
        result: dict[str, Any] = {"disk": disk_usage(self.root)}
        if self._storage_at is None or now - self._storage_at >= STORAGE_INTERVAL_SECONDS:
            sizes = {}
            for label, (path, limit) in self.storage.items():
                size = directory_size(path)
                sizes[label] = {
                    "path": str(path),
                    "bytes": None if size is None else size["bytes"],
                    "files": None if size is None else size["files"],
                    "limit_bytes": limit,
                }
            self._storage_cache, self._storage_at = sizes, now
        result["directories"] = self._storage_cache
        return result

    # Sampling

    def sample(self) -> dict[str, Any]:
        """Take one sample; return the full page document."""

        with self._lock:
            return self._sample()

    def _sample(self) -> dict[str, Any]:
        now = self.monotonic()
        wall = self.clock()
        reader = self.reader
        cores = reader.cpu_count()

        system = reader.system_cpu()
        system_percent = None
        if system is not None and self._system is not None:
            busy = system[0] - self._system[0]
            total = system[1] - self._system[1]
            if total > 0:
                system_percent = max(0.0, min(100.0, busy / total * 100.0))
        if system is not None:
            self._system = system

        processes: list[dict[str, Any]] = []
        by_purpose: dict[str, float] = {}
        threads: list[dict[str, Any]] = []
        keys = []
        for stat, info in self._project_processes():
            key = (stat.pid, stat.start_ticks)
            keys.append(key)
            rate = self._cpu.rate(key, stat.cpu_seconds, now)
            percent = None if rate is None else rate * 100.0
            if stat.pid == self.pid and self.own_threads:
                breakdown, threads = self._own_thread_breakdown(now, rate)
                if breakdown:
                    for purpose, value in breakdown.items():
                        by_purpose[purpose] = by_purpose.get(purpose, 0.0) + value
                elif percent is not None:
                    by_purpose[info.purpose] = by_purpose.get(info.purpose, 0.0) + percent
            elif percent is not None:
                by_purpose[info.purpose] = by_purpose.get(info.purpose, 0.0) + percent
            processes.append({
                "pid": stat.pid,
                "kind": info.kind,
                "label": info.label,
                "purpose": info.purpose,
                "profile": info.profile,
                "cpu_percent": _round(percent, 2),
                "cpu_seconds": round(stat.cpu_seconds, 1),
                "rss_bytes": stat.rss_bytes,
                "is_self": stat.pid == self.pid,
            })
        self._cpu.forget_except(keys)
        processes.sort(key=lambda row: (-(row["cpu_percent"] or 0), row["label"]))
        project_percent = sum(p["cpu_percent"] or 0 for p in processes) if any(
            p["cpu_percent"] is not None for p in processes) else None

        interfaces = reader.network()
        rx = sum(v[0] for v in interfaces.values())
        tx = sum(v[1] for v in interfaces.values())
        rx_rate = self._net.rate("rx", rx, now) if interfaces else None
        tx_rate = self._net.rate("tx", tx, now) if interfaces else None

        traffic_now = self.traffic.snapshot() if self.traffic is not None else {}
        client_rates: dict[str, dict[str, Any]] = {}
        served = received = 0.0
        have_rates = False
        for client_id, values in traffic_now.items():
            out_rate = self._traffic_rates.rate(("out", client_id), values["bytes_out"], now)
            in_rate = self._traffic_rates.rate(("in", client_id), values["bytes_in"], now)
            if out_rate is not None:
                have_rates = True
                served += out_rate
                received += in_rate or 0.0
                client_rates[client_id] = {"out_bps": round(out_rate, 1), "in_bps": round(in_rate or 0.0, 1)}

        clients = dict(self.clients()) if self.clients is not None else {}
        client_points = {}
        for client_id, record in clients.items():
            resources = record.get("resources") if isinstance(record, Mapping) else None
            point = dict(client_rates.get(client_id) or {})
            if isinstance(resources, Mapping):
                for name in ("process_cpu_percent", "system_cpu_percent"):
                    if resources.get(name) is not None:
                        point[name] = resources[name]
            if point:
                client_points[client_id] = point
        for client_id, point in client_rates.items():
            client_points.setdefault(client_id, point)

        point = {
            "t": round(wall, 1),
            "cpu": {key: round(value, 2) for key, value in sorted(by_purpose.items())},
            "project_cpu": _round(project_percent, 2),
            "system_cpu": _round(system_percent, 1),
            "net_rx_bps": _round(rx_rate),
            "net_tx_bps": _round(tx_rate),
            "served_bps": round(served, 1) if have_rates else None,
            "received_bps": round(received, 1) if have_rates else None,
            "clients": client_points,
        }
        if project_percent is not None or system_percent is not None:
            self.recent.append(point)
            self._add_long(point, wall)

        memory_total, memory_available = reader.memory()
        traffic_totals = _add_traffic(self._traffic_base, traffic_now)
        document = {
            "schema_version": SCHEMA_VERSION,
            "role": self.role,
            "generated_at": wall,
            "started_at": self.started_at,
            "totals_since": self._totals_since or self.started_at,
            "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "purposes": {key: {"label": p.label, "description": p.description} for key, p in PURPOSES.items()},
            "system": {
                "cpu_count": cores,
                "cpu_percent": _round(system_percent, 1),
                "load_1m": reader.load_1m(),
                "memory_total_bytes": memory_total,
                "memory_available_bytes": memory_available,
                "temperature_c": reader.temperature_c(),
                "uptime_seconds": reader.uptime(),
            },
            "cpu": {
                "project_percent": _round(project_percent, 2),
                "by_purpose": point["cpu"],
                "processes": processes,
                "threads": threads,
            },
            "network": {
                "interfaces": {name: {"rx_bytes": v[0], "tx_bytes": v[1]} for name, v in sorted(interfaces.items())},
                "rx_bps": point["net_rx_bps"],
                "tx_bps": point["net_tx_bps"],
            },
            "traffic": {
                "session": traffic_now,
                "totals": traffic_totals,
                "rates": client_rates,
                "served_bps": point["served_bps"],
                "received_bps": point["received_bps"],
            },
            "storage": self._storage(now),
            "history": {"recent": list(self.recent), "long": list(self.long)},
        }
        self._latest = document
        return document

    def _add_long(self, point: Mapping[str, Any], wall: float) -> None:
        bucket = wall - (wall % LONG_BUCKET_SECONDS)
        if self._bucket_start is not None and bucket != self._bucket_start and self._bucket:
            self.long.append(average_points(self._bucket, self._bucket_start))
            self._bucket = []
        self._bucket_start = bucket
        self._bucket.append(dict(point))

    def persisted_state(self) -> dict[str, Any]:
        with self._lock:
            traffic_now = self.traffic.snapshot() if self.traffic is not None else {}
            return {
                "schema_version": SCHEMA_VERSION,
                "since": self._totals_since or self.started_at,
                "traffic": _add_traffic(self._traffic_base, traffic_now),
                "long": list(self.long),
            }


def average_points(points: list[Mapping[str, Any]], start: float) -> dict[str, Any]:
    """One history point averaging *points* (a 5-minute bucket)."""

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 2) if values else None

    def numbers(name: str) -> list[float]:
        return [float(p[name]) for p in points if isinstance(p.get(name), int | float)]

    purposes = sorted({key for p in points for key in (p.get("cpu") or {})})
    count = len(points)
    clients: dict[str, dict[str, Any]] = {}
    for client_id in sorted({cid for p in points for cid in (p.get("clients") or {})}):
        fields = sorted({name for p in points for name in ((p.get("clients") or {}).get(client_id) or {})})
        clients[client_id] = {}
        for name in fields:
            values = [float(v) for p in points
                      if isinstance(v := ((p.get("clients") or {}).get(client_id) or {}).get(name), int | float)]
            clients[client_id][name] = mean(values)
    return {
        "t": round(start, 1),
        # Purposes absent from a sample used no CPU then.
        "cpu": {key: round(sum(float((p.get("cpu") or {}).get(key) or 0) for p in points) / count, 2)
                for key in purposes},
        "project_cpu": mean(numbers("project_cpu")),
        "system_cpu": mean(numbers("system_cpu")),
        "net_rx_bps": mean(numbers("net_rx_bps")),
        "net_tx_bps": mean(numbers("net_tx_bps")),
        "served_bps": mean(numbers("served_bps")),
        "received_bps": mean(numbers("received_bps")),
        "clients": clients,
    }


def _python_thread_names() -> dict[int, str]:
    return {t.native_id: t.name for t in threading.enumerate() if t.native_id is not None}


def load_persisted(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        return {}
    return data


class StatsPublisher:
    """Runs a :class:`StatsSampler` in a thread and publishes its documents."""

    def __init__(self, sampler: StatsSampler, *, live_path: Path, persist_path: Path | None,
                 interval: float = SAMPLE_INTERVAL_SECONDS) -> None:
        self.sampler = sampler
        self.live_path = live_path
        self.persist_path = persist_path
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples = 0
        self._persisted_at = time.monotonic()

    def step(self) -> dict[str, Any]:
        document = self.sampler.sample()
        self._samples += 1
        # The first two samples (the first has no rates yet) go out at once.
        if self._samples <= 2 or self._samples % PUBLISH_EVERY_SAMPLES == 0:
            try:
                write_json_atomic(self.live_path, document)
            except OSError as exc:
                LOGGER.warning("Could not write stats to %s: %s", self.live_path, exc)
        if time.monotonic() - self._persisted_at >= PERSIST_INTERVAL_SECONDS:
            self.persist()
        return document

    def persist(self) -> None:
        self._persisted_at = time.monotonic()
        if self.persist_path is None:
            return
        try:
            write_json_atomic(self.persist_path, self.sampler.persisted_state(), durable=True)
        except OSError as exc:
            LOGGER.warning("Could not save stats history to %s: %s", self.persist_path, exc)

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self.step()
            except Exception:  # pragma: no cover - logged, sampling continues
                LOGGER.exception("Stats sampling failed")
            self._stop.wait(self.interval)

    def start(self) -> threading.Thread:
        self._thread = threading.Thread(target=self.run, name="stats-sampler", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop.set()
        self.persist()


# ── Client side ──────────────────────────────────────────────────────────────


class ClientResourceSampler:
    """This client's figures for the heartbeat ``resources`` document."""

    def __init__(self, *, cache_root: str | os.PathLike[str] | None = None, cache_limit_bytes: int | None = None,
                 reader: ProcReader | None = None, monotonic: Callable[[], float] = time.monotonic,
                 pid: int | None = None) -> None:
        self.cache_root = None if cache_root is None else Path(cache_root)
        self.cache_limit_bytes = cache_limit_bytes
        self.reader = reader or ProcReader()
        self.monotonic = monotonic
        self.pid = os.getpid() if pid is None else pid
        self.started = monotonic()
        self._rates = _Rates()
        self._system: tuple[float, float] | None = None
        self._cache_bytes: int | None = None
        self._cache_at: float | None = None

    def sample(self, *, bytes_received: int = 0, bytes_sent: int = 0) -> dict[str, Any]:
        now = self.monotonic()
        reader = self.reader
        values: dict[str, Any] = {
            "uptime_seconds": round(now - self.started, 1),
            "cpu_count": reader.cpu_count(),
            "bytes_received": int(bytes_received),
            "bytes_sent": int(bytes_sent),
        }
        stat = reader.process(self.pid)
        if stat is not None:
            rate = self._rates.rate("self", stat.cpu_seconds, now)
            values["process_cpu_percent"] = _round(None if rate is None else rate * 100.0, 2)
            values["process_rss_bytes"] = stat.rss_bytes
        system = reader.system_cpu()
        if system is not None:
            if self._system is not None and system[1] > self._system[1]:
                values["system_cpu_percent"] = round(
                    max(0.0, min(100.0, (system[0] - self._system[0]) / (system[1] - self._system[1]) * 100.0)), 1)
            self._system = system
        values["load_1m"] = reader.load_1m()
        values["memory_total_bytes"], values["memory_available_bytes"] = reader.memory()
        values["temperature_c"] = reader.temperature_c()
        interfaces = reader.network()
        if interfaces:
            values["net_rx_bytes"] = sum(v[0] for v in interfaces.values())
            values["net_tx_bytes"] = sum(v[1] for v in interfaces.values())
        disk = disk_usage(self.cache_root or PROJECT_ROOT)
        if disk is not None:
            values["disk_total_bytes"] = disk["total_bytes"]
            values["disk_free_bytes"] = disk["free_bytes"]
        if self.cache_root is not None:
            if self._cache_at is None or now - self._cache_at >= STORAGE_INTERVAL_SECONDS:
                size = directory_size(self.cache_root)
                self._cache_bytes = None if size is None else size["bytes"]
                self._cache_at = now
            values["cache_bytes"] = self._cache_bytes
        values["cache_limit_bytes"] = self.cache_limit_bytes
        return {key: value for key, value in values.items() if value is not None}


__all__ = [
    "PURPOSES",
    "ClientResourceSampler",
    "ProcReader",
    "StatsPublisher",
    "StatsSampler",
    "TrafficCounter",
    "classify_process",
    "directory_size",
    "disk_usage",
    "history_path",
    "read_stats",
    "stats_path",
    "thread_purpose",
    "traffic_kind",
]
