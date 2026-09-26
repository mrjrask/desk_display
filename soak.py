"""Soak monitoring, release gates and rollback triggers for the server/client system.

``sample`` records what a device is doing once per interval as JSON lines:
the render server's view of its clients, renders and playlists (when run
where the admin token is readable), and this device's own processes and
disk. ``gates`` reads one or more of those logs and decides go or no-go
against the release gates, flagging any rollback trigger that fired.

    python3 soak.py sample --out soak/server.jsonl --hours 48
    python3 soak.py gates soak/*.jsonl --clients office,den,shelf

See docs/soak-and-release.md. Standard library only, so it runs on any role.
Nothing it writes contains a credential: the admin token is only ever sent
in a request header, and every server payload is scrubbed before it is kept.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_VERSION = 1
# The project's long-running processes, by what their command line runs.
PROCESSES = {
    "server": "display_server",
    "client": "display_client",
    "standalone": "main.py",
    "feeds": "feed_server",
    "config_ui": "config_ui",
}
CACHES = ("cache/artifacts", "cache/client")


# ── Sampling ────────────────────────────────────────────────────────────────


def _env_file_value(path: Path, name: str) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        key, sep, value = line.strip().partition("=")
        if sep and key.strip().removeprefix("export ").strip() == name:
            return value.strip().strip("'\"") or None
    return None


def _get(url: str, token: str | None, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator-supplied URL
        return json.loads(response.read().decode("utf-8"))


def summarize_server(status: Mapping[str, Any], render: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep the numbers the gates need, and nothing that could be a secret."""

    clients = {}
    for record in status.get("clients") or ():
        reported = record.get("status") or {}
        clients[str(record.get("client_id"))] = {
            "lease_state": record.get("lease_state"),
            "playback_state": reported.get("playback_state"),
            "current_screen": reported.get("current_screen"),
            "last_sync_age_seconds": reported.get("last_sync_age_seconds"),
            "cache_age_seconds": reported.get("cache_age_seconds"),
            "errors": len(reported.get("recent_errors") or ()),
            "physical_rotation": reported.get("physical_rotation"),
        }
    result: dict[str, Any] = {"reachable": True, "clients": clients,
                              "artifacts": dict(status.get("artifacts") or {})}
    if render is not None:
        lineages = render.get("lineages") or ()
        result["render"] = {
            "renders": sum(int(row.get("renders") or 0) for row in lineages),
            "failures": sum(int(row.get("failures") or 0) for row in lineages),
            "failing": sorted(f"{row.get('screen_id')}@{row.get('render_profile')}" for row in lineages
                              if int(row.get("consecutive_failures") or 0) > 0),
            "not_fresh": sum(1 for row in lineages if row.get("state") not in (None, "fresh")),
            "queue": len(render.get("queue") or ()),
        }
        result["playlists"] = {str(row.get("client_id")): row.get("state") for row in render.get("playlists") or ()}
    return result


def _processes(proc: Path = Path("/proc")) -> dict[str, Any]:
    found: dict[str, Any] = {}
    try:
        entries = [p for p in proc.iterdir() if p.name.isdigit()]
    except OSError:
        return found
    page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    ticks = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    for entry in entries:
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            command = " ".join(a.decode("utf-8", "replace") for a in argv if a)
            if "python" not in command or "soak.py" in command:
                continue
            name = next((n for n, marker in PROCESSES.items() if marker in command), None)
            if name is None or name in found:
                continue
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            rss_pages = int((entry / "statm").read_text().split()[1])
            try:
                fds = len(os.listdir(entry / "fd"))
            except OSError:
                fds = None
            found[name] = {"pid": int(entry.name), "rss_mb": round(rss_pages * page / 2**20, 1),
                           "cpu_seconds": round((int(stat[11]) + int(stat[12])) / ticks, 1), "fds": fds}
        except (OSError, ValueError, IndexError):
            continue
    return found


def _tree_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return round(total / 2**20, 2)


def collect_sample(*, project_dir: Path = PROJECT_DIR, server_url: str | None = None,
                   admin_token: str | None = None, timeout: float = 10.0,
                   now: float | None = None, proc: Path = Path("/proc")) -> dict[str, Any]:
    sample: dict[str, Any] = {"v": SAMPLE_VERSION, "t": round(time.time() if now is None else now, 1),
                              "host": socket.gethostname()}
    if server_url:
        base = server_url.rstrip("/")
        try:
            status = _get(f"{base}/api/v1/admin/status", admin_token, timeout)
            try:
                render = _get(f"{base}/api/v1/admin/render-status", admin_token, timeout)
            except urllib.error.HTTPError:
                render = None
            sample["server"] = summarize_server(status, render)
        except (OSError, ValueError) as exc:
            reason = getattr(exc, "code", None) or type(exc).__name__
            sample["server"] = {"reachable": False, "error": str(reason)}
    sample["processes"] = _processes(proc)
    usage = shutil.disk_usage(project_dir)
    sample["disk"] = {"free_percent": round(100 * usage.free / usage.total, 1),
                      **{name: _tree_mb(project_dir / name) for name in CACHES}}
    return sample


def run_sampler(out: Path, *, interval: float, hours: float, **kwargs: Any) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + hours * 3600
    count = 0
    while True:
        sample = collect_sample(**kwargs)
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
        count += 1
        if time.time() + interval > deadline:
            return count
        time.sleep(interval)


# ── Release gates ───────────────────────────────────────────────────────────


@dataclass
class Gates:
    """Release thresholds. Every one is documented in docs/soak-and-release.md."""

    min_hours: float = 48.0
    max_gap_seconds: float = 300.0
    server_reachable_percent: float = 99.5
    client_active_percent: float = 99.0
    client_playing_percent: float = 99.0
    sync_age_p95_seconds: float = 120.0
    render_failure_percent: float = 1.0
    rss_growth_mb: float = 64.0
    min_free_disk_percent: float = 10.0
    cache_limits_mb: dict[str, float] = field(default_factory=lambda: {"cache/client": 256.0})
    unplanned_restarts: int = 0
    # Rollback triggers: any one of these is an immediate no-go.
    trigger_client_down_minutes: float = 10.0
    trigger_render_failing_minutes: float = 15.0
    trigger_playlist_stuck_minutes: float = 10.0
    trigger_rss_growth_mb: float = 256.0
    trigger_free_disk_percent: float = 5.0

    @classmethod
    def load(cls, path: Path | None) -> Gates:
        if path is None:
            return cls()
        values = json.loads(path.read_text(encoding="utf-8"))
        unknown = sorted(set(values) - set(asdict(cls())))
        if unknown:
            raise ValueError(f"unknown gate settings: {', '.join(unknown)}")
        return cls(**values)


@dataclass
class Result:
    name: str
    passed: bool
    detail: str
    trigger: bool = False  # a rollback trigger rather than a gate


def load_samples(paths: Iterable[Path]) -> list[dict[str, Any]]:
    samples = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: not a JSON sample ({exc})") from None
            if sample.get("v") != SAMPLE_VERSION:
                raise ValueError(f"{path}:{number}: sample version {sample.get('v')!r} is not supported")
            samples.append(sample)
    return samples


def _longest_run(samples: list[dict[str, Any]], bad) -> float:
    """Longest stretch, in seconds, over which ``bad(sample)`` held."""

    longest, started = 0.0, None
    for sample in samples:
        if bad(sample):
            started = sample["t"] if started is None else started
            longest = max(longest, sample["t"] - started)
        else:
            started = None
    return longest


def _percent(values: list[bool]) -> float:
    return 100.0 if not values else 100.0 * sum(values) / len(values)


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def _positive_deltas(values: list[int]) -> int:
    """Total increase of a counter that resets to zero when a process restarts."""

    total, previous = 0, None
    for value in values:
        if previous is not None:
            total += value - previous if value >= previous else value
        previous = value
    return total


def evaluate(samples: list[dict[str, Any]], gates: Gates, *,
             expected_clients: Iterable[str] = ()) -> list[Result]:
    results: list[Result] = []
    if not samples:
        return [Result("samples", False, "no samples recorded")]
    by_host: dict[str, list[dict[str, Any]]] = {}
    for sample in sorted(samples, key=lambda s: s["t"]):
        by_host.setdefault(sample.get("host") or "?", []).append(sample)
    ordered = sorted(samples, key=lambda s: s["t"])
    hours = (ordered[-1]["t"] - ordered[0]["t"]) / 3600
    results.append(Result("duration", hours >= gates.min_hours,
                          f"{hours:.1f} h recorded, {gates.min_hours:g} h required"))
    gaps = [b["t"] - a["t"] for host in by_host.values() for a, b in zip(host, host[1:])]
    worst_gap = max(gaps, default=0.0)
    results.append(Result("monitoring", worst_gap <= gates.max_gap_seconds,
                          f"longest gap between samples {worst_gap:.0f} s (limit {gates.max_gap_seconds:g} s)"))

    server = [s for s in ordered if "server" in s]
    if server:
        reachable = [bool(s["server"].get("reachable")) for s in server]
        results.append(Result("server availability", _percent(reachable) >= gates.server_reachable_percent,
                              f"{_percent(reachable):.2f}% of samples (need {gates.server_reachable_percent:g}%)"))
        up = [s for s in server if s["server"].get("reachable")]
        clients = set(expected_clients) or {c for s in up for c in s["server"].get("clients", {})}
        for client_id in sorted(clients):
            def row(sample, cid=client_id):
                return sample["server"].get("clients", {}).get(cid) or {}

            active = [row(s).get("lease_state") in ("active", "static") for s in up]
            playing = [row(s).get("playback_state") == "playing" and row(s).get("current_screen") is not None
                       for s in up]
            results.append(Result(f"{client_id} connected", _percent(active) >= gates.client_active_percent,
                                  f"lease active in {_percent(active):.2f}% of samples"))
            results.append(Result(f"{client_id} playing", _percent(playing) >= gates.client_playing_percent,
                                  f"showing content in {_percent(playing):.2f}% of samples"))
            ages = [float(row(s)["last_sync_age_seconds"]) for s in up
                    if isinstance(row(s).get("last_sync_age_seconds"), int | float)]
            p95 = _p95(ages)
            results.append(Result(f"{client_id} sync age", p95 is not None and p95 <= gates.sync_age_p95_seconds,
                                  "no sync age reported" if p95 is None else
                                  f"95th percentile {p95:.0f} s (limit {gates.sync_age_p95_seconds:g} s)"))
            down = _longest_run(up, lambda s, cid=client_id: not (
                row(s, cid).get("lease_state") in ("active", "static")
                and row(s, cid).get("current_screen") is not None))
            results.append(Result(f"{client_id} blank or disconnected", down < gates.trigger_client_down_minutes * 60,
                                  f"longest {down / 60:.1f} min (trigger {gates.trigger_client_down_minutes:g} min)",
                                  trigger=True))
            stuck = _longest_run(up, lambda s, cid=client_id: (s["server"].get("playlists") or {}).get(cid)
                                 in ("pending_delivery", "pending_acknowledgment"))
            results.append(Result(f"{client_id} playlist delivery", stuck < gates.trigger_playlist_stuck_minutes * 60,
                                  f"longest unacknowledged {stuck / 60:.1f} min "
                                  f"(trigger {gates.trigger_playlist_stuck_minutes:g} min)", trigger=True))
        rendered = [s["server"]["render"] for s in up if "render" in s["server"]]
        if rendered:
            renders = _positive_deltas([r["renders"] for r in rendered])
            failures = _positive_deltas([r["failures"] for r in rendered])
            rate = 0.0 if renders + failures == 0 else 100.0 * failures / (renders + failures)
            results.append(Result("render failures", rate <= gates.render_failure_percent,
                                  f"{failures} of {renders + failures} renders failed ({rate:.2f}%)"))
            failing = {name for r in rendered for name in r.get("failing", ())}
            worst = max((_longest_run([s for s in up if "render" in s["server"]],
                                      lambda s, n=name: n in s["server"]["render"].get("failing", ()))
                         for name in failing), default=0.0)
            results.append(Result("sustained render failure", worst < gates.trigger_render_failing_minutes * 60,
                                  f"longest {worst / 60:.1f} min (trigger {gates.trigger_render_failing_minutes:g} min)",
                                  trigger=True))

    for host, host_samples in sorted(by_host.items()):
        names = sorted({n for s in host_samples for n in s.get("processes", {})})
        for name in names:
            seen = [s["processes"][name] for s in host_samples if name in s.get("processes", {})]
            pids = [p["pid"] for p in seen]
            restarts = sum(1 for a, b in zip(pids, pids[1:]) if a != b)
            results.append(Result(f"{host} {name} restarts", restarts <= gates.unplanned_restarts,
                                  f"{restarts} restarts (allowed {gates.unplanned_restarts})"))
            window = max(1, len(seen) // 10)
            first = statistics.median(p["rss_mb"] for p in seen[:window])
            last = statistics.median(p["rss_mb"] for p in seen[-window:])
            growth = last - first
            results.append(Result(f"{host} {name} memory", growth <= gates.rss_growth_mb,
                                  f"RSS {first:.0f} -> {last:.0f} MB ({growth:+.0f} MB, limit {gates.rss_growth_mb:g})"))
            results.append(Result(f"{host} {name} memory runaway", growth < gates.trigger_rss_growth_mb,
                                  f"{growth:+.0f} MB (trigger {gates.trigger_rss_growth_mb:g} MB)", trigger=True))
        disks = [s["disk"] for s in host_samples if "disk" in s]
        if disks:
            free = min(d["free_percent"] for d in disks)
            results.append(Result(f"{host} free disk", free >= gates.min_free_disk_percent,
                                  f"lowest {free:.1f}% (need {gates.min_free_disk_percent:g}%)"))
            results.append(Result(f"{host} disk exhaustion", free > gates.trigger_free_disk_percent,
                                  f"lowest {free:.1f}% (trigger {gates.trigger_free_disk_percent:g}%)", trigger=True))
            for cache, limit in sorted(gates.cache_limits_mb.items()):
                sizes = [d[cache] for d in disks if isinstance(d.get(cache), int | float)]
                if sizes:
                    results.append(Result(f"{host} {cache} size", max(sizes) <= limit,
                                          f"largest {max(sizes):.1f} MB (limit {limit:g} MB)"))
    return results


def verdict(results: list[Result]) -> str:
    if any(r.trigger and not r.passed for r in results):
        return "ROLLBACK"
    return "GO" if all(r.passed for r in results) else "NO-GO"


def report(results: list[Result]) -> str:
    lines = [f"Verdict: {verdict(results)}", ""]
    for heading, triggers in (("Release gates", False), ("Rollback triggers", True)):
        rows = [r for r in results if r.trigger is triggers]
        if rows:
            lines.append(heading)
            lines += [f"  {'PASS' if r.passed else 'FAIL'}  {r.name}: {r.detail}" for r in rows]
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ── Command line ────────────────────────────────────────────────────────────


def _cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 soak.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample", help="record samples as JSON lines")
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--interval", type=float, default=60.0, help="seconds between samples (default 60)")
    sample.add_argument("--hours", type=float, default=48.0, help="how long to record (default 48)")
    sample.add_argument("--once", action="store_true", help="record one sample and exit")
    sample.add_argument("--server-url", help="the render server to watch, e.g. http://127.0.0.1:8765")
    sample.add_argument("--env-file", type=Path, default=PROJECT_DIR / ".env",
                        help="where DESK_DISPLAY_SERVER_ADMIN_TOKEN is read from (default .env)")
    sample.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    gates = sub.add_parser("gates", help="judge recorded samples against the release gates")
    gates.add_argument("logs", type=Path, nargs="+")
    gates.add_argument("--clients", default="", help="comma-separated client IDs that must pass")
    gates.add_argument("--gates", type=Path, help="JSON file overriding gate thresholds")
    gates.add_argument("--json", action="store_true", help="print the results as JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "sample":
        token = os.environ.get("DESK_DISPLAY_SERVER_ADMIN_TOKEN") or _env_file_value(
            args.env_file, "DESK_DISPLAY_SERVER_ADMIN_TOKEN")
        if args.server_url and not token:
            print("error: no DESK_DISPLAY_SERVER_ADMIN_TOKEN in the environment or the env file",
                  file=sys.stderr)
            return 2
        kwargs = {"project_dir": args.project_dir, "server_url": args.server_url, "admin_token": token}
        if args.once:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(collect_sample(**kwargs), sort_keys=True) + "\n")
            return 0
        count = run_sampler(args.out, interval=args.interval, hours=args.hours, **kwargs)
        print(f"recorded {count} samples in {args.out}")
        return 0

    try:
        results = evaluate(load_samples(args.logs), Gates.load(args.gates),
                           expected_clients=[c for c in args.clients.split(",") if c])
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"verdict": verdict(results), "results": [asdict(r) for r in results]}, indent=2))
    else:
        print(report(results), end="")
    return {"GO": 0, "NO-GO": 1, "ROLLBACK": 3}[verdict(results)]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
