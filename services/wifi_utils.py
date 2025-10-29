"""Wi-Fi monitoring and automatic recovery utilities."""

from __future__ import annotations

import datetime
import logging
import os
import pwd
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# ─── Behaviour configuration ───────────────────────────────────────────────────

PING_HOSTS: Sequence[str] = ("1.1.1.1", "8.8.8.8")
PING_TIMEOUT = 2  # seconds per ping
CHECK_INTERVAL_OK = 15  # seconds between healthy checks
RETRY_INTERVAL = 60  # seconds between recovery attempts
MAX_FAILS = 1  # failures before triggering recovery


# ─── Module globals ───────────────────────────────────────────────────────────

wifi_status = "no_wifi"  # one of "no_wifi", "no_internet", "ok"
current_ssid: Optional[str] = None

_STATE_LOCK = threading.Lock()
_STOP_EVENT = threading.Event()
_MONITOR_THREAD: Optional[threading.Thread] = None
_IFACE: Optional[str] = None
_USER_LOG_PATH: Optional[Path] = None
_SYSTEM_LOG_PATH = Path("/var/log/wifi_auto_recover.log")

_LOGGER = logging.getLogger(__name__)


# ─── Helpers: system/user logging ──────────────────────────────────────────────

def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append_line(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")
    except Exception as exc:  # pragma: no cover - defensive logging
        _LOGGER.debug("Unable to append to %s: %s", path, exc)


def _system_log(message: str) -> None:
    text = f"{_timestamp()} [wifi-auto-recover] {message}"
    _LOGGER.info(message)
    try:
        _append_line(_SYSTEM_LOG_PATH, text)
    except Exception:
        # _append_line already logs the failure; swallow to avoid recursion
        pass


def _user_log(message: str) -> None:
    if not _USER_LOG_PATH:
        return
    text = f"{_timestamp()} [wifi-recovery] {message}"
    _append_line(_USER_LOG_PATH, text)


def _resolve_user_log() -> Optional[Path]:
    env = os.environ.get("WIFI_RECOVERY_LOG")
    if env:
        return Path(env)

    sudo_user = os.environ.get("SUDO_USER")
    home_path: Optional[Path] = None
    if sudo_user and sudo_user != "root":
        try:
            home_path = Path(pwd.getpwnam(sudo_user).pw_dir)
        except KeyError:
            home_path = None

    if not home_path or not home_path.exists():
        candidate = Path("/home/pi")
        if candidate.exists():
            home_path = candidate
        else:
            home_path = Path("/root")

    return home_path / "wifi_recovery.log"


# ─── Helpers: interface/state detection ────────────────────────────────────────

def _run_command(args: Sequence[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=check)


def _get_wireless_interfaces() -> Sequence[str]:
    try:
        proc = _run_command(["iw", "dev"])
        interfaces = []
        for line in proc.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("Interface"):
                parts = stripped.split()
                if len(parts) >= 2:
                    interfaces.append(parts[1])
        return interfaces
    except Exception as exc:
        _LOGGER.debug("iw dev failed: %s", exc)
        return []


def _detect_interface() -> Optional[str]:
    env_iface = os.environ.get("WIFI_INTERFACE")
    if env_iface:
        return env_iface

    interfaces = _get_wireless_interfaces()
    if interfaces:
        return interfaces[0]

    return None


def _get_link_info(iface: str) -> str:
    try:
        return _run_command(["iw", "dev", iface, "link"]).stdout
    except Exception as exc:
        _LOGGER.debug("iw dev %s link failed: %s", iface, exc)
        return ""


def _extract_field(link_info: str, key: str) -> Optional[str]:
    lower_key = key.lower()
    for line in link_info.splitlines():
        if lower_key in line.lower():
            cleaned = line.split(key, 1)[-1].strip()
            return cleaned
    return None


def _get_ssid_from_link(link_info: str) -> Optional[str]:
    for line in link_info.splitlines():
        line = line.strip()
        if line.startswith("SSID:"):
            return line.split("SSID:", 1)[-1].strip()
    return None


def _get_ssid_fallback() -> Optional[str]:
    try:
        proc = _run_command(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"])
        for line in proc.stdout.splitlines():
            if not line:
                continue
            active, _, ssid = line.partition(":")
            if active == "yes" and ssid:
                return ssid
    except Exception as exc:
        _LOGGER.debug("nmcli SSID lookup failed: %s", exc)

    try:
        proc = _run_command(["iwgetid", "-r"])
        value = proc.stdout.strip()
        if value:
            return value
    except Exception as exc:
        _LOGGER.debug("iwgetid failed: %s", exc)

    return None


def _has_default_route(iface: str) -> bool:
    try:
        proc = _run_command(["ip", "route", "show", "default", "dev", iface])
        if proc.returncode != 0:
            return False
        return bool(proc.stdout.strip())
    except Exception as exc:
        _LOGGER.debug("ip route show default dev %s failed: %s", iface, exc)
        return False


def _check_dns_resolution() -> bool:
    try:
        proc = _run_command(["getent", "hosts", "dns.google"])
        return proc.returncode == 0
    except Exception as exc:
        _LOGGER.debug("getent hosts dns.google failed: %s", exc)
        return False


def _check_internet(iface: str) -> Tuple[bool, List[str]]:
    tried: List[str] = []
    for host in PING_HOSTS:
        tried.append(host)

        needs_fallback = False

        if iface:
            try:
                proc = _run_command([
                    "ping",
                    "-I",
                    iface,
                    "-c",
                    "1",
                    "-W",
                    str(PING_TIMEOUT),
                    host,
                ])
                if proc.returncode == 0:
                    return True, tried

                stderr = proc.stderr.strip() if proc.stderr else ""
                if stderr:
                    _LOGGER.debug(
                        "ping via %s to %s failed rc=%s: %s",
                        iface,
                        host,
                        proc.returncode,
                        stderr,
                    )
                else:
                    _LOGGER.debug(
                        "ping via %s to %s failed rc=%s",
                        iface,
                        host,
                        proc.returncode,
                    )

                if stderr:
                    lower_err = stderr.lower()
                    if any(
                        keyword in lower_err
                        for keyword in (
                            "operation not permitted",
                            "permission denied",
                            "must be root",
                            "requires cap_net_raw",
                        )
                    ):
                        needs_fallback = True
            except Exception as exc:
                needs_fallback = True
                _LOGGER.debug("ping via %s to %s raised: %s", iface, host, exc)
        else:
            needs_fallback = True

        if needs_fallback:
            try:
                proc = _run_command([
                    "ping",
                    "-c",
                    "1",
                    "-W",
                    str(PING_TIMEOUT),
                    host,
                ])
                if proc.returncode == 0:
                    return True, tried
                if proc.stderr:
                    _LOGGER.debug(
                        "ping to %s failed rc=%s: %s",
                        host,
                        proc.returncode,
                        proc.stderr.strip(),
                    )
            except Exception as exc:
                _LOGGER.debug("ping to %s raised: %s", host, exc)

    return False, tried


def _get_ipv4_address(iface: str) -> Optional[str]:
    try:
        proc = _run_command(["ip", "-4", "addr", "show", "dev", iface])
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("inet "):
                addr = line.split()[1]
                return addr.split("/")[0]
    except Exception as exc:
        _LOGGER.debug("ip -4 addr show %s failed: %s", iface, exc)
    return None


def _report_status(iface: str, link_info: str) -> None:
    ssid = _get_ssid_from_link(link_info) or _get_ssid_fallback() or "?"
    bssid = None
    for line in link_info.splitlines():
        line = line.strip()
        if line.startswith("Connected to"):
            parts = line.split()
            if len(parts) >= 3:
                bssid = parts[2]
                break
    signal_dbm = None
    sig = _extract_field(link_info, "signal:")
    if sig:
        signal_dbm = sig.split()[0]
    freq = _extract_field(link_info, "freq:")
    tx = _extract_field(link_info, "tx bitrate:")
    ipv4 = _get_ipv4_address(iface) or "none"
    default_route = "yes" if _has_default_route(iface) else "no"
    dns_ok = "yes" if _check_dns_resolution() else "no"

    _system_log(
        "Status iface=%s ssid=%s bssid=%s signal_dbm=%s freq_mhz=%s tx=%s ip=%s "
        "default_route=%s dns_resolve=%s"
        % (
            iface,
            ssid,
            bssid or "?",
            signal_dbm or "?",
            freq.split()[0] if freq else "?",
            tx or "?",
            ipv4,
            default_route,
            dns_ok,
        )
    )


def _disable_powersave(iface: str) -> None:
    try:
        proc = _run_command(["iw", "dev", iface, "get", "power_save"])
        if "on" in proc.stdout.lower():
            _run_command(["iw", "dev", iface, "set", "power_save", "off"])
            _system_log(f"Action: disabled_power_save iface={iface}")
    except Exception as exc:
        _LOGGER.debug("Unable to disable power save on %s: %s", iface, exc)


def _cycle_wifi(iface: str) -> None:
    _system_log(f"Action: cycle_wifi iface={iface} step=down")
    try:
        _run_command(["ip", "link", "set", iface, "down"])
    except Exception as exc:
        _LOGGER.debug("Failed to bring %s down: %s", iface, exc)
    time.sleep(2)
    _system_log(f"Action: cycle_wifi iface={iface} step=up")
    try:
        _run_command(["ip", "link", "set", iface, "up"])
    except Exception as exc:
        _LOGGER.debug("Failed to bring %s up: %s", iface, exc)
    if shutil.which("wpa_cli"):
        try:
            _run_command(["wpa_cli", "-i", iface, "reconfigure"])
            _system_log(f"Action: wpa_supplicant_reconfigure iface={iface}")
        except Exception as exc:
            _LOGGER.debug("wpa_cli reconfigure failed: %s", exc)


def _update_state(state: str, ssid: Optional[str]) -> None:
    global wifi_status, current_ssid
    with _STATE_LOCK:
        wifi_status = state
        current_ssid = ssid


def _sleep_with_stop(seconds: float) -> bool:
    return _STOP_EVENT.wait(seconds)


def _monitor_loop() -> None:
    global _IFACE

    iface = _IFACE
    if not iface:
        _LOGGER.warning("Wi-Fi monitor started without a detected interface; exiting")
        return

    _system_log(f"Startup: begin iface={iface} user_log={_USER_LOG_PATH}")
    _disable_powersave(iface)

    link_info = _get_link_info(iface)
    _report_status(iface, link_info)

    fails = 0
    recovery_started: Optional[float] = None

    while not _STOP_EVENT.is_set():
        link_info = _get_link_info(iface)
        associated = "Connected to" in link_info
        ssid = _get_ssid_from_link(link_info) or _get_ssid_fallback()

        if not associated:
            fails += 1
            _update_state("no_wifi", None)
            _system_log(f"Fail: not_associated iface={iface} fail_count={fails}/{MAX_FAILS}")
        else:
            if not ssid:
                ssid = None
            if not _has_default_route(iface):
                fails += 1
                _update_state("no_internet", ssid)
                _system_log(f"Fail: no_default_route iface={iface} fail_count={fails}/{MAX_FAILS}")
            else:
                internet_ok, tried = _check_internet(iface)
                if not internet_ok:
                    fails += 1
                    _update_state("no_internet", ssid)
                    _system_log(
                        "Fail: ping_timeout iface=%s hosts_tried='%s' timeout_s=%s fail_count=%s/%s"
                        % (iface, " ".join(tried), PING_TIMEOUT, fails, MAX_FAILS)
                    )
                else:
                    _update_state("ok", ssid)
                    if recovery_started is not None:
                        duration = int(time.time() - recovery_started)
                        _user_log(f"Recovered connection on {iface} after {duration}s.")
                        _system_log(f"Recovered: iface={iface} duration_s={duration}")
                        _report_status(iface, _get_link_info(iface))
                        recovery_started = None
                    fails = 0
                    if _sleep_with_stop(CHECK_INTERVAL_OK):
                        break
                    continue

        _report_status(iface, link_info)

        if fails >= MAX_FAILS:
            if recovery_started is None:
                recovery_started = time.time()
                _user_log(f"Lost connection on {iface} — starting recovery attempts.")
                _system_log(f"Recover: start iface={iface}")
            _cycle_wifi(iface)
            if _sleep_with_stop(RETRY_INTERVAL):
                break
        else:
            if _sleep_with_stop(5):
                break

    _system_log("Wi-Fi monitor thread exiting")


def start_monitor() -> None:
    """Start the background Wi-Fi monitor."""

    global _MONITOR_THREAD, _IFACE, _USER_LOG_PATH

    if _MONITOR_THREAD and _MONITOR_THREAD.is_alive():
        return

    _IFACE = _detect_interface()
    if not _IFACE:
        _LOGGER.warning("No wireless interface detected; Wi-Fi monitor disabled")
        return

    _USER_LOG_PATH = _resolve_user_log()
    if _USER_LOG_PATH:
        try:
            _USER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            _USER_LOG_PATH.touch(exist_ok=True)
        except Exception as exc:
            _LOGGER.debug("Unable to prepare user Wi-Fi log %s: %s", _USER_LOG_PATH, exc)

    _STOP_EVENT.clear()
    thread = threading.Thread(target=_monitor_loop, daemon=True)
    thread.start()
    _MONITOR_THREAD = thread


def get_wifi_state() -> Tuple[str, Optional[str]]:
    """Return the current Wi-Fi state and SSID."""

    with _STATE_LOCK:
        return wifi_status, current_ssid


def stop_monitor(timeout: Optional[float] = 5.0) -> None:
    """Request the Wi-Fi monitor thread to stop and wait for it to exit."""

    global _MONITOR_THREAD

    if not _MONITOR_THREAD:
        return

    _STOP_EVENT.set()
    _MONITOR_THREAD.join(timeout)
    if _MONITOR_THREAD.is_alive():
        _LOGGER.debug("Wi-Fi monitor thread did not exit before timeout.")
    _MONITOR_THREAD = None
