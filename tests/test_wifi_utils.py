from services import wifi_utils


class DummyResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_check_internet_falls_back_without_interface_binding(monkeypatch):
    calls = []

    monkeypatch.setattr(wifi_utils, "_get_tcp_probe_targets", lambda: [])

    def fake_run(args, capture_output=True, text=True, check=False):
        calls.append(tuple(args))
        if "-I" in args:
            return DummyResult(returncode=2, stderr="ping: connect: Operation not permitted")
        return DummyResult(returncode=0)

    monkeypatch.setattr(wifi_utils, "_run_command", fake_run)
    monkeypatch.setattr(wifi_utils, "PING_HOSTS", ("8.8.8.8",))

    ok, tried = wifi_utils._check_internet("wlan0")

    assert ok is True
    assert tried == ["8.8.8.8"]
    assert len(calls) == 2
    assert "-I" in calls[0]
    assert all("-I" not in arg for arg in calls[1])


def test_tcp_probe_short_circuits_before_ping(monkeypatch):
    monkeypatch.setattr(
        wifi_utils,
        "_get_tcp_probe_targets",
        lambda: [("host", 443, "tcp://host:443")],
    )

    def fake_tcp(targets, tried):
        tried.append("tcp://host:443")
        return True

    monkeypatch.setattr(wifi_utils, "_check_tcp_targets", fake_tcp)
    monkeypatch.setattr(wifi_utils, "_run_command", lambda *args, **kwargs: DummyResult())
    monkeypatch.setattr(wifi_utils, "PING_HOSTS", ("9.9.9.9",))

    ok, tried = wifi_utils._check_internet("wlan0")

    assert ok is True
    assert tried == ["tcp://host:443"]


def test_tcp_probe_runs_after_ping_failures(monkeypatch):
    targets = [("host", 443, "tcp://host:443")]
    tcp_calls = []

    def fake_targets():
        return list(targets)

    def fake_tcp(targets_arg, tried):
        tcp_calls.append(list(targets_arg))
        tried.append("tcp://host:443")
        return len(tcp_calls) > 1

    def fake_run(args, capture_output=True, text=True, check=False):
        return DummyResult(returncode=1, stderr="timeout")

    monkeypatch.setattr(wifi_utils, "_get_tcp_probe_targets", fake_targets)
    monkeypatch.setattr(wifi_utils, "_check_tcp_targets", fake_tcp)
    monkeypatch.setattr(wifi_utils, "_run_command", fake_run)
    monkeypatch.setattr(wifi_utils, "PING_HOSTS", ("1.1.1.1",))

    ok, tried = wifi_utils._check_internet("wlan0")

    assert ok is True
    assert tried == ["tcp://host:443", "1.1.1.1", "tcp://host:443"]
    assert len(tcp_calls) == 2


def test_parse_tcp_probe_targets(monkeypatch):
    monkeypatch.setenv("WIFI_TCP_PROBE_URLS", "https://example.com,foo.test")
    monkeypatch.setenv("WIFI_TCP_PROBE_HOSTS", "1.2.3.4")
    monkeypatch.setenv("WIFI_TCP_PROBE_PORT", "8443")
    monkeypatch.setenv("RPI_CONNECT_CONTROL_HOST", "control.local")

    targets = wifi_utils._get_tcp_probe_targets()

    assert ("example.com", 443, "tcp://example.com:443") in targets
    assert ("foo.test", 443, "tcp://foo.test:443") in targets
    assert ("1.2.3.4", 8443, "tcp://1.2.3.4:8443") in targets

    monkeypatch.delenv("WIFI_TCP_PROBE_HOSTS")
    targets = wifi_utils._get_tcp_probe_targets()
    assert ("control.local", 8443, "tcp://control.local:8443") in targets


def test_start_monitor_resets_state_to_ok_before_thread_start(monkeypatch):
    class DummyThread:
        def __init__(self, *args, **kwargs):
            self.started = False

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

    wifi_utils.wifi_status = "no_wifi"
    wifi_utils.current_ssid = "test-ssid"
    wifi_utils._MONITOR_THREAD = None

    monkeypatch.setattr(wifi_utils, "should_monitor_wifi", lambda: True)
    monkeypatch.setattr(wifi_utils, "_detect_interface", lambda: "wlan0")
    monkeypatch.setattr(wifi_utils, "_resolve_user_log", lambda: None)
    monkeypatch.setattr(wifi_utils.threading, "Thread", DummyThread)

    wifi_utils.start_monitor()

    assert wifi_utils.get_wifi_state() == ("ok", None)

    wifi_utils._MONITOR_THREAD = None


def test_get_power_diagnostic_reports_clear_state(monkeypatch):
    monkeypatch.setattr(wifi_utils.shutil, "which", lambda _: "/usr/bin/vcgencmd")
    monkeypatch.setattr(wifi_utils, "_run_command", lambda args: DummyResult(stdout="throttled=0x0\n"))

    assert wifi_utils.get_power_diagnostic() == "no throttling detected"


def test_get_power_diagnostic_returns_raw_when_throttled(monkeypatch):
    monkeypatch.setattr(wifi_utils.shutil, "which", lambda _: "/usr/bin/vcgencmd")
    monkeypatch.setattr(wifi_utils, "_run_command", lambda args: DummyResult(stdout="throttled=0x50005\n"))

    assert wifi_utils.get_power_diagnostic() == "throttled=0x50005"


def test_wifi_state_source_of_truth_is_wifi_utils_state():
    wifi_utils._update_state("no_internet", "OfficeWiFi")

    assert wifi_utils.get_wifi_state() == ("no_internet", "OfficeWiFi")


def test_services_package_exports_wifi_utils_only_for_wifi_monitoring():
    import services

    assert "wifi_utils" in services.__all__
    assert "network" not in services.__all__


def test_get_assigned_ipv4_prefers_detected_wireless_interface(monkeypatch):
    monkeypatch.setattr(wifi_utils, "_IFACE", None)
    monkeypatch.setattr(wifi_utils, "_detect_interface", lambda: "wlan0")
    monkeypatch.setattr(wifi_utils, "_get_default_route_interfaces", lambda: ["eth0"])

    def fake_get_ipv4(iface):
        return {"wlan0": "192.168.1.20", "eth0": "10.0.0.5"}.get(iface)

    monkeypatch.setattr(wifi_utils, "_get_ipv4_address", fake_get_ipv4)

    assert wifi_utils.get_assigned_ipv4() == "192.168.1.20"


def test_get_assigned_ipv4_falls_back_to_socket_when_interfaces_missing(monkeypatch):
    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def connect(self, _endpoint):
            return None

        def getsockname(self):
            return ("10.1.2.3", 12345)

    monkeypatch.setattr(wifi_utils, "_IFACE", None)
    monkeypatch.setattr(wifi_utils, "_detect_interface", lambda: None)
    monkeypatch.setattr(wifi_utils, "_get_default_route_interfaces", lambda: [])
    monkeypatch.setattr(wifi_utils.socket, "socket", lambda *_args, **_kwargs: FakeSocket())

    assert wifi_utils.get_assigned_ipv4() == "10.1.2.3"
