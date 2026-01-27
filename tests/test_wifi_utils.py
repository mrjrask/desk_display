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
