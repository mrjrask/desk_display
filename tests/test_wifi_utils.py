from services import wifi_utils


class DummyResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_check_internet_falls_back_without_interface_binding(monkeypatch):
    calls = []

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
