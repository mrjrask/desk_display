import importlib
import sys
import time

import pytest
import requests


def _reload_http_client(monkeypatch: pytest.MonkeyPatch, value: str | None):
    module_name = "services.http_client"
    if value is None:
        monkeypatch.delenv("HTTP_CLIENT_USE_SYSTEM_PROXIES", raising=False)
    else:
        monkeypatch.setenv("HTTP_CLIENT_USE_SYSTEM_PROXIES", value)

    if module_name in sys.modules:
        del sys.modules[module_name]

    return importlib.import_module(module_name)


def test_http_client_ignores_proxies_by_default(monkeypatch: pytest.MonkeyPatch):
    http_client = _reload_http_client(monkeypatch, None)
    try:
        session = http_client.get_session()
        assert session.trust_env is False
    finally:
        _reload_http_client(monkeypatch, None)


def test_http_client_can_opt_in_to_system_proxies(monkeypatch: pytest.MonkeyPatch):
    http_client = _reload_http_client(monkeypatch, "1")
    try:
        session = http_client.get_session()
        assert session.trust_env is True
    finally:
        _reload_http_client(monkeypatch, None)


def _fake_response(status_code: int) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = b"{}"
    return response


def test_circuit_breaker_short_circuits_after_403(monkeypatch: pytest.MonkeyPatch):
    http_client = _reload_http_client(monkeypatch, None)
    try:
        session = http_client.get_session()
        calls = []

        def fake_send(self, request, **kwargs):
            calls.append(request.url)
            return _fake_response(403)

        monkeypatch.setattr(requests.Session, "send", fake_send)

        # First request actually hits the network and gets a 403.
        response = session.get("https://blocked.example.com/a")
        assert response.status_code == 403
        assert len(calls) == 1

        # A second request to the same host, made shortly after, should be
        # short-circuited (no real network call) instead of also hitting 403.
        with pytest.raises(http_client.HostTemporarilyForbidden):
            session.get("https://blocked.example.com/b")
        assert len(calls) == 1  # no new network call was made

        # A different host is unaffected.
        response = session.get("https://other.example.com/c")
        assert response.status_code == 403
        assert len(calls) == 2
    finally:
        _reload_http_client(monkeypatch, None)


def test_circuit_breaker_clears_after_success(monkeypatch: pytest.MonkeyPatch):
    http_client = _reload_http_client(monkeypatch, None)
    try:
        session = http_client.get_session()
        statuses = iter([403, 200])

        def fake_send(self, request, **kwargs):
            return _fake_response(next(statuses))

        monkeypatch.setattr(requests.Session, "send", fake_send)

        assert session.get("https://recovering.example.com/a").status_code == 403
        with pytest.raises(http_client.HostTemporarilyForbidden):
            session.get("https://recovering.example.com/b")

        # Manually expire the cooldown (as if enough time had passed) and
        # confirm a subsequent success clears the block for later requests.
        session._forbidden_hosts_until["recovering.example.com"] = 0.0
        response = session.get("https://recovering.example.com/c")
        assert response.status_code == 200
        assert "recovering.example.com" not in session._forbidden_hosts_until
    finally:
        _reload_http_client(monkeypatch, None)


def test_named_sessions_have_isolated_circuit_breakers(monkeypatch: pytest.MonkeyPatch):
    http_client = _reload_http_client(monkeypatch, None)
    try:
        nfl_session = http_client.get_session("nfl")
        nba_session = http_client.get_session("nba")
        assert nfl_session is not nba_session

        calls = []

        def fake_send(self, request, **kwargs):
            calls.append(request.url)
            return _fake_response(403)

        monkeypatch.setattr(requests.Session, "send", fake_send)

        # NBA's request to the shared host gets 403'd and cools NBA's own
        # session down.
        assert nba_session.get("https://site.api.espn.com/a").status_code == 403
        with pytest.raises(http_client.HostTemporarilyForbidden):
            nba_session.get("https://site.api.espn.com/b")

        # NFL's separate session is unaffected and still hits the network.
        response = nfl_session.get("https://site.api.espn.com/c")
        assert response.status_code == 403
        assert len(calls) == 2

        # Requesting the same name twice returns the same session instance.
        assert http_client.get_session("nfl") is nfl_session
    finally:
        _reload_http_client(monkeypatch, None)


def test_day_scan_cooldown_starts_unblocked():
    from services.http_client import DayScanCooldown

    cooldown = DayScanCooldown(1800)
    assert cooldown.blocked() is False


def test_day_scan_cooldown_blocks_after_empty_scan(monkeypatch: pytest.MonkeyPatch):
    from services.http_client import DayScanCooldown

    cooldown = DayScanCooldown(1800)
    fake_now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: fake_now[0])

    cooldown.mark_empty()
    assert cooldown.blocked() is True

    fake_now[0] += 1799
    assert cooldown.blocked() is True

    fake_now[0] += 2
    assert cooldown.blocked() is False


def test_day_scan_cooldown_reset_clears_block(monkeypatch: pytest.MonkeyPatch):
    from services.http_client import DayScanCooldown

    cooldown = DayScanCooldown(1800)
    cooldown.mark_empty()
    assert cooldown.blocked() is True

    cooldown.reset()
    assert cooldown.blocked() is False
