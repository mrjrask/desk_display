import importlib
import queue
import sys
import threading

import pytest


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


def test_get_session_reuses_session_within_thread(monkeypatch: pytest.MonkeyPatch):
    http_client = _reload_http_client(monkeypatch, None)
    try:
        assert http_client.get_session() is http_client.get_session()
    finally:
        _reload_http_client(monkeypatch, None)


def test_get_session_uses_distinct_sessions_across_threads(monkeypatch: pytest.MonkeyPatch):
    http_client = _reload_http_client(monkeypatch, None)
    sessions: queue.Queue[tuple[object, object]] = queue.Queue()
    barrier = threading.Barrier(2)

    def collect_session() -> None:
        first = http_client.get_session()
        barrier.wait(timeout=5)
        second = http_client.get_session()
        sessions.put((first, second))

    try:
        threads = [threading.Thread(target=collect_session) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

        thread_sessions = [sessions.get_nowait() for _ in threads]
        assert all(first is second for first, second in thread_sessions)
        assert thread_sessions[0][0] is not thread_sessions[1][0]
    finally:
        _reload_http_client(monkeypatch, None)
