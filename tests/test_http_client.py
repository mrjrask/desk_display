import importlib
import sys

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
