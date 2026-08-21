import gzip

import pytest

import services.aircraft_type_db as type_db_module
from services.aircraft_type_db import lookup, refresh


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


_SAMPLE_CSV = (
    b"008187;ZS-ZWV;B738;00;BOEING 737-800;;;\n"
    b"000001;;;10;;;Miscode - VARIOUS;\n"
    b"01012A;SU-GDL;B77W;00;BOEING 777-300ER;;;\n"
)


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Point the module at a scratch DB file and reset its connection cache."""

    db_file = tmp_path / "aircraft_types.db"
    monkeypatch.setattr(type_db_module, "_db_path", lambda: str(db_file))
    type_db_module._conn = None
    type_db_module._conn_path = None
    monkeypatch.setattr(type_db_module.config, "ADSB_TYPE_DB_ENABLED", True)
    monkeypatch.setattr(type_db_module.config, "ADSB_TYPE_DB_REFRESH_DAYS", 30)
    yield
    type_db_module._conn = None
    type_db_module._conn_path = None


def test_lookup_returns_none_when_db_not_built_yet():
    assert lookup("008187") is None


def test_refresh_downloads_parses_and_populates_lookup(monkeypatch):
    def _fake_http_get(url, *, timeout=10.0, **kwargs):
        assert url == type_db_module.config.ADSB_TYPE_DB_URL
        return _FakeResponse(gzip.compress(_SAMPLE_CSV))

    monkeypatch.setattr(type_db_module, "http_get", _fake_http_get)

    assert refresh() is True
    assert lookup("008187") == "B738"
    assert lookup("01012a") == "B77W"  # case-insensitive
    assert lookup("000001") is None  # miscode row has no type, skipped
    assert lookup("ffffff") is None  # not in the database


def test_refresh_skips_when_recently_refreshed(monkeypatch):
    calls = []

    def _fake_http_get(url, *, timeout=10.0, **kwargs):
        calls.append(url)
        return _FakeResponse(gzip.compress(_SAMPLE_CSV))

    monkeypatch.setattr(type_db_module, "http_get", _fake_http_get)

    assert refresh() is True
    assert len(calls) == 1

    assert refresh() is False  # still fresh, no re-download
    assert len(calls) == 1

    assert refresh(force=True) is True  # force bypasses the freshness check
    assert len(calls) == 2


def test_refresh_swallows_network_errors(monkeypatch):
    def _fake_http_get(url, *, timeout=10.0, **kwargs):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(type_db_module, "http_get", _fake_http_get)

    assert refresh() is False
    assert lookup("008187") is None


def test_refresh_and_lookup_are_no_ops_when_disabled(monkeypatch):
    monkeypatch.setattr(type_db_module.config, "ADSB_TYPE_DB_ENABLED", False)

    def _fake_http_get(url, *, timeout=10.0, **kwargs):
        raise AssertionError("should not be called when disabled")

    monkeypatch.setattr(type_db_module, "http_get", _fake_http_get)

    assert refresh() is False
    assert lookup("008187") is None
