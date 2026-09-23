import importlib
from types import SimpleNamespace

import pytest
import requests


def _reload_uploader(monkeypatch, **env):
    for key in (
        "FEED_UPLOAD_URL",
        "FEED_UPLOAD_TOKEN",
        "FEED_SOURCE_NAME",
        "FEED_UPLOAD_INTERVAL_SECONDS",
        "FEED_UPLOAD_TIMEOUT_SECONDS",
        "FEED_UPLOAD_CYCLE_TIMEOUT_SECONDS",
        "FEED_UPLOAD_MAX_BACKOFF_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    module = importlib.import_module("scripts.screenshot_uploader")
    return importlib.reload(module)


class _FakeResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


class _FakeSession:
    def __init__(self, response=None, exc=None):
        self.calls = []
        self._response = response or _FakeResponse()
        self._exc = exc

    def post(self, url, headers=None, data=None, files=None, json=None, timeout=None):
        self.calls.append(
            SimpleNamespace(
                url=url,
                headers=headers,
                data=data,
                files=files,
                json=json,
                timeout=timeout,
            )
        )
        if self._exc is not None:
            raise self._exc
        return self._response


def test_iter_current_screenshots_filters_non_image_files(tmp_path):
    uploader = importlib.import_module("scripts.screenshot_uploader")

    (tmp_path / "date.png").write_bytes(b"x")
    (tmp_path / "weather.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    (tmp_path / "subdir").mkdir()

    found = {path.name for path in uploader._iter_current_screenshots(tmp_path)}

    assert found == {"date.png", "weather.jpg"}


def test_iter_current_screenshots_handles_missing_dir(tmp_path):
    uploader = importlib.import_module("scripts.screenshot_uploader")

    assert list(uploader._iter_current_screenshots(tmp_path / "missing")) == []


def test_content_type_for_extension():
    uploader = importlib.import_module("scripts.screenshot_uploader")

    assert uploader._content_type_for(uploader.Path("weather.jpg")) == "image/jpeg"
    assert uploader._content_type_for(uploader.Path("weather.JPEG")) == "image/jpeg"
    assert uploader._content_type_for(uploader.Path("date.png")) == "image/png"


def test_upload_file_posts_expected_request(monkeypatch, tmp_path):
    uploader = _reload_uploader(
        monkeypatch,
        FEED_UPLOAD_URL="http://192.168.1.200:5003",
        FEED_UPLOAD_TOKEN="secret-token",
        FEED_SOURCE_NAME="hyper",
    )

    screenshot = tmp_path / "date.png"
    screenshot.write_bytes(b"fake-png-bytes")

    session = _FakeSession()
    result = uploader._upload_file(session, screenshot)

    assert result is True
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call.url == "http://192.168.1.200:5003/api/feed/hyper/upload"
    assert call.headers == {"Authorization": "Bearer secret-token"}
    assert call.data == {"screen_id": "date"}
    assert call.files["file"][0] == "date.png"


@pytest.mark.parametrize(
    ("source_name", "source_id"),
    [
        ("desk/office", "desk-office"),
        ("desk?office#1", "deskoffice1"),
        ("desk office", "desk_office"),
        ("café", "café"),
    ],
)
def test_upload_file_uses_canonical_source_id(monkeypatch, tmp_path, source_name, source_id):
    uploader = _reload_uploader(
        monkeypatch,
        FEED_UPLOAD_URL="http://192.168.1.200:5003",
        FEED_UPLOAD_TOKEN="secret-token",
        FEED_SOURCE_NAME=source_name,
    )

    screenshot = tmp_path / "date.png"
    screenshot.write_bytes(b"fake-png-bytes")
    session = _FakeSession()

    assert uploader._upload_file(session, screenshot) is True
    assert session.calls[0].url == (
        f"http://192.168.1.200:5003/api/feed/{source_id}/upload"
    )


def test_upload_status_posts_json_body(monkeypatch, tmp_path):
    uploader = _reload_uploader(
        monkeypatch,
        FEED_UPLOAD_URL="http://192.168.1.200:5003",
        FEED_UPLOAD_TOKEN="secret-token",
        FEED_SOURCE_NAME="hyper",
    )

    status_path = tmp_path / "display_status.json"
    status_path.write_text('{"screen_id": "date", "loop_iteration": 3}', encoding="utf-8")

    session = _FakeSession()
    result = uploader._upload_status(session, status_path)

    assert result is True
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call.url == "http://192.168.1.200:5003/api/feed/hyper/status"
    assert call.headers == {"Authorization": "Bearer secret-token"}


def test_upload_status_returns_false_for_invalid_json(monkeypatch, tmp_path):
    uploader = _reload_uploader(
        monkeypatch,
        FEED_UPLOAD_URL="http://192.168.1.200:5003",
        FEED_SOURCE_NAME="hyper",
    )

    status_path = tmp_path / "display_status.json"
    status_path.write_text("not json", encoding="utf-8")

    session = _FakeSession()
    result = uploader._upload_status(session, status_path)

    assert result is False
    assert session.calls == []


def test_upload_file_returns_false_on_error_status(monkeypatch, tmp_path):
    uploader = _reload_uploader(
        monkeypatch,
        FEED_UPLOAD_URL="http://192.168.1.200:5003",
        FEED_SOURCE_NAME="hyper",
    )

    screenshot = tmp_path / "date.png"
    screenshot.write_bytes(b"fake-png-bytes")

    session = _FakeSession(response=_FakeResponse(status_code=401, text="Unauthorized"))
    result = uploader._upload_file(session, screenshot)

    assert result is False


def test_upload_file_returns_false_on_request_exception(monkeypatch, tmp_path):
    uploader = _reload_uploader(
        monkeypatch,
        FEED_UPLOAD_URL="http://192.168.1.200:5003",
        FEED_SOURCE_NAME="hyper",
    )

    screenshot = tmp_path / "date.png"
    screenshot.write_bytes(b"fake-png-bytes")

    session = _FakeSession(exc=requests.ConnectionError("refused"))
    result = uploader._upload_file(session, screenshot)

    assert result is False


def test_run_loop_requires_upload_url(monkeypatch):
    uploader = _reload_uploader(monkeypatch)

    assert uploader.run_loop() == 1


def test_run_loop_uploads_changed_files_and_skips_unchanged(monkeypatch, tmp_path):
    uploader = _reload_uploader(
        monkeypatch,
        FEED_UPLOAD_URL="http://192.168.1.200:5003",
        FEED_UPLOAD_TOKEN="secret-token",
        FEED_SOURCE_NAME="hyper",
    )

    current_dir = tmp_path / "current"
    current_dir.mkdir()
    screenshot = current_dir / "date.png"
    screenshot.write_bytes(b"first")

    monkeypatch.setattr(
        uploader,
        "resolve_storage_paths",
        lambda: SimpleNamespace(current_screenshot_dir=current_dir),
    )

    upload_calls = []

    def fake_upload_file(session, path):
        upload_calls.append(path.name)
        return True

    monkeypatch.setattr(uploader, "_upload_file", fake_upload_file)

    # Stop the loop after the second iteration by setting the stop event
    # once _STOP_EVENT.wait is invoked a couple of times.
    wait_calls = {"count": 0}

    def fake_wait(_seconds):
        wait_calls["count"] += 1
        if wait_calls["count"] >= 2:
            uploader._STOP_EVENT.set()

    monkeypatch.setattr(uploader._STOP_EVENT, "wait", fake_wait)
    monkeypatch.setattr(uploader._STOP_EVENT, "is_set", lambda: wait_calls["count"] >= 2)

    uploader.run_loop()

    # Uploaded once on the first pass; unchanged mtime on the second pass
    # (before the stop event trips) means no duplicate upload.
    assert upload_calls == ["date.png"]


def test_run_loop_uploads_changed_display_status_and_skips_unchanged(monkeypatch, tmp_path):
    uploader = _reload_uploader(
        monkeypatch,
        FEED_UPLOAD_URL="http://192.168.1.200:5003",
        FEED_UPLOAD_TOKEN="secret-token",
        FEED_SOURCE_NAME="hyper",
    )

    current_dir = tmp_path / "current"
    current_dir.mkdir()
    status_path = current_dir / "display_status.json"
    status_path.write_text('{"screen_id": "date"}', encoding="utf-8")

    monkeypatch.setattr(
        uploader,
        "resolve_storage_paths",
        lambda: SimpleNamespace(current_screenshot_dir=current_dir),
    )
    monkeypatch.setattr(uploader, "_upload_file", lambda session, path: True)

    status_calls = []

    def fake_upload_status(session, path):
        status_calls.append(path.name)
        return True

    monkeypatch.setattr(uploader, "_upload_status", fake_upload_status)

    wait_calls = {"count": 0}

    def fake_wait(_seconds):
        wait_calls["count"] += 1
        if wait_calls["count"] >= 2:
            uploader._STOP_EVENT.set()

    monkeypatch.setattr(uploader._STOP_EVENT, "wait", fake_wait)
    monkeypatch.setattr(uploader._STOP_EVENT, "is_set", lambda: wait_calls["count"] >= 2)

    uploader.run_loop()

    # Uploaded once on the first pass; unchanged mtime on the second pass
    # (before the stop event trips) means no duplicate upload.
    assert status_calls == ["display_status.json"]


@pytest.mark.parametrize("request_error", [requests.ConnectionError("refused"), requests.Timeout()])
def test_run_loop_aborts_cycle_after_connection_failure(monkeypatch, tmp_path, request_error):
    uploader = _reload_uploader(
        monkeypatch,
        FEED_UPLOAD_URL="http://192.168.1.200:5003",
        FEED_SOURCE_NAME="hyper",
        FEED_UPLOAD_INTERVAL_SECONDS="1",
    )
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    for name in ("date.png", "inside.png", "weather.png"):
        (current_dir / name).write_bytes(b"image")

    monkeypatch.setattr(
        uploader,
        "resolve_storage_paths",
        lambda: SimpleNamespace(current_screenshot_dir=current_dir),
    )
    session = _FakeSession(exc=request_error)
    monkeypatch.setattr(uploader.requests, "Session", lambda: session)
    monkeypatch.setattr(uploader.random, "uniform", lambda *_args: 0.0)
    waits = []

    def stop_after_wait(seconds):
        waits.append(seconds)
        uploader._STOP_EVENT.set()

    uploader._STOP_EVENT.clear()
    monkeypatch.setattr(uploader._STOP_EVENT, "wait", stop_after_wait)

    uploader.run_loop()

    assert len(session.calls) == 1
    assert waits == [1.0]
