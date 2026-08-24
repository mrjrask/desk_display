import importlib
import io
import os

from PIL import Image


def _reload_feed_server(monkeypatch, tmp_path, token="secret-token"):
    monkeypatch.setenv("FEED_STORAGE_DIR", str(tmp_path))
    if token is None:
        monkeypatch.delenv("FEED_UPLOAD_TOKEN", raising=False)
    else:
        monkeypatch.setenv("FEED_UPLOAD_TOKEN", token)
    module = importlib.import_module("feed_server")
    return importlib.reload(module)


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_sanitize_id_strips_unsafe_characters(monkeypatch, tmp_path):
    feed_server = _reload_feed_server(monkeypatch, tmp_path)

    assert feed_server._sanitize_id("hyper pi/../etc") == "hyper_pi--etc"
    assert feed_server._sanitize_id("") == "unknown"


def test_upload_requires_token_configured(monkeypatch, tmp_path):
    feed_server = _reload_feed_server(monkeypatch, tmp_path, token=None)
    client = feed_server.app.test_client()

    response = client.post(
        "/api/feed/hyper/upload",
        data={"screen_id": "date", "file": (io.BytesIO(_png_bytes()), "date.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 503


def test_upload_rejects_bad_token(monkeypatch, tmp_path):
    feed_server = _reload_feed_server(monkeypatch, tmp_path)
    client = feed_server.app.test_client()

    response = client.post(
        "/api/feed/hyper/upload",
        headers={"Authorization": "Bearer wrong-token"},
        data={"screen_id": "date", "file": (io.BytesIO(_png_bytes()), "date.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 401


def test_upload_rejects_non_image_payload(monkeypatch, tmp_path):
    feed_server = _reload_feed_server(monkeypatch, tmp_path)
    client = feed_server.app.test_client()

    response = client.post(
        "/api/feed/hyper/upload",
        headers={"Authorization": "Bearer secret-token"},
        data={"screen_id": "date", "file": (io.BytesIO(b"not an image"), "date.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400


def test_upload_then_feed_page_and_api_reflect_screenshot(monkeypatch, tmp_path):
    feed_server = _reload_feed_server(monkeypatch, tmp_path)
    client = feed_server.app.test_client()

    upload_response = client.post(
        "/api/feed/hyper/upload",
        headers={"Authorization": "Bearer secret-token"},
        data={"screen_id": "date", "file": (io.BytesIO(_png_bytes()), "date.png")},
        content_type="multipart/form-data",
    )
    assert upload_response.status_code == 200
    assert upload_response.get_json() == {"status": "ok", "source": "hyper", "screen_id": "date"}

    api_response = client.get("/api/feed/hyper/screenshots")
    payload = api_response.get_json()
    assert len(payload["screens"]) == 1
    assert payload["screens"][0]["id"] == "date"
    assert payload["screens"][0]["filename"] == "date.png"

    page_response = client.get("/feed/hyper")
    html = page_response.get_data(as_text=True)
    assert 'data-screen-id="date"' in html
    assert "/feed/hyper/file/date.png" in html

    file_response = client.get("/feed/hyper/file/date.png")
    assert file_response.status_code == 200
    assert file_response.content_type == "image/png"

    index_response = client.get("/")
    index_html = index_response.get_data(as_text=True)
    assert "hyper" in index_html
    assert "1 screen" in index_html


def test_feed_screenshot_file_blocks_path_traversal(monkeypatch, tmp_path):
    feed_server = _reload_feed_server(monkeypatch, tmp_path)
    client = feed_server.app.test_client()

    client.post(
        "/api/feed/hyper/upload",
        headers={"Authorization": "Bearer secret-token"},
        data={"screen_id": "date", "file": (io.BytesIO(_png_bytes()), "date.png")},
        content_type="multipart/form-data",
    )

    response = client.get("/feed/hyper/file/..%2F..%2Fsecret.png")
    assert response.status_code == 404


def test_upload_overwrites_previous_screenshot_for_same_screen(monkeypatch, tmp_path):
    feed_server = _reload_feed_server(monkeypatch, tmp_path)
    client = feed_server.app.test_client()

    for _ in range(2):
        response = client.post(
            "/api/feed/hyper/upload",
            headers={"Authorization": "Bearer secret-token"},
            data={"screen_id": "date", "file": (io.BytesIO(_png_bytes()), "date.png")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200

    current_dir = feed_server._source_current_dir("hyper")
    matches = list(current_dir.glob("date.*"))
    assert len(matches) == 1


def test_screenshots_ordered_like_large_screen_defaults_and_skip_missing(monkeypatch, tmp_path):
    feed_server = _reload_feed_server(monkeypatch, tmp_path)
    client = feed_server.app.test_client()

    # "nixie" precedes "weather1" in default_screens_large.json's sequence,
    # but upload them in the opposite order and confirm the page/API still
    # reflect the large-screen-defaults order rather than upload order. Skip
    # every other screen in between (e.g. "date") entirely -- no image was
    # ever uploaded for it, so it should not appear at all.
    for screen_id in ("weather1", "nixie"):
        response = client.post(
            "/api/feed/hyper/upload",
            headers={"Authorization": "Bearer secret-token"},
            data={"screen_id": screen_id, "file": (io.BytesIO(_png_bytes()), f"{screen_id}.png")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200

    payload = client.get("/api/feed/hyper/screenshots").get_json()
    ids = [screen["id"] for screen in payload["screens"]]
    assert ids == ["nixie", "weather1"]
    assert "date" not in ids

    html = client.get("/feed/hyper").get_data(as_text=True)
    assert html.index('data-screen-id="nixie"') < html.index('data-screen-id="weather1"')


def test_screenshots_dedup_to_one_entry_per_screen_id(monkeypatch, tmp_path):
    feed_server = _reload_feed_server(monkeypatch, tmp_path)
    current_dir = feed_server._source_current_dir("hyper")
    current_dir.mkdir(parents=True)

    # Simulate a leftover stale file alongside the current one for the same
    # screen id (e.g. a format change) -- only the newest should be shown.
    stale = current_dir / "date.jpg"
    stale.write_bytes(_png_bytes())
    fresh = current_dir / "date.png"
    fresh.write_bytes(_png_bytes())
    os.utime(stale, (1, 1))
    os.utime(fresh, (1000, 1000))

    entries = feed_server._build_source_screen_entries("hyper")
    assert len(entries) == 1
    assert entries[0]["filename"] == "date.png"
