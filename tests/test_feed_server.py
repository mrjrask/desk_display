import importlib
import io
import threading
import time

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


def test_concurrent_uploads_for_same_screen_do_not_corrupt_file(monkeypatch, tmp_path):
    feed_server = _reload_feed_server(monkeypatch, tmp_path)
    client = feed_server.app.test_client()

    original_save = Image.Image.save

    def slow_save(self, fp, *args, **kwargs):
        time.sleep(0.05)
        return original_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", slow_save)

    def _colored_png(color) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
        return buffer.getvalue()

    statuses: list[int] = []

    def upload(color) -> None:
        response = client.post(
            "/api/feed/hyper/upload",
            headers={"Authorization": "Bearer secret-token"},
            data={"screen_id": "date", "file": (io.BytesIO(_colored_png(color)), "date.png")},
            content_type="multipart/form-data",
        )
        statuses.append(response.status_code)

    threads = [
        threading.Thread(target=upload, args=(color,))
        for color in [(255, 0, 0), (0, 255, 0)]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert statuses == [200, 200]

    target = feed_server._source_current_dir("hyper") / "date.png"
    image = Image.open(target)
    image.load()
    assert image.size == (16, 16)
    assert image.getpixel((0, 0)) in [(255, 0, 0), (0, 255, 0)]


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
