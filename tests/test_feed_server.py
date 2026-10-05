import importlib
import io
import os
import threading
import time

import pytest
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


def test_feed_source_page_greyscales_screenshots_when_heartbeat_is_stale(monkeypatch, tmp_path):
    feed_server = _reload_feed_server(monkeypatch, tmp_path)
    current_dir = tmp_path / "hyper" / "current"
    current_dir.mkdir(parents=True)
    (current_dir / "date.png").write_bytes(_png_bytes())
    monkeypatch.setattr(
        feed_server,
        "_load_source_display_status",
        lambda _source: {"screen_id": "date", "is_stale": True},
    )

    response = feed_server.app.test_client().get("/feed/hyper")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert '<main id="feed" class="is-stale">' in html
    assert "#feed.is-stale > img" in html
    assert 'feed.classList.toggle("is-stale", Boolean(status.is_stale));' in html


def test_feed_sources_api_reflects_screen_count_and_heartbeat(monkeypatch, tmp_path):
    feed_server = _reload_feed_server(monkeypatch, tmp_path)
    client = feed_server.app.test_client()

    upload_response = client.post(
        "/api/feed/hyper/upload",
        headers={"Authorization": "Bearer secret-token"},
        data={"screen_id": "date", "file": (io.BytesIO(_png_bytes()), "date.png")},
        content_type="multipart/form-data",
    )
    assert upload_response.status_code == 200

    response = client.get("/api/feed/sources")
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["sources"]) == 1
    source = payload["sources"][0]
    assert source["name"] == "hyper"
    assert source["screen_count"] == 1
    assert source["elapsed"] is not None
    assert "display_status" in source


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


def test_feed_source_page_sizes_column_to_screenshot_width(monkeypatch, tmp_path):
    feed_server = _reload_feed_server(monkeypatch, tmp_path)
    current_dir = tmp_path / "hyper" / "current"
    current_dir.mkdir(parents=True)
    (current_dir / "date.png").write_bytes(_png_bytes())

    html = feed_server.app.test_client().get("/feed/hyper").get_data(as_text=True)

    assert "width: var(--feed-width, auto);" in html
    assert "FeedWidth.watch(feed);" in html
    assert "window.resizeBy(delta, 0);" in html


@pytest.fixture
def chromium():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as playwright:
        kwargs = {}
        executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
        bundled = os.path.join(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""), "chromium")
        if not executable and os.environ.get("PLAYWRIGHT_BROWSERS_PATH") and os.path.isfile(bundled):
            executable = bundled
        if executable:
            kwargs["executable_path"] = executable
        try:
            instance = playwright.chromium.launch(**kwargs)
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"Chromium is not available: {exc}")
        yield instance
        instance.close()


def test_browser_feed_column_matches_screenshot_width(monkeypatch, tmp_path, chromium):
    from werkzeug.serving import make_server

    feed_server = _reload_feed_server(monkeypatch, tmp_path)
    current_dir = tmp_path / "hyper" / "current"
    current_dir.mkdir(parents=True)
    Image.new("RGB", (320, 240), (0, 0, 255)).save(current_dir / "date.png")

    server = make_server("127.0.0.1", 0, feed_server.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        page = chromium.new_page(viewport={"width": 1200, "height": 800})
        page.goto(f"http://127.0.0.1:{server.server_port}/feed/hyper")
        page.wait_for_function("document.querySelector('#feed img').complete")
        page.wait_for_function("document.body.getBoundingClientRect().width === 320")
        box = page.locator("#feed img").bounding_box()
        assert box["width"] == 320
        # Centered in the wider window.
        assert box["x"] == (1200 - 320) / 2

        # A narrower window still shrinks the screenshot to fit.
        page.set_viewport_size({"width": 200, "height": 800})
        assert page.locator("#feed img").bounding_box()["width"] == 200
    finally:
        server.shutdown()
