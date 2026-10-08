"""Phone-width layout checks for the config UI pages."""
from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("flask")

from tests import test_remote_playlists_ui as _ui

env = _ui.env
live_server = _ui.live_server
browser = _ui.browser

TEMPLATES = pathlib.Path(__file__).resolve().parent.parent / "templates"
PAGES = (
    "screen_config", "playlists", "clients", "client_wizard", "screenshots",
    "live", "stats", "login", "feed_index",
)
URLS = ("/", "/playlists", "/clients", "/clients/add", "/screenshots", "/live", "/stats")


@pytest.mark.parametrize("name", PAGES)
def test_page_is_mobile_ready(name):
    html = (TEMPLATES / f"{name}.html").read_text(encoding="utf-8")
    assert 'name="viewport"' in html
    assert '{% include "_mobile_style.html" %}' in html


def test_mobile_stylesheet_only_changes_narrow_layouts():
    css = (TEMPLATES / "_mobile_style.html").read_text(encoding="utf-8")
    # Every layout rule except the global safety nets lives inside a media query.
    outside = css
    while "@media" in outside:
        start = outside.index("@media")
        depth, i = 0, outside.index("{", start)
        end = i
        for end in range(i, len(outside)):
            depth += {"{": 1, "}": -1}.get(outside[end], 0)
            if depth == 0:
                break
        outside = outside[:start] + outside[end + 1:]
    assert "padding" not in outside and "display" not in outside


@pytest.mark.parametrize("url", URLS)
@pytest.mark.parametrize("size", [(390, 844), (320, 640), (820, 1180)])
def test_no_horizontal_scroll_on_phones(live_server, browser, url, size):
    page = browser.new_page(viewport={"width": size[0], "height": size[1]})
    page.goto(f"{live_server}{url}")
    page.wait_for_timeout(300)
    overflow = page.evaluate(
        "document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert overflow <= 0, f"{url} overflows {size[0]}px by {overflow}px"
    page.close()
