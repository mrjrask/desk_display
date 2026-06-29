from PIL import Image

import utils
from utils import load_team_logo


def _write_logo(path):
    Image.new("RGBA", (8, 6), (255, 0, 0, 255)).save(path)


def test_load_team_logo_reuses_cached_image_without_reopening(monkeypatch, tmp_path):
    logo_path = tmp_path / "CHC.png"
    _write_logo(logo_path)
    utils._TEAM_LOGO_CACHE.clear()
    real_open = Image.open
    opened = []

    def tracked_open(path, *args, **kwargs):
        opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(utils.Image, "open", tracked_open)

    first = load_team_logo(str(tmp_path), "CHC", height=12)
    second = load_team_logo(str(tmp_path), "CHC", height=12)

    assert first is not None
    assert second is not None
    assert first is not second
    assert opened == [str(logo_path)]


def test_load_team_logo_cached_copy_protects_cached_image(monkeypatch, tmp_path):
    logo_path = tmp_path / "CHC.png"
    _write_logo(logo_path)
    utils._TEAM_LOGO_CACHE.clear()
    real_open = Image.open
    opened = []

    def tracked_open(path, *args, **kwargs):
        opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(utils.Image, "open", tracked_open)

    first = load_team_logo(str(tmp_path), "CHC", height=12)
    assert first is not None
    first.paste((0, 255, 0, 255), (0, 0, first.width, first.height))

    second = load_team_logo(str(tmp_path), "CHC", height=12)

    assert second is not None
    assert second.getpixel((0, 0)) == (255, 0, 0, 255)
    assert opened == [str(logo_path)]


def test_load_team_logo_caches_misses(monkeypatch, tmp_path):
    utils._TEAM_LOGO_CACHE.clear()
    opened = []

    def tracked_open(path, *args, **kwargs):  # pragma: no cover - should not be called
        opened.append(path)
        raise AssertionError("missing logo should not be opened")

    monkeypatch.setattr(utils.Image, "open", tracked_open)

    assert load_team_logo(str(tmp_path), "CHC", height=12) is None
    assert load_team_logo(str(tmp_path), "CHC", height=12) is None
    assert opened == []


def test_load_team_logo_continues_after_cached_corrupt_variant(monkeypatch, tmp_path):
    corrupt_logo_path = tmp_path / "chc.png"
    valid_logo_path = tmp_path / "CHC.png"
    corrupt_logo_path.write_bytes(b"not a real png")
    _write_logo(valid_logo_path)
    utils._TEAM_LOGO_CACHE.clear()
    real_open = Image.open
    opened = []

    def tracked_open(path, *args, **kwargs):
        opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(utils.Image, "open", tracked_open)

    first = load_team_logo(str(tmp_path), "chc", height=12)
    second = load_team_logo(str(tmp_path), "chc", height=12)

    assert first is not None
    assert second is not None
    assert first is not second
    assert opened == [str(corrupt_logo_path), str(valid_logo_path)]
