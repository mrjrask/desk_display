import datetime as dt

from screens.on_this_day import _build_sections, draw_on_this_day


class DummyDisplay:
    width = 320
    height = 240

    def __init__(self):
        self.frames = []

    def clear(self):
        pass

    def image(self, img):
        self.frames.append(img.copy())


def test_on_this_day_has_requested_curated_sections_for_july_6(monkeypatch):
    monkeypatch.setattr("screens.on_this_day._wiki_items", lambda *args, **kwargs: [])

    sections = _build_sections(dt.date(2026, 7, 6))

    assert "🇺🇸 American History" in sections
    assert "🏙️ Chicago History" in sections
    assert "🏟️ Sports History" in sections
    assert "🎂 Famous Birthdays" in sections
    assert "💾 Tech & Science" in sections


def test_on_this_day_renderer_produces_frame(monkeypatch):
    monkeypatch.setattr("screens.on_this_day._wiki_items", lambda *args, **kwargs: [])
    display = DummyDisplay()

    screen = draw_on_this_day(display, transition=False, today=dt.date(2026, 7, 6))

    assert screen.image.size == (320, 240)
    assert display.frames
