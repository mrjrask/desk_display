from tools.maintenance import render_all_screens
from PIL import Image
from utils import ScreenImage


def test_non_interactive_resolution_prompt_is_silent(monkeypatch):
    class _FakeStdin:
        closed = False

        def isatty(self) -> bool:
            return False

    monkeypatch.delenv("DISPLAY_WIDTH", raising=False)
    monkeypatch.delenv("DISPLAY_HEIGHT", raising=False)
    monkeypatch.delenv("DISPLAY_RESOLUTION", raising=False)
    monkeypatch.setattr(render_all_screens.sys, "stdin", _FakeStdin())
    monkeypatch.setattr(render_all_screens, "_read_resolution_selection_from_stdin", lambda: None)

    def _unexpected_menu_call() -> None:
        raise AssertionError("resolution menu should not print in non-interactive mode")

    monkeypatch.setattr(render_all_screens, "_print_resolution_menu", _unexpected_menu_call)

    render_all_screens._maybe_prompt_resolution_selection()


def test_resolution_cli_option_applies_dimensions(monkeypatch):
    applied = []

    def _record(token: str) -> bool:
        applied.append(token)
        return True

    monkeypatch.setattr(render_all_screens, "_apply_resolution_token", _record)
    monkeypatch.setattr(render_all_screens, "_render_all_screens_impl", lambda **_kwargs: 0)

    exit_code = render_all_screens.main(["--resolution", "1080p", "--no-archive", "--no-sync-screenshots"])

    assert exit_code == 0
    assert applied == ["1080p"]


def test_extract_image_adds_notification_border_for_led_override(monkeypatch):
    monkeypatch.setattr(render_all_screens.utils, "LED_INDICATOR_LEVEL", 1.0)
    base = Image.new("RGB", (6, 6), "black")
    result = ScreenImage(base, led_override=(0.0, 1.0, 0.0))

    extracted = render_all_screens._extract_image(result, render_all_screens.HeadlessDisplay())

    assert extracted is not None
    assert extracted.getpixel((0, 0)) == (0, 255, 0)
    assert extracted.getpixel((3, 3)) == (0, 0, 0)
