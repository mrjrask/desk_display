from PIL import Image

from scripts import render_screens
from utils import ScreenImage


def test_non_interactive_resolution_prompt_is_silent(monkeypatch):
    class _FakeStdin:
        closed = False

        def isatty(self) -> bool:
            return False

    monkeypatch.delenv("DISPLAY_WIDTH", raising=False)
    monkeypatch.delenv("DISPLAY_HEIGHT", raising=False)
    monkeypatch.delenv("DISPLAY_RESOLUTION", raising=False)
    monkeypatch.setattr(render_screens.sys, "stdin", _FakeStdin())
    monkeypatch.setattr(render_screens, "_read_resolution_selection_from_stdin", lambda: None)

    def _unexpected_menu_call() -> None:
        raise AssertionError("resolution menu should not print in non-interactive mode")

    monkeypatch.setattr(render_screens, "_print_resolution_menu", _unexpected_menu_call)

    render_screens._maybe_prompt_resolution_selection()


def test_resolution_cli_option_applies_dimensions(monkeypatch):
    applied = []

    def _record(token: str) -> bool:
        applied.append(token)
        return True

    monkeypatch.setattr(render_screens, "_apply_resolution_token", _record)
    monkeypatch.setattr(render_screens, "_render_all_screens_impl", lambda **_kwargs: 0)

    exit_code = render_screens.main(["--resolution", "1080p", "--no-archive", "--no-sync-screenshots"])

    assert exit_code == 0
    assert applied == ["1080p"]


def test_cli_defaults_to_rendering_all_screens():
    parser = render_screens._build_arg_parser()
    args = parser.parse_args([])
    assert args.ignore_schedule is True


def test_main_prompts_for_screenshot_webpage_when_sync_flag_omitted(monkeypatch):
    prompted_with = []
    captured = {}

    monkeypatch.setattr(render_screens, "_maybe_prompt_resolution_selection", lambda: None)
    monkeypatch.setattr(
        render_screens,
        "_maybe_prompt_screenshot_webpage_sync",
        lambda default: prompted_with.append(default) or False,
    )
    monkeypatch.setattr(
        render_screens,
        "_render_all_screens_impl",
        lambda **kwargs: captured.update(kwargs) or 0,
    )

    exit_code = render_screens.main(["--no-archive"])

    assert exit_code == 0
    assert prompted_with == [render_screens.config.ENABLE_SCREENSHOTS]
    assert captured["sync_screenshots"] is False


def test_main_does_not_prompt_for_screenshot_webpage_when_sync_flag_set(monkeypatch):
    monkeypatch.setattr(render_screens, "_maybe_prompt_resolution_selection", lambda: None)

    def _unexpected_prompt(_default: bool) -> bool:
        raise AssertionError("webpage sync prompt should not run with explicit sync flag")

    monkeypatch.setattr(render_screens, "_maybe_prompt_screenshot_webpage_sync", _unexpected_prompt)
    captured = {}
    monkeypatch.setattr(
        render_screens,
        "_render_all_screens_impl",
        lambda **kwargs: captured.update(kwargs) or 0,
    )

    exit_code = render_screens.main(["--no-archive", "--no-sync-screenshots"])

    assert exit_code == 0
    assert captured["sync_screenshots"] is False



def test_build_cache_includes_scoreboards_payload(monkeypatch):
    monkeypatch.setattr(render_screens.data_fetch, "fetch_weather", lambda: None)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_bears_standings", lambda: None)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_blackhawks_last_game", lambda: None)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_blackhawks_live_game", lambda: None)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_blackhawks_next_game", lambda: None)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_blackhawks_next_home_game", lambda: None)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_blackhawks_standings", lambda: None)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_wolves_games", dict)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_bulls_last_game", lambda: None)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_bulls_live_game", lambda: None)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_bulls_next_game", lambda: None)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_bulls_next_home_game", lambda: None)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_bulls_standings", lambda: None)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_cubs_games", dict)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_cubs_standings", lambda: None)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_sox_games", dict)
    monkeypatch.setattr(render_screens.data_fetch, "fetch_sox_standings", lambda: None)

    payload = {"scoreboards": {"nba": [{"id": "game-1"}], "nfl": []}}
    monkeypatch.setattr(
        render_screens.data_provider,
        "read_sports_payloads",
        lambda ttl_seconds=0: payload,
    )

    cache = render_screens.build_cache()

    assert cache["scoreboards"]["nba"] == [{"id": "game-1"}]


def test_extract_image_prefers_screenimage_screenshot_image():
    display = render_screens.HeadlessDisplay(320, 240)
    display_frame = Image.new("RGB", (320, 240), "black")
    screenshot_image = Image.new("RGB", (320, 900), "blue")

    extracted = render_screens._extract_image(
        ScreenImage(display_frame, displayed=True, screenshot_image=screenshot_image),
        display,
    )

    assert extracted is screenshot_image
    assert extracted.size == (320, 900)
