from tools.maintenance import render_screens as render_all_screens


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


def test_cli_defaults_to_rendering_all_screens():
    parser = render_all_screens._build_arg_parser()
    args = parser.parse_args([])
    assert args.ignore_schedule is True
