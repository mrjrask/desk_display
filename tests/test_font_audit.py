import ast

from scripts.font_audit import FontCallScanner, build_known_font_sizes


def test_build_known_font_sizes_supports_annotated_assignments(tmp_path):
    (tmp_path / "config.py").write_text(
        "\n".join(
            [
                "BASE: int = 20",
                "FONT_TITLE: object = _load_font('DejaVuSans.ttf', size=BASE + 2)",
                "FONT_SUBTITLE: object = _load_emoji_font(size=18)",
                "FONT_ALIAS: object = FONT_TITLE",
            ]
        ),
        encoding="utf-8",
    )

    sizes = build_known_font_sizes(tmp_path)

    assert sizes["FONT_TITLE"].size == 22
    assert sizes["FONT_SUBTITLE"].size == 18
    assert sizes["FONT_ALIAS"].size == 22


def test_font_call_scanner_tracks_annassign_ints_for_truetype_size():
    source = "\n".join(
        [
            "from PIL import ImageFont",
            "def make_font():",
            "    point_size: int = 16",
            "    return ImageFont.truetype('DejaVuSans.ttf', point_size + 4)",
        ]
    )
    tree = ast.parse(source)

    scanner = FontCallScanner({})
    scanner.visit(tree)

    assert len(scanner.truetype_calls) == 1
    assert scanner.truetype_calls[0].resolved_size == 20
