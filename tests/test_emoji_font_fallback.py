import config


def test_load_emoji_font_uses_bitmap_fallback_for_macos_invalid_pixel(monkeypatch):
    mac_path = "/System/Library/Fonts/Apple Color Emoji.ttc"
    sentinel_font = object()

    monkeypatch.setattr(config, "_try_load_font", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(config.glob, "glob", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        config.os.path,
        "isfile",
        lambda path: path == mac_path,
    )

    def fake_truetype(path, _size):
        if path == mac_path:
            raise OSError("invalid pixel size")
        raise OSError("font missing")

    monkeypatch.setattr(config.ImageFont, "truetype", fake_truetype)

    attempted_sizes = []

    def fake_bitmap_loader(path, native_size, _scaled_size):
        assert path == mac_path
        attempted_sizes.append(native_size)
        if native_size == 160:
            return sentinel_font
        raise OSError("unsupported native size")

    monkeypatch.setattr(config, "_BitmapEmojiFont", fake_bitmap_loader)

    loaded = config._load_emoji_font(30)

    assert loaded is sentinel_font
    assert attempted_sizes
    assert 160 in attempted_sizes
