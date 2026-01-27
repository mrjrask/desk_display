import importlib
import json


def _write_style_config(path, payload):
    path.write_text(json.dumps(payload, indent=2))


def test_get_screen_font_applies_override(tmp_path, monkeypatch):
    style_path = tmp_path / "screens_style.json"
    payload = {
        "screens": {
            "NBA Scoreboard": {
                "fonts": {
                    "score": {"family": "DejaVuSans.ttf", "size": 12}
                }
            }
        }
    }
    _write_style_config(style_path, payload)
    monkeypatch.setenv("SCREENS_STYLE_PATH", str(style_path))

    import config

    module = importlib.reload(config)
    font = module.get_screen_font(
        "NBA Scoreboard",
        "score",
        base_font=module.FONT_TEAM_SPORTS,
        default_size=39,
    )
    assert getattr(font, "size", None) == 12


def test_get_screen_image_scale_uses_override(tmp_path, monkeypatch):
    style_path = tmp_path / "screens_style.json"
    payload = {
        "screens": {
            "NFL Scoreboard": {
                "images": {"team_logo": {"scale": 1.5}}
            }
        }
    }
    _write_style_config(style_path, payload)
    monkeypatch.setenv("SCREENS_STYLE_PATH", str(style_path))

    import config

    module = importlib.reload(config)
    scale = module.get_screen_image_scale("NFL Scoreboard", "team_logo", 1.0)
    assert scale == 1.5


def test_reload_style_config_refreshes_cache(tmp_path, monkeypatch):
    style_path = tmp_path / "screens_style.json"
    _write_style_config(style_path, {"screens": {}})
    monkeypatch.setenv("SCREENS_STYLE_PATH", str(style_path))

    import config

    module = importlib.reload(config)
    assert module.get_screen_image_scale("NBA Scoreboard", "team_logo", 1.0) == 1.0

    payload = {
        "screens": {
            "NBA Scoreboard": {
                "images": {"team_logo": {"scale": 0.8}}
            }
        }
    }
    _write_style_config(style_path, payload)
    module.reload_style_config()
    assert module.get_screen_image_scale("NBA Scoreboard", "team_logo", 1.0) == 0.8


def test_get_screen_background_color_uses_hex_override(tmp_path, monkeypatch):
    style_path = tmp_path / "screens_style.json"
    payload = {"screens": {"weather1": {"background": "#1a2b3c"}}}
    _write_style_config(style_path, payload)
    monkeypatch.setenv("SCREENS_STYLE_PATH", str(style_path))

    import config

    module = importlib.reload(config)
    color = module.get_screen_background_color("weather1", (0, 0, 0))
    assert color == (26, 43, 60)
