"""Quad-view mode that renders four screens simultaneously in a 2x2 grid."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PIL import Image, ImageDraw, ImageFont

from config import HEIGHT, WIDTH
from utils import ScreenImage

RenderFunc = Callable[[], Optional[Image.Image | ScreenImage]]


@dataclass
class _TileSpec:
    label: str
    render: RenderFunc


def _extract_image(rendered: Optional[Image.Image | ScreenImage]) -> Optional[Image.Image]:
    if isinstance(rendered, ScreenImage):
        return rendered.image
    if isinstance(rendered, Image.Image):
        return rendered
    return None


def _error_tile(size: tuple[int, int], label: str) -> Image.Image:
    img = Image.new("RGB", size, (20, 20, 20))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(65, 65, 65), width=1)
    draw.text((8, 8), label, fill=(220, 220, 220), font=font)
    draw.text((8, 22), "unavailable", fill=(170, 170, 170), font=font)
    return img


def draw_quad_screen(display, tiles: list[_TileSpec], *, transition: bool = False) -> ScreenImage:
    """Render a four-tile dashboard using the provided tile renderers."""

    frame = Image.new("RGB", (WIDTH, HEIGHT), "black")
    draw = ImageDraw.Draw(frame)

    cols = 2
    rows = 2
    tile_w = WIDTH // cols
    tile_h = HEIGHT // rows

    for index in range(cols * rows):
        row = index // cols
        col = index % cols
        x0 = col * tile_w
        y0 = row * tile_h
        x1 = WIDTH if col == cols - 1 else (x0 + tile_w)
        y1 = HEIGHT if row == rows - 1 else (y0 + tile_h)
        region_size = (x1 - x0, y1 - y0)

        tile = tiles[index] if index < len(tiles) else None
        if tile is None:
            tile_img = _error_tile(region_size, "empty")
        else:
            try:
                source = _extract_image(tile.render())
            except Exception:
                source = None
            tile_img = source.resize(region_size, Image.Resampling.LANCZOS) if source else _error_tile(region_size, tile.label)

        frame.paste(tile_img, (x0, y0))
        if x0 > 0:
            draw.line((x0, y0, x0, y1), fill=(70, 70, 70), width=1)
        if y0 > 0:
            draw.line((x0, y0, x1, y0), fill=(70, 70, 70), width=1)

    display.image(frame)
    if transition:
        display.show()

    return ScreenImage(frame, displayed=bool(transition))
