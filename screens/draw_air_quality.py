"""Render the outdoor air quality screen."""
from __future__ import annotations

import textwrap
from typing import Optional

from PIL import Image, ImageDraw

import config
from config import FONT_WEATHER_DETAILS_SMALL, FONT_WEATHER_DETAILS_SMALL_BOLD, FONT_WEATHER_DETAILS_TINY, HEIGHT, WIDTH, get_screen_background_color
from services.air_quality import AirQualityReport, fetch_air_quality
from utils import ScreenImage, log_call

CATEGORY_COLORS = {
    "Good": ((0, 170, 80), (0, 0, 0)),
    "Moderate": ((255, 214, 0), (0, 0, 0)),
    "Unhealthy for Sensitive Groups": ((255, 126, 0), (0, 0, 0)),
    "Unhealthy": ((220, 40, 40), (255, 255, 255)),
    "Very Unhealthy": ((143, 63, 151), (255, 255, 255)),
    "Hazardous": ((126, 0, 35), (255, 255, 255)),
    "Unknown": ((80, 80, 80), (255, 255, 255)),
}


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    if draw.textsize(text, font=font)[0] <= max_width:
        return text
    while text and draw.textsize(text + "…", font=font)[0] > max_width:
        text = text[:-1]
    return text + "…" if text else "…"


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int) -> list[str]:
    approx_chars = max(8, int(max_width / max(1, draw.textsize("M", font=font)[0])))
    lines: list[str] = []
    for line in textwrap.wrap(text, width=approx_chars):
        lines.append(_fit_text(draw, line, font, max_width))
        if len(lines) >= max_lines:
            break
    return lines or [text]


def _render_report(report: AirQualityReport) -> Image.Image:
    background = get_screen_background_color("air quality", (0, 0, 0))
    img = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)
    margin = max(4, WIDTH // 32)
    badge_fill, badge_text = CATEGORY_COLORS.get(report.aqi_category, CATEGORY_COLORS["Unknown"])

    title_font = FONT_WEATHER_DETAILS_SMALL_BOLD
    body_font = FONT_WEATHER_DETAILS_SMALL
    tiny_font = FONT_WEATHER_DETAILS_TINY
    title = "AIR QUALITY"
    draw.text((margin, margin), title, font=title_font, fill=(235, 235, 235))

    badge_top = margin + draw.textsize(title, font=title_font)[1] + max(4, HEIGHT // 40)
    badge_h = max(34, HEIGHT // 4)
    draw.rounded_rectangle((margin, badge_top, WIDTH - margin, badge_top + badge_h), radius=8, fill=badge_fill)

    aqi_text = "AQI --" if report.aqi_value is None else f"AQI {report.aqi_value}"
    cat_text = _fit_text(draw, report.aqi_category, title_font, WIDTH - margin * 4)
    aqi_w, aqi_h = draw.textsize(aqi_text, font=title_font)
    cat_w, cat_h = draw.textsize(cat_text, font=body_font)
    center_y = badge_top + badge_h // 2
    draw.text(((WIDTH - aqi_w) // 2, center_y - aqi_h - 1), aqi_text, font=title_font, fill=badge_text)
    draw.text(((WIDTH - cat_w) // 2, center_y + 2), cat_text, font=body_font, fill=badge_text)

    y = badge_top + badge_h + max(6, HEIGHT // 32)
    details: list[str] = []
    if report.primary_pollutant:
        details.append(f"Main: {report.primary_pollutant}")
    if report.pollen_level:
        details.append(f"Pollen: {report.pollen_level}")
    if report.pollutant_breakdown:
        pieces = [f"{name} {value:g}" for name, value in report.pollutant_breakdown[:3]]
        details.append(" • ".join(pieces))

    for detail in details[:3]:
        line = _fit_text(draw, detail, tiny_font, WIDTH - margin * 2)
        draw.text((margin, y), line, font=tiny_font, fill=(210, 220, 230))
        y += draw.textsize(line, font=tiny_font)[1] + 2

    advisory = report.advisory_text or "Check local conditions."
    lines = _wrap(draw, advisory, body_font, WIDTH - margin * 2, 2)
    rec_top = max(y + 2, HEIGHT - margin - len(lines) * (draw.textsize("Ag", font=body_font)[1] + 2))
    for line in lines:
        line_w = draw.textsize(line, font=body_font)[0]
        draw.text(((WIDTH - line_w) // 2, rec_top), line, font=body_font, fill=(255, 255, 255))
        rec_top += draw.textsize(line, font=body_font)[1] + 2
    return img


@log_call
def draw_air_quality_screen(display, report: Optional[AirQualityReport] = None, transition: bool = False):
    """Draw AQI, pollen (when available), and window guidance."""

    if report is None and config.ENABLE_AIR_QUALITY and config.AIR_QUALITY_LATITUDE is not None and config.AIR_QUALITY_LONGITUDE is not None:
        report = fetch_air_quality(
            config.AIR_QUALITY_LATITUDE,
            config.AIR_QUALITY_LONGITUDE,
            include_pollen=config.AIR_QUALITY_ENABLE_POLLEN,
        )
    if report is None:
        report = AirQualityReport(None, "Unknown", None, advisory_text="Air quality unavailable.")
    img = _render_report(report)
    return ScreenImage(img, displayed=False)
