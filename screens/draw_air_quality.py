"""Render the outdoor air quality screen."""
from __future__ import annotations

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



def _format_value(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "--"
    text = f"{value:.1f}" if abs(value - round(value)) >= 0.05 else f"{value:.0f}"
    return f"{text}{suffix}"


def _display_metrics(report: AirQualityReport) -> list[tuple[str, str]]:
    """Return the compact set of readings shown beneath the AQI badge."""

    return [
        ("Health", report.aqi_category),
        ("Pollutant", report.primary_pollutant or "--"),
        ("PM2.5", _format_value(report.us_aqi_pm2_5)),
        ("Advice", report.advisory_text or "Check local conditions."),
    ]


def _render_report(report: AirQualityReport) -> Image.Image:
    background = get_screen_background_color("air quality", (0, 0, 0))
    img = Image.new("RGB", (WIDTH, HEIGHT), background)
    draw = ImageDraw.Draw(img)
    margin = max(4, WIDTH // 32)
    badge_fill, badge_text = CATEGORY_COLORS.get(report.aqi_category, CATEGORY_COLORS["Unknown"])

    header_font = FONT_WEATHER_DETAILS_SMALL_BOLD
    value_font = FONT_WEATHER_DETAILS_SMALL
    label_font = FONT_WEATHER_DETAILS_TINY
    label_color = (165, 185, 205)
    value_color = (235, 242, 248)
    card_fill = (12, 28, 42)
    card_outline = (34, 70, 98)

    title = "AIR QUALITY"
    draw.text((margin, margin), title, font=header_font, fill=(235, 235, 235))
    title_h = draw.textsize(title, font=header_font)[1]

    badge_top = margin + title_h + max(4, HEIGHT // 40)
    badge_h = max(26, HEIGHT // 5)
    draw.rounded_rectangle((margin, badge_top, WIDTH - margin, badge_top + badge_h), radius=8, fill=badge_fill)

    aqi_text = "AQI --" if report.aqi_value is None else f"AQI {report.aqi_value}"
    cat_text = _fit_text(draw, report.aqi_category, value_font, WIDTH - margin * 4)
    aqi_w, aqi_h = draw.textsize(aqi_text, font=header_font)
    cat_w, cat_h = draw.textsize(cat_text, font=label_font)
    center_y = badge_top + badge_h // 2
    draw.text(((WIDTH - aqi_w) // 2, center_y - aqi_h - 1), aqi_text, font=header_font, fill=badge_text)
    draw.text(((WIDTH - cat_w) // 2, center_y + 2), cat_text, font=label_font, fill=badge_text)

    card_top = badge_top + badge_h + max(4, HEIGHT // 40)
    card_bottom = HEIGHT - margin
    draw.rounded_rectangle((margin, card_top, WIDTH - margin, card_bottom), radius=8, fill=card_fill, outline=card_outline)

    metrics = _display_metrics(report)

    row_h = max(1, (card_bottom - card_top - 8) // len(metrics))
    label_w = max(draw.textsize(label.upper(), font=label_font)[0] for label, _value in metrics) + 10
    content_x = margin + 6
    value_x = content_x + label_w
    value_max_w = WIDTH - margin - 6 - value_x

    for index, (label, value) in enumerate(metrics):
        y0 = card_top + 4 + index * row_h
        if index > 0:
            draw.line((margin + 6, y0, WIDTH - margin - 6, y0), fill=(26, 54, 76))
        label_h = draw.textsize(label, font=label_font)[1]
        value_h = draw.textsize(value, font=value_font)[1]
        label_y = y0 + (row_h - label_h) // 2
        value_y = y0 + (row_h - value_h) // 2
        draw.text((content_x, label_y), label.upper(), font=label_font, fill=label_color)
        value_text = _fit_text(draw, value, value_font, value_max_w)
        draw.text((value_x, value_y), value_text, font=value_font, fill=value_color)
    return img

@log_call
def draw_air_quality_screen(display, report: Optional[AirQualityReport] = None, transition: bool = False):
    """Draw current AQI, health category, pollutant, PM2.5 AQI, and recommendation."""

    if report is None and config.ENABLE_AIR_QUALITY and config.AIR_QUALITY_LATITUDE is not None and config.AIR_QUALITY_LONGITUDE is not None:
        report = fetch_air_quality(
            config.AIR_QUALITY_LATITUDE,
            config.AIR_QUALITY_LONGITUDE,
            api_key=config.AIRNOW_API_KEY,
            include_pollen=config.AIR_QUALITY_ENABLE_POLLEN,
        )
    if report is None:
        report = AirQualityReport(None, "Unknown", None, advisory_text="Air quality unavailable.")
    img = _render_report(report)
    return ScreenImage(img, displayed=False)
