"""Render the outdoor air quality screen."""
from __future__ import annotations

from typing import Optional

from PIL import Image, ImageDraw

import config
from config import (
    FONT_WEATHER_DETAILS_SMALL,
    FONT_WEATHER_DETAILS_SMALL_BOLD,
    FONT_WEATHER_DETAILS_TINY,
    HEIGHT,
    WIDTH,
    get_screen_background_color,
)
from services.air_quality import AirQualityReport, fetch_air_quality
from utils import ScreenImage, log_call, wrap_text

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


def _wrap_advice_lines(
    draw: ImageDraw.ImageDraw, text: str, font, max_width: int, max_lines: int
) -> list[str]:
    """Word-wrap advisory text, only falling back to an ellipsis if it still
    doesn't fit within ``max_lines``."""

    lines = wrap_text(text, font, max_width) or [text]
    if len(lines) <= max_lines:
        return lines
    kept = lines[: max_lines - 1]
    remainder = " ".join(lines[max_lines - 1 :])
    kept.append(_fit_text(draw, remainder, font, max_width))
    return kept



def _format_value(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "--"
    text = f"{value:.1f}" if abs(value - round(value)) >= 0.05 else f"{value:.0f}"
    return f"{text}{suffix}"


def _display_metrics(report: AirQualityReport) -> list[tuple[str, str]]:
    """Return the compact set of readings shown beneath the AQI badge."""

    return [
        ("Top Pollutant", report.primary_pollutant or "--"),
        ("PM2.5", _format_value(report.us_aqi_pm2_5)),
        ("PM10", _format_value(report.us_aqi_pm10)),
        ("Ozone", _format_value(report.us_aqi_ozone)),
        ("Advice", report.advisory_text or "Check local conditions."),
    ]


def _chart_layout(
    value_ends: list[int],
    *,
    value_x: int,
    right_edge: int,
    chart_gap: int,
    chart_min_w: int,
) -> tuple[int, int, bool]:
    """Size aligned component charts from the room left after their values."""

    chart_x = max(value_ends, default=value_x) + chart_gap
    chart_w = right_edge - chart_x
    if chart_w >= chart_min_w:
        return chart_x, chart_w, True

    chart_x = right_edge - chart_min_w
    chart_w = chart_min_w
    return chart_x, chart_w, chart_x > value_x + chart_gap


def _component_history_points(
    report: AirQualityReport, component_index: int
) -> list[tuple[float, float]]:
    """Return timestamped samples for one displayed AQI component."""

    points = []
    for sample in report.component_history:
        if len(sample) != 4:
            continue
        timestamp, *values = sample
        value = values[component_index]
        if value is not None:
            points.append((timestamp, float(value)))
    return points


def _metric_value_max_width(
    label: str,
    *,
    chart_labels: set[str],
    charts_enabled: bool,
    chart_x: int,
    chart_gap: int,
    value_x: int,
    right_edge: int,
) -> int:
    """Reserve chart space only on rows that actually render a chart."""

    if charts_enabled and label in chart_labels:
        return chart_x - chart_gap - value_x
    return right_edge - value_x


def _draw_component_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    points: list[tuple[float, float]],
) -> None:
    """Draw a compact AQI history chart styled to match Weather 2."""

    x0, y0, x1, y1 = box
    if x1 - x0 < 12 or y1 - y0 < 8:
        return

    grid_color = (28, 64, 88)
    muted_color = (68, 105, 130)
    line_color = (115, 210, 255)
    draw.rounded_rectangle(box, radius=max(2, min(5, (y1 - y0) // 3)), outline=grid_color)
    inner_x0, inner_y0 = x0 + 2, y0 + 2
    inner_x1, inner_y1 = x1 - 2, y1 - 2
    chart_w = max(1, inner_x1 - inner_x0)
    chart_h = max(1, inner_y1 - inner_y0)
    mid_y = inner_y0 + chart_h // 2

    for tick in range(1, 4):
        tick_x = inner_x0 + int(round(chart_w * tick / 4))
        draw.line((tick_x, inner_y0, tick_x, inner_y1), fill=(43, 88, 116))
        draw.line((tick_x, inner_y1 - 2, tick_x, inner_y1), fill=muted_color)
    draw.line((inner_x0, mid_y, inner_x1, mid_y), fill=muted_color)
    if len(points) < 2:
        return

    start_time, end_time = points[0][0], points[-1][0]
    if end_time <= start_time:
        end_time = start_time + 600
    values = [value for _timestamp, value in points]
    low, high = min(values), max(values)
    if low == high:
        padding = max(1.0, abs(high) * 0.05)
        low, high = low - padding, high + padding

    coordinates = [
        (
            inner_x0 + int(round((timestamp - start_time) / (end_time - start_time) * chart_w)),
            inner_y1 - int(round((value - low) / (high - low) * chart_h)),
        )
        for timestamp, value in points
    ]
    draw.line(coordinates, fill=line_color, width=2 if y1 - y0 >= 14 else 1)
    for x, y in coordinates[-12:]:
        draw.point((x, y), fill=(245, 250, 255))


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

    label_w = (
        max(draw.textsize(label.upper(), font=label_font)[0] for label, _value in metrics) + 10
    )
    content_x = margin + 6
    value_x = content_x + label_w
    right_edge = WIDTH - margin - 6
    chart_gap = max(6, WIDTH // 80)
    chart_min_w = max(36, WIDTH // 8)
    chart_labels = {"PM2.5", "PM10", "Ozone"}
    value_ends = [
        value_x + draw.textsize(value, font=value_font)[0]
        for label, value in metrics
        if label in chart_labels
    ]
    chart_x, _chart_w, charts_enabled = _chart_layout(
        value_ends,
        value_x=value_x,
        right_edge=right_edge,
        chart_gap=chart_gap,
        chart_min_w=chart_min_w,
    )

    # Removing the redundant Health row leaves its height to be shared by the
    # three component rows.  Advice gets extra height so its recommendation
    # can wrap onto a second line instead of being truncated.
    row_weights = (3, 4, 4, 4, 5)
    row_space = card_bottom - card_top - 8
    total_weight = sum(row_weights)
    row_tops = [card_top + 4]
    consumed_weight = 0
    for weight in row_weights[:-1]:
        consumed_weight += weight
        row_tops.append(card_top + 4 + round(row_space * consumed_weight / total_weight))
    row_bottoms = row_tops[1:] + [card_top + 4 + row_space]

    for index, (label, value) in enumerate(metrics):
        y0, y1 = row_tops[index], row_bottoms[index]
        row_h = y1 - y0
        if index > 0:
            draw.line((margin + 6, y0, right_edge, y0), fill=(26, 54, 76))
        label_h = draw.textsize(label, font=label_font)[1]
        value_h = draw.textsize(value, font=value_font)[1]
        label_y = y0 + (row_h - label_h) // 2
        value_y = y0 + (row_h - value_h) // 2
        draw.text((content_x, label_y), label.upper(), font=label_font, fill=label_color)
        # Advice needs its full row width; unlike the component values it has
        # neither a chart nor a reason to align with the wider Top Pollutant
        # label column.
        row_value_x = value_x
        if label == "Advice":
            row_value_x = content_x + draw.textsize(label.upper(), font=label_font)[0] + 10
        available_width = _metric_value_max_width(
            label,
            chart_labels=chart_labels,
            charts_enabled=charts_enabled,
            chart_x=chart_x,
            chart_gap=chart_gap,
            value_x=row_value_x,
            right_edge=right_edge,
        )
        if label == "Advice":
            line_gap = 2
            max_lines = max(1, min(2, row_h // (value_h + line_gap)))
            lines = _wrap_advice_lines(draw, value, value_font, available_width, max_lines)
            block_h = len(lines) * value_h + (len(lines) - 1) * line_gap
            line_y = y0 + max(0, (row_h - block_h) // 2)
            for line in lines:
                draw.text((row_value_x, line_y), line, font=value_font, fill=value_color)
                line_y += value_h + line_gap
        else:
            value_text = _fit_text(draw, value, value_font, available_width)
            draw.text((row_value_x, value_y), value_text, font=value_font, fill=value_color)
        if charts_enabled and label in chart_labels:
            chart_h = max(8, min(row_h - 6, HEIGHT // 18))
            chart_y = y0 + (row_h - chart_h) // 2
            _draw_component_chart(
                draw,
                (chart_x, chart_y, right_edge, chart_y + chart_h),
                _component_history_points(report, ("PM2.5", "PM10", "Ozone").index(label)),
            )
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
