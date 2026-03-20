"""NHL team standings screens."""
import config
from screens.mlb_team_standings import (
    LOGO_SZ,
    draw_standings_screen1 as _base_screen1,
    draw_standings_screen2 as _base_screen2,
    _format_int,
)
from config import FONT_STAND1_RANK, DISPLAY_SCALE, WIDTH, HEIGHT, is_hyperpixel_4_square_layout, is_hyperpixel_next_layout
from utils import log_call, clone_font

_IS_HYPERPIXEL_4_SQUARE = is_hyperpixel_4_square_layout()
_IS_HYPERPIXEL_LAYOUT = is_hyperpixel_next_layout()
_IS_HYPERPIXEL_4 = _IS_HYPERPIXEL_LAYOUT and not _IS_HYPERPIXEL_4_SQUARE
_IS_1080P_LAYOUT = config.is_hdmi_1080p_layout()
_IS_WAVESHARE_DISPLAY = config.get_display_profile_id() == "display_hat_mini"

NHL_LOGO_SZ = max(1, int(round(LOGO_SZ * 0.5)))
NHL_STAND1_LOGO_SZ = NHL_LOGO_SZ * 3 if _IS_HYPERPIXEL_LAYOUT else NHL_LOGO_SZ
if _IS_HYPERPIXEL_4:
    NHL_STAND1_LOGO_SZ = max(1, int(round(NHL_STAND1_LOGO_SZ * 1.5)))
if _IS_WAVESHARE_DISPLAY:
    NHL_LOGO_SZ = max(1, int(round(NHL_LOGO_SZ * 2.0)))
    NHL_STAND1_LOGO_SZ = max(1, int(round(NHL_STAND1_LOGO_SZ * 2.0)))

NHL_POINTS_FONT = clone_font(
    FONT_STAND1_RANK,
    max(1, int(round(getattr(FONT_STAND1_RANK, "size", 22) / max(1.0, DISPLAY_SCALE)))),
)


def _strip_pct_leading_zero(rec, *, precision=3):
    """Return a copy of the record with pct formatted without a leading zero."""

    if not rec:
        return rec

    league_record = rec.get("leagueRecord")
    if not isinstance(league_record, dict):
        return rec

    pct_val = league_record.get("pct")
    if pct_val in (None, ""):
        return rec

    try:
        pct_txt = f"{float(pct_val):.{precision}f}".lstrip("0")
    except Exception:
        pct_txt = str(pct_val).lstrip("0")

    updated_record = {**league_record, "pct": pct_txt}
    return {**rec, "leagueRecord": updated_record}


def _format_division_name(rec, default_name):
    name = None
    if isinstance(rec, dict):
        division_info = rec.get("division")
        if isinstance(division_info, dict):
            division_info = division_info.get("name") or division_info.get("abbreviation")

        name = (
            rec.get("divisionName")
            or rec.get("divisionAbbrev")
            or (division_info if isinstance(division_info, str) else None)
        )

    name = name or default_name
    if not name:
        return "division"

    cleaned = str(name).replace("Division", "").strip()
    return cleaned or str(name)


def _format_conference_name(rec):
    name = None
    if isinstance(rec, dict):
        conference_info = rec.get("conference")
        if isinstance(conference_info, dict):
            conference_info = conference_info.get("name") or conference_info.get(
                "abbreviation"
            )

        name = rec.get("conferenceName") or rec.get("conferenceAbbrev") or conference_info

    if not name:
        return "conference"

    lower_name = str(name).lower()
    if "conference" in lower_name:
        trimmed = str(name).replace("Conference", "").strip()
        if trimmed.lower() in ("western", "west"):
            return "the West"
        if trimmed.lower() in ("eastern", "east"):
            return "the East"
        return f"{trimmed} Conf." if trimmed else "conference"

    if "conf" in lower_name:
        return name

    name = str(name).replace("Conference", "").strip()
    return f"{name} Conf." if name else "conference"


@log_call
def draw_nhl_standings_screen1(display, rec, logo_path, division_name, *, logo_scale: float = 1.0, transition=False):
    """Wrap the generic standings screen for NHL teams (no GB/WC columns)."""

    rec_clean = _strip_pct_leading_zero(rec)

    division_rank = None
    conference_rank = None
    if isinstance(rec_clean, dict):
        division_rank = rec_clean.get("divisionRank")
        conference_rank = rec_clean.get("conferenceRank")

    division_display = _format_division_name(rec_clean, division_name)
    conference_display = _format_conference_name(rec_clean)
    rec_for_display = (
        {
            **rec_clean,
            "divisionName": division_display,
            "conferenceName": conference_display,
            "divisionRank": division_rank if division_rank not in (None, "") else "-",
            "conferenceRank": conference_rank if conference_rank not in (None, "") else "-",
        }
        if rec_clean
        else rec_clean
    )

    return _base_screen1(
        display,
        rec_for_display,
        logo_path,
        division_display,
        show_games_back=False,
        show_wild_card=False,
        points_font=NHL_POINTS_FONT,
        ot_label="OTL",
        points_label="points",
        conference_label="conference",
        show_conference_rank=True,
        record_details_fn=_format_nhl_record,
        logo_size=max(1, int(round(NHL_STAND1_LOGO_SZ * max(0.1, logo_scale)))),
        font_size_offset=(30 if _IS_1080P_LAYOUT else (12 if _IS_HYPERPIXEL_4 else (8 if _IS_HYPERPIXEL_4_SQUARE else 0))),
        transition=transition,
    )


def _nhl_record_details(rec, base_rec):
    pts_val = _format_int(rec.get("points"))
    return f"{base_rec} ({pts_val} pts)"


def _format_nhl_record(rec, _record_line):
    record = rec.get("leagueRecord", {}) if isinstance(rec, dict) else {}

    wins = _format_int(record.get("wins"))
    losses = _format_int(record.get("losses"))

    ties = record.get("ties")
    otl = record.get("ot")
    extra = ties if ties not in (None, "", "-", 0, "0") else otl

    parts = [wins, losses]
    if extra not in (None, "", "-", 0, "0"):
        parts.append(_format_int(extra))

    return "-".join(parts)


@log_call
def draw_nhl_standings_screen2(display, rec, logo_path, *, transition=False):
    """Customize standings screen 2 for NHL teams."""

    return _base_screen2(
        display,
        _strip_pct_leading_zero(rec),
        logo_path,
        record_details_fn=_nhl_record_details,
        split_order=("division", "conference", "home", "away"),
        show_streak=False,
        show_points=False,
        logo_size=NHL_LOGO_SZ,
        transition=transition,
    )

__all__ = ["draw_nhl_standings_screen1", "draw_nhl_standings_screen2"]
