"""NBA team standings screens."""
from screens.mlb_team_standings import (
    LOGO_SZ,
    draw_standings_screen1 as _base_screen1,
    draw_standings_screen2 as _base_screen2,
)
from config import is_hyperpixel_4_square_layout, is_hyperpixel_next_layout
from utils import log_call

_IS_HYPERPIXEL_4_SQUARE = is_hyperpixel_4_square_layout()
_IS_HYPERPIXEL_LAYOUT = is_hyperpixel_next_layout()

NBA_LOGO_SZ = max(1, int(round(LOGO_SZ * 0.5)))
NBA_STAND1_LOGO_SZ = NBA_LOGO_SZ * 3 if _IS_HYPERPIXEL_LAYOUT else NBA_LOGO_SZ


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


def _format_record_with_pct(rec, *, precision=3):
    """Return a simple wins-losses record string with winning percentage."""

    league_record = rec.get("leagueRecord", {}) if isinstance(rec, dict) else {}
    wins = league_record.get("wins", "-")
    losses = league_record.get("losses", "-")
    ties = league_record.get("ties")

    if ties in (None, "", "-", 0, "0"):
        ot_val = league_record.get("ot")
        ties = ot_val if ot_val not in (None, "", "-", 0, "0") else None

    base_record = f"{wins}-{losses}"
    if ties not in (None, "", "-", 0, "0"):
        base_record = f"{base_record}-{ties}"

    pct_raw = league_record.get("pct", "-")
    try:
        pct_txt = f"{float(pct_raw):.{precision}f}".lstrip("0")
    except Exception:
        pct_txt = str(pct_raw).lstrip("0")

    return f"{base_record} ({pct_txt})"


@log_call
def draw_nba_standings_screen1(display, rec, logo_path, division_name, *, logo_scale: float = 1.0, transition=False):
    """Wrap the generic standings screen for NBA teams (no games-back row)."""
    return _base_screen1(
        display,
        _strip_pct_leading_zero(rec),
        logo_path,
        division_name,
        conference_label=None,
        place_gb_before_rank=True,
        show_games_back=False,
        record_details_fn=lambda record, base_line: _format_record_with_pct(
            record, precision=3
        ),
        show_pct=False,
        show_streak=True,
        logo_size=max(1, int(round(NBA_STAND1_LOGO_SZ * max(0.1, logo_scale)))),
        font_size_offset=4 if _IS_HYPERPIXEL_4_SQUARE else 0,
        transition=transition,
    )


@log_call
def draw_nba_standings_screen2(display, rec, logo_path, *, transition=False):
    """Customize standings screen 2 for NBA teams."""

    return _base_screen2(
        display,
        _strip_pct_leading_zero(rec),
        logo_path,
        logo_size=NBA_LOGO_SZ,
        pct_precision=3,
        transition=transition,
    )

__all__ = ["draw_nba_standings_screen1", "draw_nba_standings_screen2"]
