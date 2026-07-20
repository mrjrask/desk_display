# Image asset size policy

Tracked runtime images should be stored near the largest size the display code needs so startup and draw paths do not repeatedly downscale large source files.

Canonical limits:

| Asset class | Paths | Max edge | Max file size |
| --- | --- | ---: | ---: |
| Team logos and country flags | `images/ahl/`, `images/nba/`, `images/nhl/`, `images/mlb/`, `images/nfl/`, `images/oly/` | 128 px | 75 KB |
| League, conference, event, and trophy logos | League/event files in the same folders, such as `NBA.png`, `nhl.png`, `AFC.png`, `NL.png`, `WC.png`, and `OLY.png` | 160 px | 100 KB |
| Weather icons | `images/weather/` | 128 px | 75 KB |
| Nixie digits | `images/nixie/` | 256 px | 75 KB |

Use `tools/adjust_image_assets.py` to generate optimized copies outside the repository, then visually review those copies before replacing tracked files. The default team-logo/flag max edge is 128 px; known league/event logos are capped at 160 px.

Run `python tools/check_image_assets.py` before committing image changes. The check reports tracked image assets that exceed the policy without rewriting files; pass `--fail-on-violation` when you want a non-zero exit for strict local or CI gates.
