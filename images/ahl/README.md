# AHL logos

The Wolves scoreboard reuses the Hawks drawing helpers but temporarily points them at this
directory. Every abbreviation configured in `_AHL_TEAM_ABBR_OVERRIDES` needs a matching PNG.
The lookup checks `<UPPERCASE>.png` first and then `<lowercase>.png`; filenames with any other
case do not match. Prefer the uppercase form for new and renamed assets.

At minimum ship:

- `AHL.png` — the fallback crest used when a club-specific file is missing.
- `<WOLVES TRI>.png` — the Wolves logo, where the tri-code is `AHL_TEAM_TRICODE` (defaults to `CHI`).

Add more logo PNGs as the schedule feed reveals new opponent abbreviations.

If a team intentionally has no dedicated logo, add its abbreviation and a non-empty explanation
to `_AHL_FALLBACK_ONLY_ABBRS` in `data_fetch.py`. The asset-consistency test permits only those
documented exceptions to use `AHL.png`.
