# Airline logos for desk_display (images/air/)

Drop the `air/` folder from this zip straight into your desk_display checkout as
`images/air/` (i.e. files end up at `images/air/UAL.png`, `images/air/AAL.png`,
etc.). That matches how `screens/draw_adsb_stats.py` looks logos up: it reads
the 3-letter ICAO code parsed from each aircraft's callsign
(`_airline_code()` in `services/adsb.py`, e.g. "UAL1234" -> "UAL") and looks
for `images/air/<code>.png` (case-insensitive; `.jpg`/`.jpeg` also work).

## Format

- 160x160 px, PNG, transparent background (RGBA) — square so it fits the
  tile's icon slot cleanly at any of the sizes the ADS-B screen renders at
  (it downsizes on the fly to whatever `icon_dim` the current layout needs,
  typically well under 100px, so 160px gives clean headroom without bloating
  the repo).
- Filenames are ICAO codes, uppercase, matching what `_airline_code()` parses
  from callsigns.

## The 20 codes included

Chicago Executive (PWK) is a GA/business-jet reliever field with no
scheduled airline service, so there's no "top airlines at PWK" list to draw
from — flights there show up on ADS-B as tail numbers, which the screen
already falls back to showing as plain text. The list below instead covers
the carriers you're actually likely to see overhead from a Chicago-area
receiver: the biggest scheduled passenger operators at O'Hare (a major
United/American hub) and Midway (Southwest's base), plus the heaviest cargo
operators.

**Passenger**
| Code | Airline |
|---|---|
| UAL | United Airlines |
| AAL | American Airlines |
| SWA | Southwest Airlines |
| DAL | Delta Air Lines |
| ASA | Alaska Airlines |
| JBU | JetBlue Airways |
| NKS | Spirit Airlines |
| FFT | Frontier Airlines |
| ACA | Air Canada |
| BAW | British Airways |
| DLH | Lufthansa |
| AFR | Air France |
| KLM | KLM Royal Dutch Airlines |
| UAE | Emirates |
| SKW | SkyWest Airlines (regional feeder for United Express/Delta Connection/Alaska SkyWest) |
| ENY | Envoy Air (American Eagle regional feeder) |

**Cargo**
| Code | Airline |
|---|---|
| FDX | FedEx Express |
| UPS | UPS Airlines |
| GTI | Atlas Air |
| CKS | Kalitta Air |

Not a hard science — "top 20" was picked by scheduled flight volume/hub
presence at ORD + MDW, not a ranked traffic count. Swap any code out easily:
drop a differently-named `<ICAO>.png` into `images/air/` and it'll pick up
automatically.

## Source & license

Logos are from [Jxck-S/airline-logos](https://github.com/Jxck-S/airline-logos)
(`flightaware_logos/`), an aggregation of airline branding scraped for flight-
tracking/identification use. Per that repo: "provided here for educational
and identification purposes only... considered Fair Use under copyright law
as it is non-commercial, transformative (aggregating for identification),
and does not impede the owners' ability to profit from their branding." Same
basis desk_display already uses for its Olympics country-flag icons. Logos
remain the property of their respective airlines.
