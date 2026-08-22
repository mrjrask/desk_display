# Airline logos

Used by the "Live Now (by airline)" ADS-B tile (`screens/draw_adsb_stats.py`).

Drop a `png` or `jpg` file named for the flight's ICAO airline code (the
leading letters of its callsign — e.g. `UAL123` -> `UAL`), matched
case-insensitively:

```
images/air/UAL.png   # United
images/air/SWA.jpg   # Southwest
```

An airline with no matching logo file falls back to showing its code as
plain text instead.
