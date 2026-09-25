# Changelog

All notable changes to Desk Display are documented in this file.

## v0.1 — 2026-09-25

This release preserves the pre-server/client standalone Desk Display baseline.

### Included

- A standalone, always-on dashboard application for Raspberry Pi, Linux, macOS,
  and Windows.
- Output profiles for Pimoroni Display HAT Mini, Adafruit miniPiTFT, Waveshare
  OLED/LCD HAT (A), HyperPixel/kernel displays, HDMI, SDL windows, direct
  framebuffers, and hardware-independent headless rendering.
- Configurable weather, date/time, indoor sensor, finance, travel, news, sports
  schedule, scoreboard, standings, playoff, ADS-B, and quad-layout screens.
- A browser-based configuration UI for screen rotation, playlists, layouts,
  import/export, diagnostics, and screenshot review.
- The standalone Feed service for publishing screenshot feeds uploaded by Desk
  Display devices.

### Known limitations

- Physical display, GPIO/button/touch, framebuffer, I2C sensor, and systemd
  installer behavior must be validated on the corresponding Raspberry Pi and
  display hardware; these checks cannot be completed in headless CI.
- Live weather, maps, finance, news, travel, sports, and Feed upload behavior
  depends on network availability, third-party services, and any required API
  credentials.
- The application remains a standalone deployment in this baseline. It does
  not include the later server/client architecture.
