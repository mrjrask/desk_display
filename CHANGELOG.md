# Changelog

All notable changes to Desk Display are documented in this file.

## Unreleased — server/client architecture

Desk Display can now run as a render server with any number of thin display
clients, or both on one machine, alongside the unchanged standalone install.
See the README's [deployment modes](README.md#deployment-modes) and
[OPERATIONS.md](OPERATIONS.md) for installing, upgrading and rolling back.

- Role-specific configuration (`.env.server.example`, `.env.client.example`)
  validated at startup, with secrets excluded from every client-facing payload.
- `display_server.py`: registration and leases, per-client credentials
  (provisioned by default), rate limits, manifests and content-addressed
  artifacts, and a render coordinator that renders each artifact once for
  every client that shares it.
- Server-managed playlists and assignments in the configuration UI, with
  per-client capability warnings, provisioning, rotation and revocation.
- `display_client.py`: offline-first playback from a local cache, render
  packages (static, animated and client-timed clocks), touch focus on quads,
  and physical rotation applied only at presentation.
- Migration: `scripts/migrate_standalone_config.py` moves a standalone rotation
  onto the server, and `scripts/convert_env.py` converts an existing `.env`.
- Installers for the server, client and combined modes, `scripts/upgrade.sh`,
  and mode-aware uninstall and cleanup.

## v0.1 — 2026-09-25

This release preserves the pre-server/client standalone Desk Display baseline.

- Commit: `9e193dc22ce0caa88a66493dafa42f90ae3b0db9`.
- Tag: `v0.1` is a lightweight tag (it points straight at the commit, with no
  tag message or signature) and has a hosted GitHub release. It is never moved
  or reused. [Restoring v0.1](OPERATIONS.md#restoring-v01) explains how to go
  back to it.

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
