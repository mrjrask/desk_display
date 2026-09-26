# Soak test, release gates and rollback

This is the runbook for Phase 20b: a hardware soak test of the server/client
release on the real devices, judged against measurable release gates, with
explicit rollback triggers. Everything it runs was prepared and tested in
Phase 20a, so it needs no changes on the devices.

The automated end-to-end suite (`tests/test_end_to_end.py`) must pass on the
release commit first. It checks, with fixture data and no network:

- server renders match the standalone display pixel for pixel on every
  profile, including every frame of scrolling and animated screens;
- one server serving several clients with shared and different playlists
  and profiles, one render per screen and profile, and independent playback;
- playlist edits reaching clients and being acknowledged;
- all four rotations showing the same logical picture, with taps mapped back;
- server and client restarts, a long outage with cache expiry and recovery,
  stale data, failed renders and a corrupt client cache;
- credential revocation, incompatible protocol and manifest versions, and
  that no provider credential ever reaches a client;
- a server backup restored with `python3 install_modes.py restore`.

## Topology

| Device | Role | Profile |
| --- | --- | --- |
| Server Pi | `combined`: render server plus its own panel | HyperPixel 4 Square (`hyperpixel4_square`) |
| Client 1 | `client` | HyperPixel 4 (`hyperpixel4`) |
| Client 2 | `client` | Display HAT Mini (`display_hat_mini`) |
| Client 3 | `client` | Waveshare (`waveshare_lcd_320x240` or `waveshare_oled_128x64`) |

Install each with `Installers/install.sh --mode combined` or `--mode client`
(see [OPERATIONS.md](../OPERATIONS.md#server-and-client-operations)), give
every client a playlist, and let everything run for an hour before starting
the clock.

## Recording

Run the sampler on every device for the whole soak. On the server it also
records the server's view of every client:

```bash
# Server (reads DESK_DISPLAY_SERVER_ADMIN_TOKEN from .env; never writes it)
nohup python3 soak.py sample --out soak/server.jsonl --hours 48 \
    --server-url http://127.0.0.1:8765 &

# Each client
nohup python3 soak.py sample --out soak/$(hostname).jsonl --hours 48 &
```

A sample is one JSON line a minute: each client's lease, playback state,
current screen, sync and cache age and error count; render and failure
counts; playlist delivery; and this device's process memory, CPU, open
files, cache sizes and free disk. Samples never contain a credential.

## Hardware checklist

While it runs, check by hand on each panel and note the result:

- [ ] Touch: tapping a quad tile on a HyperPixel expands it, and a tap returns.
- [ ] Buttons: skip and previous on the Display HAT Mini.
- [ ] Scrolling screens scroll smoothly and hold at the end.
- [ ] Dark hours dim or blank the backlight and restore it.
- [ ] Unplug the server's network for 30 minutes: clients keep playing and
      clocks keep time; they reconnect by themselves afterwards.
- [ ] Reboot one client: it plays from its cache before it reconnects.
- [ ] Power-cycle the server: clients never go blank.
- [ ] Stop a client for longer than its lease (`DESK_DISPLAY_CLIENT_LEASE_SECONDS`):
      the server marks it expired and it re-enrolls when started.

## Judging the run

Copy the logs to one place and run:

```bash
python3 soak.py gates soak/*.jsonl --clients server-panel,client-1,client-2,client-3
```

It prints `GO`, `NO-GO` or `ROLLBACK` with every gate and trigger, and exits
0, 1 or 3. `--json` prints the same as JSON; `--gates file.json` overrides
thresholds (for a shorter rehearsal, for example `{"min_hours": 2}`).

### Release gates

All must pass for a go.

| Setting | Default | Gate |
| --- | --- | --- |
| `min_hours` | 48 | The soak lasted at least this long |
| `max_gap_seconds` | 300 | No gap between samples on any device longer than this |
| `server_reachable_percent` | 99.5 | The server answered in this share of samples |
| `client_active_percent` | 99 | Each client held an active lease in this share of samples |
| `client_playing_percent` | 99 | Each client was showing content in this share of samples |
| `sync_age_p95_seconds` | 120 | 95th percentile of each client's time since its last sync |
| `render_failure_percent` | 1 | Failed renders as a share of all renders |
| `rss_growth_mb` | 64 | Memory growth of each process from the first to the last tenth of the run |
| `min_free_disk_percent` | 10 | Lowest free disk on every device |
| `cache_limits_mb` | `{"cache/client": 256}` | Largest size of each cache |
| `unplanned_restarts` | 0 | Process restarts not caused by a planned test step |

Planned restarts (the reboot and power-cycle checks above) change process
IDs. Run those steps outside the recorded window, or record them in a
separate log and judge the main log alone.

### Rollback triggers

Any one of these is an immediate `ROLLBACK`, during the soak or after release.

| Setting | Default | Trigger |
| --- | --- | --- |
| `trigger_client_down_minutes` | 10 | A client blank, on its diagnostic screen or disconnected this long while the server is up |
| `trigger_render_failing_minutes` | 15 | A screen failing to render continuously this long |
| `trigger_playlist_stuck_minutes` | 10 | A playlist change not acknowledged by its client this long |
| `trigger_rss_growth_mb` | 256 | A process growing by this much memory |
| `trigger_free_disk_percent` | 5 | Free disk falling to this share or below |

Also roll back, whatever the tool says, if a credential appears anywhere it
should not (a client's files, a log, the configuration UI), or if a device
cannot be brought back without re-imaging.

## Go, no-go and rollback

- **GO:** every gate passes, no trigger fired and the hardware checklist is
  complete. Publish the release notes (supported profiles, known
  limitations, migration steps, backup requirements and rollback
  instructions) and tag the release.
- **NO-GO:** fix what failed, then soak again from the start.
- **ROLLBACK:** return every device to the last good release. For a failed
  upgrade, restore the snapshot the upgrade took
  (`python3 install_modes.py restore .runtime/server/backups/upgrade-<time>`)
  and check out the previous commit; to return to the original standalone
  application, follow [Restoring v0.1](../OPERATIONS.md#restoring-v01).
