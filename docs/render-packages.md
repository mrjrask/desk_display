# Render packages and client playback

A render package tells a display client how to play a screen that moves,
without the client rendering any content itself. The server renders every
pixel into embedded images. The client only composes them over time and
applies its own clock. The wire format that delivers packages is in
[remote-display-protocol.md](remote-display-protocol.md).

## Screen classes

Every screen has a remote class (`rendering/screen_classes.py`) that decides
what the server ships for it:

| Class | Ships | Examples |
| --- | --- | --- |
| `static` | A still PNG | Weather, next-game and standings cards |
| `periodic` | A still PNG re-rendered every 60 s | Live games and `adsb live` |
| `scrolling_canvas` | Still + `scroll` package | Scoreboards, standings tables, On This Day |
| `ticker_overlay` | Still + `ticker` package | Screens with moving text lanes |
| `finite_animation` | Still + `animation` package | Intro animations, slides |
| `composite`, `interactive_focus` | Still + `composite` package | Quads; `interactive_focus` tiles open on tap |
| `client_timed` | Still + `clock` package | `date`, `nixie`: the client draws the time |
| `unsupported` | Nothing | `inside` (reads the standalone's local sensor) |

Every screen always has a still. A client that cannot play a package, or
has animation turned off, shows the still.

## Package document

Media type `application/vnd.desk-display.render-package+json`, schema
version 1 (`RENDER_PACKAGE_SCHEMA_VERSION`). Defined in
`remote_display/render_package.py`.

| Field | Meaning |
| --- | --- |
| `type` | `render_package` |
| `render_package_schema_version` | 1 |
| `screen_id`, `render_profile`, `width`, `height`, `color_mode` | The profile's canonical logical size and colour mode, exactly |
| `render_key_digest` | Ties the package to the artifact it belongs to |
| `classification` | The screen class above |
| `kind` | `scroll`, `ticker`, `animation`, `composite` or `clock` |
| `assets` | `{id: {media_type, width, height, color_mode, sha256, length, data}}`, base64 PNG, de-duplicated by hash |
| `<kind>` | Exactly one body, named after the kind |

All geometry is in logical coordinates. Physical rotation stays on the client.

On a 1-bit profile (`color_mode` `1`), the images the client composes into
moving frames stay in colour: the scroll canvas, the ticker base and strips,
and the composite base and tile frames are `RGB` assets. The client builds
each frame in colour and dithers it to 1-bit as it shows it, exactly as the
standalone display dithers each frame. Dithering a whole canvas once would
give scrolled frames a different pattern.

### Kinds

| Kind | Body rules |
| --- | --- |
| `scroll` | A canvas as wide as the screen and taller than it. `viewport` is the logical size, `step_px` is 1 to the screen height, `frame_seconds` is 0.001 to 5, pauses are at most 600 s, and `direction` is `up` or `down`. |
| `ticker` | A full-frame base image and 1 to 12 lanes. Each lane's strip is exactly the lane's height. |
| `animation` | Either `frames` (2 to 24 frames, `loops` 1 to 100) or a `slide`, not both. |
| `composite` | 1 to 4 tiles whose frames match the tile bounds. A tile's optional `focus_screen` makes it tappable. |
| `clock` | A layout with exactly `face`, `time_zone`, `time_format` (`12` or `24`), `show_ip` and `background_color`. |

### Limits

| Limit | Value |
| --- | --- |
| Package size | 16 MiB |
| Assets | 160 |
| Pixels per asset | 24 million |
| Animation frames | 24 |
| Frames per composite tile | 10 |
| Ticker lanes | 12 |
| Asset colour modes | `1`, `L`, `RGB`, `RGBA` |

The client validates every package before playing it (`validate_package`).
It checks the type and version, the profile and exact size, the screen and
render key, a known kind and class, a single body, every asset's PNG
signature, hash, length, size and mode, and the kind's rules above. A
package that fails is never played.

### What a clock package never contains

A clock package carries a face, a time zone and a format. The client draws
the time from its own clock, so the date and nixie screens keep ticking
while the server is unreachable. The package never contains the server's IP
address or its update state. `show_ip` is only a flag: the address, if any,
comes from the client itself.

## Playback

`playback/package_player.py` renders a frame as a pure function of the time
since the screen started, so a frame can be recomputed exactly after any
delay.

- `scroll`, `slide` and `frames` play once, then hold on the last frame for
  the screen's hold time. `ticker`, `composite` and `clock` run for the
  longer of their motion and the hold time.
- The picture is updated at the package's own pace: a scroll's
  `frame_seconds` (at least 0.01 s), 0.045 s for tickers, the composite's
  `frame_seconds` or an animation's shortest frame (at least 0.03 s), 1 s for
  the nixie clock and 5 s for other clocks, and 1/30 s for a slide.
- A slide picks left-to-right or right-to-left at random.

### Fallbacks

`remote_display/fallbacks.py` decides how each screen plays on a client, and
the configuration UI shows the same result as assignment warnings:

| Mode | When |
| --- | --- |
| `full` | The client supports what the screen needs |
| `still` | A moving screen on a client without animation. Clocks are exempt and always tick. |
| `unavailable` | An `unsupported` or unknown screen; the client skips it |

Also:

- Without touch, quad tiles do not expand.
- On a 1-bit display, colour-dependent screens (weather radar, air quality)
  get a warning.

At playback time:

- A missing, corrupt or unplayable package shows the still and logs `Render
  package for <screen> is not playable; showing its still`.
- An error mid-animation holds the last frame.
- A screen with nothing cached is skipped.

## Local interaction

With touch (`DESK_DISPLAY_CLIENT_TOUCH`: `auto` turns it on for HyperPixel
profiles):

- A tap on a tile of an interactive quad opens that tile's screen. The
  rotation does not advance, and the quad resumes after the tile's time or
  on the next tap.
- Elsewhere, the left half of the screen goes back and the right half skips.

Buttons skip (B, Y, right, next) or go back (A, X, left, previous). Taps
and focus changes stay on the client. The server learns only the current
screen and playback state from the heartbeat, and renders quad tiles in
advance as interaction dependencies for touch clients.

## Rotation

The server renders every artifact in the profile's canonical logical
orientation, and a render key never includes rotation. The client rotates
each frame just before its driver draws it, and maps touches back to logical
coordinates. `DISPLAY_ROTATION` (0, 90, 180 or 270) is the client's own
setting. With `DISPLAY_ROTATION_STRICT=1` and a kernel overlay that already
rotates, the client applies no second rotation. The heartbeat reports the
applied rotation for diagnosis only.

## Client cache

The cache lives in `cache/client/` (`DESK_DISPLAY_CLIENT_CACHE_DIR`):

| Path | Holds |
| --- | --- |
| `playlist/current.json`, `playlist/previous.json`, `override.json` | The assigned playlist, the last one, and any local override |
| `playback.json` | Where playback was, to resume after a restart |
| `manifests/0.json` to `2.json` | The active manifest and two before it |
| `artifacts/<sha256>.png\|json` | Stills and packages |
| `client_credential` | The lease credential (mode 600) |
| `staging/` | Downloads in progress |

It stays within `DESK_DISPLAY_CLIENT_CACHE_MAX_MB` (default 256) by evicting
least-recently-used artifacts. Nothing the three kept manifests reference is
evicted.

A new playlist and manifest become active together, and only when every
required screen is usable: its still, its package when the client plays
packages, and the quad tiles a touch client can open. Until then the client
keeps playing the previous content and logs `Not activating yet: ...`.

## Offline behavior

The client starts from its cache before it contacts the server, and keeps
playing while the server is unreachable. With nothing cached it shows a
diagnostic screen with its ID, the server host, its sync state and version,
but never a credential. States are "waiting for first sync", "server
unreachable" and "cached content unavailable".

`DESK_DISPLAY_OFFLINE_MAX_AGE_HOURS` (0 keeps content forever) makes a
disconnected client stop showing content older than that and show "offline;
cached content expired" instead. The client retries with jittered exponential
backoff, up to 300 s between tries. When a sync succeeds it resumes at the same place in the
rotation, or just after the screen it last showed.
