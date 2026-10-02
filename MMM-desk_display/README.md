# MMM-desk_display

A [MagicMirror²](https://magicmirror.builders/) module that plays the screens
rendered by a [desk_display](https://github.com/mrjrask/desk_display) render
server. The mirror registers as one of the server's remote display clients,
just like `display_client.py` on a Pi panel: it gets its own lease, its own
assigned playlist, and shows up on the config UI's **Clients** page.

## How it works

The node helper speaks protocol v1 from
[`docs/remote-display-protocol.md`](../docs/remote-display-protocol.md):

1. `POST /api/v1/register` with the client's capabilities, using its
   provisioned `ddc_...` credential, to get a lease credential.
2. `GET /config` for the assigned playlist, `POST /heartbeat` with the
   playlist's screens as demand, and `GET /manifest` (with `If-None-Match`).
3. Downloads each screen's PNG from `/artifacts/<sha256>.png`, keeping it only
   when its length, SHA-256 and PNG signature match the manifest.
4. Repeats a full sync every `syncInterval` seconds, sends heartbeats in
   between, and syncs at once when a heartbeat reports a new manifest or
   playlist. A 401 drops the lease and registers again; failures back off
   (honouring `Retry-After`) while the mirror keeps playing its cache.

The browser plays the cached images with the same scheduler as a desk_display
client (`lib/schedule.js`, a port of `schedule.py`):

- Screens play in the playlist's own order and frequencies: a screen with
  frequency *N* plays on cycles 1, 1+*N*, 1+2*N*, …
- Alternates replace every *N*th showing of their base screen (never in the
  first cycle), and screens with a hide-after time stop at that time.
- Playback starts at the top of the playlist labelled **Starter**. A changed
  playlist continues after the screen on show.
- Each screen holds for `screenSeconds` plus its `extra_seconds`. A screen
  without an image yet (or withdrawn by the server, such as a live game that
  ended) is skipped.

The module registers with `supports_animation: false`, so the server sends a
static image for every screen; scrolling and animated render packages play as
their still frame.

The `date` and `nixie` clocks are the exception. A still would show the time
it was rendered, so while a clock is on show the module asks the server for
the face drawn now (`GET /api/v1/clients/<id>/clock/<screen>.png`): nixie
every second, the date face each minute. If the server is unreachable, or is
older than this feature, the clock shows its cached still.

## Install

```bash
cd ~/MagicMirror/modules
git clone https://github.com/mrjrask/desk_display.git
ln -s desk_display/MMM-desk_display MMM-desk_display
```

No `npm install` is needed: the module only uses Node's built-ins (Node 18 or
later, which MagicMirror² already requires).

## Provision the mirror on the server

On the render server, create a client for the mirror and pick a display
profile whose size suits the space it will fill on the mirror:

```bash
python3 -m remote_display.provisioning provision magicmirror --profile hyperpixel4_square \
    --server-url https://render.lan:8765
```

The render server listens only on 127.0.0.1 by default. For a mirror on
another machine, set `DESK_DISPLAY_SERVER_HOST=0.0.0.0` in the server's `.env`
(or put it behind a reverse proxy) and restart it.

Copy the `DESK_DISPLAY_CLIENT_TOKEN` value it prints (shown once) into
`enrollmentToken` below, then assign the client a playlist on the config UI's
**Playlists** page. A server using `DESK_DISPLAY_SERVER_ENROLLMENT=shared`
takes the shared `DESK_DISPLAY_SERVER_AUTH_TOKEN` instead.

## Configure

```js
{
  module: "MMM-desk_display",
  position: "top_right",
  config: {
    serverUrl: "https://render.lan:8765",
    clientId: "magicmirror",
    enrollmentToken: "ddc_...",
    displayProfile: "hyperpixel4_square",
    width: 360            // scale the 720×720 render down to 360 px wide
  }
}
```

| Option | Default | Meaning |
| --- | --- | --- |
| `serverUrl` | `http://127.0.0.1:8765` | The render server. Use `https://` for anything beyond loopback; plain HTTP elsewhere works but logs a warning. |
| `clientId` | `magicmirror` | The provisioned client ID. Several modules with different IDs can run side by side. |
| `enrollmentToken` | `""` | The client's provisioned `ddc_...` credential (or the shared server token). |
| `displayProfile` | `hyperpixel4_square` | Profile the server renders for: `display_hat_mini` (320×240), `adafruit_minipitft_114` (240×135), `hyperpixel4` (800×480), `hyperpixel4_square` (720×720), `waveshare_lcd_320x240`, `waveshare_oled_128x64`, `hdmi_1080p` (1920×1080), `fallback_hd` (1280×720), `fallback_default` (320×240). A static client must use the profile it was provisioned with. |
| `width`, `height` | profile size | CSS size of the frame (number of pixels or any CSS length). Give only `width` to keep the profile's aspect ratio. |
| `screenSeconds` | `4` | Base hold per screen, as desk_display's `SCREEN_DELAY`. |
| `fadeMs` | `400` | Cross-fade between screens. |
| `syncInterval`, `heartbeatInterval` | `30`, `60` | Seconds; capped by what the server advertises. |
| `requestTimeoutMs` | `15000` | Per-request timeout. |
| `showStatus` | `true` | Show connection status text until the first screens arrive. |

For a server with a private CA, start MagicMirror with
`NODE_EXTRA_CA_CERTS=/path/to/ca.pem`.

## Files

The lease credential (mode 600), the last playlist and manifest, and the
verified PNGs live in `MMM-desk_display/.cache/<clientId>/`. On restart the
mirror plays that cache before it reaches the server. Delete the folder to
start fresh.

## Tests

```bash
npm test
```
