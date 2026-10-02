"use strict";
const test = require("node:test");
const assert = require("node:assert");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { DeskDisplayClient } = require("../lib/client");
const { colorCycle, scrollDone, scrollOffset, scrollSeconds, slideSeconds, slideX, brightColor } = require("../lib/motion");

const SIG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const png = (text) => Buffer.concat([SIG, Buffer.from(text)]);
const sha = (data) => crypto.createHash("sha256").update(data).digest("hex");

// Expected values recorded from playback/package_player.py and
// screens/draw_date_time.py with the same inputs.
const SCROLL = { canvasHeight: 2000, stepPx: 3, frameSeconds: 0.02, pauseStartSeconds: 1.5, pauseEndSeconds: 2, direction: "down" };

test("a scroll holds, steps down its canvas and holds, as on a desk_display client", () => {
  assert.deepStrictEqual([0, 1.5, 1.51, 2.0, 5, 9.99, 20].map((t) => scrollOffset(SCROLL, 720, t)), [0, 0, 0, 75, 525, 1272, 1280]);
  const up = { ...SCROLL, direction: "up" };
  assert.deepStrictEqual([0, 2.0, 20].map((t) => scrollOffset(up, 720, t)), [1280, 1205, 0]);
  assert.ok(Math.abs(scrollSeconds(SCROLL, 720) - 12.04) < 1e-9);
  assert.strictEqual(scrollDone(SCROLL, 720, 5), false);
  assert.strictEqual(scrollDone(SCROLL, 720, 20), true);
  assert.strictEqual(scrollDone(up, 720, 0), false);
});

test("a logo crosses the screen and then rests centred", () => {
  const slide = { spriteWidth: 200, speedPxPerSecond: 300 };
  assert.deepStrictEqual([0, 0.1, 1.5, 3.07, 3.1].map((t) => slideX(slide, 720, t)), [-200, -170, 250, null, null]);
  assert.deepStrictEqual([0, 0.1, 3.0].map((t) => slideX(slide, 720, t, "rtl")), [720, 690, -180]);
  assert.ok(Math.abs(slideSeconds(slide, 720) - 920 / 300) < 1e-9);
});

test("the date face cycles colours at each display's pace", () => {
  assert.deepStrictEqual(colorCycle("hyperpixel4_square", 4), { interval: 0.2, steps: 18 });
  assert.deepStrictEqual(colorCycle("hdmi_1080p", 4), { interval: 0.08, steps: 47 });
  assert.deepStrictEqual(colorCycle("display_hat_mini", 4), { interval: 0.12, steps: 10 });
  for (let i = 0; i < 50; i += 1) {
    const [r, g, b] = brightColor();
    assert.ok(0.2126 * r + 0.7152 * g + 0.0722 * b >= 160 && Math.min(r, g, b) >= 80 && Math.max(r, g, b) <= 255);
  }
});

function reply (status, body) {
  const data = Buffer.isBuffer(body) ? body : Buffer.from(JSON.stringify(body));
  return { status, headers: new Headers(), text: async () => data.toString(), arrayBuffer: async () => data.buffer.slice(data.byteOffset, data.byteOffset + data.length) };
}

function asset (data, width, height) {
  return { media_type: "image/png", width, height, color_mode: "RGB", sha256: sha(data), length: data.length, data: data.toString("base64") };
}

/* A server whose manifest has a scroll, a logo slide and a frame animation. */
function motionServer ({ adjustment } = {}) {
  const canvas = png("tall canvas");
  const sprite = png("logo");
  const packages = {
    scroll: { type: "render_package", width: 720, height: 720, kind: "scroll", assets: { a0: asset(canvas, 720, 2000) },
      scroll: { canvas: "a0", viewport: [720, 720], step_px: 3, frame_seconds: 0.02, pause_start_seconds: 1.5, pause_end_seconds: 2, direction: "down", vertical_speed_adjustment: 0 } },
    slide: { type: "render_package", width: 720, height: 720, kind: "animation", assets: { a0: asset(sprite, 200, 120) },
      animation: { slide: { sprite: "a0", y: 300, speed_px_per_second: 300, background: [0, 0, 0] } } },
    frames: { type: "render_package", width: 720, height: 720, kind: "animation", assets: {},
      animation: { frames: [], loops: 1 } }
  };
  const bodies = {};
  const artifacts = Object.entries(packages).map(([screen, pkg], i) => {
    const still = png(`still ${screen}`);
    const body = Buffer.from(JSON.stringify(pkg));
    bodies[sha(still)] = still;
    bodies[sha(body)] = body;
    return {
      screen_id: screen, role: "requested", artifact_type: "static_image", media_type: "image/png",
      url: `/api/v1/clients/mm/artifacts/${sha(still)}.png`, sha256: sha(still), length: still.length,
      package: { url: `/api/v1/clients/mm/artifacts/${sha(body)}.json`, sha256: sha(body), length: body.length,
        media_type: "application/vnd.desk-display.render-package+json", kind: pkg.kind, classification: "x", render_package_schema_version: 1 }
    };
  });
  const downloads = [];
  const configuration = adjustment === undefined ? {} : { vertical_speed_adjustment: adjustment };
  const fetchImpl = async (url) => {
    const p = new URL(url).pathname;
    if (p === "/api/v1/register") return reply(201, { client_credential: "lease", lease_expires_at: "2030-01-01T00:00:00Z", assignment_state: "assigned" });
    if (p.endsWith("/config")) return reply(200, { playlist: { playlist_id: "p", playlist_revision: "r1", document: { screens: { scroll: 1, slide: 1, frames: 1 } } } });
    if (p.endsWith("/heartbeat")) return reply(200, { manifest_revision: "m-1" });
    if (p.endsWith("/manifest")) return reply(200, { type: "client_manifest", manifest_revision: "m-1", configuration, requested_screens: Object.keys(packages), artifacts });
    const match = /artifacts\/([0-9a-f]{64})\./.exec(p);
    if (match && bodies[match[1]]) {
      downloads.push(p);
      return reply(200, bodies[match[1]]);
    }
    return reply(404, { error: "not_found" });
  };
  return { fetchImpl, downloads, canvas, sprite };
}

function client (server) {
  const cacheDir = fs.mkdtempSync(path.join(os.tmpdir(), "mmdd-motion-"));
  return new DeskDisplayClient({ serverUrl: "http://127.0.0.1:8765", clientId: "mm", displayProfile: "hyperpixel4_square", enrollmentToken: "ddc_x", cacheDir },
    { fetchImpl: server.fetchImpl, log: { warn () {} } });
}

test("scroll and logo-slide packages are kept for the browser; other animations stay stills", async () => {
  const server = motionServer();
  const c = client(server);
  await c.fullSync();
  const images = c.images();
  assert.deepStrictEqual(images.scroll.scroll, {
    sha256: sha(server.canvas), canvasHeight: 2000, stepPx: 3, frameSeconds: 0.02,
    pauseStartSeconds: 1.5, pauseEndSeconds: 2, direction: "down"
  });
  assert.deepStrictEqual(images.slide.slide, { sha256: sha(server.sprite), spriteWidth: 200, y: 300, speedPxPerSecond: 300, background: [0, 0, 0] });
  assert.strictEqual(images.frames.scroll, undefined);
  assert.strictEqual(images.frames.slide, undefined);
  assert.deepStrictEqual(fs.readFileSync(c.artifactFile(sha(server.canvas))), server.canvas);
  assert.strictEqual(c.recentErrors.size, 0);

  // Kept packages are not downloaded again, and survive pruning.
  const before = server.downloads.length;
  c.state.manifest = null;
  await c.fetchManifest();
  assert.strictEqual(server.downloads.length, before);
  assert.ok(c.images().scroll.scroll);
});

test("a display's own scroll adjustment re-paces the scroll", async () => {
  const c = client(motionServer({ adjustment: 1 }));
  await c.fullSync();
  // Paced by the server at 0 (×1), played at +1 (×2): half the frame time.
  assert.ok(Math.abs(c.images().scroll.scroll.frameSeconds - 0.01) < 1e-12);
});

test("a package that fails its hash leaves the screen on its still", async () => {
  const server = motionServer();
  const real = server.fetchImpl;
  server.fetchImpl = async (url, init) => {
    const res = await real(url, init);
    if (url.endsWith(".json")) return reply(200, Buffer.from("{\"tampered\": true}"));
    return res;
  };
  const c = client(server);
  await c.fullSync();
  assert.strictEqual(c.images().scroll.scroll, undefined);
  assert.ok(c.recentErrors.has("package_invalid"));
});
