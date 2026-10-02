"use strict";
const test = require("node:test");
const assert = require("node:assert");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const Module = require("node:module");

// MagicMirror provides these two modules to node helpers.
const STUBS = { node_helper: { create: (definition) => definition }, logger: { warn () {}, info () {}, log () {} } };
const resolve = Module._resolveFilename;
Module._resolveFilename = function (request, ...rest) {
  return request in STUBS ? request : resolve.call(this, request, ...rest);
};
for (const [name, exports] of Object.entries(STUBS)) {
  require.cache[name] = { id: name, filename: name, loaded: true, exports };
}

const PNG = Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), Buffer.from("helper image")]);
const SHA = crypto.createHash("sha256").update(PNG).digest("hex");
const CLIENT_ID = `test-helper-${process.pid}`;
const CACHE = path.join(__dirname, "..", ".cache", CLIENT_ID);

function reply (status, body) {
  const data = Buffer.isBuffer(body) ? body : Buffer.from(body === undefined ? "" : JSON.stringify(body));
  return { status, headers: new Headers(), text: async () => data.toString(), arrayBuffer: async () => data.buffer.slice(data.byteOffset, data.byteOffset + data.length) };
}

/* A server whose manifest matches the mirror's cache and that offers live clocks. */
function fakeFetch () {
  return async (url, init) => {
    const p = new URL(url).pathname;
    const lease = { lease_expires_at: "2030-01-01T00:00:00Z", live_clock_faces: ["date", "nixie"], heartbeat_interval_seconds: 60, sync_interval_seconds: 30, assignment_state: "assigned" };
    if (p === "/api/v1/register") return reply(201, { client_credential: "lease-1", ...lease });
    if (p.endsWith("/config")) return reply(200, { ...lease, assigned_playlist: { playlist_id: "p", playlist_revision: "r1" }, playlist: { playlist_id: "p", playlist_revision: "r1", document: { screens: { date: 1 } } } });
    if (p.endsWith("/heartbeat")) return reply(200, { ...lease, manifest_revision: "m-1" });
    if (p.endsWith("/manifest")) return reply(304);
    return reply(404, {});
  };
}

function freshHelper (sent) {
  delete require.cache[require.resolve("../node_helper.js")];
  const helper = require("../node_helper.js");
  const routes = {};
  Object.assign(helper, {
    name: "MMM-desk_display",
    expressApp: { get: (route, handler) => { routes[route] = handler; } },
    sendSocketNotification: (notification, payload) => sent.push([notification, payload])
  });
  helper.start();
  return { helper, routes };
}

function seedCache () {
  fs.rmSync(CACHE, { recursive: true, force: true });
  fs.mkdirSync(path.join(CACHE, "artifacts"), { recursive: true });
  fs.writeFileSync(path.join(CACHE, "artifacts", `${SHA}.png`), PNG);
  fs.writeFileSync(path.join(CACHE, "state.json"), JSON.stringify({
    playlist: { playlist_id: "p", playlist_revision: "r1", document: { screens: { date: 1 } } },
    manifest: {
      type: "client_manifest", manifest_revision: "m-1", requested_screens: ["date"],
      artifacts: [{ screen_id: "date", role: "requested", artifact_type: "static_image", media_type: "image/png", url: `/api/v1/clients/${CLIENT_ID}/artifacts/${SHA}.png`, sha256: SHA, length: PNG.length }]
    },
    lastSyncAt: null
  }));
}

test.after(() => fs.rmSync(CACHE, { recursive: true, force: true }));

test("live clocks reach the browser even when the cache already matches the server", async () => {
  seedCache();
  const realFetch = globalThis.fetch;
  globalThis.fetch = fakeFetch();
  const sent = [];
  const { helper } = freshHelper(sent);
  try {
    helper.socketNotificationReceived("DD_START", { serverUrl: "http://127.0.0.1:8765", clientId: CLIENT_ID, enrollmentToken: "ddc_x", displayProfile: "hdmi_1080p" });
    const first = sent.find(([n]) => n === "DD_SCREENS")[1];
    assert.strictEqual(first.images.date.liveUrl, undefined, "nothing is known about live clocks before the first sync");
    assert.strictEqual(first.images.date.url, `/MMM-desk_display/${CLIENT_ID}/images/${SHA}.png`);
    for (let i = 0; i < 50 && sent.filter(([n]) => n === "DD_SCREENS").length < 2; i += 1) await new Promise((r) => setTimeout(r, 10));
    const second = sent.filter(([n]) => n === "DD_SCREENS")[1];
    assert.ok(second, "a sync that changed no revision still republishes what the lease taught");
    assert.strictEqual(second[1].images.date.liveUrl, `/MMM-desk_display/${CLIENT_ID}/clock/date.png`);
  } finally {
    helper.stop();
    globalThis.fetch = realFetch;
  }
});

test("only a found image is cacheable", () => {
  seedCache();
  const { helper, routes } = freshHelper([]);
  helper._registerRoute({ clientId: CLIENT_ID, artifactFile: (sha) => path.join(CACHE, "artifacts", `${sha}.png`), liveClocks: [] });
  const handler = routes[`/MMM-desk_display/${CLIENT_ID}/images/:file`];
  assert.ok(handler, "images are served under images/");

  function call (file) {
    const res = { headers: {}, status: 200, sentFile: null, headersSent: false };
    res.set = (k, v) => { res.headers[k] = v; return res; };
    res.sendStatus = (code) => { res.status = code; res.headersSent = true; return res; };
    res.sendFile = (file, options, done) => {
      if (!fs.existsSync(file)) return done(new Error("ENOENT"));
      res.sentFile = file;
      Object.assign(res.headers, options.headers);
      res.headersSent = true;
      return done();
    };
    handler({ params: { file } }, res);
    return res;
  }

  const found = call(`${SHA}.png`);
  assert.strictEqual(found.sentFile, path.join(CACHE, "artifacts", `${SHA}.png`));
  assert.match(found.headers["Cache-Control"], /immutable/);
  const missing = call(`${"0".repeat(64)}.png`);
  assert.strictEqual(missing.status, 404);
  assert.strictEqual(missing.headers["Cache-Control"], "no-store");
  assert.strictEqual(call("../state.json").headers["Cache-Control"], "no-store");
});
