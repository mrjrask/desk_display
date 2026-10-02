"use strict";
const test = require("node:test");
const assert = require("node:assert");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { DeskDisplayClient, documentScreens, scheduleEntries } = require("../lib/client");

const PNG = Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), Buffer.from("fake image body")]);
const SHA = crypto.createHash("sha256").update(PNG).digest("hex");

function reply (status, body, headers = {}) {
  const data = Buffer.isBuffer(body) ? body : Buffer.from(body === undefined ? "" : JSON.stringify(body));
  return {
    status,
    headers: new Headers(headers),
    text: async () => data.toString(),
    arrayBuffer: async () => data.buffer.slice(data.byteOffset, data.byteOffset + data.length)
  };
}

function fakeServer ({ image = PNG } = {}) {
  const calls = [];
  let lease = 0;
  const fetchImpl = async (url, init) => {
    const p = new URL(url).pathname;
    const auth = (init.headers.Authorization || "").replace("Bearer ", "");
    const body = init.body ? JSON.parse(init.body) : null;
    calls.push({ method: init.method, path: p, auth, body });
    if (p === "/api/v1/register") {
      lease += 1;
      return reply(201, { client_credential: `lease-${lease}`, heartbeat_interval_seconds: 20, sync_interval_seconds: 10, assignment_state: "assigned" });
    }
    if (auth !== `lease-${lease}`) return reply(401, { error: "unauthorized" });
    if (p.endsWith("/config")) {
      return reply(200, { assigned_playlist: { playlist_id: "p", playlist_revision: "r1" }, playlist: { playlist_id: "p", playlist_revision: "r1", document: { screens: { date: 1, "news headlines": { frequency: 2, extra_seconds: 5 } } } } });
    }
    if (p.endsWith("/heartbeat")) return reply(200, { manifest_revision: "m-1", assignment_state: "assigned", assigned_playlist: { playlist_revision: "r1" } });
    if (p.endsWith("/manifest")) {
      if (init.headers["If-None-Match"] === "\"m-1\"") return reply(304);
      return reply(200, {
        type: "client_manifest",
        manifest_revision: "m-1",
        requested_screens: ["date"],
        artifacts: [{ screen_id: "date", role: "requested", artifact_type: "static_image", media_type: "image/png", url: `/api/v1/clients/mm/artifacts/${SHA}.png`, sha256: SHA, length: PNG.length }]
      });
    }
    if (p.includes("/artifacts/")) return reply(200, image);
    return reply(404, { error: "not_found" });
  };
  return { fetchImpl, calls, expire: () => { lease += 100; } };
}

function client (server) {
  const cacheDir = fs.mkdtempSync(path.join(os.tmpdir(), "mmdd-"));
  return new DeskDisplayClient({ serverUrl: "http://127.0.0.1:8765/", clientId: "mm", displayProfile: "hyperpixel4_square", enrollmentToken: "ddc_secret", cacheDir },
    { fetchImpl: server.fetchImpl, log: { warn () {} } });
}

test("full sync registers, sends demand and caches a verified image", async () => {
  const server = fakeServer();
  const c = client(server);
  assert.strictEqual(await c.fullSync(), true);
  assert.strictEqual(server.calls[0].auth, "ddc_secret");
  assert.deepStrictEqual(server.calls[0].body.capabilities.logical_width, 720);
  const hb = server.calls.find((x) => x.path.endsWith("/heartbeat"));
  assert.deepStrictEqual(hb.body.demand.required_screens, ["date", "news headlines"]);
  assert.deepStrictEqual(c.playableEntries().map((e) => e.screenId), ["date"]);
  assert.strictEqual(c.syncInterval(30), 10);
  assert.strictEqual(await c.fullSync(), false, "304 manifest means nothing changed");
  assert.strictEqual(await c.heartbeatOnly(), false);
});

test("a 401 drops the lease, re-registers and retries once", async () => {
  const server = fakeServer();
  const c = client(server);
  await c.fullSync();
  server.expire();
  await c.heartbeatOnly();
  const registrations = server.calls.filter((x) => x.path === "/api/v1/register");
  assert.strictEqual(registrations.length, 2);
});

test("an artifact with the wrong hash is never cached", async () => {
  const c = client(fakeServer({ image: Buffer.concat([PNG, Buffer.from("x")]) }));
  await c.fullSync();
  assert.deepStrictEqual(c.playableEntries(), []);
  assert.deepStrictEqual(fs.readdirSync(c.artifactDir), []);
});

test("schedule follows Config-page order and frequency", () => {
  const doc = {
    screens: { a: 1, b: { frequency: 2, extra_seconds: 3 }, c: 0, d: 1 },
    playlists: { g: { steps: [{ screen: "a" }] } },
    sequence: [{ playlist: "g" }]
  };
  assert.deepStrictEqual(scheduleEntries(doc).map((e) => e.screenId), ["b", "d", "a"]);
  assert.deepStrictEqual(documentScreens({ screens: { a: 1, b: { frequency: 0, alt: { screen: "z" } } } }), [["a"], ["z"]]);
});
