/* Node helper for MMM-desk_display: runs the desk_display sync loop, serves
 * verified artifacts to the browser from the local cache, and relays live
 * clock faces from the server. */
"use strict";

const path = require("node:path");
const NodeHelper = require("node_helper");
const Log = require("logger");
const { DeskDisplayClient } = require("./lib/client");

const MAX_BACKOFF_SECONDS = 300;

module.exports = NodeHelper.create({
  start () {
    this.clients = new Map();
  },

  stop () {
    this.stopped = true; // a sync still in flight must not schedule another
    for (const entry of this.clients.values()) clearTimeout(entry.timer);
  },

  socketNotificationReceived (notification, payload) {
    if (notification === "DD_START") this._startClient(payload);
    else if (notification === "DD_SHOWING") {
      const entry = this.clients.get(payload.clientId);
      if (entry) entry.client.currentScreen = payload.screenId || null;
    }
  },

  _startClient (config) {
    const existing = this.clients.get(config.clientId);
    if (existing) {
      // A second browser connected: just resend what we have.
      this._publish(existing);
      return;
    }
    let client;
    try {
      client = new DeskDisplayClient({ ...config, cacheDir: path.join(__dirname, ".cache") }, { log: Log });
    } catch (err) {
      this.sendSocketNotification("DD_STATUS", { clientId: config.clientId, message: err.message, error: true });
      return;
    }
    const entry = { client, config, timer: null, failures: 0, nextFullAt: 0, lastPublished: null };
    this.clients.set(config.clientId, entry);
    if (!/^https:\/\//.test(client.serverUrl) && !/^http:\/\/(127\.0\.0\.1|localhost|\[::1\])(:|\/|$)/.test(client.serverUrl)) {
      Log.warn(`[MMM-desk_display] ${client.serverUrl} is plain HTTP beyond loopback; credentials travel unencrypted`);
    }
    this._registerRoute(client);
    // Play the cached manifest at once, before contacting the server.
    this._publish(entry);
    this._tick(entry);
  },

  _registerRoute (client) {
    const base = `/${this.name}/${encodeURIComponent(client.clientId)}`;
    // The date or nixie face now; the browser falls back to the cached still on any error.
    this.expressApp.get(`${base}/clock/:file`, async (req, res) => {
      res.set("Cache-Control", "no-store");
      const match = /^(date|nixie)\.png$/.exec(req.params.file);
      if (!match || !client.liveClocks.includes(match[1])) return res.sendStatus(404);
      const seed = /^\d{1,9}$/.test(req.query.colors || "") ? Number(req.query.colors) : null;
      try {
        res.type("png").send(await client.fetchClock(match[1], seed, { layers: req.query.layers === "1" }));
      } catch (err) {
        client.noteError(err);
        if (!res.headersSent) res.sendStatus(502);
      }
      return undefined;
    });
    // Images live under images/: before this route allowed dotfiles, the old
    // /<sha>.png route answered every image with a 404 marked cacheable for a
    // year, and browsers still hold those. Only a found image may be cached.
    this.expressApp.get(`${base}/images/:file`, (req, res) => {
      const match = /^([0-9a-f]{64})\.png$/.exec(req.params.file);
      if (!match) return res.set("Cache-Control", "no-store").sendStatus(404);
      // The cache lives under .cache/, which sendFile refuses unless dotfiles are allowed.
      return res.sendFile(client.artifactFile(match[1]), {
        dotfiles: "allow",
        headers: { "Cache-Control": "private, max-age=31536000, immutable" }
      }, (err) => { if (err && !res.headersSent) res.set("Cache-Control", "no-store").sendStatus(404); });
    });
  },

  async _tick (entry) {
    const { client, config } = entry;
    let delay;
    try {
      if (Date.now() >= entry.nextFullAt || await client.heartbeatOnly()) {
        await client.fullSync();
        entry.nextFullAt = Date.now() + client.syncInterval(config.syncInterval) * 1000;
      }
      entry.failures = 0;
      // Offered after every pass: _publish sends only when something the
      // browser uses changed, such as a newly downloaded image or the live
      // clocks the lease advertises.
      this._publish(entry);
      delay = Math.min(client.heartbeatInterval(config.heartbeatInterval), client.syncInterval(config.syncInterval));
    } catch (err) {
      entry.failures += 1;
      client.noteError(err);
      entry.nextFullAt = 0; // after a failure the next pass is a full sync
      const limit = Math.min(MAX_BACKOFF_SECONDS, 2 * 2 ** (entry.failures - 1));
      delay = Math.max(err.retryAfter || 0, limit * (0.5 + Math.random() / 2));
      Log.warn(`[MMM-desk_display] sync failed (${err.code || "error"}): ${err.message}; retrying in ${Math.round(delay)}s`);
      this.sendSocketNotification("DD_STATUS", { clientId: client.clientId, message: err.message, error: true });
      if (!entry.lastPublished) this._publish(entry);
    }
    if (!this.stopped) entry.timer = setTimeout(() => this._tick(entry), delay * 1000);
  },

  _publish (entry) {
    const { client } = entry;
    const base = `/${this.name}/${encodeURIComponent(client.clientId)}`;
    const playback = client.playback();
    for (const [screenId, image] of Object.entries(playback.images)) {
      image.url = `${base}/images/${image.sha256}.png`;
      if (image.scroll) image.scroll.url = `${base}/images/${image.scroll.sha256}.png`;
      if (image.slide) image.slide.url = `${base}/images/${image.slide.sha256}.png`;
      const url = (sha) => `${base}/images/${sha}.png`;
      if (image.frames) image.frames.urls = image.frames.sha256s.map(url);
      if (image.ticker) {
        image.ticker.baseUrl = url(image.ticker.base);
        for (const lane of image.ticker.lanes) lane.url = url(lane.strip);
      }
      if (image.composite) {
        image.composite.baseUrl = url(image.composite.base);
        for (const tile of image.composite.tiles) tile.urls = tile.frames.map(url);
      }
      if (playback.liveClocks.includes(screenId)) image.liveUrl = `${base}/clock/${encodeURIComponent(screenId)}.png`;
    }
    const key = JSON.stringify(playback);
    if (entry.lastPublished === key) return;
    entry.lastPublished = key;
    this.sendSocketNotification("DD_SCREENS", {
      clientId: client.clientId,
      width: client.width,
      height: client.height,
      ...playback
    });
  }
});
