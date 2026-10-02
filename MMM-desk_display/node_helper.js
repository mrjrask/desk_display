/* Node helper for MMM-desk_display: runs the desk_display sync loop and serves
 * verified artifacts to the browser from the local cache. */
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
    const route = `/${this.name}/${encodeURIComponent(client.clientId)}/:file`;
    this.expressApp.get(route, (req, res) => {
      const match = /^([0-9a-f]{64})\.png$/.exec(req.params.file);
      if (!match) return res.sendStatus(404);
      res.set("Cache-Control", "private, max-age=31536000, immutable");
      return res.sendFile(client.artifactFile(match[1]), (err) => { if (err && !res.headersSent) res.sendStatus(404); });
    });
  },

  async _tick (entry) {
    const { client, config } = entry;
    let delay;
    try {
      let changed = false;
      if (Date.now() >= entry.nextFullAt) {
        changed = await client.fullSync();
        entry.nextFullAt = Date.now() + client.syncInterval(config.syncInterval) * 1000;
      } else if (await client.heartbeatOnly()) {
        changed = await client.fullSync();
        entry.nextFullAt = Date.now() + client.syncInterval(config.syncInterval) * 1000;
      }
      entry.failures = 0;
      if (changed || !entry.lastPublished) this._publish(entry);
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
    entry.timer = setTimeout(() => this._tick(entry), delay * 1000);
  },

  _publish (entry) {
    const { client } = entry;
    const base = `/${this.name}/${encodeURIComponent(client.clientId)}`;
    const screens = client.playableEntries().map((e) => ({ ...e, url: `${base}/${e.sha256}.png` }));
    const key = JSON.stringify(screens);
    if (entry.lastPublished === key) return;
    entry.lastPublished = key;
    this.sendSocketNotification("DD_SCREENS", {
      clientId: client.clientId,
      width: client.width,
      height: client.height,
      screens
    });
  }
});
