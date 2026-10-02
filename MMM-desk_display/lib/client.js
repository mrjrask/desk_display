/* Desk Display remote-display protocol client (protocol v1).
 *
 * Speaks the same wire contract as display_client.py / remote_display/client_sync.py
 * (see docs/remote-display-protocol.md): register for a lease, fetch config and
 * manifest, heartbeat with demand, and download static PNG artifacts into a local
 * cache, verifying length and SHA-256 before use.
 *
 * Render packages (animations) are not played: the client registers with
 * supports_animation=false, so the server ships a static image per screen.
 * The date and nixie clocks are fetched live from a server that offers them
 * (`live_clock_faces`), since a still would show the time it was rendered.
 */
"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const { buildEntries, starterScreenIds } = require("./schedule");

const NETWORK_PROTOCOL_VERSION = 1;
const CLIENT_SOFTWARE_VERSION = "0.1";
const RENDER_PACKAGE_VERSIONS = [1];
const MAX_ARTIFACT_BYTES = 16 * 1024 * 1024;
const MIN_INTERVAL = 5;
const MAX_INTERVAL = 3600;
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

// Canonical logical sizes and colour modes, mirroring display_profiles.PROFILE_PRESETS.
const PROFILES = {
  display_hat_mini: [320, 240, "RGB"],
  adafruit_minipitft_114: [240, 135, "RGB"],
  hyperpixel4: [800, 480, "RGB"],
  hyperpixel4_square: [720, 720, "RGB"],
  waveshare_lcd_320x240: [320, 240, "RGB"],
  waveshare_oled_128x64: [128, 64, "1"],
  hdmi_1080p: [1920, 1080, "RGB"],
  fallback_hd: [1280, 720, "RGB"],
  fallback_default: [320, 240, "RGB"]
};

class SyncError extends Error {
  constructor (code, message, { status = null, retryAfter = null } = {}) {
    super(message);
    this.code = code;
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

function clampInterval (value, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return Math.min(MAX_INTERVAL, Math.max(MIN_INTERVAL, n));
}

function isEnabled (spec) {
  if (typeof spec === "boolean") return spec;
  if (typeof spec === "number") return spec > 0;
  if (spec && typeof spec === "object") {
    const f = spec.frequency === undefined ? 1 : spec.frequency;
    return typeof f === "number" && f > 0;
  }
  return false;
}

function frequencyOf (spec) {
  if (typeof spec === "boolean") return spec ? 1 : 0;
  if (typeof spec === "number") return Math.max(0, Math.floor(spec));
  if (spec && typeof spec === "object") {
    const f = Number(spec.frequency === undefined ? 1 : spec.frequency);
    return Number.isFinite(f) ? Math.max(0, Math.floor(f)) : 0;
  }
  return 0;
}

/* (required, alternates) screen IDs, as remote_display.playlist_store.document_screens. */
function documentScreens (document) {
  const required = new Set();
  const alternates = new Set();
  for (const [sid, spec] of Object.entries((document && document.screens) || {})) {
    if (isEnabled(spec)) required.add(sid);
    if (spec && typeof spec === "object" && spec.alt && typeof spec.alt === "object") {
      const alt = spec.alt.screen;
      for (const a of (typeof alt === "string" ? [alt] : alt || [])) alternates.add(a);
    }
  }
  for (const playlist of Object.values((document && document.playlists) || {})) {
    for (const step of (playlist && Array.isArray(playlist.steps) ? playlist.steps : [])) {
      if (step && typeof step.screen === "string") required.add(step.screen);
    }
  }
  return [[...required].sort(), [...alternates].filter((a) => !required.has(a)).sort()];
}

class DeskDisplayClient {
  constructor (options, { fetchImpl = globalThis.fetch, log = console } = {}) {
    const profile = PROFILES[options.displayProfile];
    if (!profile) throw new Error(`Unknown displayProfile ${options.displayProfile}`);
    if (!options.serverUrl) throw new Error("serverUrl is required");
    if (!options.clientId) throw new Error("clientId is required");
    this.serverUrl = options.serverUrl.replace(/\/+$/, "");
    this.clientId = options.clientId;
    this.profileId = options.displayProfile;
    [this.width, this.height, this.colorMode] = profile;
    this.enrollmentToken = options.enrollmentToken || "";
    this.timeoutMs = options.requestTimeoutMs || 15000;
    // DESK_DISPLAY_CONTENT_TIMEZONE on a desk_display client: hide-after times without an offset.
    this.contentTimeZone = options.contentTimeZone || "America/Chicago";
    this.cacheDir = path.join(options.cacheDir, this.clientId);
    this.artifactDir = path.join(this.cacheDir, "artifacts");
    fs.mkdirSync(this.artifactDir, { recursive: true });
    this.fetch = fetchImpl;
    this.log = log;
    this.credential = this._readFile("client_credential");
    this.state = this._readJson("state.json") || { playlist: null, manifest: null, lastSyncAt: null };
    this.advertised = {};
    this.liveClocks = [];
    this.unassigned = false;
    this.offeredRevision = undefined;
    this.currentScreen = null;
    this.recentErrors = new Map();
  }

  // ── persistence ───────────────────────────────────────────────────────────

  _readFile (name) {
    try { return fs.readFileSync(path.join(this.cacheDir, name), "utf8").trim() || null; } catch { return null; }
  }

  _readJson (name) {
    try { return JSON.parse(fs.readFileSync(path.join(this.cacheDir, name), "utf8")); } catch { return null; }
  }

  _writeAtomic (name, data, mode = 0o600) {
    const target = path.join(this.cacheDir, name);
    const tmp = `${target}.tmp`;
    fs.writeFileSync(tmp, data, { mode });
    fs.renameSync(tmp, target);
  }

  _saveCredential (value) {
    this.credential = value;
    if (value) this._writeAtomic("client_credential", value);
    else fs.rmSync(path.join(this.cacheDir, "client_credential"), { force: true });
  }

  _saveState () {
    this._writeAtomic("state.json", JSON.stringify(this.state));
  }

  // ── wire documents ────────────────────────────────────────────────────────

  capabilities () {
    return {
      type: "client_capabilities",
      version: 1,
      protocol_version: NETWORK_PROTOCOL_VERSION,
      client_software_version: CLIENT_SOFTWARE_VERSION,
      client_id: this.clientId,
      display_profile: this.profileId,
      logical_width: this.width,
      logical_height: this.height,
      image_formats: ["PNG"],
      color_modes: [this.colorMode],
      render_package_versions: RENDER_PACKAGE_VERSIONS,
      supports_animation: false,
      has_touch: false,
      buttons: [],
      hardware: { model: "MagicMirror", driver: "MMM-desk_display" }
    };
  }

  demand () {
    const playlist = this.state.playlist;
    // An unassigned client keeps playing its cache but reports no demand.
    if (this.unassigned || !playlist || !playlist.document || !playlist.playlist_revision) return null;
    const [required, alternates] = documentScreens(playlist.document);
    return {
      type: "client_demand",
      version: 1,
      client_id: this.clientId,
      playlist_revision: playlist.playlist_revision,
      required_screens: required,
      alternate_screens: alternates,
      touch_targets: [],
      package_capabilities: {
        render_package_versions: RENDER_PACKAGE_VERSIONS,
        image_formats: ["PNG"],
        supports_animation: false,
        max_package_bytes: MAX_ARTIFACT_BYTES
      },
      sync_interval_seconds: Math.round(this.syncInterval(30))
    };
  }

  status () {
    const now = Date.now();
    const manifest = this.state.manifest;
    const playlist = this.state.playlist;
    return {
      type: "client_status",
      version: 1,
      client_id: this.clientId,
      playback_state: Object.keys(this.images()).length ? "playing" : "starting",
      accepted_revisions: {
        manifest_revision: (manifest && manifest.manifest_revision) || null,
        playlist_revision: (playlist && playlist.playlist_revision) || null,
        config_revision: null
      },
      current_screen: this.currentScreen,
      current_playlist: (playlist && playlist.playlist_id) || null,
      last_sync_age_seconds: this.state.lastSyncAt ? (now - this.state.lastSyncAt) / 1000 : null,
      cache_age_seconds: null,
      physical_rotation: 0,
      recent_errors: [...this.recentErrors.values()].slice(-8).map((e) => ({
        code: e.code,
        message: e.message.slice(0, 200),
        count: e.count,
        last_seen_age_seconds: (now - e.at) / 1000
      }))
    };
  }

  noteError (err) {
    const code = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(err.code || "") ? err.code : "sync_failed";
    const prev = this.recentErrors.get(code);
    this.recentErrors.delete(code);
    this.recentErrors.set(code, { code, message: String(err.message || code), count: (prev ? prev.count : 0) + 1, at: Date.now() });
    while (this.recentErrors.size > 8) this.recentErrors.delete(this.recentErrors.keys().next().value);
  }

  // ── intervals ─────────────────────────────────────────────────────────────

  syncInterval (local) {
    const own = clampInterval(local, 30);
    const adv = this.advertised.sync_interval_seconds;
    return adv ? Math.min(own, clampInterval(adv, own)) : own;
  }

  heartbeatInterval (local) {
    const own = clampInterval(local, 60);
    const adv = this.advertised.heartbeat_interval_seconds;
    return adv ? Math.min(own, clampInterval(adv, own)) : own;
  }

  _absorb (payload) {
    if (!payload || typeof payload !== "object") return;
    for (const key of ["heartbeat_interval_seconds", "sync_interval_seconds"]) {
      if (typeof payload[key] === "number") this.advertised[key] = payload[key];
    }
    // Register, config and heartbeat responses carry the lease, and live
    // clocks with it (a manifest's configuration has no lease_expires_at).
    if ("lease_expires_at" in payload) {
      const faces = Array.isArray(payload.live_clock_faces) ? payload.live_clock_faces : [];
      this.liveClocks = faces.filter((f) => f === "date" || f === "nixie");
    }
    if (payload.assignment_state === "unassigned") this.unassigned = true;
    else if (payload.assignment_state === "assigned") this.unassigned = false;
    if (payload.configuration) this._absorb(payload.configuration);
  }

  // ── HTTP ──────────────────────────────────────────────────────────────────

  async _request (method, urlPath, { auth, body, headers = {}, raw = false } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    const h = { Accept: "application/json", ...headers };
    if (auth) h.Authorization = `Bearer ${auth}`;
    if (body !== undefined) h["Content-Type"] = "application/json";
    let res;
    try {
      res = await this.fetch(`${this.serverUrl}${urlPath}`, {
        method,
        headers: h,
        body: body === undefined ? undefined : JSON.stringify(body),
        redirect: "manual",
        signal: controller.signal
      });
      if (raw || res.status === 304) return { status: res.status, res };
      const text = await res.text();
      let json = null;
      try { json = text ? JSON.parse(text) : null; } catch { json = null; }
      return { status: res.status, json, res };
    } catch (err) {
      throw new SyncError("network_error", `${method} ${urlPath}: ${err.message}`);
    } finally {
      clearTimeout(timer);
    }
  }

  _fail (response, what) {
    const json = response.json || {};
    let retryAfter = Number(json.retry_after_seconds) || null;
    const header = response.res && response.res.headers && response.res.headers.get("retry-after");
    if (header) {
      const secs = Number(header);
      const fromHeader = Number.isFinite(secs) ? secs : (Date.parse(header) - Date.now()) / 1000;
      if (Number.isFinite(fromHeader)) retryAfter = Math.max(retryAfter || 0, fromHeader);
    }
    if (retryAfter) retryAfter = Math.min(3600, Math.max(0, retryAfter));
    return new SyncError(json.error || `http_${response.status}`,
      `${what} failed (${response.status}): ${json.message || json.error || "no detail"}`,
      { status: response.status, retryAfter });
  }

  async _clientRequest (method, suffix, opts = {}) {
    if (!this.credential) await this.register();
    const urlPath = `/api/v1/clients/${encodeURIComponent(this.clientId)}${suffix}`;
    let response = await this._request(method, urlPath, { ...opts, auth: this.credential });
    if (response.status === 401) {
      // The lease ended: drop it, register again and retry once.
      this._saveCredential(null);
      await this.register();
      response = await this._request(method, urlPath, { ...opts, auth: this.credential });
    }
    return response;
  }

  // ── protocol steps ────────────────────────────────────────────────────────

  async register () {
    if (!this.enrollmentToken) throw new SyncError("no_enrollment_token", "enrollmentToken is not configured");
    const body = { capabilities: this.capabilities() };
    const demand = this.demand();
    if (demand) body.demand = demand;
    if (this.credential) body.client_credential = this.credential;
    const response = await this._request("POST", "/api/v1/register", { auth: this.enrollmentToken, body });
    if (response.status !== 200 && response.status !== 201) throw this._fail(response, "registration");
    const payload = response.json || {};
    if (typeof payload.client_credential !== "string" || !payload.client_credential) {
      throw new SyncError("invalid_response", "registration returned no client credential");
    }
    this._saveCredential(payload.client_credential);
    this._absorb(payload);
    return payload;
  }

  async fetchConfig () {
    const response = await this._clientRequest("GET", "/config");
    if (response.status !== 200) throw this._fail(response, "config");
    const payload = response.json || {};
    this._absorb(payload);
    this.state.playlist = payload.playlist || null;
    const assigned = payload.assigned_playlist;
    this.offeredRevision = assigned && typeof assigned === "object" ? assigned.playlist_revision : null;
    return payload;
  }

  async heartbeat () {
    const body = { status: this.status() };
    const demand = this.demand();
    if (demand) body.demand = demand;
    const response = await this._clientRequest("POST", "/heartbeat", { body });
    if (response.status !== 200) throw this._fail(response, "heartbeat");
    this._absorb(response.json);
    return response.json || {};
  }

  async fetchManifest () {
    const current = this.state.manifest;
    const headers = current && current.manifest_revision ? { "If-None-Match": `"${current.manifest_revision}"` } : {};
    const response = await this._clientRequest("GET", "/manifest", { headers });
    if (response.status === 304) {
      // Unchanged, but retry any image an earlier pass failed to download.
      await this._downloadArtifacts(current, { onlyMissing: true });
      return false;
    }
    if (response.status !== 200) throw this._fail(response, "manifest");
    const manifest = response.json || {};
    if (manifest.type !== "client_manifest") throw new SyncError("invalid_response", "manifest has the wrong type");
    this._absorb(manifest);
    await this._downloadArtifacts(manifest);
    this.state.manifest = manifest;
    this._pruneArtifacts();
    return true;
  }

  _imageEntries (manifest) {
    const prefix = `/api/v1/clients/${encodeURIComponent(this.clientId)}/artifacts/`;
    return ((manifest && manifest.artifacts) || []).filter((a) =>
      a && a.artifact_type === "static_image" && a.media_type === "image/png" &&
      typeof a.url === "string" && a.url.startsWith(prefix) && /^[0-9a-f]{64}$/.test(a.sha256 || "") &&
      Number.isInteger(a.length) && a.length > 0 && a.length <= MAX_ARTIFACT_BYTES);
  }

  artifactFile (sha256) {
    return path.join(this.artifactDir, `${sha256}.png`);
  }

  _hasValid (entry) {
    try {
      const data = fs.readFileSync(this.artifactFile(entry.sha256));
      return data.length === entry.length && crypto.createHash("sha256").update(data).digest("hex") === entry.sha256;
    } catch { return false; }
  }

  async _downloadArtifacts (manifest, { onlyMissing = false } = {}) {
    for (const entry of this._imageEntries(manifest)) {
      if (onlyMissing ? fs.existsSync(this.artifactFile(entry.sha256)) : this._hasValid(entry)) continue;
      const suffix = entry.url.slice(`/api/v1/clients/${encodeURIComponent(this.clientId)}`.length);
      let response;
      try {
        response = await this._clientRequest("GET", suffix, { raw: true, headers: { Accept: "image/png" } });
      } catch (err) {
        // One failed download must not cost the rest of the manifest.
        this.noteError(err);
        continue;
      }
      if (response.status !== 200) {
        this.noteError(new SyncError("artifact_download_failed", `${entry.screen_id}: HTTP ${response.status}`));
        continue;
      }
      let data;
      try {
        data = Buffer.from(await response.res.arrayBuffer());
      } catch (err) {
        this.noteError(new SyncError("network_error", `${entry.screen_id}: ${err.message}`));
        continue;
      }
      const digest = crypto.createHash("sha256").update(data).digest("hex");
      const isPng = data.subarray(0, 8).equals(PNG_SIGNATURE);
      if (data.length !== entry.length || digest !== entry.sha256 || !isPng) {
        this.noteError(new SyncError("artifact_invalid", `${entry.screen_id}: length or hash mismatch`));
        continue;
      }
      const target = this.artifactFile(entry.sha256);
      fs.writeFileSync(`${target}.tmp`, data, { mode: 0o644 });
      fs.renameSync(`${target}.tmp`, target);
    }
  }

  _pruneArtifacts () {
    const keep = new Set(this._imageEntries(this.state.manifest).map((a) => `${a.sha256}.png`));
    for (const name of fs.readdirSync(this.artifactDir)) {
      if (!keep.has(name)) fs.rmSync(path.join(this.artifactDir, name), { force: true });
    }
  }

  /* One full pass: config, heartbeat (with demand), manifest and downloads.
   * Returns true when the playable content may have changed. */
  async fullSync () {
    const before = this.state.manifest && this.state.manifest.manifest_revision;
    const beforePlaylist = this.state.playlist && this.state.playlist.playlist_revision;
    await this.fetchConfig();
    await this.heartbeat();
    await this.fetchManifest();
    this.state.lastSyncAt = Date.now();
    this._saveState();
    const after = this.state.manifest && this.state.manifest.manifest_revision;
    const afterPlaylist = this.state.playlist && this.state.playlist.playlist_revision;
    return before !== after || beforePlaylist !== afterPlaylist;
  }

  /* Heartbeat only. Returns true when a full sync is needed. */
  async heartbeatOnly () {
    const wasUnassigned = this.unassigned;
    const payload = await this.heartbeat();
    this.unassigned = wasUnassigned; // compared below; the full sync absorbs it
    const manifest = this.state.manifest;
    if (payload.manifest_revision && (!manifest || payload.manifest_revision !== manifest.manifest_revision)) return true;
    if ((payload.assignment_state === "unassigned") !== this.unassigned) return true;
    const assigned = payload.assigned_playlist;
    return Boolean(assigned && typeof assigned === "object" && assigned.playlist_revision !== this.offeredRevision);
  }

  /* The date or nixie face at the current time, drawn by the server. A seed
   * keeps the date face's colours for one showing. Throws a SyncError. */
  async fetchClock (screenId, colorsSeed = null) {
    if (!this.liveClocks.includes(screenId)) throw new SyncError("no_live_clock", `the server draws no live ${screenId}`);
    const query = Number.isInteger(colorsSeed) && colorsSeed >= 0 ? `?colors=${colorsSeed}` : "";
    const response = await this._clientRequest("GET", `/clock/${encodeURIComponent(screenId)}.png${query}`,
      { raw: true, headers: { Accept: "image/png" } });
    if (response.status !== 200) throw new SyncError("live_clock_failed", `${screenId} clock: HTTP ${response.status}`, { status: response.status });
    const data = Buffer.from(await response.res.arrayBuffer());
    if (data.length > MAX_ARTIFACT_BYTES || !data.subarray(0, 8).equals(PNG_SIGNATURE)) {
      throw new SyncError("live_clock_invalid", `${screenId} clock is not a PNG`);
    }
    return data;
  }

  /* Verified images on disk for the screens this client requested, by screen ID. */
  images () {
    const manifest = this.state.manifest;
    const images = {};
    for (const entry of this._imageEntries(manifest)) {
      if (entry.role === "requested" && fs.existsSync(this.artifactFile(entry.sha256))) {
        images[entry.screen_id] = { sha256: entry.sha256, width: entry.width, height: entry.height,
          state: entry.state, generatedAt: entry.generated_at };
      }
    }
    return images;
  }

  /* What the browser needs to play: the schedule entries, the Starter
   * playlist and the images. Without a playlist document the manifest's
   * screens play once each per cycle. */
  playback () {
    const images = this.images();
    const doc = this.state.playlist && this.state.playlist.document;
    let entries = doc ? buildEntries(doc, { timeZone: this.contentTimeZone }) : [];
    if (!entries.length && this.state.manifest) {
      entries = (this.state.manifest.requested_screens || Object.keys(images)).map((screenId) => (
        { screenId, frequency: 1, extraSeconds: 0, hideAfter: null, alternate: null }));
    }
    return { entries, starter: doc ? starterScreenIds(doc) : [], images, liveClocks: [...this.liveClocks] };
  }
}

module.exports = { DeskDisplayClient, SyncError, PROFILES, documentScreens };
