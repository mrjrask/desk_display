/* Playlist scheduling for MMM-desk_display, a port of desk_display's schedule.py.
 *
 * Shared by the node helper (CommonJS) and the browser module (global
 * DeskDisplaySchedule), so both read a playlist document the same way:
 *
 * - Screens play in Config-page order: ungrouped first, then playlists in
 *   sequence order.
 * - Frequency N plays on cycles 1, 1+N, 1+2N, …; frequency 0 never plays on
 *   its own but can still be an alternate.
 * - An alternate (`alt: {screen, frequency}`) replaces every frequency-th
 *   presentation of its base screen, round-robin over its screen list,
 *   except in cycle 1, which always shows the bases.
 * - `hide_after_enabled` / `hide_after_at` retire a screen at that time
 *   (in the content time zone when the value has no offset).
 * - Playback starts at the top of the playlist labelled "Starter".
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.DeskDisplaySchedule = api;
}(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const REPLACEMENT_ONLY_SCREENS = new Set(["cubs no game", "sox no game"]);
  const STARTER_LABEL = "starter";
  const CONTENT_TIME_ZONE = "America/Chicago";

  function toInt (value) {
    if (typeof value === "boolean") return value ? 1 : 0;
    const n = Number(value);
    return Number.isFinite(n) ? Math.trunc(n) : NaN;
  }

  /* Milliseconds since the epoch for a wall-clock time in *timeZone*. */
  /* Like Python's datetime(..., tzinfo=ZoneInfo(zone)) with fold=0: an
   * ambiguous time is its first occurrence, and a time skipped by a
   * spring-forward gap uses the offset from before the change. */
  function zonedTime (year, month, day, hour, minute, second, timeZone) {
    const format = new Intl.DateTimeFormat("en-US", {
      timeZone, hourCycle: "h23", year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit"
    });
    const shown = (at) => {
      const parts = {};
      for (const p of format.formatToParts(new Date(at))) parts[p.type] = Number(p.value);
      return Date.UTC(parts.year, parts.month - 1, parts.day, parts.hour, parts.minute, parts.second);
    };
    const offsetAt = (at) => shown(at) - at;
    const wall = Date.UTC(year, month - 1, day, hour, minute, second);
    const DAY = 86400000;
    const before = offsetAt(wall - DAY);
    const after = offsetAt(wall + DAY);
    if (shown(wall - before) === wall) return wall - before;
    if (shown(wall - after) === wall) return wall - after;
    return wall - before; // in the gap
  }

  /* `hide_after_at` as epoch milliseconds, or null. */
  function parseHideAfter (value, timeZone = CONTENT_TIME_ZONE) {
    if (typeof value !== "string" || !value.trim()) return null;
    const text = value.trim();
    if (/(Z|[+-]\d{2}:?\d{2})$/i.test(text)) {
      const at = Date.parse(text);
      return Number.isFinite(at) ? at : null;
    }
    const m = /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?)?$/.exec(text);
    if (!m) return null;
    try {
      return zonedTime(+m[1], +m[2], +m[3], +(m[4] || 0), +(m[5] || 0), +(m[6] || 0), timeZone);
    } catch {
      return null;
    }
  }

  /* Screen IDs in Config-page order, as schedule.build_scheduler orders them. */
  function orderedScreenIds (document) {
    const screens = (document && document.screens) || {};
    const playlists = (document && document.playlists) || {};
    const order = [];
    for (const item of (Array.isArray(document && document.sequence) ? document.sequence : [])) {
      const id = item && item.playlist;
      if (typeof id === "string" && id in playlists && !order.includes(id)) order.push(id);
    }
    for (const id of Object.keys(playlists)) if (!order.includes(id)) order.push(id);
    const assignment = {};
    for (const id of order) {
      const steps = playlists[id] && Array.isArray(playlists[id].steps) ? playlists[id].steps : [];
      for (const step of steps) {
        const sid = step && step.screen;
        if (typeof sid === "string" && sid in screens && !(sid in assignment)) assignment[sid] = id;
      }
    }
    const ordered = [];
    for (const group of ["", ...order]) {
      for (const sid of Object.keys(screens)) {
        if ((assignment[sid] || "") === group && !ordered.includes(sid)) ordered.push(sid);
      }
    }
    return ordered;
  }

  /* Schedule entries for a playlist document: the enabled base screens in
   * play order, each with its frequency, extra seconds, hide time and
   * alternates. Malformed entries are skipped rather than failing the whole
   * playlist (the server validates documents when they are saved). */
  function buildEntries (document, { timeZone = CONTENT_TIME_ZONE } = {}) {
    const screens = (document && document.screens) || {};
    const entries = [];
    for (const screenId of orderedScreenIds(document)) {
      if (REPLACEMENT_ONLY_SCREENS.has(screenId)) continue;
      const spec = screens[screenId];
      const isObject = spec !== null && typeof spec === "object";
      const frequency = toInt(isObject ? (spec.frequency === undefined ? 1 : spec.frequency) : spec);
      if (!(frequency > 0)) continue;
      let extraSeconds = 0;
      let hideAfter = null;
      let alternate = null;
      if (isObject) {
        extraSeconds = Math.max(0, toInt(spec.extra_seconds || 0) || 0);
        if (spec.hide_after_enabled) hideAfter = parseHideAfter(spec.hide_after_at, timeZone);
        const alt = spec.alt;
        if (alt && typeof alt === "object") {
          const ids = (typeof alt.screen === "string" ? [alt.screen] : Array.isArray(alt.screen) ? alt.screen : [])
            .filter((s) => typeof s === "string");
          const altFrequency = toInt(alt.frequency);
          if (ids.length && altFrequency > 0) alternate = { screenIds: ids, frequency: altFrequency, cursor: 0 };
        }
      }
      entries.push({ screenId, frequency, extraSeconds, hideAfter, alternate, presentations: 0 });
    }
    return entries;
  }

  /* Screen IDs of the playlist labelled "Starter", in step order. */
  function starterScreenIds (document) {
    const playlists = (document && document.playlists) || {};
    for (const playlist of Object.values(playlists)) {
      if (!playlist || typeof playlist.label !== "string") continue;
      if (playlist.label.trim().toLowerCase() !== STARTER_LABEL) continue;
      return (Array.isArray(playlist.steps) ? playlist.steps : [])
        .map((step) => step && step.screen)
        .filter((s) => typeof s === "string");
    }
    return [];
  }

  class Scheduler {
    constructor (entries) {
      this.entries = entries.map((e) => ({
        ...e,
        presentations: 0,
        alternate: e.alternate ? { ...e.alternate, cursor: 0 } : null
      }));
      this.cycle = 1;
      this.pending = null;
      this.extra = new Map();
      for (const e of this.entries) this.extra.set(e.screenId, Math.max(this.extra.get(e.screenId) || 0, e.extraSeconds));
    }

    /* Every screen this schedule can show, alternates included. */
    requestedIds () {
      const ids = new Set();
      for (const e of this.entries) {
        ids.add(e.screenId);
        if (e.alternate) e.alternate.screenIds.forEach((s) => ids.add(s));
      }
      return ids;
    }

    extraSecondsFor (screenId) {
      return this.extra.get(screenId) || 0;
    }

    _hidden (entry, now) {
      return entry.hideAfter !== null && now >= entry.hideAfter;
    }

    _queue (now) {
      this.pending = [];
      this.entries.forEach((entry, index) => {
        if (!this._hidden(entry, now) && (this.cycle - 1) % entry.frequency === 0) this.pending.push(index);
      });
    }

    _advance (now) {
      this.cycle += 1;
      this._queue(now);
      if (this.pending.length) return true;
      const active = this.entries.filter((e) => !this._hidden(e, now)).map((e) => e.frequency);
      if (!active.length) return false;
      this.cycle = Math.min(...active.map((f) => this.cycle + (f - ((this.cycle - 1) % f))));
      this._queue(now);
      return this.pending.length > 0;
    }

    _resolve (entry, isAvailable) {
      entry.presentations += 1;
      const alt = entry.alternate;
      if (this.cycle !== 1 && alt && entry.presentations % alt.frequency === 0) {
        for (let i = 0; i < alt.screenIds.length; i += 1) {
          const id = alt.screenIds[alt.cursor];
          alt.cursor = (alt.cursor + 1) % alt.screenIds.length;
          if (isAvailable(id)) return id;
        }
      }
      return isAvailable(entry.screenId) ? entry.screenId : null;
    }

    /* The next screen to show from one cycle, skipping unavailable ones; null
     * when the rest of the cycle had nothing (call again for the next one). */
    _nextFromCycle (isAvailable, now) {
      if (!this.entries.length) return null;
      if (this.pending === null) this._queue(now);
      if (!this.pending.length && !this._advance(now)) return null;
      while (this.pending.length) {
        const entry = this.entries[this.pending.shift()];
        if (this._hidden(entry, now)) continue;
        const id = this._resolve(entry, isAvailable);
        if (id !== null) return id;
      }
      return null;
    }

    /* The next screen, as ClientPlayer.next: one more cycle if this one is spent. */
    next (isAvailable, now = Date.now()) {
      const id = this._nextFromCycle(isAvailable, now);
      return id !== null ? id : this._nextFromCycle(isAvailable, now);
    }

    /* Begin cycle 1 at the first of *screenIds* that has a slot in it. */
    startAt (screenIds, now = Date.now()) {
      if (!this.entries.length || !screenIds || !screenIds.length) return false;
      if (this.pending === null) this._queue(now);
      for (const id of screenIds) {
        const position = this.pending.findIndex((i) => this.entries[i].screenId === id);
        if (position >= 0) {
          this.pending.splice(0, position);
          return true;
        }
      }
      return false;
    }

    /* Continue after *screenId*'s slot in the current cycle, if it has one. */
    seekAfter (screenId, now = Date.now()) {
      if (!this.entries.length) return false;
      if (this.pending === null) this._queue(now);
      const position = this.pending.findIndex((i) => this.entries[i].screenId === screenId);
      if (position < 0) return false;
      this.pending.splice(0, position + 1);
      return true;
    }
  }

  return { Scheduler, buildEntries, orderedScreenIds, parseHideAfter, starterScreenIds };
}));
