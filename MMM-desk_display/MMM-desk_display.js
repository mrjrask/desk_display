/* MagicMirror² module: MMM-desk_display
 * Plays screens rendered by a desk_display render server (display_server.py),
 * acting as one of its remote display clients. */
/* global Module, Log, DeskDisplaySchedule */
const DD_RETRY_MS = 5000;
const DD_CLOCK_WAIT_MS = 1500;  // show a clock's still if the live face is slower than this

Module.register("MMM-desk_display", {
  defaults: {
    serverUrl: "http://127.0.0.1:8765",
    clientId: "magicmirror",
    enrollmentToken: "",          // the provisioned ddc_... credential for this client
    displayProfile: "hyperpixel4_square",
    width: null,                  // CSS size of the frame; defaults to the profile size
    height: null,
    screenSeconds: 4,             // base hold per screen (desk_display's SCREEN_DELAY)
    fadeMs: 400,
    syncInterval: 30,             // capped by what the server advertises
    heartbeatInterval: 60,
    requestTimeoutMs: 15000,
    showStatus: true
  },

  getScripts () {
    return [this.file("lib/schedule.js")];
  },

  getStyles () {
    return ["MMM-desk_display.css"];
  },

  start () {
    this.images = {};
    this.liveClocks = [];
    this.scheduler = null;
    this.scheduleKey = null;
    this.profileSize = null;
    this.statusMessage = "Connecting to desk_display…";
    this.timer = null;
    this.clockTimer = null;
    this.current = null;
    this.sendSocketNotification("DD_START", {
      serverUrl: this.config.serverUrl,
      clientId: this.config.clientId,
      enrollmentToken: this.config.enrollmentToken,
      displayProfile: this.config.displayProfile,
      syncInterval: this.config.syncInterval,
      heartbeatInterval: this.config.heartbeatInterval,
      requestTimeoutMs: this.config.requestTimeoutMs
    });
  },

  socketNotificationReceived (notification, payload) {
    if (!payload || payload.clientId !== this.config.clientId) return;
    if (notification === "DD_SCREENS") {
      const hadScreens = Object.keys(this.images).length > 0;
      this.images = payload.images || {};
      this.liveClocks = payload.liveClocks || [];
      this.profileSize = [payload.width, payload.height];
      this.reschedule(payload.entries || [], payload.starter || []);
      const hasScreens = Object.keys(this.images).length > 0;
      this.statusMessage = hasScreens ? null : "Waiting for the server to render screens…";
      if (!hadScreens || !this.frame) this.updateDom(0);
      if (!this.timer) this.advance();
    } else if (notification === "DD_STATUS") {
      if (payload.error) Log.warn(`MMM-desk_display: ${payload.message}`);
      if (!Object.keys(this.images).length) {
        this.statusMessage = payload.message;
        this.updateDom(0);
      }
    }
  },

  /* A new or changed playlist restarts the schedule, as on a desk_display
   * client: at the top of the Starter playlist the first time, otherwise
   * after the screen on show. */
  reschedule (entries, starter) {
    const key = JSON.stringify([entries, starter]);
    if (key === this.scheduleKey) return;
    const first = this.scheduler === null;
    this.scheduleKey = key;
    this.scheduler = new DeskDisplaySchedule.Scheduler(entries);
    if (first) this.scheduler.startAt(starter);
    else if (this.current) this.scheduler.seekAfter(this.current);
  },

  advance () {
    clearTimeout(this.timer);
    this.timer = null;
    this.stopClock();
    if (!this.scheduler) return;
    const screenId = this.scheduler.next((id) => Boolean(this.images[id]));
    if (screenId === null) {
      // Nothing playable yet: look again shortly (new images restart us too).
      if (Object.keys(this.images).length) this.timer = setTimeout(() => this.advance(), DD_RETRY_MS);
      return;
    }
    this.show(screenId);
    const seconds = Math.max(1, Number(this.config.screenSeconds) + this.scheduler.extraSecondsFor(screenId));
    this.timer = setTimeout(() => this.advance(), seconds * 1000);
  },

  /* Put *src* on screen once it loads; *instant* replaces the picture
   * without a fade, *fallback* is tried if *src* fails, and *done(loaded)*
   * is called when the attempt ends. */
  place (src, alt, { instant = false, fallback = null, done = null } = {}) {
    if (!this.frame) return;
    const showing = this.showing;
    const img = document.createElement("img");
    img.alt = alt;
    if (instant) img.classList.add("dd-instant");
    img.onload = () => {
      if (showing !== this.showing) {
        img.remove(); // loaded after the next screen took over
        return;
      }
      this.loadedShowing = showing;
      const swap = () => {
        img.classList.add("dd-visible");
        const old = [...this.frame.querySelectorAll("img")].filter((i) => i !== img);
        setTimeout(() => old.forEach((i) => i.remove()), instant ? 0 : this.config.fadeMs + 50);
      };
      if (instant) swap();
      else requestAnimationFrame(swap);
      if (done) done(true);
    };
    img.onerror = () => {
      if (fallback) {
        const fallbackSrc = fallback;
        fallback = null;
        img.src = fallbackSrc;
      } else {
        img.remove();
        if (done) done(false);
      }
    };
    img.src = src;
    this.frame.appendChild(img);
  },

  show (screenId) {
    const image = this.images[screenId];
    this.current = screenId;
    this.showing = (this.showing || 0) + 1;
    this.sendSocketNotification("DD_SHOWING", { clientId: this.config.clientId, screenId });
    if (image.liveUrl) {
      // A clock: show it drawn now, keep it ticking, and fall back to the still.
      this.clockSeed = Math.floor(Math.random() * 1e9);
      const showing = this.showing;
      this.place(this.clockUrl(image), screenId, { fallback: image.url });
      this.scheduleClock(screenId, showing);
      setTimeout(() => {
        if (this.showing === showing && this.loadedShowing !== showing) this.place(image.url, screenId);
      }, DD_CLOCK_WAIT_MS);
    } else {
      this.place(image.url, screenId);
    }
  },

  clockUrl (image) {
    return `${image.liveUrl}?colors=${this.clockSeed}&t=${Date.now()}`;
  },

  /* Nixie shows seconds; the date face changes on the minute. */
  scheduleClock (screenId, showing) {
    const period = screenId === "nixie" ? 1000 : 60000;
    const delay = period - (Date.now() % period) + 20;
    this.clockTimer = setTimeout(() => this.tickClock(screenId, showing), delay);
  },

  tickClock (screenId, showing) {
    this.clockTimer = null;
    const image = this.images[screenId];
    const current = () => this.showing === showing && this.timer !== null;
    if (!current() || !image || !image.liveUrl || !this.frame) return;
    // Swapped in without a fade once loaded; a failed tick keeps the last picture.
    this.place(this.clockUrl(image), screenId, {
      instant: true,
      done: () => { if (current()) this.scheduleClock(screenId, showing); }
    });
  },

  stopClock () {
    clearTimeout(this.clockTimer);
    this.clockTimer = null;
  },

  getDom () {
    const wrapper = document.createElement("div");
    const [pw, ph] = this.profileSize || [null, null];
    const width = this.config.width || (pw ? `${pw}px` : null);
    const height = this.config.height || (ph ? `${ph}px` : null);
    const frame = document.createElement("div");
    frame.className = "dd-frame";
    frame.style.setProperty("--dd-fade", `${this.config.fadeMs}ms`);
    if (width) frame.style.width = typeof width === "number" ? `${width}px` : width;
    if (height) frame.style.height = typeof height === "number" ? `${height}px` : height;
    if (pw && ph && this.config.width && !this.config.height) {
      frame.style.height = "auto";
      frame.style.aspectRatio = `${pw} / ${ph}`;
    }
    wrapper.appendChild(frame);
    this.frame = frame;
    if (this.statusMessage && this.config.showStatus) {
      const status = document.createElement("div");
      status.className = "dd-status";
      status.textContent = this.statusMessage;
      wrapper.appendChild(status);
    }
    const image = this.current && this.images[this.current];
    if (image) {
      const img = document.createElement("img");
      img.src = image.url;
      img.className = "dd-visible";
      frame.appendChild(img);
    }
    return wrapper;
  },

  suspend () {
    clearTimeout(this.timer);
    this.timer = null;
    this.stopClock();
  },

  resume () {
    if (!this.timer) this.advance();
  }
});
