/* MagicMirror² module: MMM-desk_display
 * Plays screens rendered by a desk_display render server (display_server.py),
 * acting as one of its remote display clients. */
/* global Module, Log, DeskDisplaySchedule, DeskDisplayMotion */
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
    contentTimeZone: "America/Chicago", // the server's DESK_DISPLAY_CONTENT_TIMEZONE
    showStatus: true
  },

  getScripts () {
    return [this.file("lib/schedule.js"), this.file("lib/motion.js")];
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
    this.motionStop = null;
    this.current = null;
    this.sendSocketNotification("DD_START", {
      serverUrl: this.config.serverUrl,
      clientId: this.config.clientId,
      enrollmentToken: this.config.enrollmentToken,
      displayProfile: this.config.displayProfile,
      syncInterval: this.config.syncInterval,
      heartbeatInterval: this.config.heartbeatInterval,
      requestTimeoutMs: this.config.requestTimeoutMs,
      contentTimeZone: this.config.contentTimeZone
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
    // As on a desk_display client, a finite motion holds after it ends and a
    // ticker or quad runs for its own window or the hold, whichever is longer.
    const [width, height] = this.profileSize || [0, 0];
    const hold = Number(this.config.screenSeconds) + this.scheduler.extraSecondsFor(screenId);
    const seconds = Math.max(1, DeskDisplayMotion.showSeconds(this.images[screenId], width, height, hold));
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
      if (!this.present(img, showing, instant)) return;
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

  /* Fade *el* (in the frame) in over what is there, unless the next screen
   * has taken over since *showing*; returns whether it was shown. */
  present (el, showing, instant = false) {
    if (showing !== this.showing || !this.frame) {
      el.remove();
      return false;
    }
    if (!el.parentNode) this.frame.appendChild(el);
    if (instant) el.classList.add("dd-instant");
    this.loadedShowing = showing;
    const swap = () => {
      el.classList.add("dd-visible");
      const old = [...this.frame.children].filter((i) => i !== el);
      setTimeout(() => old.forEach((i) => i.remove()), instant ? 0 : this.config.fadeMs + 50);
    };
    if (instant) swap();
    else requestAnimationFrame(swap);
    return true;
  },

  /* A canvas the size of the screen's logical pixels, styled like an image. */
  canvas () {
    const [width, height] = this.profileSize;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    return canvas;
  },

  /* Run *draw(seconds since start)* every animation frame until it returns
   * true, the next screen takes over, or the motion is stopped. */
  animate (showing, draw) {
    let frame = null;
    const started = performance.now();
    const step = () => {
      frame = null;
      if (showing !== this.showing) return;
      if (!draw((performance.now() - started) / 1000)) frame = requestAnimationFrame(step);
    };
    step();
    this.motionStop = () => { if (frame !== null) cancelAnimationFrame(frame); };
  },

  /* Load *src* as an image, then call *ready(img)*; on failure show the still. */
  withImage (src, screenId, showing, ready) {
    const img = new Image();
    img.onload = () => { if (showing === this.showing) ready(img); };
    img.onerror = () => { if (showing === this.showing) this.place(this.images[screenId].url, screenId); };
    img.src = src;
  },

  /* Load every one of *urls* as an image, then call *ready(imgs)* in the
   * same order; if any fails, show the still. */
  withImages (urls, screenId, showing, ready) {
    const loaded = new Map();
    let failed = false;
    const unique = [...new Set(urls)];
    unique.forEach((src) => {
      const img = new Image();
      img.onload = () => {
        loaded.set(src, img);
        if (!failed && loaded.size === unique.length && showing === this.showing) ready(urls.map((u) => loaded.get(u)));
      };
      img.onerror = () => {
        if (failed) return;
        failed = true;
        if (showing === this.showing) this.place(this.images[screenId].url, screenId);
      };
      img.src = src;
    });
  },

  /* A frame animation (radar, standings overviews): it loops, then holds its last frame. */
  playFrames (screenId, frames, showing) {
    const [width, height] = this.profileSize;
    this.withImages(frames.urls, screenId, showing, (imgs) => {
      const canvas = this.canvas();
      const ctx = canvas.getContext("2d");
      const last = frames.durationsMs.length - 1;
      let shown = null;
      this.animate(showing, (t) => {
        const index = DeskDisplayMotion.frameIndex(frames, t);
        if (index !== shown) ctx.drawImage(imgs[index], 0, 0, width, height);
        shown = index;
        return t >= DeskDisplayMotion.framesSeconds(frames) && index === last;
      });
      this.present(canvas, showing);
    });
  },

  /* A news ticker: the still base, with each lane's strip looping across it. */
  playTicker (screenId, ticker, showing) {
    const [width, height] = this.profileSize;
    const urls = [ticker.baseUrl, ...ticker.lanes.map((lane) => lane.url)];
    this.withImages(urls, screenId, showing, ([base, ...strips]) => {
      const canvas = this.canvas();
      const ctx = canvas.getContext("2d");
      ctx.drawImage(base, 0, 0, width, height);
      const shown = ticker.lanes.map(() => null);
      this.animate(showing, (t) => {
        ticker.lanes.forEach((lane, i) => {
          const offset = DeskDisplayMotion.tickerOffset(lane, t);
          if (offset === shown[i]) return;
          shown[i] = offset;
          const [left, top, right, bottom] = lane.bounds;
          ctx.save();
          ctx.beginPath();
          ctx.rect(left, top, right - left, bottom - top);
          ctx.clip();
          ctx.fillStyle = "#000";
          ctx.fillRect(left, top, right - left, bottom - top);
          for (let x = -offset; x < right - left; x += lane.stripWidth) ctx.drawImage(strips[i], left + x, top);
          ctx.restore();
        });
        return false; // runs until the next screen
      });
      this.present(canvas, showing);
    });
  },

  /* A quad: the still base, with each tile stepping through its own frames. */
  playComposite (screenId, composite, showing) {
    const [width, height] = this.profileSize;
    const urls = [composite.baseUrl, ...composite.tiles.flatMap((tile) => tile.urls)];
    this.withImages(urls, screenId, showing, ([base, ...tileImgs]) => {
      const canvas = this.canvas();
      const ctx = canvas.getContext("2d");
      ctx.drawImage(base, 0, 0, width, height);
      let at = 0;
      const frames = composite.tiles.map((tile) => {
        const imgs = tileImgs.slice(at, at + tile.urls.length);
        at += tile.urls.length;
        return imgs;
      });
      const shown = composite.tiles.map(() => null);
      const still = composite.tiles.every((tile) => tile.urls.length === 1);
      this.animate(showing, (t) => {
        DeskDisplayMotion.compositeFrames(composite, t).forEach((index, i) => {
          if (index === shown[i]) return;
          shown[i] = index;
          const [left, top] = composite.tiles[i].bounds;
          ctx.drawImage(frames[i][index], left, top);
        });
        return still; // a quad of still tiles never redraws
      });
      this.present(canvas, showing);
    });
  },

  /* A tall screen: hold, step down its full-height canvas, hold. */
  playScroll (screenId, scroll, showing) {
    const [width, height] = this.profileSize;
    this.withImage(scroll.url, screenId, showing, (img) => {
      const canvas = this.canvas();
      const ctx = canvas.getContext("2d");
      let last = null;
      this.animate(showing, (t) => {
        const offset = DeskDisplayMotion.scrollOffset(scroll, height, t);
        if (offset !== last) ctx.drawImage(img, 0, offset, width, height, 0, 0, width, height);
        last = offset;
        return DeskDisplayMotion.scrollDone(scroll, height, t);
      });
      this.present(canvas, showing);
    });
  },

  /* A logo screen: the logo crosses from a random side and rests centred. */
  playSlide (screenId, slide, showing) {
    const [width, height] = this.profileSize;
    const direction = Math.random() < 0.5 ? "ltr" : "rtl";
    this.withImage(slide.url, screenId, showing, (img) => {
      const canvas = this.canvas();
      const ctx = canvas.getContext("2d");
      const [r, g, b] = slide.background;
      let last;
      this.animate(showing, (t) => {
        const x = DeskDisplayMotion.slideX(slide, width, t, direction);
        if (x !== last) {
          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
          ctx.fillRect(0, 0, width, height);
          ctx.drawImage(img, x === null ? Math.trunc((width - img.width) / 2) : x, slide.y);
        }
        last = x;
        return x === null;
      });
      this.present(canvas, showing);
    });
  },

  /* The date face from layers the server draws each minute, recoloured
   * here: fresh colours every interval for a few seconds, as on a desk_display
   * client. A server without layers sends the plain face. */
  playDate (screenId, image, showing) {
    const [width, height] = this.profileSize;
    const { interval, steps } = DeskDisplayMotion.colorCycle(this.config.displayProfile, this.config.screenSeconds);
    const current = () => showing === this.showing && this.timer !== null;
    const canvas = this.canvas();
    const ctx = canvas.getContext("2d");
    const tint = document.createElement("canvas");
    tint.width = width;
    tint.height = height;
    const tintCtx = tint.getContext("2d");
    let sheet = null;
    let step = 0;
    let colours = [DeskDisplayMotion.brightColor(), DeskDisplayMotion.brightColor()];
    let cycleTimer = null;
    let minuteTimer = null;

    const compose = () => {
      ctx.globalCompositeOperation = "source-over";
      ctx.drawImage(sheet, 0, 0, width, height, 0, 0, width, height);
      if (sheet.height < height * 3) return; // a plain face
      colours.forEach(([r, g, b], layer) => {
        // colour × coverage, added onto the face drawn without text colour
        tintCtx.globalCompositeOperation = "source-over";
        tintCtx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        tintCtx.fillRect(0, 0, width, height);
        tintCtx.globalCompositeOperation = "multiply";
        tintCtx.drawImage(sheet, 0, height * (layer + 1), width, height, 0, 0, width, height);
        ctx.globalCompositeOperation = "lighter";
        ctx.drawImage(tint, 0, 0);
      });
      ctx.globalCompositeOperation = "source-over";
    };
    const load = async () => {
      const response = await fetch(`${image.liveUrl}?layers=1&t=${Date.now()}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const bitmap = await createImageBitmap(await response.blob());
      if (bitmap.width !== width || (bitmap.height !== height && bitmap.height !== height * 3)) {
        throw new Error("unexpected clock size");
      }
      return bitmap;
    };
    const cycle = () => {
      cycleTimer = null;
      if (!current()) return;
      step += 1;
      colours = [DeskDisplayMotion.brightColor(), DeskDisplayMotion.brightColor()];
      compose();
      if (step < steps) cycleTimer = setTimeout(cycle, interval * 1000);
    };
    const nextMinute = () => {
      minuteTimer = setTimeout(async () => {
        if (!current()) return;
        try {
          sheet = await load();
          if (current()) compose();
        } catch {
          // keep the face already on show
        }
        if (current()) nextMinute();
      }, 60000 - (Date.now() % 60000) + 20);
    };
    this.motionStop = () => {
      clearTimeout(cycleTimer);
      clearTimeout(minuteTimer);
    };
    load().then((bitmap) => {
      if (!current()) return;
      sheet = bitmap;
      compose();
      this.present(canvas, showing);
      if (bitmap.height === height * 3) cycleTimer = setTimeout(cycle, interval * 1000);
      nextMinute();
    }, () => {
      if (showing === this.showing && this.loadedShowing !== showing) this.place(image.url, screenId);
    });
  },

  show (screenId) {
    const image = this.images[screenId];
    this.current = screenId;
    this.showing = (this.showing || 0) + 1;
    this.sendSocketNotification("DD_SHOWING", { clientId: this.config.clientId, screenId });
    if (image.scroll) {
      this.playScroll(screenId, image.scroll, this.showing);
    } else if (image.slide) {
      this.playSlide(screenId, image.slide, this.showing);
    } else if (image.frames) {
      this.playFrames(screenId, image.frames, this.showing);
    } else if (image.ticker) {
      this.playTicker(screenId, image.ticker, this.showing);
    } else if (image.composite) {
      this.playComposite(screenId, image.composite, this.showing);
    } else if (image.liveUrl && screenId === "date" && this.profileSize) {
      const showing = this.showing;
      this.playDate(screenId, image, showing);
      setTimeout(() => {
        if (this.showing === showing && this.loadedShowing !== showing) this.place(image.url, screenId);
      }, DD_CLOCK_WAIT_MS);
    } else if (image.liveUrl) {
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
    if (this.motionStop) this.motionStop();
    this.motionStop = null;
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
