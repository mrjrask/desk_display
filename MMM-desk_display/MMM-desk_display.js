/* MagicMirror² module: MMM-desk_display
 * Plays screens rendered by a desk_display render server (display_server.py),
 * acting as one of its remote display clients. */
/* global Module, Log */
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

  getStyles () {
    return ["MMM-desk_display.css"];
  },

  start () {
    this.screens = [];
    this.profileSize = null;
    this.statusMessage = "Connecting to desk_display…";
    this.cycle = 1;
    this.position = -1;
    this.timer = null;
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
      const hadScreens = this.screens.length > 0;
      this.screens = payload.screens;
      this.profileSize = [payload.width, payload.height];
      this.statusMessage = this.screens.length ? null : "Waiting for the server to render screens…";
      if (!hadScreens || !this.frame) this.updateDom(0);
      if (!this.timer) this.advance();
    } else if (notification === "DD_STATUS") {
      if (payload.error) Log.warn(`MMM-desk_display: ${payload.message}`);
      if (!this.screens.length) {
        this.statusMessage = payload.message;
        this.updateDom(0);
      }
    }
  },

  /* Next screen in desk_display's frequency cycles: a screen with frequency N
   * plays on cycles 1, 1+N, 1+2N, … */
  nextIndex () {
    const n = this.screens.length;
    for (let tries = 0; tries < n * 64; tries += 1) {
      this.position += 1;
      if (this.position >= n) {
        this.position = 0;
        this.cycle += 1;
      }
      const screen = this.screens[this.position];
      if ((this.cycle - 1) % Math.max(1, screen.frequency) === 0) return this.position;
    }
    return n ? 0 : -1;
  },

  advance () {
    clearTimeout(this.timer);
    this.timer = null;
    if (!this.screens.length) return;
    const index = this.nextIndex();
    const screen = this.screens[index];
    this.show(screen);
    const seconds = Math.max(1, Number(this.config.screenSeconds) + (screen.extraSeconds || 0));
    this.timer = setTimeout(() => this.advance(), seconds * 1000);
  },

  show (screen) {
    if (!this.frame) return;
    const img = document.createElement("img");
    img.alt = screen.screenId;
    img.onload = () => {
      requestAnimationFrame(() => img.classList.add("dd-visible"));
      const old = [...this.frame.querySelectorAll("img")].filter((i) => i !== img);
      setTimeout(() => old.forEach((i) => i.remove()), this.config.fadeMs + 50);
    };
    img.onerror = () => img.remove();
    img.src = screen.url;
    this.frame.appendChild(img);
    this.current = screen.screenId;
    this.sendSocketNotification("DD_SHOWING", { clientId: this.config.clientId, screenId: screen.screenId });
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
    const screen = this.screens.find((s) => s.screenId === this.current);
    if (screen) {
      const img = document.createElement("img");
      img.src = screen.url;
      img.className = "dd-visible";
      frame.appendChild(img);
    }
    return wrapper;
  },

  suspend () {
    clearTimeout(this.timer);
    this.timer = null;
  },

  resume () {
    if (!this.timer) this.advance();
  }
});
