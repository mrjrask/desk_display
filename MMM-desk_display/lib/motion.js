/* Motion for MMM-desk_display: vertical scrolls and the date face's colour
 * cycle, timed as a desk_display client times them
 * (playback/package_player.py, screens/draw_date_time.py).
 *
 * Shared by the browser module (global DeskDisplayMotion) and the tests.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.DeskDisplayMotion = api;
}(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // Panels driven through the kernel (and the HyperPixel layouts) cycle calmly.
  const CALM_CYCLE_PROFILES = new Set(["hyperpixel4", "hyperpixel4_square", "waveshare_lcd_320x240"]);

  /* Rows a scroll has travelled *t* seconds after it appeared, as
   * PackagePlayback._scroll_offset: it holds, steps down, then holds. */
  function scrollOffset (scroll, viewportHeight, t) {
    const maxOffset = Math.max(0, scroll.canvasHeight - viewportHeight);
    const moving = t - scroll.pauseStartSeconds;
    const travelled = moving <= 0 ? 0 : Math.min(maxOffset, Math.floor(moving / scroll.frameSeconds) * scroll.stepPx);
    return scroll.direction === "up" ? maxOffset - travelled : travelled;
  }

  /* How long the scroll's own motion lasts, pauses included. */
  function scrollSeconds (scroll, viewportHeight) {
    const steps = Math.ceil(Math.max(0, scroll.canvasHeight - viewportHeight) / scroll.stepPx);
    return scroll.pauseStartSeconds + steps * scroll.frameSeconds + scroll.pauseEndSeconds;
  }

  /* Whether the scroll has reached the end of its travel at *t*. */
  function scrollDone (scroll, viewportHeight, t) {
    const maxOffset = Math.max(0, scroll.canvasHeight - viewportHeight);
    const end = scroll.direction === "up" ? 0 : maxOffset;
    return scrollOffset(scroll, viewportHeight, t) === end && t > scroll.pauseStartSeconds;
  }

  /* How long a logo takes to cross the screen. */
  function slideSeconds (slide, viewportWidth) {
    return (viewportWidth + slide.spriteWidth) / slide.speedPxPerSecond;
  }

  /* The logo's left edge *t* seconds in, as PackagePlayback._slide_x: it
   * enters from one side, crosses, and then rests centred (null). */
  function slideX (slide, viewportWidth, t, direction = "ltr") {
    const travelled = t * slide.speedPxPerSecond;
    if (travelled >= viewportWidth + slide.spriteWidth) return null;
    return Math.trunc(direction === "ltr" ? -slide.spriteWidth + travelled : viewportWidth - travelled);
  }

  /* {interval, steps} of the date face's colour cycle, as
   * draw_date_time._color_cycle_profile with SCREEN_DELAY = *screenSeconds*:
   * fresh colours every interval for *steps* redraws, then they hold. */
  function colorCycle (profileId, screenSeconds) {
    const delay = Number(screenSeconds) || 0;
    let interval;
    let windowSeconds;
    if (CALM_CYCLE_PROFILES.has(profileId)) {
      interval = 0.2;
      windowSeconds = Math.max(0.5, delay - 0.2);
    } else if (profileId === "display_hat_mini") {
      interval = 0.12;
      windowSeconds = Math.min(Math.max(0.4, delay * 0.35), 1.2);
    } else {
      interval = 0.08;
      windowSeconds = Math.max(0.5, delay - 0.2);
    }
    return { interval, steps: Math.max(1, Math.floor(windowSeconds / interval)) };
  }

  /* A random colour bright enough to read on black, as utils.bright_color. */
  function brightColor (random = Math.random) {
    const channel = () => 80 + Math.floor(random() * 176);
    for (let i = 0; i < 20; i += 1) {
      const r = channel();
      const g = channel();
      const b = channel();
      if (0.2126 * r + 0.7152 * g + 0.0722 * b >= 160) return [r, g, b];
    }
    return [255, 255, 255];
  }

  return { brightColor, colorCycle, scrollDone, scrollOffset, scrollSeconds, slideSeconds, slideX };
}));
