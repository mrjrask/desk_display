"use strict";
const test = require("node:test");
const assert = require("node:assert");
const { Scheduler, buildEntries, parseHideAfter, starterScreenIds } = require("../lib/schedule");

// Play orders recorded from desk_display's schedule.py (build_scheduler,
// start_at(starter_screen_ids(doc)), next_available) with the listed screens
// unavailable. Each document has a Starter playlist and alternates.
const PYTHON_CASES = require("./schedule-cases.json");

function play (doc, unavailable = [], count = 24, now = Date.now()) {
  const scheduler = new Scheduler(buildEntries(doc));
  scheduler.startAt(starterScreenIds(doc), now);
  const skip = new Set(unavailable);
  return Array.from({ length: count }, () => scheduler.next((id) => !skip.has(id), now));
}

test("plays in the same order as desk_display's scheduler", () => {
  assert.ok(PYTHON_CASES.length >= 3);
  for (const c of PYTHON_CASES) assert.deepStrictEqual(play(c.doc, c.unavailable, c.seq.length), c.seq);
});

test("an alternate replaces every Nth showing of its base, but never in cycle 1", () => {
  const doc = { screens: { a: { frequency: 1, alt: { screen: ["x", "y"], frequency: 2 } }, b: 1, x: 0, y: 0 } };
  assert.deepStrictEqual(play(doc, [], 8), ["a", "b", "x", "b", "a", "b", "y", "b"]);
  // An unavailable alternate gives way to the next one, then to the base.
  assert.deepStrictEqual(play(doc, ["x"], 8), ["a", "b", "y", "b", "a", "b", "y", "b"]);
  assert.deepStrictEqual(play(doc, ["x", "y"], 4), ["a", "b", "a", "b"]);
});

test("playback starts at the top of the Starter playlist", () => {
  const doc = {
    screens: { a: 1, b: 1, c: 1 },
    playlists: { p: { label: " starter ", steps: [{ screen: "b" }, { screen: "c" }] } },
    sequence: [{ playlist: "p" }]
  };
  assert.deepStrictEqual(starterScreenIds(doc), ["b", "c"]);
  assert.deepStrictEqual(play(doc, [], 5), ["b", "c", "a", "b", "c"]);
});

test("seekAfter continues after the screen on show", () => {
  const scheduler = new Scheduler(buildEntries({ screens: { a: 1, b: 1, c: 1 } }));
  assert.strictEqual(scheduler.seekAfter("b"), true);
  assert.strictEqual(scheduler.next(() => true), "c");
  assert.strictEqual(scheduler.next(() => true), "a");
});

test("extra seconds, replacement-only and zero-frequency screens", () => {
  const entries = buildEntries({ screens: { a: { frequency: 1, extra_seconds: 4 }, "cubs no game": 1, z: 0, t: true } });
  assert.deepStrictEqual(entries.map((e) => e.screenId), ["a", "t"]);
  assert.strictEqual(new Scheduler(entries).extraSecondsFor("a"), 4);
});

test("hide_after retires a screen, reading times without an offset as Central time", () => {
  assert.strictEqual(new Date(parseHideAfter("2026-10-02T18:30")).toISOString(), "2026-10-02T23:30:00.000Z");
  assert.strictEqual(new Date(parseHideAfter("2026-01-15 08:00:00")).toISOString(), "2026-01-15T14:00:00.000Z");
  assert.strictEqual(parseHideAfter("2026-10-02T18:30:00-04:00"), Date.parse("2026-10-02T22:30:00Z"));
  assert.strictEqual(parseHideAfter("soon"), null);
  // As Python's fold=0: a skipped time takes the offset from before the
  // change, and a repeated one is its first occurrence.
  assert.strictEqual(new Date(parseHideAfter("2026-03-08T02:30")).toISOString(), "2026-03-08T08:30:00.000Z");
  assert.strictEqual(new Date(parseHideAfter("2026-11-01T01:30")).toISOString(), "2026-11-01T06:30:00.000Z");
  assert.strictEqual(new Date(parseHideAfter("2026-10-02T18:30", "America/New_York")).toISOString(), "2026-10-02T22:30:00.000Z");
  const doc = { screens: { a: 1, b: { frequency: 1, hide_after_enabled: true, hide_after_at: "2026-10-02T18:30" } } };
  assert.deepStrictEqual(play(doc, [], 3, Date.parse("2026-10-02T23:00:00Z")), ["a", "b", "a"]);
  assert.deepStrictEqual(play(doc, [], 3, Date.parse("2026-10-02T23:31:00Z")), ["a", "a", "a"]);
});

test("nothing playable gives null", () => {
  assert.strictEqual(new Scheduler([]).next(() => true), null);
  assert.strictEqual(new Scheduler(buildEntries({ screens: { a: 1 } })).next(() => false), null);
});
