/** Execute the actual Route Lab UI with the real synthetic engine in a small DOM. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const html = fs.readFileSync(path.join(root, "public-demo/index.html"), "utf8");
const scripts = Object.fromEntries(
  ["data", "engine", "app"].map((name) => [
    name,
    fs.readFileSync(path.join(root, `public-demo/${name}.js`), "utf8"),
  ]),
);
const normalize = (value) => JSON.parse(JSON.stringify(value));
let count = 0;
async function test(name, fn) {
  await fn();
  console.log(`ok ${++count} - ${name}`);
}

function browser({ reducedMotion = false, missingEngine = false } = {}) {
  const nodes = new Map(),
    frames = new Map(),
    listeners = new Map();
  let nextFrame = 0,
    blob = null,
    download = null;
  class Element {
    constructor(tag = "div") {
      this.tag = tag;
      this.handlers = {};
      this.children = [];
      this.attributes = {};
      this.style = {};
      this.value = "";
      this.checked = false;
      this.disabled = false;
      this.hidden = false;
      this._text = "";
    }
    set id(value) {
      this._id = value;
      nodes.set(value, this);
    }
    get id() {
      return this._id;
    }
    set textContent(value) {
      this._text = String(value);
      this.children = [];
    }
    get textContent() {
      return (
        this._text +
        this.children
          .map((c) => (typeof c === "string" ? c : c.textContent))
          .join("")
      );
    }
    append(...children) {
      this.children.push(...children);
    }
    replaceChildren(...children) {
      this._text = "";
      this.children = children;
      if (this.tag === "select") this.value = children[0]?.value || "";
    }
    setAttribute(name, value) {
      this.attributes[name] = String(value);
    }
    addEventListener(name, fn) {
      this.handlers[name] = fn;
    }
    getBoundingClientRect() {
      return {
        width: this.id === "field" ? 700 : this.id === "timeline" ? 500 : 300,
      };
    }
    click() {
      if (this.disabled) return;
      this.handlers.click?.();
      if (this.download) download = this;
    }
  }
  for (const match of html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"[^>]*>/g)) {
    const e = new Element(match[1]);
    e.id = match[2];
  }
  const sandbox = {
    console,
    Blob,
    URL: {
      createObjectURL(value) {
        blob = value;
        return "blob:synthetic-route-test";
      },
      revokeObjectURL() {},
    },
    document: {
      getElementById: (id) => nodes.get(id),
      createElement: (tag) => new Element(tag),
      createElementNS: (_, tag) => new Element(tag),
    },
    requestAnimationFrame(fn) {
      const id = ++nextFrame;
      frames.set(id, fn);
      return id;
    },
    cancelAnimationFrame(id) {
      frames.delete(id);
    },
    setTimeout() {},
    addEventListener(name, fn) {
      listeners.set(name, fn);
    },
    matchMedia() {
      return { matches: reducedMotion };
    },
  };
  sandbox.window = sandbox;
  const context = vm.createContext(sandbox);
  vm.runInContext(scripts.data, context, { filename: "data.js" });
  if (!missingEngine)
    vm.runInContext(scripts.engine, context, { filename: "engine.js" });
  vm.runInContext(scripts.app, context, { filename: "app.js" });
  const change = (id, value, event = "change") => {
    const e = nodes.get(id);
    if (id === "truth") e.checked = value;
    else e.value = String(value);
    e.handlers[event]();
  };
  return {
    nodes,
    context,
    frames,
    listeners,
    change,
    get blob() {
      return blob;
    },
    get download() {
      return download;
    },
    frameAt(timestamp) {
      assert.ok(frames.size, "animation frame is scheduled");
      const [id, fn] = frames.entries().next().value;
      frames.delete(id);
      fn(timestamp);
    },
  };
}
function descendants(node) {
  return node.children.flatMap((child) =>
    typeof child === "string" ? [] : [child, ...descendants(child)],
  );
}
function expected(
  b,
  {
    method = "damped_velocity",
    horizon = 12,
    damping = 0.9,
    game = 105,
    play = 1,
    player = "all",
  } = {},
) {
  const e = b.context.NFLTrajectoryEngine,
    d = b.context.NFL_TRAJECTORY_DATA;
  const run = e.run(d, { method, horizon, damping });
  const belongs = (r) =>
    r.game_id === game &&
    r.play_id === play &&
    (player === "all" || r.nfl_id === Number(player)) &&
    r.frame_id <= horizon;
  return {
    run,
    metrics: e.evaluate(
      run.predictions.filter(belongs),
      d.labels.filter(belongs),
    ),
  };
}
await test("initial UI computes real selected-play and whole-holdout metrics", () => {
  const b = browser(),
    value = expected(b);
  assert.equal(b.nodes.get("error").hidden, true);
  assert.equal(
    b.nodes.get("rmse").textContent,
    `${value.metrics.coordinate_rmse_yards.toFixed(4)} yd`,
  );
  assert.equal(
    b.nodes.get("all-rmse").textContent,
    `${value.run.metrics.coordinate_rmse_yards.toFixed(4)} yd`,
  );
  assert.match(
    b.nodes.get("metric-scope").textContent,
    /3 tracks × 12 future frames · 72 coordinate errors/,
  );
  assert.match(b.nodes.get("all-scope").textContent, /144 forecast rows/);
  assert.equal(b.nodes.get("track-table").children.length, 3);
  assert.ok(
    descendants(b.nodes.get("field")).some((n) => n.tag === "polyline"),
  );
});
await test("method and damping controls produce engine values, including endpoint equivalence", () => {
  const b = browser();
  b.change("damping", 1, "input");
  assert.equal(
    b.nodes.get("rmse").textContent,
    b.nodes.get("cv-rmse").textContent,
  );
  b.change("damping", 0, "input");
  assert.equal(
    b.nodes.get("rmse").textContent,
    b.nodes.get("hold-rmse").textContent,
  );
  b.change("method", "hold_last_position");
  assert.equal(b.nodes.get("damping").disabled, true);
  assert.equal(
    b.nodes.get("rmse").textContent,
    `${expected(b, { method: "hold_last_position", damping: 0 }).metrics.coordinate_rmse_yards.toFixed(4)} yd`,
  );
  b.change("method", "damped_velocity");
  assert.equal(b.nodes.get("damping").disabled, false);
});
await test("horizon changes update requests, frame limits, score denominator and timeline", () => {
  const b = browser();
  b.change("horizon", 1, "input");
  assert.equal(b.nodes.get("frame").max, "1");
  assert.equal(b.nodes.get("frame-value").textContent, "1 / 1");
  assert.match(b.nodes.get("metric-scope").textContent, /6 coordinate errors/);
  assert.match(b.nodes.get("all-scope").textContent, /12 forecast rows/);
  assert.equal(
    b.nodes.get("rmse").textContent,
    `${expected(b, { horizon: 1 }).metrics.coordinate_rmse_yards.toFixed(4)} yd`,
  );
  assert.ok(!JSON.stringify(b.nodes.get("timeline").children).includes("NaN"));
  b.change("horizon", 12, "input");
  assert.equal(b.nodes.get("frame").max, "12");
});
await test("play and player selection change only the intended evaluation scope", () => {
  const b = browser();
  const global = b.nodes.get("all-rmse").textContent;
  b.change("play-select", "106:2");
  b.change("player-select", "2");
  assert.match(b.nodes.get("field-title").textContent, /106.*02/);
  assert.equal(
    b.nodes.get("rmse").textContent,
    `${expected(b, { game: 106, play: 2, player: "2" }).metrics.coordinate_rmse_yards.toFixed(4)} yd`,
  );
  assert.match(
    b.nodes.get("metric-scope").textContent,
    /1 track × 12 future frames · 24/,
  );
  assert.equal(b.nodes.get("all-rmse").textContent, global);
  assert.equal(b.nodes.get("track-2").attributes["aria-pressed"], "true");
});
await test("track table buttons are live keyboard-compatible selection controls", () => {
  const b = browser();
  b.nodes.get("track-3").click();
  assert.equal(b.nodes.get("player-select").value, "3");
  assert.equal(b.nodes.get("track-3").attributes["aria-pressed"], "true");
  assert.match(b.nodes.get("field").attributes["aria-label"], /Track C/);
  assert.equal(
    b.nodes.get("rmse").textContent,
    `${expected(b, { player: "3" }).metrics.coordinate_rmse_yards.toFixed(4)} yd`,
  );
});
await test("future-truth reveal changes display without changing forecast or score", async () => {
  const b = browser();
  b.nodes.get("export").click();
  const before = JSON.parse(await b.blob.text());
  const blue = () =>
    descendants(b.nodes.get("field")).filter(
      (n) => n.tag === "polyline" && n.attributes.stroke === "#b7d5ff",
    );
  assert.equal(blue().length, 0);
  b.change("truth", true);
  assert.equal(blue().length, 3);
  assert.equal(b.nodes.get("truth-legend").hidden, false);
  b.nodes.get("export").click();
  const after = JSON.parse(await b.blob.text());
  assert.deepEqual(before.predictions, after.predictions);
  assert.deepEqual(before.selected_metrics, after.selected_metrics);
  b.change("truth", false);
  assert.equal(blue().length, 0);
});
await test("hidden future labels cannot change field crop or predicted paths", () => {
  const b = browser();
  const before = JSON.stringify(b.nodes.get("field").children);
  for (const row of b.context.NFL_TRAJECTORY_DATA.labels) {
    row.x += 10;
    row.y -= 5;
  }
  b.change("method", "damped_velocity");
  assert.equal(JSON.stringify(b.nodes.get("field").children), before);
  assert.ok(Number.parseFloat(b.nodes.get("rmse").textContent) > 5);
});
await test("scrubbing and single-frame stepping change visible time, not aggregate metrics", () => {
  const b = browser(),
    score = b.nodes.get("rmse").textContent;
  b.change("frame", 0, "input");
  assert.equal(b.nodes.get("field-clock").textContent, "t + 0.0 s");
  assert.equal(b.nodes.get("previous").disabled, true);
  b.nodes.get("next").click();
  assert.equal(b.nodes.get("frame-value").textContent, "1 / 12");
  b.nodes.get("previous").click();
  assert.equal(b.nodes.get("frame-value").textContent, "0 / 12");
  b.change("frame", 7, "input");
  assert.match(b.nodes.get("field").attributes["aria-label"], /frame 7 of 12/);
  assert.equal(b.nodes.get("rmse").textContent, score);
});
await test("playback advances at10Hz, pauses exactly, resumes and stops at horizon", () => {
  const b = browser();
  b.nodes.get("play").click();
  b.frameAt(0);
  b.frameAt(99);
  assert.equal(b.nodes.get("frame-value").textContent, "0 / 12");
  b.frameAt(100);
  assert.equal(b.nodes.get("frame-value").textContent, "1 / 12");
  b.frameAt(300);
  assert.equal(b.nodes.get("frame-value").textContent, "3 / 12");
  b.nodes.get("play").click();
  assert.equal(b.frames.size, 0);
  assert.match(b.nodes.get("play").textContent, /Play/);
  b.nodes.get("play").click();
  b.frameAt(1000);
  b.frameAt(1900);
  assert.equal(b.nodes.get("frame-value").textContent, "12 / 12");
  assert.equal(b.frames.size, 0);
  assert.match(b.nodes.get("play").textContent, /Play/);
});
await test("manual scrub and configuration changes cancel an active animation", () => {
  const b = browser();
  b.nodes.get("play").click();
  b.frameAt(0);
  b.frameAt(200);
  b.change("frame", 5, "input");
  assert.equal(b.frames.size, 0);
  assert.equal(b.nodes.get("frame-value").textContent, "5 / 12");
  b.nodes.get("play").click();
  b.frameAt(500);
  b.change("horizon", 3, "input");
  assert.equal(b.frames.size, 0);
  assert.equal(b.nodes.get("frame-value").textContent, "3 / 3");
});
await test("reduced-motion preference disables playback but preserves manual inspection", () => {
  const b = browser({ reducedMotion: true });
  assert.equal(b.nodes.get("play").disabled, true);
  assert.match(b.nodes.get("motion-note").textContent, /Reduced-motion/);
  b.change("frame", 4, "input");
  assert.equal(b.nodes.get("frame-value").textContent, "4 / 12");
  assert.equal(b.frames.size, 0);
  assert.equal(b.nodes.get("export").disabled, false);
});
await test("full-field view and resize render finite accessible SVG without recomputing scores", () => {
  const b = browser(),
    score = b.nodes.get("rmse").textContent;
  b.change("view", "field");
  b.listeners.get("resize")();
  assert.equal(b.nodes.get("rmse").textContent, score);
  assert.ok(!JSON.stringify(b.nodes.get("field").children).includes("NaN"));
  assert.match(
    b.nodes.get("field").attributes["aria-label"],
    /coordinates in yards/,
  );
  const outline = descendants(b.nodes.get("field")).find(
    (n) => n.tag === "rect" && n.attributes.stroke === "#d6dec9",
  );
  const xScale = Number(outline.attributes.width) / 120,
    yScale = Number(outline.attributes.height) / 53.333;
  assert.ok(
    Math.abs(xScale - yScale) < 1e-12,
    "full-field x/y yards use the same pixel scale",
  );
});
await test("invalid configuration clears stale results and export, and reset recovers", () => {
  const b = browser();
  b.change("horizon", 0, "input");
  assert.equal(b.nodes.get("error").hidden, false);
  assert.equal(b.nodes.get("rmse").textContent, "—");
  assert.equal(b.nodes.get("field").children.length, 0);
  assert.equal(b.nodes.get("export").disabled, true);
  assert.equal(b.nodes.get("play").disabled, true);
  assert.equal(b.nodes.get("track-table").children.length, 0);
  b.nodes.get("reset").click();
  assert.equal(b.nodes.get("error").hidden, true);
  assert.equal(b.nodes.get("export").disabled, false);
  assert.equal(
    b.nodes.get("rmse").textContent,
    `${expected(b).metrics.coordinate_rmse_yards.toFixed(4)} yd`,
  );
});
await test("missing engine reports an explicit unavailable result, never fabricated output", () => {
  const b = browser({ missingEngine: true });
  assert.equal(b.nodes.get("error").hidden, false);
  assert.equal(b.nodes.get("rmse").textContent, "—");
  assert.equal(b.nodes.get("export").disabled, true);
  assert.equal(b.nodes.get("field").children.length, 0);
});
await test("JSON export replays actual predictions and metrics with separate input/label arrays", async () => {
  const b = browser();
  b.change("play-select", "106:1");
  b.change("player-select", "1");
  b.change("horizon", 7, "input");
  b.change("damping", 0.7, "input");
  b.nodes.get("export").click();
  const report = JSON.parse(await b.blob.text());
  assert.equal(report.evidence_type, "SYNTHETIC_ONLY");
  assert.equal(report.sampling_hz, 10);
  assert.equal(report.predictions.length, 7);
  assert.equal(report.selection.game, 106);
  assert.equal(report.selection.horizon, 7);
  assert.ok(
    report.requests.every(
      (r) =>
        Object.keys(r).sort().join(",") === "frame_id,game_id,nfl_id,play_id",
    ),
  );
  const e = b.context.NFLTrajectoryEngine;
  assert.deepEqual(
    normalize(
      e.predict(report.observations, report.requests, {
        method: report.selection.method,
        damping: report.selection.damping,
      }),
    ),
    report.predictions,
  );
  assert.deepEqual(
    normalize(e.evaluate(report.predictions, report.evaluation_labels)),
    report.selected_metrics.damped_velocity,
  );
  assert.equal(report.error_timeline.length, 7);
  assert.equal(report.per_track.length, 3);
  assert.equal(b.download.download, "synthetic-route-game106-play1.json");
});
await test("HTML scope, native controls, minimum text sizes and local assets are explicit", () => {
  assert.match(html, /SYNTHETIC_ONLY/);
  assert.match(html, /No parameters are fitted/);
  assert.match(html, /Future coordinates enter afterward/);
  const scriptPaths = [
    ...html.matchAll(/<script\s+defer\s+src="([^"]+)"/g),
  ].map((m) => m[1]);
  assert.deepEqual(scriptPaths, ["data.js", "engine.js", "app.js"]);
  assert.ok(
    !/\b(?:fetch|XMLHttpRequest|WebSocket|localStorage|sessionStorage)\s*[.(]/.test(
      scripts.app,
    ),
  );
  assert.ok(!scripts.app.includes("innerHTML"));
  const css = fs.readFileSync(
    path.join(root, "public-demo/styles.css"),
    "utf8",
  );
  for (const size of css.matchAll(/font(?:-size)?:\s*(\d+)px/g))
    assert.ok(Number(size[1]) >= 12);
  assert.match(css, /:focus-visible/);
  assert.match(css, /prefers-reduced-motion/);
});
console.log(
  `\n${count} Route Lab UI checks passed using the actual engine. No browser layout or research-performance claim.`,
);
