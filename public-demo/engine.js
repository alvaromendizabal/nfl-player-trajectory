/* Public synthetic motion references. No trained/private model or network access. */
(function (root) {
  "use strict";
  const KEYS = ["game_id", "play_id", "nfl_id", "frame_id"];
  const METHODS = [
    "constant_velocity",
    "hold_last_position",
    "damped_velocity",
  ];
  const own = (object, key) =>
    Object.prototype.hasOwnProperty.call(object, key);
  function fail(message) {
    throw new Error(message);
  }
  function exactKeys(row, expected, kind) {
    if (!row || typeof row !== "object" || Array.isArray(row))
      fail(`${kind} must be an object`);
    const keys = Object.keys(row).sort();
    if (keys.join("|") !== [...expected].sort().join("|"))
      fail(`${kind} keys do not match the public contract`);
  }
  function integer(value, name) {
    if (!Number.isSafeInteger(value) || value <= 0)
      fail(`${name} must be a positive safe integer`);
    return value;
  }
  function finite(value) {
    if (typeof value !== "number" || !Number.isFinite(value))
      fail("Coordinates must be finite numbers");
    return value;
  }
  function rowKey(row, coordinates = false) {
    exactKeys(
      row,
      coordinates ? [...KEYS, "x", "y"] : KEYS,
      coordinates ? "Coordinate row" : "Identifier-only request",
    );
    const key = KEYS.map((name) => integer(row[name], name));
    if (coordinates) {
      finite(row.x);
      finite(row.y);
    }
    return key.join(":");
  }
  const entityKey = (row) =>
    KEYS.slice(0, 3)
      .map((name) => row[name])
      .join(":");
  function rowsRequired(rows, name) {
    if (!Array.isArray(rows) || !rows.length || rows.length > 100000)
      fail(`${name} must contain 1..100000 rows`);
  }
  // Python round(value, 6) rounds the exact binary64 value, ties to even.
  // Integer arithmetic avoids JavaScript toFixed/Math.round midpoint differences.
  function roundSix(value) {
    finite(value);
    if (value === 0) return value;
    const negative = value < 0;
    const bytes = new DataView(new ArrayBuffer(8));
    bytes.setFloat64(0, Math.abs(value), false);
    const bits = bytes.getBigUint64(0, false);
    const exponent = Number((bits >> 52n) & 2047n);
    let significand = bits & ((1n << 52n) - 1n);
    if (exponent) significand += 1n << 52n;
    const shift = (exponent ? exponent - 1023 : -1022) - 52;
    let scaled = significand * 1000000n;
    let rounded;
    if (shift >= 0) rounded = scaled << BigInt(shift);
    else {
      const divisor = 1n << BigInt(-shift);
      rounded = scaled / divisor;
      const twiceRemainder = 2n * (scaled % divisor);
      if (
        twiceRemainder > divisor ||
        (twiceRemainder === divisor && rounded % 2n)
      )
        rounded += 1n;
    }
    const result = Number(`${negative ? "-" : ""}${rounded}e-6`);
    return finite(result);
  }
  function options(settings = {}, maxHorizon = 12) {
    if (!settings || typeof settings !== "object" || Array.isArray(settings))
      fail("Settings must be an object");
    const method = settings.method ?? "constant_velocity";
    const damping = settings.damping ?? 0.9;
    const horizon = settings.horizon ?? maxHorizon;
    if (!METHODS.includes(method)) fail("Unknown public reference method");
    if (
      typeof damping !== "number" ||
      !Number.isFinite(damping) ||
      damping < 0 ||
      damping > 1
    )
      fail("Damping must be in 0..1");
    if (!Number.isInteger(horizon) || horizon < 1 || horizon > maxHorizon)
      fail(`Horizon must be in 1..${maxHorizon}`);
    return { method, damping, horizon };
  }
  function predict(observations, requests, settings = {}) {
    const { method, damping } = options(settings, 120);
    rowsRequired(observations, "Observations");
    rowsRequired(requests, "Requests");
    const history = new Map();
    const seen = new Set();
    for (const row of observations) {
      const key = rowKey(row, true);
      if (seen.has(key)) fail("Duplicate observation key");
      seen.add(key);
      const entity = entityKey(row);
      if (!history.has(entity)) history.set(entity, []);
      history.get(entity).push(row);
    }
    for (const records of history.values()) {
      records.sort((a, b) => a.frame_id - b.frame_id);
      if (
        records.length < 2 ||
        records.some((row, i) => row.frame_id !== i + 1)
      )
        fail("Each player needs at least two contiguous observation frames");
    }
    seen.clear();
    return requests.map((row) => {
      const key = rowKey(row);
      if (seen.has(key)) fail("Duplicate request key");
      seen.add(key);
      const records = history.get(entityKey(row));
      if (!records) fail("Requested player has no observed history");
      if (row.frame_id > 120) fail("Demonstration horizon exceeds 120 frames");
      const [previous, last] = records.slice(-2);
      let multiplier = method === "hold_last_position" ? 0 : row.frame_id;
      if (method === "damped_velocity" && damping !== 1) {
        multiplier = 0;
        let velocity = damping;
        for (let frame = 1; frame <= row.frame_id; frame++) {
          multiplier += velocity;
          velocity *= damping;
        }
      }
      return {
        game_id: row.game_id,
        play_id: row.play_id,
        nfl_id: row.nfl_id,
        frame_id: row.frame_id,
        x: roundSix(
          multiplier === 0
            ? last.x
            : last.x + multiplier * (last.x - previous.x),
        ),
        y: roundSix(
          multiplier === 0
            ? last.y
            : last.y + multiplier * (last.y - previous.y),
        ),
      };
    });
  }
  // Compensated sum keeps pooled metric accumulation stable across row order.
  function stableSum(values) {
    const partials = [];
    for (let value of values) {
      let index = 0;
      for (let other of partials) {
        if (Math.abs(value) < Math.abs(other)) [value, other] = [other, value];
        const high = value + other;
        const low = other - (high - value);
        if (low) partials[index++] = low;
        value = high;
      }
      partials.length = index;
      partials.push(value);
    }
    return finite(partials.reduceRight((sum, value) => sum + value, 0));
  }
  function evaluate(predictions, labels) {
    rowsRequired(predictions, "Predictions");
    rowsRequired(labels, "Labels");
    const truth = new Map();
    for (const row of labels) {
      const key = rowKey(row, true);
      if (truth.has(key)) fail("Duplicate label key");
      truth.set(key, row);
    }
    const seen = new Set(),
      squared = [],
      displacement = [],
      endpoints = new Map();
    for (const row of predictions) {
      const key = rowKey(row, true);
      if (seen.has(key) || !truth.has(key))
        fail("Prediction keys must match labels exactly once");
      seen.add(key);
      const target = truth.get(key),
        dx = finite(row.x - target.x),
        dy = finite(row.y - target.y);
      squared.push(finite(dx * dx), finite(dy * dy));
      const distance = finite(Math.hypot(dx, dy));
      displacement.push(distance);
      const entity = entityKey(row);
      if (!endpoints.has(entity) || endpoints.get(entity).frame < row.frame_id)
        endpoints.set(entity, { frame: row.frame_id, distance });
    }
    if (seen.size !== truth.size)
      fail("Prediction keys must cover every label");
    return {
      coordinate_rmse_yards: Math.sqrt(
        stableSum(squared) / (2 * predictions.length),
      ),
      ade_yards: stableSum(displacement) / predictions.length,
      fde_yards:
        stableSum(
          [...endpoints.values()].map((endpoint) => endpoint.distance),
        ) / endpoints.size,
      rows: predictions.length,
      entities: endpoints.size,
      coordinate_denominator: 2 * predictions.length,
      definitions: {
        coordinate_rmse: "sqrt(sum(dx² + dy²)/(2 × rows))",
        ade: "mean Euclidean distance over forecast rows",
        fde: "mean Euclidean distance at the last requested frame of each player/play",
      },
    };
  }
  function validateData(data) {
    if (
      !data ||
      data.schema_version !== 1 ||
      data.evidence_type !== "SYNTHETIC_ONLY" ||
      data.units !== "yards" ||
      data.sampling_hz !== 10 ||
      data.observed_frames !== 8 ||
      data.future_frames !== 12
    )
      fail("Unsupported synthetic fixture contract");
    if (
      !data.splits ||
      typeof data.splits !== "object" ||
      Array.isArray(data.splits)
    )
      fail("Game partitions are required");
    const games = Object.keys(data.splits)
      .map(Number)
      .sort((a, b) => a - b);
    if (games.join(",") !== "101,102,103,104,105,106")
      fail("Fixture requires exactly six synthetic games");
    const expected = {
      101: "train",
      102: "train",
      103: "train",
      104: "validation",
      105: "holdout",
      106: "holdout",
    };
    for (const game of games)
      if (data.splits[game] !== expected[game])
        fail("Game partition assignment changed");
    for (const [name, frames, coordinates] of [
      ["observations", 8, true],
      ["requests", 12, false],
      ["labels", 12, true],
    ]) {
      rowsRequired(data[name], name);
      if (data[name].length !== 6 * 2 * 3 * frames)
        fail(`Incomplete ${name} fixture`);
      const seen = new Set();
      for (const row of data[name]) {
        const key = rowKey(row, coordinates);
        if (seen.has(key)) fail(`Duplicate ${name} key`);
        seen.add(key);
        if (
          !own(expected, row.game_id) ||
          ![1, 2].includes(row.play_id) ||
          ![1, 2, 3].includes(row.nfl_id) ||
          row.frame_id > frames
        )
          fail(`Out-of-domain ${name} key`);
      }
    }
    if (
      !data.provenance ||
      data.provenance.training_fits !== 0 ||
      data.provenance.official_evaluation !== false ||
      !/^[a-f0-9]{64}$/.test(data.provenance.source_sha256)
    )
      fail("Synthetic source provenance is missing");
    return {
      game_disjoint: true,
      observations: 288,
      requests: 432,
      labels: 432,
      holdout_games: [105, 106],
    };
  }
  function run(data, settings = {}) {
    validateData(data);
    const resolved = options(settings);
    const holdout = (row) => data.splits[row.game_id] === "holdout";
    const observations = data.observations.filter(holdout);
    const requests = data.requests.filter(
      (row) => holdout(row) && row.frame_id <= resolved.horizon,
    );
    // Labels are deliberately selected only after prediction. They never enter predict().
    const predictions = predict(observations, requests, resolved);
    const labels = data.labels.filter(
      (row) => holdout(row) && row.frame_id <= resolved.horizon,
    );
    return {
      schema_version: 1,
      evidence_type: "SYNTHETIC_ONLY",
      settings: resolved,
      holdout_games: [105, 106],
      sampling_hz: 10,
      units: "yards",
      predictions,
      metrics: evaluate(predictions, labels),
      provenance: { ...data.provenance },
      checks: {
        game_disjoint: true,
        request_order_preserved: predictions.every((row, i) =>
          KEYS.every((name) => row[name] === requests[i][name]),
        ),
        prediction_input_scope:
          "Observed x/y plus identifier-only future requests",
        labels_not_passed_to_predictor: true,
        all_predictions_finite: predictions.every(
          (row) => Number.isFinite(row.x) && Number.isFinite(row.y),
        ),
      },
      method_scope:
        resolved.method === "damped_velocity"
          ? "New generic illustration: last observed displacement multiplied by damping^step; no fitted parameters."
          : "Public Python observation-only reference; six-decimal predictions match its contract.",
      limitations: [
        "Simulated trajectories; not NFL tracking data or a competition benchmark.",
        "No models are trained or selected by heldout score.",
        "Changing settings explores the same illustrative holdout repeatedly; it does not create an independent evaluation.",
        "No field-boundary clipping or interaction model is applied.",
      ],
    };
  }
  const api = {
    KEYS: [...KEYS],
    METHODS: [...METHODS],
    predict,
    evaluate,
    validateData,
    run,
  };
  root.NFLTrajectoryEngine = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : window);
