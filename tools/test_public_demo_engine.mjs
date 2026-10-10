import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const context = vm.createContext({});
for (const name of ["data.js", "engine.js"])
  vm.runInContext(
    fs.readFileSync(path.join(root, "public-demo", name), "utf8"),
    context,
    { filename: name },
  );
const engine = context.NFLTrajectoryEngine,
  data = context.NFL_TRAJECTORY_DATA;
const plain = (value) => JSON.parse(JSON.stringify(value));
let count = 0;
function test(name, callback) {
  callback();
  count++;
  console.log(`PASS ${name}`);
}
function near(actual, expected, tolerance = 1e-12) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${actual} != ${expected}`,
  );
}

const pythonCode = String.raw`
import importlib.util,json,sys,hashlib
from pathlib import Path
source=Path(sys.argv[1])/"src/nfl_trajectory/portfolio_demo.py"
spec=importlib.util.spec_from_file_location("authentic_public_demo",source)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
data=module.synthetic_data(2026)
fixture=dict(observations=data.observations,requests=data.requests,labels=data.labels,splits=data.splits)
results={}
for method in module.MODELS:
    predictions=module.predict_observed(data.observations,data.requests,method)
    metrics=[]
    for horizon in range(1,13):
        selected=lambda row: data.splits[row["game_id"]]=="holdout" and row["frame_id"]<=horizon
        metrics.append(module.coordinate_rmse(list(filter(selected,predictions)),list(filter(selected,data.labels))))
    results[method]=dict(predictions=predictions,holdout_by_horizon=metrics)
extra=json.load(sys.stdin)
rounding={method:module.predict_observed(extra["observations"],extra["requests"],method) for method in module.MODELS}
print(json.dumps(dict(fixture=fixture,results=results,rounding=rounding,sha256=hashlib.sha256(source.read_bytes()).hexdigest()),allow_nan=False))
`;
// Include exact and nearby binary midpoint boundaries; using the actual Python
// predictor proves rounding behavior rather than testing a duplicate JS helper.
const roundingValues = [
  0.0078125, 0.0234375, -0.0078125, -0.0234375, 1.2345645, -1.2345645,
  0.0000005, 1e-300, 1e20,
];
const roundingObservations = roundingValues.flatMap((value, index) =>
  [1, 2].map((frame) => ({
    game_id: 1,
    play_id: 1,
    nfl_id: index + 1,
    frame_id: frame,
    x: value,
    y: -value,
  })),
);
const roundingRequests = roundingValues.map((_, index) => ({
  game_id: 1,
  play_id: 1,
  nfl_id: index + 1,
  frame_id: 1,
}));
const py = spawnSync(
  process.env.PYTHON || "python3",
  ["-S", "-B", "-c", pythonCode, root],
  {
    input: JSON.stringify({
      observations: roundingObservations,
      requests: roundingRequests,
    }),
    encoding: "utf8",
    maxBuffer: 4 * 1024 * 1024,
  },
);
assert.equal(py.status, 0, py.stderr);
const reference = JSON.parse(py.stdout);

test("fixture is the unchanged Python-generated dataset with authenticated public source", () => {
  const checked = engine.validateData(data);
  assert.deepEqual(plain(checked), {
    game_disjoint: true,
    observations: 288,
    requests: 432,
    labels: 432,
    holdout_games: [105, 106],
  });
  for (const name of ["observations", "requests", "labels", "splits"])
    assert.deepEqual(plain(data[name]), reference.fixture[name]);
  assert.equal(data.provenance.source_sha256, reference.sha256);
  assert.equal(
    reference.sha256,
    createHash("sha256")
      .update(fs.readFileSync(path.join(root, data.provenance.source)))
      .digest("hex"),
  );
  assert.equal(data.provenance.official_evaluation, false);
  const generated = spawnSync(
    process.env.PYTHON || "python3",
    ["-S", "-B", path.join(root, "tools/generate_public_demo.py"), "--check"],
    { encoding: "utf8" },
  );
  assert.equal(generated.status, 0, generated.stderr);
});
test("all 864 constant-velocity and hold rows exactly match the actual Python predictor", () => {
  for (const method of ["constant_velocity", "hold_last_position"])
    assert.deepEqual(
      plain(engine.predict(data.observations, data.requests, { method })),
      reference.results[method].predictions,
    );
});
test("Python binary64 rounding parity includes signed decimal midpoint boundaries", () => {
  for (const method of ["constant_velocity", "hold_last_position"])
    assert.deepEqual(
      plain(engine.predict(roundingObservations, roundingRequests, { method })),
      plain(reference.rounding[method]),
    );
});
test("all 24 whole-holdout horizon scores match Python pooled coordinate RMSE", () => {
  for (const method of ["constant_velocity", "hold_last_position"])
    for (let horizon = 1; horizon <= 12; horizon++) {
      const result = engine.run(data, { method, horizon });
      near(
        result.metrics.coordinate_rmse_yards,
        reference.results[method].holdout_by_horizon[horizon - 1],
      );
      assert.equal(result.metrics.rows, 12 * horizon);
      assert.equal(result.metrics.coordinate_denominator, 24 * horizon);
      assert.equal(result.metrics.entities, 12);
      assert.deepEqual(plain(result.holdout_games), [105, 106]);
      assert.ok(
        result.predictions.every(
          (row) => [105, 106].includes(row.game_id) && row.frame_id <= horizon,
        ),
      );
    }
});
test("damping endpoints reproduce hold and constant velocity; the intermediate formula is explicit", () => {
  assert.deepEqual(
    plain(
      engine.predict(data.observations, data.requests, {
        method: "damped_velocity",
        damping: 0,
      }),
    ),
    reference.results.hold_last_position.predictions,
  );
  assert.deepEqual(
    plain(
      engine.predict(data.observations, data.requests, {
        method: "damped_velocity",
        damping: 1,
      }),
    ),
    reference.results.constant_velocity.predictions,
  );
  const observations = [
    { game_id: 1, play_id: 1, nfl_id: 1, frame_id: 1, x: 0, y: 0 },
    { game_id: 1, play_id: 1, nfl_id: 1, frame_id: 2, x: 2, y: 4 },
  ];
  const request = { game_id: 1, play_id: 1, nfl_id: 1, frame_id: 3 };
  const prediction = engine.predict(observations, [request], {
    method: "damped_velocity",
    damping: 0.5,
  })[0];
  assert.equal(prediction.x, 3.75);
  assert.equal(prediction.y, 7.5);
});
test("predictions remain independent of future labels and fixture seed metadata", () => {
  const changed = plain(data);
  changed.seed = 17;
  changed.labels = changed.labels.map((row) => ({
    ...row,
    x: row.x + 10,
    y: row.y - 7,
  }));
  const original = engine.run(data),
    altered = engine.run(changed);
  assert.deepEqual(plain(original.predictions), plain(altered.predictions));
  assert.notEqual(
    original.metrics.coordinate_rmse_yards,
    altered.metrics.coordinate_rmse_yards,
  );
  const forbidden = plain(data.requests[0]);
  forbidden.x = 999;
  assert.throws(
    () => engine.predict(data.observations, [forbidden]),
    /Identifier-only request/,
  );
  assert.equal(original.checks.labels_not_passed_to_predictor, true);
});
test("shuffled observations are sorted while prediction request order is preserved", () => {
  const requested = [...data.requests].reverse();
  const result = engine.predict([...data.observations].reverse(), requested);
  assert.deepEqual(
    plain(result),
    [...reference.results.constant_velocity.predictions].reverse(),
  );
  assert.deepEqual(
    plain(engine.evaluate([...result].reverse(), [...data.labels].reverse())),
    plain(engine.evaluate(result, data.labels)),
  );
});
test("RMSE, ADE and endpoint FDE have separate keyed definitions", () => {
  const labels = [1, 2].map((frame_id) => ({
    game_id: 1,
    play_id: 1,
    nfl_id: 1,
    frame_id,
    x: 0,
    y: 0,
  }));
  const predictions = labels.map((row, i) => ({
    ...row,
    x: i ? 0 : 3,
    y: i ? 0 : 4,
  }));
  const metrics = engine.evaluate(predictions, labels);
  assert.equal(metrics.coordinate_rmse_yards, 2.5);
  assert.equal(metrics.ade_yards, 2.5);
  assert.equal(metrics.fde_yards, 0);
  assert.equal(metrics.coordinate_denominator, 4);
  const endpoint = [{ ...labels[0], x: 3, y: 4 }];
  near(
    engine.evaluate(endpoint, [labels[0]]).coordinate_rmse_yards,
    Math.sqrt(12.5),
  );
  assert.equal(engine.evaluate(endpoint, [labels[0]]).fde_yards, 5);
});
test("duplicate, missing, nonfinite and misaligned metric populations reject", () => {
  const predicted = engine.predict(data.observations, data.requests);
  for (const labels of [
    [],
    data.labels.slice(1),
    [...data.labels, data.labels[0]],
    data.labels.map((row, i) => (i ? row : { ...row, x: NaN })),
  ])
    assert.throws(() => engine.evaluate(predicted, labels));
  for (const predictions of [
    [],
    predicted.slice(1),
    [...predicted, predicted[0]],
    predicted.map((row, i) => (i ? row : { ...row, nfl_id: 99 })),
    predicted.map((row, i) => (i ? row : { ...row, y: Infinity })),
  ])
    assert.throws(() => engine.evaluate(predictions, data.labels));
});
test("forecast rejects malformed identifiers, duplicate/gapped histories and unsupported settings", () => {
  const obs = data.observations.slice(0, 8),
    request = data.requests.slice(0, 12);
  for (const observations of [
    [],
    obs.slice(1),
    obs.slice(0, 1),
    [...obs, obs[0]],
    obs.map((row, i) => (i ? row : { ...row, x: NaN })),
    obs.map((row, i) => (i ? row : { ...row, y: "1" })),
  ])
    assert.throws(() => engine.predict(observations, request));
  for (const key of [true, 0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1])
    assert.throws(() => engine.predict(obs, [{ ...request[0], game_id: key }]));
  for (const requests of [
    [],
    [request[0], request[0]],
    [{ ...request[0], nfl_id: 99 }],
    [{ ...request[0], frame_id: 121 }],
    [{ ...request[0], extra: 1 }],
  ])
    assert.throws(() => engine.predict(obs, requests));
  for (const settings of [
    { method: "trained_model" },
    { damping: NaN },
    { damping: -0.1 },
    { damping: 1.1 },
    { horizon: 0 },
    { horizon: 13 },
    { horizon: 1.5 },
  ])
    assert.throws(() => engine.run(data, settings));
});
test("fixture validation rejects changed split membership, missing groups and bad provenance", () => {
  for (const mutate of [
    (d) => (d.splits[105] = "train"),
    (d) => d.observations.pop(),
    (d) => (d.requests[0].nfl_id = 99),
    (d) => (d.labels[0].x = Infinity),
    (d) => (d.provenance.training_fits = 1),
    (d) => (d.provenance.source_sha256 = "bad"),
    (d) => (d.requests[0] = d.requests[1]),
  ]) {
    const bad = plain(data);
    mutate(bad);
    assert.throws(() => engine.validateData(bad));
  }
});
test("engine is deterministic, does not mutate inputs, and performs no external I/O", () => {
  const before = JSON.stringify(data);
  const first = JSON.stringify(
    engine.run(data, { method: "damped_velocity", damping: 0.65, horizon: 7 }),
  );
  assert.equal(
    JSON.stringify(
      engine.run(data, {
        method: "damped_velocity",
        damping: 0.65,
        horizon: 7,
      }),
    ),
    first,
  );
  assert.equal(JSON.stringify(data), before);
  const source = fs.readFileSync(
    path.join(root, "public-demo/engine.js"),
    "utf8",
  );
  assert.doesNotMatch(
    source,
    /\bfetch\s*\(|XMLHttpRequest|WebSocket|Math\.random|\beval\s*\(/,
  );
  assert.ok(
    JSON.parse(first).limitations.some((text) =>
      text.includes("same illustrative holdout repeatedly"),
    ),
  );
});
console.log(
  JSON.stringify(
    {
      status: "PASS",
      tests: count,
      scope:
        "Actual browser engine; independent unchanged Python generator/predictor/coordinate-RMSE parity. Synthetic evidence only.",
    },
    null,
    2,
  ),
);
