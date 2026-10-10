# Reproducibility and public demonstrations

## Run Route Lab in the browser

[Open Route Lab](https://alvaro-nfl-route-lab.tartmacaw2.chatgpt.site) without installation.
To run the same source locally, serve the repository root with `python -m http.server 8000`, then open
`http://localhost:8000/public-demo/`. No backend, installation, account or model download is
needed. Select a synthetic play and player, animate or scrub the paths, adjust the 1–12-frame
forecast horizon at 10 Hz and compare three fixed rules: constant velocity, hold last position
and damped velocity. Damping ranges from zero (hold) to one (constant velocity); each
future step scales the last observed displacement by the damping raised to that step.
Future positions are evaluation-only.

The browser computes coordinate RMSE, average Euclidean displacement error and final
displacement error, with per-player inspection and JSON export. These are synthetic
diagnostics, not NFL model scores. The Python report below remains the original two-reference
example; the browser adds interactive damped motion.

The browser fixture comes directly from the unchanged Python synthetic generator: six
games × two plays × three players, with eight observed and twelve future frames.
The browser exposes the four plays from the two held-out games. No forecasting rule is fitted.
Repeated interactive comparisons are exploratory demonstrations, not independent test results.

```bash
python3 tools/generate_public_demo.py --check
node tools/test_public_demo_engine.mjs
node tools/test_public_demo.mjs
```

The first command checks the committed fixture against its generator and source hash.
Omit `--check` to regenerate `public-demo/data.js`; review that diff before publishing.
The engine checks compare constant-velocity and hold outputs with the Python example.
Damped velocity is an additional generic rule, not a historical model component.

## Run the deterministic Python report

Use Python 3.11 from the repository root:

```bash
python scripts/run_portfolio_demo.py --output demo_output
```

Open `demo_output/dashboard.html` in a browser. The script uses only Python's standard library,
runs on a CPU in seconds, and needs no package installation, account, network connection,
competition data, or model download. It imports the lightweight demo directly, so the research
CLI and its optional dependencies are not loaded.

This is a **synthetic engineering demonstration**. Its metrics do not reproduce, estimate,
or validate the private neural ensemble's competition score.

## What the demonstration does

The generator creates six simulated games, each with two plays and three players. Every player
has eight observed frames and twelve future frames at 10 Hz. Coordinates are in yards. Motion
has constant acceleration, so a constant-velocity forecast is informative but imperfect.

Two fixed references receive the same observed coordinates and identifier-only requests:

- **Constant velocity:** extrapolate the difference between the final two observed positions.
- **Hold last position:** repeat the final observed position throughout the forecast.

Neither reference fits parameters or selects a configuration using future labels. Whole games
are assigned to train, validation, and holdout partitions before evaluation: three, one, and
two games respectively. These partitions demonstrate a game-disjoint evaluation boundary;
the demo performs zero training fits. All metrics are calculated after prediction.

Default seed `2026` produces these **synthetic holdout results** on 144 player-frame rows:

| Reference | Coordinate RMSE, yards |
|---|---:|
| Constant velocity | 0.263475522816 |
| Hold last position | 2.097543662939 |

The score pools squared errors over every requested row and both coordinates:

```text
coordinate RMSE = sqrt(sum((pred_x - x)^2 + (pred_y - y)^2) / (2 * rows))
```

It is not a mean of per-game RMSE values. Predictions and labels are joined by all four
identifiers, so changing label row order cannot change the result.

## Data contract and checks

| Field or boundary | Enforced behavior |
|---|---|
| Identifiers | Positive integer `game_id`, `play_id`, `nfl_id`, and `frame_id`; no booleans |
| Observations | At least two contiguous frames, numbered from one, for each requested player |
| Coordinates | Finite numeric `x` and `y`; nonfinite inputs and forecasts fail |
| Forecast frames | Relative to the observation cutoff: frame one is the next 0.1-second step; maximum 120 |
| Keys | Duplicate observation, request, prediction, or label keys fail |
| Alignment | Forecast rows preserve request order; metric keys must cover labels exactly once |
| Labels | The predictor reads identifiers from requests, never attached future `x` or `y` |
| Partitions | All rows from a game stay in the same partition |

The implementation deliberately keeps observations, requests, and labels separate. The test
suite uses guarded requests that raise if a future coordinate is read, perturbs every future
label, and verifies that predictions remain unchanged. An independent CSV calculation checks
the exported metric, including its two-coordinate denominator.

## Output files

| File | Purpose |
|---|---|
| `dashboard.html` | Self-contained report with inline styles and trajectory graphics |
| `trajectories.svg` | Accessible, standalone comparison of three players in one heldout play |
| `metrics.json` | Per-partition row counts, game IDs, reference scores, and check results |
| `observations.csv` | Simulated input positions before the forecast cutoff |
| `labels.csv` | Simulated future positions used only for scoring and plotting |
| `predictions.csv` | Both references' forecasts with their full identifiers |
| `splits.csv` | One game-to-partition assignment per simulated game |
| `manifest.json` | Source SHA-256 and byte lengths/SHA-256 values for the seven other outputs |

The manifest excludes its own hash; the terminal receipt includes that hash. Artifacts contain
no timestamps or absolute workspace paths. Replaying the same source, seed, and Python runtime
in a second output directory produces byte-identical files. The manifest's source hash changes
when the implementation changes. Existing identical files are accepted; conflicting files
are preserved and the command stops with a clear message. Use a new output directory after
changing the seed or implementation.

To exercise the standalone checks with all site packages disabled:

```bash
python -S -m unittest discover -s tests -p test_portfolio_demo.py -v
```

Checks cover hand-calculated metrics, observation-only prediction, invalid input rejection,
whole-game partitioning, CSV replay, manifest hashes, self-contained graphics, and the real
CLI in a separate process. The research test suite and dependency lock remain separate from
this small, dependency-free path.

## Full public package environment

The research package requires **Python >=3.11,<3.12**, as declared in `pyproject.toml`.
Use its committed lockfile for the broader software and notebook checks:

```bash
uv sync --frozen --group dev
uv run --frozen python scripts/quality.py
```

This broader gate includes synthetic integration and public notebook execution. It is
separate from the dependency-free browser and report paths above; it does not regenerate
the private ensemble score. Additional research self-tests and their optional dependencies
are recorded in the existing quality workflow.

## Public reproducibility boundary

The repository also publishes research code, experiment records, historical runners, and a
small fitted baseline at `docs/results/model.json`. Those assets have their own scope and
dependencies; they should not be confused with this synthetic demonstration.

The private neural ensemble's fitted weights and unpublished inference recipe are not part
of this public demo. Reproducing that score would additionally require the appropriate
competition data access, exact private artifacts, and the recorded evaluation/submission
environment. Licensed competition data is not bundled here. A successful demo establishes
that its public data contract, forecast references, metric calculation, and artifact replay
work as documented; historical private model performance remains a separate evidence claim.
