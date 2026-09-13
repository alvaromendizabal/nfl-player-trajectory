# Round 4 · Direct observed state and landing-frame geometry

**Status: implemented; private-data benefit is not yet measured. Feature research remains open.**

## Why this experiment follows Round 3

Round 3 completed successfully after the storage-parity correction. All three planned first-fold feature contrasts failed. Do not launch its second or third folds unchanged. The prior 0.807004 arrival result came from a different ridge/sample comparison; it did not transfer to the Round 3 tree study. The control for this next study is the completed Round 3 control, not a retrospectively selected arrival combination.

An audit of the supplied diagnostic adapter found that all 62 `STATE_NAMES` fields enter its base matrix multiplied by `tau = query time + observation age`. This includes role indicators, input-support flags, positions and velocity. It was an intentional linear response representation, not evidence of corrupt raw data. But it is not the same representation as presenting observed state separately from forecast time. For example, a receiver-role indicator becomes 0.1 at 0.1 seconds and 1.5 at 1.5 seconds. The original role information can theoretically be recovered using time; that does not mean a finite tree learns it equally efficiently.

**Hypothesis H1:** exposing the same state directly makes fixed-estimator prediction more effective. This is a representation change, not a new raw signal.

**Hypothesis H2:** compact goal-relative geometry gives useful radial/lateral motion and arrival context beyond explicit state. Geometry refers to the organizer-supplied ball-landing point, not an independently observed live landing outcome or a fabricated coverage assignment.

## Frozen arms and isolation

| Arm | Input | Columns before train-only screening | New fits |
|---|---|---:|---:|
| Control | Preserved Round 3 72-column control | 72 | 0; reload x/y models |
| Direct state | Control + 62 unmultiplied observed-state fields | 134 | 2 coordinate fits |
| Goal geometry | Direct state + 26 goal-frame values/support flags | 160 | 2 coordinate fits |

The full 24-column arrival addition and both 324-column relationship treatments are excluded. This is not a rerun of their failed comparisons. Previously rejected turning, braking, orientation-response and earlier-origin mixtures are not appended.

Rows, labels, ordering, first-fold games, loss, residual target, tree settings and optimizer exposure are identical to the completed Round 3 control. No new player/game history lookup, data source, sample expansion, hyperparameter search, row weighting, ensemble or prediction clipping is introduced. It is not a full existing-model refit or leaderboard evaluation.

The exact first-fold population is 19,713 training rows and 6,326 evaluation rows from 16 evaluation games. Only first-fold plays already present within the selected 1,024-play parent cache receive new features. Later folds are neither fitted nor scored by this package. All game splits are reused and exploratory.

## Input availability and leakage rules

`build_features(raw, keys, cutoff=...)` receives observed input rows plus request keys. There is no truth, error, model-prediction or target-statistic argument. Geometry uses observed anchors/velocity, known horizon, synchronized receiver information and supplied landing coordinates. Per-play normalization follows existing play-direction rules. Identifiers are keys only.

The 62 direct fields are the existing normalized state values, not inversions of time-multiplied float32 caches. They are rebuilt from the hash-verified observed CSVs using the preserved raw adapter. The 26 geometry columns contain 20 numerical descriptors and six support flags. Invalid axes/velocities receive explicit masks; a missing measurement is not silently equated with measured zero. Kinematic projections are hypotheses, not constraints that all players must meet.

Targets are reused from verified Round 3 caches. No raw output CSV, additional season, external label source or reserved outer holdout is opened. Screening uses training rows only. Feature-support/range plots use one count per training player/play; outcome plots are descriptive and may not define new role-gated treatments.

## Exact historical precision, not a relaxed tolerance

Each prior play checkpoint may store X as float32 (reused Round 1 examples) or float64 (new Round 3 examples). Recomputed float64 legacy features undergo that shard's recorded storage conversion and then require exact equality. A one-ULP representable mismatch stops. No old X, y, source, folds or cache is rewritten. This preserves the tested correction and explicitly exercises both dtypes.

## Estimator and checkpoints

Use the existing fixed HistGradientBoostingRegressor settings: squared error, learning rate 0.06, 120 iterations, 15 maximum leaves, minimum leaf size 30, L2 1.0, 127 bins, no early stopping, warm start, seed 20260911. Screen near-constant columns using training data only. Two CPU threads; no GPU. Save every 30 iterations. Model manifests bind source, data, schema, keep-mask, settings and numerical environment. Replays cannot fit missing models.

The runner uses the existing exact script lock and cached packages through offline `uv`. It may materialize a separate script-cache environment; it does not change `.venv`, update a dependency lock, download packages/interpreters, or invoke the full final fitter. Failure to resolve that cache is a diagnostic stop, not permission to reinstall the project.

## Comparisons and acceptance

1. Direct state versus preserved control: isolates direct-state availability.
2. Goal geometry versus direct state: isolates incremental geometry.
3. Goal geometry versus preserved control: prevents approving an increment over an inferior intermediate control.

For each contrast use the official coordinate RMSE, `sqrt(sum(dx² + dy²)/(2N))`, all requested rows, and paired game-bootstrap differences. Use 10,000 resamples and the inherited six-look interval quantiles (three current contrasts plus three potential future pooled looks). A contrast passes the exploratory screen only if relative RMSE improves by at least 1% and its adjusted upper difference bound is below zero. Goal geometry earns further consideration only when comparisons 2 AND 3 pass. No post-hoc threshold, seed, role, horizon or mixture tuning is allowed.

These adjusted intervals do not undo reuse of the same games across several research rounds. They are exploratory decision aids, not independent confirmatory claims or evidence that the winner has been beaten.

## Bounded stages

Preflight 120 seconds; 32-play training smoke 180 seconds; remaining first-fold feature preparation 360 seconds; four-model fitting 240 seconds; fresh-process replay 120 seconds. These are hard limits, not promised runtimes. The runner emits 15-second heartbeats and locks both its own run and the existing parent run descriptor.

Atomic, verified completed checkpoints survive interruption and are reused. An orphan file or deterministic mismatch is a stop requiring diagnosis; the package does not discard it or pretend it is complete. Return the failure report instead of retrying an unchanged error.

## Decision and publication boundary

Return this first-fold report before running anything larger. A passing representation can be considered for fixed-protocol chronological replication and integration into a replayable temporal forecasting path. If both hypotheses fail, stop this interface unchanged rather than trying another long list of arbitrary tree columns. Learned player interactions, trajectory parameterization, fold-safe historical context, and other realistic domain families remain open.

The overall project target remains 0.46340 under comparable evaluation. The current diagnostic score is not the historical ~0.62 model or a Kaggle score. No record, scientific acceptance, repository update or AWS deployment is implied by this package.
