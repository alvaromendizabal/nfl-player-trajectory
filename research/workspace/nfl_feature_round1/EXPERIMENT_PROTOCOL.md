# Round 1 — observed origins and role-conditioned arrival

## Objective and scope
Target the historical final private coordinate RMSE of **0.46340**, without claiming that a small internal screen is a comparable score. Keep feature engineering open. This round creates reproducible raw-input views and candidate features, then runs a deliberately inexpensive diagnostic. It does not replace the existing temporal neural model or recover the missing historical 0.62708 blend.

The uploaded report records code synchronized to `402843faa1722460aa84d1bbaf27c4050a7b7ff9`, 49 raw files (864,820,234 bytes), a ready Python 3.11 environment, and a successful 32-play temporal-edge smoke. **Do not repeat that synchronization, restore data, reinstall packages, or rerun the completed scientific job.** GitHub main matched that revision when checked for this deliverable.

The new origin smoke tests a different prerequisite: truncating the observed input and reconstructing labels without giving the model withheld coordinates.

## Evidence behind the hypothesis
The uploaded inspection receipt reports matched coordinate-only versus velocity-supervised RMSE of 0.8616038787 and 0.7203815957 on 83,938 rows / 41 games. This is an old completed experiment, not a result of the new code. Its numerical-model-forward-replay flag is false; restored checkpoint bytes alone are not a forward replay.

Using the receipt's per-horizon row counts and coordinate RMSE, the four second bins account for approximately 16.43%, 51.65%, 31.08%, and 0.84% of squared error. Rows after the first second are **24.80% of rows but 83.57% of squared error**. This concentration motivates longer-horizon representation research. The slices were already inspected, so that old validation set is not an untouched final test.

The first-place author's solution used compact, physically meaningful observed features and reported a substantial contribution from starting some training predictions earlier than the pass cutoff. It also used carefully designed architecture, auxiliary losses, augmentation and a large ensemble. This evidence does **not** justify assuming that extra columns alone can close the whole gap.

## Frozen questions
1. Does adding role-conditioned arrival response features to the same fixed-ridge control lower coordinate RMSE?
2. With the same feature schema and labeled training-row budget, does a mixture of original and earlier-origin training views improve predictions evaluated only at the real pass origin?

These are not the earlier failed soft-affinity ridge correction: no affinity assignments, existing neural predictions, cached target statistics or parent residual model are used. Nevertheless, this is still a limited linear diagnostic. A negative result cannot rule out a nonlinear temporal/interaction encoder or observed-origin augmentation for that encoder.

## Input provenance and information boundary
Read the exact trusted existing cache only after SHA256 verification, to obtain `split` and `(game_id, play_id)` join keys. The frozen training subset has 4,951 plays / 94 games. Do not use cached `static` target encodings, fitted baselines, targets or validation rows for this round's features or sampling. The pickle file is already owner-trusted; the helper will not unpickle an arbitrary or changed artifact.

Read the selected raw weekly CSVs with the uploaded report's full-file SHA256 fingerprints. Reading a weekly file or the combined cache places other rows in memory, but only selected original training plays are transformed or used. No statistics, sampling decisions, screening, fitting or evaluation use the old outer-validation plays. The earlier observed error slices are historical motivation only.

At offset `k`, the new origin is `max(observed frame_id) - k`. Feature construction receives only rows at or before that origin. Original target flags, role, landing point and supplied horizon are known task descriptors. `num_frames_output + k` defines the pseudo-task horizon. These supplied descriptors need not be known to a live real-time system at the artificially early time; this is expressly a competition-training construction, not a causal live-service claim.

Shifted labels consist of observed coordinates withheld after the new origin plus original post-pass output coordinates with their frame IDs shifted by `k`. Only originally scored players receive labels. All original post-pass labels must remain present. Missing withheld frames stay missing; no coordinates are interpolated or forward-filled to invent targets. A shifted view losing an originally scored player is logged as ineligible. A bad original view is a hard failure.

Use offsets **0, 5, 10 and 20 frames** at 10 Hz. No predictions are clipped to the field, no long-horizon rows are removed, and no shifted-origin view enters evaluation.

## Candidate representation
The control is a **72-column diagnostic representation**, not the complete existing 7,999-column bank. It contains observed terminal motion, orientation/availability, synchronous targeted-receiver and passer context, recent 5/10/20-frame summaries, role indicators, time response and simple constant-velocity geometry. A constant-velocity displacement reference is subtracted from labels and added conceptually to predictions; it is computed directly from observed state, not a trained parent model.

The **24 candidate arrival columns** are six per raw role: two-dimensional displacement corrections under a constant-velocity arrival hypothesis, constant-acceleration arrival hypothesis, and receiver-velocity-gap hypothesis. Let `r` be the ball displacement from the latest actual observed player position, `v` the observed velocity, `H` the time from that observation to the supplied arrival horizon, and `tau` the time from that observation to a requested frame. The first two corrections are `(r-vH)*tau/H` and `(r-vH)*(tau/H)^2`, divided by a fixed yard scale. They are hypotheses; no player is forced to reach the landing point. Role gating allows the fitted response to differ between receiver and defender.

These ideas overlap prior arrival/role feature families in the repository. **Do not add 24 to the historical bank as 24 distinct new mechanisms.** Some role gates may be structurally zero because only two roles are scored. The training-fold screen reports the retained dimension and drops constant columns. Zero activation is not automatically missing data. Window summaries have finite-difference and observed support limits; a later temporal encoder should retain full per-channel/time masks rather than assume these coarse summaries exhaust temporal information.

The important new data representation is the truncated-origin example with correct clocks, recomputed motion, role/landmark context, and no post-origin state in predictors.

## Frozen diagnostic protocol
Select **256 plays**, without reading outcomes, round-robin across games with SHA256-based deterministic ordering seeded by 20260911. The preceding engineering smoke uses 32 plays. The full cache is not fitted. Sampling is for training-side diagnosis and does not estimate performance on the full competition population.

Group by game; split on distinct dates (`game_id // 100`) using three expanding windows. Training ends at approximately 50%, 67%, and 84% of ordered dates; corresponding validation blocks end at 67%, 84%, and 100%. No same-date game crosses a train/validation boundary. Every original requested forecast row for each selected evaluation play remains in that fold. At least eight distinct dates and sufficient training/evaluation games are required.

Three arms:
- `control`: 72 columns, original-origin training rows.
- `arrival`: 96 columns, exactly the same original-origin training rows.
- `arrival_origin`: the same 96-column schema, half original-origin and half earlier-origin training rows.

Use the same labeled training-row count in each arm of a fold, capped at 10,000. Train-only standardization and constant-column screening are fitted separately for each arm. The first contrast necessarily changes feature width; it does not hold effective model capacity exactly constant. The second holds declared input width and labeled row budget fixed, but changes the data distribution and can activate previously constant features.

Fit ridge to coordinate residuals, optimizing mean coordinate squared error plus a fixed L2 penalty of 0.01 on standardized coefficients. No alpha search, ensembling, feature ranking by evaluation outcomes or test-based rule adjustment. Up to nine fits total. Save model arrays and every evaluation prediction/key privately; reload numeric arrays with `allow_pickle=False` and verify exact forward prediction replay after each fit.

Metric: `sqrt(sum(dx²+dy²)/(2N))`. Do not average fold RMSEs to produce pooled RMSE. Paired bootstrap resamples games, 2,000 times, keeping both arms aligned. Its uncertainty is conditional on fitted predictions and does not include fitting-seed uncertainty.

Exploratory continuation gate for each predeclared contrast: >=1% pooled RMSE reduction, negative upper paired-game 95% difference bound, and improvement in at least two folds. There are two contrasts and the intervals are not multiplicity-adjusted. This is a small hypothesis screen, not confirmatory proof. No Kaggle or established-model score is produced.

## Budgets, persistence and decisions
CPU only, existing environment, two numerical-library threads. Preflight cap 120 seconds; new-origin smoke cap 180 seconds; research preparation cap 600 seconds; screen cap 600 seconds. These are stop limits, not runtime promises. At most 512 plays or 900 seconds can be explicitly requested; do not increase limits without inspecting an interrupted run. The supplied notebook stays at 256 / 600.

A parent watchdog caps stalled numerical/file operations, emits heartbeats every 15 seconds, and terminates its worker at the limit. Per-play datasets and per-arm fitted artifacts are atomic and hash-checked. An output-directory lock prevents concurrent stages. A changed source, environment or scientific protocol is a stop, not an excuse to reuse stale artifacts. Repeating an unchanged successful screen must produce zero new fits and identical stored predictions.

If origin construction fails, diagnose it before fitting. If the small linear screen fails its gate, do not expand that ridge treatment unchanged. Diagnose support and underfitting; decide separately whether primary-source evidence justifies a matched established-neural-model test. If it passes, it earns that stronger-model integration—not promotion, a claim that all features are mature, or an unlimited training job.

## Primary sources
- Official final private leaderboard: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/leaderboard
- First-place author's feature/origin/augmentation/architecture account: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution
- Fourth-place author's conditional ball-node and target-specific interaction account: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/discussion/651814
- Song et al., 2026, NFL/AWS factorized temporal/player attention: https://arxiv.org/html/2603.25901v1 . Its coverage labels and privileged context are not assumed available here; its classification metrics are not our coordinate metric.
- Existing feature inventory and stopped treatments: https://github.com/alvaromendizabal/nfl-player-trajectory/blob/402843faa1722460aa84d1bbaf27c4050a7b7ff9/docs/FEATURE_STATUS.md
- Existing observed-role audit: https://github.com/alvaromendizabal/nfl-player-trajectory/blob/402843faa1722460aa84d1bbaf27c4050a7b7ff9/scripts/audit_roles.py

The earlier-origin implementation and diagnostic code in this kit are new local engineering work. No competition data, model weights or copied winning implementation is distributed.
