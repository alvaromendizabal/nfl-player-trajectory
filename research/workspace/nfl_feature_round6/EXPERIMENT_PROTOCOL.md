# Frozen Round 6 protocol: pass-axis state and observed history

## Why this question

Round 5 failed temporal replication. Its aggregate diagnostics localize the
unresolved errors but cannot identify which direct-state columns caused the
failure. This experiment does not claim that the failure mechanism is known.
It tests a separate, literature-motivated representation under the saved control.

The third-place author's feature list includes distance to the passing line,
projection onto that line, and player/QB/receiver triangle geometry. The winning
author used compact ball- and receiver-relative temporal inputs, static role and
horizon information, losses, and augmentation. These sources motivate reference
frames and relationships, not the claim that hand-engineered feature count alone
will close the score gap. Neither source demonstrates the RMSE contribution of
our formulas or this small tree experiment.

Primary sources, accessed during this preparation:
- Official third-place write-up: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/3rd-place-solution
- Official first-place write-up: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution
- Organizer task and exact coordinate RMSE: https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview

## Mechanism and availability

At a scored player's last observed frame, require the passer at that exact frame.
Let Q be that observed passer position, B the supplied landing point, and
u=(B-Q)/||B-Q||. For player P, project P-Q onto u and its perpendicular. Project
observed velocity, orientation, and receiver-relative position/velocity the same
way. Include the normalized signed double-area of triangle Q/P/R. This line is a
coordinate reference; it is not a claim about the true 3D ball trajectory, a
route label, or a defensive assignment.

The supplied landing point is legitimate input for this task. It would not be
known in every live application. No new external data or competition annotations
are downloaded. No predicted or future player positions enter the feature API.

Require simultaneous observations, not stale as-of joins. Missing or degenerate
passer/landing geometry is zero-filled only with a separate validity indicator.
A passer-to-landing distance below 0.1 yard marks the axis invalid. This fixed
numerical threshold is not tuned on validation. Use actual 5-, 10-, and 15-frame
endpoint offsets. Missing endpoints are masked. A difference between two valid
endpoints is an interval change, not a fabricated instantaneous derivative across
a gap. The final observed axis stays fixed through the history. All transformations
use canonical yards, seconds, and stated constant scales. No feature learns
statistics from evaluation games.

## Exact arms and attribution

The retained 72-column control is replayed from Round 5, with no refitting.
Arm `pass_axis`: control + 21 state columns = 93 before screening.
Arm `pass_axis_history`: control + 21 state + 27 history columns = 120 before screening.
The 27 history columns contain 18 interval-change values and 9 support indicators.
No field is asserted to be a novel raw signal or absent from every historical
repository feature bank. The tested novelty is this compact, explicit reference
frame in the current diagnostic interface.

Three comparisons are declared: state vs control, state/history vs control,
and state/history vs state. The last is the history-removal ablation. The static
and history effects are conditional on this tree representation, not an estimate
of every feature family's value in every forecasting system. A positive result
would still require finer subfamily attribution and independent validation before
promotion. Counts and training variability are not feature importance.

## Data and estimator

Reuse exactly the 1,024 selected original-origin plays and cached forecast keys,
labels, and chronological folds. Use later folds 2 and 3 only. Do not repeat the
first-fold fit or change the previous selection. Raw output CSVs are not opened.
Raw observed-input SHA256 and the original per-shard float32/float64 conversion
are checked before exact legacy-feature parity. New-feature checkpoints are
key-aligned and content-verified. Existing models remain immutable.

Both arms use the existing fixed HistGradientBoostingRegressor setup: squared
error, learning rate 0.06, 120 iterations, 15 leaves, minimum 30 rows per leaf,
L2=1.0, 127 bins, no early stopping, seed 20260911. This is not an algorithm search.
Fit constant-column screening exclusively on each training fold; no evaluation
ranking, clipping, dimensionality search, target encoding, or row filtering is
performed. Preserve all forecast rows and equal original coordinate-loss weights.

## Decision rule and compute

At most 8 new coordinate fits (2 arms x 2 coordinates x 2 folds). Replay controls.
Run smoke on 32 training plays before preparing all features. Each later fold has
a 240-second supervisor limit; preprocessing 360 seconds, smoke 180, preflight150,
replay180. Use 2 CPU threads and checkpoint every 30 boosting iterations. Save
partial successes. No GPU, managed training job, or background process is launched.

If BOTH treatments are at least 5% worse on fold 2, record futility and do not fit
fold 3. Otherwise evaluate both treatments on fold 3 unchanged. Do not revise the
settings after reviewing fold 2. A single modest negative fold remains part of
the result rather than being hidden.

For a representation to earn further work: both later folds must improve and
pooled RMSE must improve by >=1%, with adjusted upper paired-game difference
bound <0. Use 10,000 game-block draws, seed20260911, and nine planned looks
(three contrasts x two folds plus later pooling). History must earn this rule
against BOTH state-only and control. A worse intermediate arm cannot make the
history treatment appear acceptable. The 95% nominal family adjustment does not
correct the adaptive use of these games across prior research. These are
exploratory intervals on reused games, not untouched confirmation or a Kaggle
result. No model is automatically promoted, blended, or submitted.

## Interpretation limits and checkpoint exit

Aggregate Round 5 errors support diagnosis, not a causal allocation of the
leaderboard gap to features vs training, architecture, sample size, or ensembling.
The controlled experiment can answer only whether these representations improve
this fixed model on these rows. Do not make a top-score guarantee.

After fitting, independently reload every new checkpoint in a fresh process.
Replay may not fit missing work. Export aggregate receipts. If both additions
fail, stop this pass-axis tree interface unchanged and review the structured
sequence-model integration gap instead of producing endless append-only banks.
Feature engineering remains open; failed flat summaries do not exhaust learned
spatiotemporal representations.
