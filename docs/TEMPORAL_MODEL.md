# Temporal prediction with attention across players

This experiment tests whether a joint temporal representation improves the
preserved development model. It keeps all 192 training games and the same 67,857
forecast rows from 32 later development games. The previously inspected 48-game
holdout and Kaggle private score do not select its features, epochs, or weights.

## Features and football hypotheses

| Representation | Information available at prediction time | Hypothesis |
|---|---|---|
| 20-frame observed sequence | Player displacement, finite-difference velocity/acceleration, supplied speed/acceleration, circular orientation/direction | Recent changes distinguish running, braking, turning and tracking the ball |
| Landing-relative dynamics | Relative position, distance, radial/tangential velocity and acceleration | Players adjust toward or away from the supplied catch location |
| Synchronized role anchors | Receiver/passer displacement at matching observed frame IDs, with availability masks | Preserve movement relative to the intended receiver and passer |
| Directed player-pair geometry | Relative position/velocity, separation, closing speed, closest approach, projected arrival separation, side and role | Let attention learn which coverage and route interactions matter |
| Static context | Field position, landing displacement, known horizon, observed-history availability, prediction flag | Distinguish field geometry and the duration of the forecast |
| Learned role/side embeddings | Organizer-supplied role and side | Learn category-specific responses without arbitrary ordinal category codes |
| Historical target encodings | Player/role motion-error summaries from strictly earlier training dates | Estimate persistent bias while representing uncertainty from sparse history |

There are 30 dynamic channels, 20 static channels (including ten historical
statistics), and 12 directed pair channels. The model also uses explicit history,
player and output masks. Channel counts describe inputs, not independent causal
contributions. This experiment changes both architecture and representation.

All numeric scaling uses fixed physical units. There is no development-fitted
normalizer, feature selector or category vocabulary. Player identifiers align
rows and retrieve historical statistics; their integer values are not covariates.
Names, retrospective descriptions and future player coordinates are not inputs.
Body/position metadata is deferred because earlier controlled tree experiments
did not demonstrate a benefit; that result does not rule out its future use.

## Target encoding protocol

The existing `historical_encodings` function already implements a chronological
encoder for player identity and role. The target is the player's average
canonical, time-normalized displacement error relative to constant velocity,
with one observation per player/play. This target does not depend on a fitted
model. For each category it emits count, cold-start flag, two smoothed error means
and dispersion. Means shrink toward zero using 20 pseudo-observations.

Each training date is encoded **before** any outcome from that date updates the
tables. Validation sees only the final training state; its outcomes never update
the encoder. Unknown players receive the cold prior. Scored-player encodings
enter this model; unscored context players have explicit cold priors. The role
and side embeddings remain available for all observed players.

This is deliberately stricter than randomly splitting forecast rows for target
encoding. Adjacent frames, players in the same play, and games on the same date
cannot teach an encoder the answer to a supposedly held-out row. The
[scikit-learn target-encoder documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)
explains why fitting and transforming on the same labels leaks information;
the existing date-ordered implementation also respects this project's time split.

Availability does not prove selection or benefit: the original development
tree used a role-error prior but did not select player-history inputs. The
temporal model accepts both. Its incremental benefit requires a later matched
training ablation, not an interpretation of attention weights alone.

## Model and objective

Three temporal convolutions encode each player's observed sequence. Two
four-head attention blocks combine players, with a learned attention bias from
directed pair geometry. Stable player slots are shared across time; there is no
player-order positional embedding. A continuous-time decoder predicts a
two-coordinate residual above the existing training-only role-ridge baseline.
Multiplication by forecast time makes the correction zero at the observed
endpoint. Output requests are decoded individually, including horizons longer
than 48 frames. No field clipping drops valid out-of-bounds measurements.

The objective is coordinate MSE, with equal expected weight per requested
coordinate across variable-length plays. Reporting uses the official
`sqrt(sum(dx² + dy²) / (2N))` in yards. The minibatch denominator uses the fixed
mean training row count so short plays do not receive systematic extra weight.

Lateral reflection transforms all positions, vector components, angles,
historical error vectors, baseline displacements and target residuals together.
The first experiment uses this exact field symmetry. Arbitrary rotations,
earlier forecast cutoffs, likelihood objectives and auxiliary losses remain
separate future experiments; they are not claimed as implemented here.

The design is informed by the competition's
[first-place writeup](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution),
and by joint forecasting work such as
[QCNeXt](https://arxiv.org/abs/2306.10508) and
[Wayformer](https://arxiv.org/abs/2207.05844).
This is a compact implementation inspired by those ideas, not a reproduction
of their architectures, training budgets or scores. Road-scene benchmark gains
do not establish an NFL score. Newer publication dates alone do not establish
better features.

## Bounded and reproducible execution

```bash
uv run --locked scripts/train_temporal.py --self-test
uv run --locked scripts/train_temporal.py --publish
uv run --frozen python scripts/evaluate_temporal.py --publish
```

The last command requires the hash-verified original development error artifact.
Research data and checkpoints remain private. `train_temporal.py` has a separate
Python 3.11/PyTorch 2.8 CPU lock; the submitted tree notebook is unchanged.

One seed, width 96, batch size 64 plays, 40 epochs, AdamW, a fixed cosine schedule
with three-epoch warmup, and EMA decay 0.98 are declared before fitting. The initial decoder saturated and was stopped after 24 epochs (319 seconds).
[The stability diagnostic](results/temporal_stability.json) preserves the failed
learning curve, activations, gradients and checkpoint identity. Final encoder
and decoder LayerNorm were added after this training-side diagnosis. The
corrected attempt has an 850-second limit, leaving both attempts within the
original 20-minute training budget. The limit is checked between batches. Epoch checkpoints
and a budget-stop checkpoint contain model, optimizer, EMA, batch cursor and RNG
state. Hash and source-signature checks reject stale or corrupted checkpoints.
Repeated execution resumes the same experiment; it does not start a hidden search.

The [checkpoint verification receipt](results/temporal_backup.json) records the
completed private S3 snapshot. All 69 new content objects and the snapshot manifest
were checked against local hashes, remote byte counts, and remote object ETags.
The snapshot extends the existing capacity-experiment backup and preserves the
trained checkpoint, prepared inputs, row-level errors, and stopped-run evidence.

The **final epoch EMA** is the predeclared model. Intermediate development scores
describe learning; the lowest displayed score does not select a checkpoint.
Evaluation compares exact forecast keys, the official metric, whole-game paired
bootstrap intervals, roles, forecast times and cold-player slices. Uncertainty
is conditional on this already-inspected development experiment. A single seed
cannot establish a reproducible competition-level gain.

A secondary 50/50 blend with the preserved tree predictions was declared while
the temporal run was still training, before its final score. This is one
exploratory comparison with fixed equal weights, not a blend optimized on
development. It adds no model training. Any ensemble promotion requires
training-side out-of-fold confirmation.

Tests cover exclusion of future coordinates, lateral reflection, shuffled and
missing observations, output horizons beyond 48 frames, player permutation,
padding-independent predictions and loss, opposite role-conditioned motion on
synthetic data (a constant bias cannot pass), and exact
optimizer/RNG recovery. The locked neural tests run in CI separately from the
base project's dependency environment.

## Promotion decision

The measured result is in [temporal_model.json](results/temporal_model.json),
with exact reference comparisons in
[temporal_evaluation.json](results/temporal_evaluation.json) and canonical
[notebook 02, section 9](../notebooks/02_motion_benchmarks.ipynb).
Even a lower development score requires confirmation on the three chronological
training-side folds with fold-local preprocessing. A new submission additionally
requires complete inference packaging and organizer-gateway validation.
The existing Kaggle private score remains 0.70090 until a new hidden-test run
establishes otherwise. Approximately 0.46 is the target, not a promised outcome.
