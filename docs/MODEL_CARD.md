# Research model card

**Task:** post-throw x/y player trajectory prediction for NFL Big Data Bowl 2026
Prediction. Inputs are pre-throw tracking, player roles, the supplied ball landing
point, and requested forecast horizon. This is not a system that infers an unknown
ball landing point at release time.

## Measured research comparisons

| Representation and estimator | Development coordinate RMSE |
|---|---:|
| Original role-conditioned physical ridge | 0.9895688 |
| Original landing residual ridge, 64 features | 0.9268684 |
| Sequential core correction, 128 features | 0.900446 |
| Sequential context correction, 186 features | 0.8643361 |
| Joint linear profile without metadata, 236 features | 0.8222565 |
| Fixed shallow boosting, landing 64 | 0.8011972 |
| Same boosting settings, engineered union 250 | 0.7280160 |

The controlled tree comparison attributes a 9.13% RMSE reduction to the feature
representation at fixed estimator settings. It does not attribute the difference
between role ridge and a tree entirely to feature engineering. Wider-budget
results and their training-only choice are reported in notebook 02.

All development comparisons use 32 games and 67,857 frames; game-level uncertainty
and paired differences accompany the full reports. Training comprises 192 games.
Three chronological inner folds select feature variants. The final 48 games remain
unscored. One labelled season and repeatedly inspected development data limit
generalization claims. There is no verified leaderboard rank.

## Current inference artifact

The research bundle selects a 236-feature joint linear profile without metadata
using pooled inner-fold RMSE. Missing telemetry uses an independently fitted
212-feature positional profile. The bundle includes its physical baseline,
frozen historical tables, route representation, selected columns, scaling,
coefficients, training/evaluation game manifest, and source hashes.

The stronger fixed-tree experiments are separate diagnostics until their raw
feature and standalone inference contracts are validated. The exporter must not
advertise their score while returning linear predictions.

## Robustness and limitations

Historical target-derived features exclude the complete current date and freeze
evaluation lookup tables. Route representations are fitted inside each training
fold. Body age uses game date; observed-frame histories cannot cross the throw.
Optional-field dependency tests compare actual feature values, not just feature
names. Future x/y columns in a request are ignored as inputs.

Raw inference validation covers every development frame plus missing metadata,
missing telemetry, and cold player history. The organizer sample gateway checks
interface shape, ordering, finiteness, and package/standalone parity; it has no
labels. Sample-year diversity does not establish multi-season accuracy.

Permutation importance expresses conditional model reliance, not causal football
effects. Correlated features reduce individual identifiability. Metadata and
route-only corrections are weak in some comparisons; failed avenues are retained
in the research record. Private competition data and fitted artifacts are not
redistributed in the public repository.

## Intended review and next gate

Use the notebooks to assess football reasoning, leakage prevention, controlled
feature gains, engineering, and reproducibility. This is a research artifact,
not a certified production or player-evaluation system. Final model selection,
one-time reserved holdout evaluation, and final inference validation follow the
feature completion decision.
