# Next experiment, chosen from measured evidence

## Completed

The feature experiment is complete. Landing-aware residual ridge achieved 0.9269
coordinate RMSE versus 0.9896 for role ridge. Its 64 selected columns came from 2,843
candidates screened on training data. The same 32 validation games are retained,
and the 48-game holdout remains locked. The executed notebooks contain the evidence.

## Immediate controlled comparison

Protect the winning landing feature block. Add receiver/defender interaction signals
without silently deleting useful motion or landing columns. Include a capacity-matched
noninteraction control, and measure whether role conditioning improves defensive
coverage. Derive closest approach, relative velocity, time-to-arrival mismatch,
turning response, and nonlinear forecast-time behavior from observed inputs only.

Choose settings using forward-chaining folds within the training games. Fit every
preprocessing decision and feature selector inside each fold. Refit the original
baseline on that fold's training games too: existing residual caches from a baseline
trained on the full training partition are not out-of-fold data. Persist fold models,
selectors, signatures, and evaluation receipts so a retry does not repeat valid work.

This new comparison is planned, not implemented or measured by the report/export release.
The current `nfl features` command reproduces the completed three-ridge experiment.

## Temporal challenger

Represent observed player histories and relative player/landing geometry jointly.
Use time masks and player-presence masks, football-specific coordinate normalization,
and a residual head targeting x/y mean squared error to match coordinate RMSE.
A GRU/temporal encoder is a useful compute-control ablation; a spatiotemporal attention
model is the next interaction hypothesis. Preserve optimizer, scheduler, random state,
and data-order state in addition to weights for faithful interrupted-training recovery.

The competition's third-place writeup reports a spatiotemporal Transformer and
auxiliary losses. Wayformer studies attention-based fusion in driving motion prediction.
These motivate experiments; neither establishes a measured NFL improvement here.

- [Competition third-place writeup](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/3rd-place-solution)
- [Wayformer primary research](https://waymo.com/research/wayformer/)

## Advancement criterion

Compare identical validation keys using pooled coordinate RMSE, game-cluster paired
uncertainty, ADE/FDE, role/time error slices, and inference cost. Do not tune another
model by repeatedly peeking at the reserved holdout. After model selection and
feature rules are locked, evaluate the holdout once and report it regardless of outcome.
The user generates/downloads inference artifacts through notebook 02 and handles
Kaggle submission personally; a static sample CSV is not a hidden-test submission.
