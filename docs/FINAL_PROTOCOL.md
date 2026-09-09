# Final refit and reserved evaluation

The feature gate is closed. This protocol records the next experiment before
fitting a final model or viewing reserved outcomes. The executable preparation
is `scripts/prepare_final.py`; its compact, verified receipt is
[final_protocol.json](results/final_protocol.json).

## Frozen choices

Combine the original 192 training games and 32 development games for the final
fit. Preserve the later 48 holdout games and their date boundary. The development
set has been inspected during research; it is eligible for the final fit but
cannot supply a new generalization estimate.

Use the validated shallow histogram boosting settings: 100 iterations, learning
rate 0.07, depth 4, 15 leaves, minimum leaf 60, L2 1, 63 bins, squared-error loss,
seed 2026, and no early stopping. Fit independent x/y residual models around a
newly fitted role-conditioned physical baseline. No additional parameter search
is part of this final experiment. Added complexity has no measured advantage
under the completed protocol.

Refit the ordered **6,308-column metadata-free schema** and its independently
fitted **5,572-column positional fallback**. The research trees' 953 and 981
active inputs are properties of those existing fits. Restricting a new fit to
those active inputs would silently change the frozen representation.

Refit all learned preprocessing on the combined training partition. In
particular, fit the physical baseline, 16 route components, and eight route
prototypes again. Recompute historical residual/count encodings from strictly
earlier dates, excluding every observation from the current date and using the
recorded smoothing of 20. Freeze history lookups for all evaluation callbacks;
the gateway need not request plays in chronological order. Preserve column names
and order without another screening or selection pass. Prune unused columns
only after fitting and verifying original/portable prediction parity.

## Preparation that has been implemented

Preparation rechecks the complete research evidence and feature-gate receipt.
It verifies the frozen model lineage, both availability schemas, original game
assignments, and estimator settings. The cached feature and baseline identifiers
must have identical row order and partition labels. Every forecast row and game
must agree with the data audit. Feature-bank entities must exactly match their
authorized targets. Missing weeks, duplicated games, incorrect labels, and any
holdout rows fail the review.

The final protocol binds the numerical source, dependency lock, research freeze,
raw pre-throw observations, training caches, original split, and input-review
receipt by SHA-256. It reads no raw output CSV or reserved input week during
preparation. It writes `artifacts/final/protocol.json` once. Repeating the same
command reuses the verified input review and preserves the protocol's bytes and
modification time. Changed inputs or a changed plan fail instead of silently
replacing a final experiment. An intentional protocol revision requires a
documented review and retention of the original evidence.

```bash
.venv/bin/python scripts/prepare_final.py --publish
```

This command verifies and publishes preparation only. The published status
distinguishes `prepared` from a fitted or evaluated model.

## Executed final preprocessing

`scripts/refit_final.py --publish` refits the physical baseline, strictly
earlier-date history encodings, and route representation on the frozen 224-game
partition. Each component has its own source-bound, checksum-verified checkpoint.
An interruption during route fitting preserves the completed baseline and history.
The command refuses to refit once `artifacts/final/model_seal.json` exists.

The completed fit covers 927,340 x/y coordinates and 38,080 player/play
trajectories. History tables contain 1,111 players and two roles. The route
encoder fits 16 components and eight prototypes. The first training date retains
the cold-history prior. The [preprocessing receipt](results/final_preprocessing.json)
records the frozen protocol, implementation hashes, training games, and output
hashes. Research-era fitted components remain preserved in their original paths.

The final fitter now consumes these components through verified stage receipts.
The validated research predictor and its development score remain the current
accuracy evidence. Full-scale final fitting has not yet run.

## Implemented final fitting

`uv run --locked scripts/fit_final.py --publish` preserves both ordered frozen
schemas and the validated sklearn 1.8.0 settings. It materializes one 6,308-column
matrix from the new baseline, earlier-date histories, and route encoder; the
5,572-column fallback uses its exact positional subset. Weekly identifiers,
complete training coverage, finite values, and coordinate signs are checked.
The full fit requires a 128 GiB worker; memory and storage checks run before the
large allocation.

Each profile's x and y estimators has an independent checksum-verified checkpoint.
Portable conversion checks every training prediction against the original
estimator before pruning inactive inputs. Export signatures include the actual
coordinate-model hashes, so recovering a changed model invalidates its export.
A completed rerun skips materialization. The immutable fit plan binds source,
locks, preprocessing, and environment; the final seal prohibits refitting.

Two separate executed checks support this implementation:

- `scripts/fit_final.py --self-test`: real sklearn fits on 1,024 synthetic rows;
  controlled interruption, preserved x checkpoint, recovered y, exact portable
  parity, and a repeat with no materialization. CI runs this in its locked environment.
- `scripts/fit_final.py --validate-data --publish`: raw-input reconstruction of
  both full frozen schemas on one complete play from each of the 15 training
  weeks: 738 forecast rows, with zero feature differences. The
  [feature validation receipt](results/final_feature_validation.json) binds this
  check to the current protocol and preprocessing.

These are software and feature-path checks. They do not constitute a final
463,670-row model fit or an out-of-time accuracy result. Full fitting, raw final
inference validation, prediction sealing, and reserved scoring remain.

## Evaluation rules recorded before fitting

Seal the final protocol, fitted models, inference source, and complete keyed
predictions before opening reserved outcomes. Resume a failed evaluation only
with the same sealed inputs. Preserve any error-correction trail; a software
failure cannot become permission to select another model using holdout scores.
The scoring and sealing implementation must enforce this rule before it runs.

Report the official coordinate RMSE, pooling squared x/y errors across every
forecast frame. Never average per-game or per-week RMSEs. Add frame- and
trajectory-weighted ADE, trajectory FDE, p95 displacement, coordinate MAE, and
descriptive role, horizon, and week slices. Compare the frozen final predictor
with the positional constant-velocity baseline and the refitted role baseline.
Do not select among models using these reserved comparisons.

Use a 2,000-resample game-cluster percentile bootstrap with seed 2026 for the
95% interval. It quantifies sampling uncertainty within the labelled 2023
season; it does not establish across-season generalization. Availability stress
checks cover complete inputs, missing metadata, missing telemetry, and cold
history. Verify the final package and standalone predictor through the
organizer's unchanged unlabelled gateway. Keep interface evidence separate from
accuracy and retain owner control over Kaggle submission.

Report the frozen model regardless of its reserved score. A disappointing
holdout result is a result to explain, not a reason to tune on the holdout.
