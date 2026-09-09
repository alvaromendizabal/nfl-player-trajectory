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

This command verifies and publishes preparation only. Final fitting, prediction
sealing, and reserved scoring are subsequent implementation milestones. The
published status distinguishes `prepared` from a fitted or evaluated model.

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
