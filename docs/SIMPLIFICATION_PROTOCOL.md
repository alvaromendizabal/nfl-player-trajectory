# Combined feature omission protocol

This follow-up is recorded before fitting the combined omission. In the current
wide-profile refits, removing direct observed-history columns and removing generic
forecast crosses each slightly improved all three chronological inner folds.
Their individual gains are below the 0.5% high-value threshold. Test whether those
gains survive together before concluding that further simplification has small value.

Run exactly one combined omission: remove `history_lags`, `history_summaries`,
`robust_history`, and `forecast_interactions` from the selected metadata-free wide
profile. These are the existing `observed_histories` and `forecast_crosses` groups.
Keep every other selected column in its existing order, with no replacement.
Derived motion and horizon information in other families remains available.

Use the existing three inner folds and development, their fitted physical
baselines and training-only transforms, and the unchanged shallow-boosting
settings. Refit both coordinate regressors. Verify the parent prediction on
freshly materialized evaluation inputs before each fit. Preserve model, error,
and report receipts separately, with source/input hashes and per-fold backups.

Apply the existing stopping tolerance: pooled inner RMSE improvement of at least
0.5%, with no individual inner fold costing more than 1%, requires a smaller
representation follow-up and keeps the feature gate open. A lesser gain supplies
additional evidence of diminishing returns. Development metrics and paired game
intervals are descriptive; they do not select the representation. Do not search
other combinations in response to development results. The 48 reserved games
remain excluded.

The feature completion decision must include this study in addition to the
individual wide-group refits. This adds a requirement; it does not relax the
previously recorded width, availability, or group-removal tolerances.
