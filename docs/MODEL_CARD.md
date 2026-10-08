# Model card

## Task and intended use

Predict post-throw player x/y trajectories for **NFL Big Data Bowl 2026 Prediction**. The organizer supplies observed tracking, player/play context, requested future frames and the ball landing point. That point is legitimate task input; the system does not claim to infer it in a live deployment.

This is a completed research and engineering portfolio, not a deployed coaching product or validated player-assessment system. Coordinate RMSE across both x and y, in yards, is the primary metric.

## Accepted system and results

The final system uses the preserved **20-model neural ensemble**, with temporal motion encoding, player interactions, static context, auxiliary motion supervision, augmentation and game-grouped validation. Public research is adapted with [attribution](SOURCES.md); the final private recipe and weights are withheld.

| Scope | Rows | Coordinate RMSE |
|---|---:|---:|
| Recorded private late submission 56928100 | Hidden | **0.46468** |
| Supported-population OOF, final policy | 561,607 | **0.4629258204** |
| Complete-population OOF, final policy | 562,936 | **0.5242026277** |

Local evaluation covers **272 games**. The broader population retains long-horizon and missing-role cases. Private scoring, local OOF and the synthetic demonstration are distinct measurements. [Results and provenance](RESULTS.md) explain the scopes and submission history.

## Validation and reliability

The research used game-heldout comparisons, fixed candidate definitions, matched controls, paired whole-game bootstrap uncertainty, role/horizon diagnostics and confirmation stages. Development evidence was inspected repeatedly. Epoch selection and repeated research decisions mean OOF is not an untouched-test guarantee.

Schemas, source hashes, finite-output checks, row-key alignment, prediction-time feature restrictions, checkpoint lineage and inference parity protect the delivery path. Checkpoints preserve each runner's declared resume state. Negative experiments remain recorded rather than being presented as accepted gains.

## Inference engineering

Shared preparation yielded **4.784× acceleration** for the fixed ensemble on **96 plays / 3,723 rows**, with **zero maximum coordinate difference**. The [aggregate receipt](results/frontier_submission.json) gives the timings. This is not an across-hardware latency guarantee.

The canonical research environment used AWS SageMaker and an NVIDIA L4, with mixed precision, profiling, resource checks, telemetry and recoverable artifacts. No production throughput or uptime commitment is claimed.

## Limitations

- Labeled competition data covers the 2023 season; across-season accuracy is not established.
- Difficult long-horizon and unsupported-role cases materially affect complete-population error.
- Competition inputs differ from unassisted live forecasting.
- Late private scores do not imply an official competition rank.
- The public release cannot reproduce the final neural score without the private model and licensed data.

## Public reproducibility

The [synthetic demo](REPRODUCIBILITY.md) verifies observation-only reference inference, game partitioning, keyed evaluation, coordinate RMSE and deterministic artifacts on invented data. It is independent of the private neural model.

Selected historical implementation and a small fitted baseline are already public. This closeout does not publish private champion weights, raw tracking, private prediction arrays, nonpublic execution bundles or the unpublished production recipe. The [data card](DATA_CARD.md) and [closeout](PROJECT_CLOSEOUT.md) explain the boundaries.
