# NFL Big Data Bowl 2026 - Prediction

Predict each selected player's post-throw x/y trajectory from observed tracking,
the organizer-supplied landing point, player roles, and forecast horizon.
This implements the [Prediction competition](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction).
The repository name describes its target; the separate Analytics competition is
not this project's entry point.

**Feature research, with measurable attribution:** 7,999 candidates across 20
families; training-only screening; three chronological inner folds; strict family
ablations; fixed-estimator comparisons; and explicit input-availability contracts.

**Reserved evaluation:** **0.80467 coordinate RMSE in yards** on 99,266 forecast
frames from 48 later games (95% game-bootstrap interval: 0.66313–1.00113).
The protocol, fitted models, inference code, and all predictions were sealed and
verified in private S3 before outcomes were opened. This is a temporal holdout
result, not a Kaggle leaderboard score.

**Kaggle evaluation:** **0.70090 private coordinate RMSE in yards**, verified in
Submission Details for Version 1 on September 9, 2026. Kaggle reports
**Succeeded (after deadline)**. The submissions list displays a public score of
0.00000; the [official leaderboard](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/leaderboard)
states that the private leaderboard uses all test data. Use the private result
when reporting this submission's test accuracy. The
[submission record](docs/results/kaggle_submission.json) preserves both fields
and the hosted validation evidence. This late submission establishes no official
competition rank. Its test set differs from the reserved holdout above.

## Review in five minutes

Start with [02 · Why the features work](notebooks/02_motion_benchmarks.ipynb), then
[01 · Data and football hypotheses](notebooks/01_data_analysis.ipynb).
[00 · Readiness](notebooks/00_project_readiness.ipynb) distinguishes real evidence
from synthetic software demonstrations. The canonical notebooks contain tables,
Plotly views, and embedded static figures for GitHub review.

| Controlled comparison | Reference RMSE | Engineered RMSE | Interpretation |
|---|---:|---:|---|
| Fixed shallow boosting: landing 64 → screened wide representation without metadata | 0.80120 | **0.68805** | **14.12% improvement from features at identical estimator settings** |
| Same boosting settings: landing 64 → engineered union 250 | 0.80120 | 0.72802 | Earlier compact representation; 9.13% feature gain |
| Residual ridge: landing 64 → joint representation without metadata 236 | 0.92687 | **0.82226** | Expanded features with the same ridge regularization; joint fit |
| Sequential linear correction: core 128 → context 186 | 0.90045 | **0.86434** | Context adds useful conditional information |

All values are coordinate RMSE in yards on the same 67,857 development frames
from 32 games. The current tree comparison's paired 95% game-bootstrap RMSE
difference is **−0.12780 to −0.09973 yards**. Notebook 02 reports the
complete width search: 512, 1,024, 2,048, 4,096, and the entire eligible pool.
The final width increment improves pooled inner-fold RMSE by only **0.071%**,
with mixed fold results. Increasing width has reached the predeclared stopping
tolerance; the feature gate also requires group refits and inference checks.

## What the research demonstrates

Motion history alone leaves substantial error. Arrival feasibility, landing-relative
geometry, coverage relationships, and role-specific destination features supply
complementary signal. Train-only route components and geometric player-set pools
test learned and interaction representations without changing estimator capacity.
Metadata is weak in several controlled comparisons; individual selected columns
are less stable than the strongest families. Failed ideas and caveats remain visible.

The first interaction experiment replaced 23 of 64 landing columns. Its worse
score did not isolate the value of interactions. The expanded study corrects that
confounding with nested additions and removals without replacement.

**The feature-research gate is closed:** all 15 evidence criteria pass, including
the complete wide-group refits and the combined omission of two weak direct
groups. That combined removal improves pooled inner RMSE by only **0.170%**,
with mixed fold results, below the predeclared 0.5% follow-up threshold.
The complete bank contains 7,999
candidates; 6,385 survive development-training screening and redundancy removal.
The selected metadata-free refit uses 6,308 columns. Its trees use **953 active
inputs**, which are exported without changing any predictions. These are distinct
counts: screening, refitting, and lossless inference pruning serve different purposes.

Raw inference reproduces **0.68805 RMSE on all 67,857 development frames**.
Removing metadata or clearing player history leaves predictions unchanged.
The independently fitted positional fallback scores **0.69396**, using 981
active inputs. The same predictor passes the organizer's unlabelled gateway on
5,837 requested rows across 143 plays, with exact package/standalone parity.
These are research-phase scores, not final-model accuracy or leaderboard claims.

## Validation and engineering

Games remain intact. Training contains 192 games; development contains 32 later
games; the final model refits all 224 before the reserved 48-game evaluation. Each chronological inner fold refits
the physical baseline, historical encodings, route representations, screening,
scaling, and estimator. Target histories exclude the entire current date, and
evaluation histories stay frozen. One labelled season does not establish
across-season generalization.

The [official metric](https://www.kaggle.com/code/metric/nfl-2025) is
`sqrt(sum(dx² + dy²) / (2N))`, with lower values better. The project's metric
matches all three examples published in the organizer's scorer. ADE, FDE, p95 displacement,
role/horizon slices, and paired game-cluster intervals supply additional context.
Inference replay checks fresh raw features against saved experiment predictions.

The locked Python 3.11 project has **299 automated tests** in the final release, including leakage, geometry, missing-input dependencies, artifact
integrity, recovery, and standalone parity. CI checks lint, formatting, types,
warnings as errors, and notebook execution. Structured UTC logs, atomic writes,
locks, source/input hashes, and content-addressed S3 checkpoints make long work
auditable and resumable. Numerical diagnostics have separate pinned script locks.

See the [research plan](docs/RESEARCH_PLAN.md), [model card](docs/MODEL_CARD.md),
[validation record](docs/VALIDATION.md), and [reproduction guide](START_HERE.md).

## Final experiment and limitations

Feature definitions and refit columns are frozen in the
[selection manifest](docs/results/feature_freeze.json).
The [final protocol](docs/FINAL_PROTOCOL.md) now verifies and freezes the combined
224-game refit partition, both ordered availability schemas, input hashes, and
evaluation rules. Its [preparation receipt](docs/results/final_protocol.json)
records the actual inputs checked. Final preprocessing has now refitted the
physical baseline, chronological histories, and route encoder on all 224 games;
the [preprocessing receipt](docs/results/final_preprocessing.json) records their
coverage and hashes. Both residual-tree fits have now completed on AWS.
The [final-fit receipt](docs/results/final_fit.json) verifies 463,670 training rows
for each profile, 951 active inputs per exported model, and exactly matching
original/portable predictions. The scores above belong to the research phase;
the separate [reserved evaluation](docs/results/final_evaluation.json) is now complete.

The final fitter is implemented with independent coordinate checkpoints and
hash-bound portable conversion. Its real sklearn recovery test passes. A
[raw-input feature check](docs/results/final_feature_validation.json) covers 738
rows across all 15 training weeks and both complete frozen schemas, with zero
feature differences. The completed full-scale fit and all four coordinate
checkpoints are preserved in a verified private S3 snapshot.

| Frozen holdout comparison | Coordinate RMSE (yards) |
|---|---:|
| Final metadata-free profile | **0.80467** |
| Metadata omitted | 0.80467 |
| Telemetry omitted: positional fallback | 0.81821 |
| Cold player histories | 0.80469 |
| Refitted role-conditioned baseline | 1.07252 |
| Constant velocity | 1.81762 |

The final model improves on both reference baselines in all 48 games. The
24.97% reduction against role ridge combines features and estimator effects;
the **14.12% controlled development improvement** above isolates feature changes.
Final frame-weighted ADE is **0.59023 yards**, trajectory FDE **1.04068 yards**,
and p95 displacement **2.07112 yards**.

The holdout is harder than development: weekly RMSE is 0.96175, 0.71397, and
0.67802. Coverage players score 0.88805 versus 0.54145 for targeted receivers.
Only **0.28% of rows** occur beyond three seconds, yet they contribute **23.67%
of squared error**. One play contributes **24.40%**. Every row remains in the
headline metric and interval. Notebook 02 shows sample support and error mass
alongside the horizon curve; these are descriptive findings, not new selection
criteria. One labelled season does not establish cross-season robustness, and
there is no verified leaderboard rank.

## Owner-controlled export

The final cell of notebook 02 defaults to `GENERATE_EXPORT = False`. Enabling it
exports the current verified local final predictor and provides a download.
It never submits to Kaggle. Quality checks use a separate output directory.

The organizer's sample gateway test is reported separately from accuracy and
the offline quality suite. Unlabelled sample predictions are not hidden-test
scores. Code is MIT licensed; competition data has separate terms and is not
redistributed. [Sources and attribution](docs/SOURCES.md) describe the evidence.
