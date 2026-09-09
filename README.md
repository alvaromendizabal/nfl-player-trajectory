# NFL Big Data Bowl 2026 - Prediction

Predict each selected player's post-throw x/y trajectory from observed tracking,
the organizer-supplied landing point, player roles, and forecast horizon.
This implements the [Prediction competition](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction).
The repository name describes its target; the separate Analytics competition is
not this project's entry point.

**Feature research, with measurable attribution:** 7,999 candidates across 20
families; training-only screening; three chronological inner folds; strict family
ablations; fixed-estimator comparisons; and explicit input-availability contracts.

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

The feature gate remains open pending the complete wide-group review and a
targeted combined omission of two consistently weak direct groups. The complete bank contains 7,999
candidates; 6,385 survive development-training screening and redundancy removal.
The selected metadata-free refit uses 6,308 columns. Its trees use **953 active
inputs**, which are exported without changing any predictions. These are distinct
counts: screening, refitting, and lossless inference pruning serve different purposes.

Raw inference reproduces **0.68805 RMSE on all 67,857 development frames**.
Removing metadata or clearing player history leaves predictions unchanged.
The independently fitted positional fallback scores **0.69396**, using 981
active inputs. The same predictor passes the organizer's unlabelled gateway on
5,837 requested rows across 143 plays, with exact package/standalone parity.
There is no final-model or leaderboard claim.

## Validation and engineering

Games remain intact. Training contains 192 games; development contains 32 later
games; the final 48 games remain unscored. Each chronological inner fold refits
the physical baseline, historical encodings, route representations, screening,
scaling, and estimator. Target histories exclude the entire current date, and
evaluation histories stay frozen. One labelled season does not establish
across-season generalization.

The official metric is `sqrt(sum(dx² + dy²) / (2N))`. ADE, FDE, p95 displacement,
role/horizon slices, and paired game-cluster intervals supply additional context.
Inference replay checks fresh raw features against saved experiment predictions.

The locked Python 3.11 project has **237 automated tests** at the current research
milestone, including leakage, geometry, missing-input dependencies, artifact
integrity, recovery, and standalone parity. CI checks lint, formatting, types,
warnings as errors, and notebook execution. Structured UTC logs, atomic writes,
locks, source/input hashes, and content-addressed S3 checkpoints make long work
auditable and resumable. Numerical diagnostics have separate pinned script locks.

See the [research plan](docs/RESEARCH_PLAN.md), [model card](docs/MODEL_CARD.md),
[validation record](docs/VALIDATION.md), and [reproduction guide](START_HERE.md).

## Owner-controlled export

The final cell of notebook 02 defaults to `GENERATE_EXPORT = False`. Enabling it
exports the current verified local research predictor and provides a download.
It never submits to Kaggle. Quality checks use a separate output directory.

The organizer's sample gateway test is reported separately from accuracy and
the offline quality suite. Unlabelled sample predictions are not hidden-test
scores. Code is MIT licensed; competition data has separate terms and is not
redistributed. [Sources and attribution](docs/SOURCES.md) describe the evidence.
