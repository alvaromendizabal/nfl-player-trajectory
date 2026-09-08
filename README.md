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
| Fixed shallow boosting: landing 64 → engineered union 250 | 0.80120 | **0.72802** | **9.13% improvement from features at identical estimator settings** |
| Residual ridge: landing 64 → joint representation without metadata 236 | 0.92687 | **0.82226** | Expanded features with the same ridge regularization; joint fit |
| Sequential linear correction: core 128 → context 186 | 0.90045 | **0.86434** | Context adds useful conditional information |

All values are coordinate RMSE in yards on the same 67,857 development frames
from 32 games. The tree comparison's paired 95% game-bootstrap difference is
−0.08342 to −0.06295 yards. The wider 512/1,024/2,048-column search is reported
separately in notebook 02; its outcome is not inferred from feature count.

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

The feature gate remains open pending the final width and robustness review.
There is no final-model or leaderboard claim. The current inference bundle is a
236-feature joint linear profile with a 212-feature positional fallback; the
stronger tree remains an explicitly labelled feature diagnostic until its own
inference path is validated.

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

The locked Python 3.11 project has **228 automated tests** at the current research
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
