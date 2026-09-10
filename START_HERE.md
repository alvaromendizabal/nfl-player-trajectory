# NFL Big Data Bowl 2026 - Prediction

This is the [Prediction project](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction). The canonical notebooks are 00, 01, and 02;
the repository name describes the player's trajectory prediction target.

**Current objective: approach 0.46 private RMSE.** The recorded Kaggle result is
**0.70090**. The new temporal research has not been submitted to the hidden test.

**Latest learned-motion experiment:** all six matched fits completed. Motion
supervision scores **0.66029**, versus **0.69635** for the equally
trained position-only control on three chronological internal folds: a
**5.18% reduction**. The declared primary gate **passes**.
Read the [executed results](docs/MOTION_SUPERVISION_RESULTS.md),
[extended domain research](docs/MOTION_LEARNING_RESEARCH.md), and notebook 02
section **14**. **The feature gate remains open and 0.46 is not reached.**

**Previous matched motion experiment:** six end-to-end continuation fits are
complete. Smoothed-state inputs score **0.71631**, versus **0.71745**
for the equally trained control on the same three internal folds. The declared
feature gate **fails**. This experiment separates extra training from feature
effects; it does not replace the stronger prior correction result. Read the
[measured result](docs/REPRESENTATION_RESEARCH.md) and notebook 02 section **13**.
The role audit preserves four distinct observed categories despite an outdated
slot name. **Feature research remains open; no new Kaggle submission was made.**

**Previous domain checkpoint:** the new 330-candidate domain screen tests motion,
arrival feasibility, coverage dynamics and field geometry under identical model
capacity. Explicit motion helps across all three chronological folds; coverage
summaries hurt this frozen representation. The predeclared all-feature/tree
blend scores **0.64160**, compared with the previous blend's **0.67866** on the
same internal rows. The standalone all-feature model fails its consistency gate.
Open [the domain review](docs/DOMAIN_RESEARCH.md) for the sources, attribution,
and unfinished feature avenues, then notebook 02 sections **11–12** for the
executed evidence. This is progress within feature research, not completion or
a new Kaggle result.
The 30-fit motion follow-up identifies **smoothed observed state** as the
subfamily that helps across all three folds in both addition and removal.
Its complete-motion/tree blend scores **0.64409**; all 60 feature comparisons
remain visible rather than selecting the smallest observed arm.

## Review the evidence

Open [notebook 02](notebooks/02_motion_benchmarks.ipynb) for feature attribution,
then [notebook 01](notebooks/01_data_analysis.ipynb) for data and feature rationale.
Public aggregates support review without private tracking or cloud credentials.
All 15 original tree feature-research criteria pass, and that feature/selection
manifest is frozen. The temporal model has its own open feature gate. Final
preprocessing and both residual-tree profiles have been fitted
on all 224 authorized games. The sealed reserved-holdout evaluation is complete:
**0.80467 coordinate RMSE in yards** on 99,266 forecast frames from 48 later
games (95% game-bootstrap interval: 0.66313–1.00113).

The [final evaluation](docs/results/final_evaluation.json),
[completed AWS fit](docs/results/final_training_cloud.json), and
[notebook publication receipt](docs/results/final_notebooks.json) record the
completed stages. All three canonical notebooks are executed. The
[final export check](docs/results/final_export.json) records exact parity on
5,837 organizer sample predictions and successful checkpoint reuse.

The development score of **0.68805** belongs to feature research; it is not the
reserved-holdout score. Historical research and training receipts retain their
original `holdout_evaluation: not_run` values to preserve the sequence of
evidence. Use the separate final evaluation receipt for the current result.

The final notebook was submitted to Kaggle on September 9, 2026 as a late
submission. Its private, offline CPU run completed all 5,837 organizer sample
predictions. The [Kaggle submission record](docs/results/kaggle_submission.json)
identifies Version 1 and the exact source and notebook hashes. The hidden-test
rerun **succeeded**, with **0.70090 private coordinate RMSE in yards**.
The submissions list's **0.00000 public score** is a separate field. Open the
[submissions page](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/submissions),
click the Version 1 submission row, and read **Private score** in Submission
Details. The [official leaderboard](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/leaderboard)
states that private scoring uses all test data. The public zero's internal cause
is not established by the published scorer; it is not evidence of a failed run.
This after-deadline result does not establish an official competition rank.

Training, reserved evaluation, export, hosted validation, and Kaggle hidden-test
scoring are complete. The reproduction commands below document completed work;
they are not instructions to rerun training merely to review the project.

**The first temporal model has been trained and evaluated.** On the same 67,857
development rows, it scores **0.69056**, compared with the existing tree's
**0.68805**. A fixed 50/50 exploratory blend reaches **0.65567** (4.71% lower
RMSE). The neural model alone has not earned replacement of the tree. The
[comparison](docs/results/temporal_evaluation.json) and
[protocol](docs/TEMPORAL_MODEL.md) preserve this distinction. No new model has
been submitted; the Kaggle private result remains **0.70090**.

**The next six-fit study is complete.** Notebook 02 section **10** reports all
three chronological training-side folds, with fold-local baselines and encodings.
Across 98 evaluation games and 202,361 forecast rows:

| Model | Pooled coordinate RMSE |
|---|---:|
| Attention without the six target-derived statistics | 0.77414 |
| Attention with the statistics | 0.72063 |
| Preserved tree | 0.70260 |
| Fixed equal tree/attention blend | **0.67866** |

The statistics help every fold and reduce pooled attention RMSE by **6.91%**.
The blend reduces pooled tree RMSE by **3.41%**, with a paired 95% difference
interval of **−0.03141 to −0.01610 yards**. The later third fold is weaker:
attention scores 0.79213 versus the tree's 0.73530, and the blend's 0.73019 has
an interval crossing zero. Attention alone does not earn replacement of the tree.
The [declared study gates](docs/TEMPORAL_RESEARCH.md) pass for target-statistic
retention and further blend inference research; those gates do not establish
deployment readiness. Other feature ablations, seed replication and inference validation
remain open. These reused folds and one seed establish conditional evidence.

Notebook 02 section **9, “Temporal attention challenger,”** presents this new
experiment separately from section 8's completed historical refit and holdout.
The run completed 40 epochs in 8.9 minutes. A first attempt was stopped after
5.3 minutes when its decoder saturated; its diagnostic and checkpoint are
preserved. Both attempts remained inside the 20-minute training budget.
Checkpoint reuse reproduced the model and predictions exactly without another
training epoch.

## Where training and evaluation happen

Notebook 02 combines feature research with the final model report. Its **section
8, “Final refit and reserved evaluation,”** shows the completed fit and the
48-game holdout results. Its number does not indicate an unfinished training
stage. The resource-intensive fitting runs through the Python scripts below on
AWS; opening the public notebook reviews their verified results.

| Stage | Executed implementation | Result |
|---|---|---|
| Final preprocessing | [refit_final.py](scripts/refit_final.py) | Baseline, histories and route representation refitted on 224 games |
| Final model training | [fit_final.py](scripts/fit_final.py) | Four HistGradientBoostingRegressor fits: x/y for two input-availability profiles |
| Reserved evaluation | [evaluate_final.py](scripts/evaluate_final.py) | Frozen predictions scored on 48 later games: 0.80467 RMSE |
| Submission notebook generation | [final_export.py](kaggle/final_export.py) | Embeds the trained parameters and prediction code; no retraining or AWS connection at prediction time |
| Kaggle hidden-test scoring | [Submission record](docs/results/kaggle_submission.json) | Succeeded after deadline; private coordinate RMSE 0.70090 |

A local organizer-gateway pass checks the callback and output format. A saved
Kaggle run additionally checks the hosted environment. Only an actual
competition submission exercises Kaggle's hidden-test rerun and produces its
score. These are separate checks; a local pass does not establish a hidden-test
result.

The [published scorer](https://www.kaggle.com/code/metric/nfl-2025) and this
project both use `sqrt(sum(dx² + dy²) / (2N))`. Lower is better. Regression
tests reproduce all three numerical examples in the organizer's scorer.
The 0.80467 reserved score and 0.70090 Kaggle private score evaluate different
datasets; their difference is not a controlled model-improvement comparison.

## Reproduce in the existing project

Use the locked Python 3.11 environment created by `scripts/bootstrap.py`.
Private data, trained artifacts, and stage receipts belong in the existing
workspace or its verified S3 restore. Keep the canonical repository checkout;
do not create alternate notebook copies or overwrite owner exports.

The canonical research sequence is:

```bash
.venv/bin/nfl feature-research
.venv/bin/nfl context-research
.venv/bin/nfl representation-research
.venv/bin/nfl research-report
uv run --locked scripts/nonlinear_probe.py
```

Run each of the following scripts for `inner_1`, `inner_2`, `inner_3`, and
`development` via its `--fold` argument:

The full-bank budget and omission fits require **128 GiB RAM**. The cloud runner
checks the actual container memory limit before starting this phase. Reviewing
published notebooks and predicting individual plays have much smaller requirements.

```bash
uv run --locked scripts/ablate_features.py --fold inner_1
uv run --locked scripts/joint_feature_fit.py --fold inner_1
uv run --locked scripts/feature_budget.py --fold inner_1
```

Then attribute the selected width, convert its verified trees, replay raw
inference, and check the organizer's unlabelled sample before publishing:

```bash
uv run --locked scripts/feature_attribution.py
uv run --locked scripts/prepare_tree.py --self-test
uv run --locked scripts/prepare_tree.py
.venv/bin/python scripts/validate_research.py
# Run this command for each of inner_1, inner_2, inner_3, and development.
uv run --locked scripts/ablate_wide_features.py --fold inner_1
# Also run this recorded combined omission for each of the four folds.
uv run --locked scripts/feature_simplification.py --fold inner_1
uv run --locked scripts/validate_gateway.py
.venv/bin/python scripts/review_feature_gate.py
.venv/bin/python scripts/notebooks.py --publish
.venv/bin/python scripts/quality.py
.venv/bin/nfl backup
```

These commands document reproduction of the completed feature research.
The project's bounded SageMaker runner restores
checksum-verified inputs, excludes holdout tracking, resumes completed stages,
and checkpoints each full-bank fold and major phase. S3 snapshots reference content-addressed
objects; the canonical restore command verifies those hashes. Do not use a
checkpoint from changed numerical source as if it were current.

The main dependency lock is unchanged. The nonlinear and gateway scripts use
separate PEP 723 locks, invoked with `uv run --locked`. Raw model pickle files are
private, source-verified artifacts from this run, not files to load from strangers.

## Prepare the final experiment

After restoring the verified research artifacts:

```bash
.venv/bin/python scripts/prepare_final.py --publish
```

This checks the complete feature gate, all 224 refit games, cache row alignment,
both frozen schemas, and input fingerprints. It preserves an immutable protocol
and reuses its verified input review on a repeated run. It does not fit or score
the final model. The [final protocol](docs/FINAL_PROTOCOL.md) records the exact
refit and reserved-evaluation rules established before fitting and scoring.

Refit and checkpoint the final baseline, chronological histories, and route
encoder with:

```bash
.venv/bin/python scripts/refit_final.py --publish
```

This stage has been executed on the full training partition. Repeating it
reuses each verified component. The separately locked final fitter is now
implemented:

```bash
uv run --locked scripts/fit_final.py --self-test
.venv/bin/python scripts/fit_final.py --validate-data --publish
# Requires verified private preprocessing and a 128 GiB worker.
uv run --locked scripts/fit_final.py --publish
```

The numerical self-test and 15-week raw feature check have been executed.
The full-scale final fit completed on the verified AWS worker. Each coordinate fit and portable
conversion resumes independently; unchanged completed fits skip materialization.
These commands do not score the reserved holdout or create the owner's export.

## Reproduce the sealed final report

After restoring the final checkpoint with this exact source revision, these
commands validate and reuse completed stages:

```bash
.venv/bin/python scripts/evaluate_final.py seal
.venv/bin/python scripts/evaluate_final.py evaluate --publish
.venv/bin/python -c "from pathlib import Path; from nfl_trajectory.final_results import publish_final_results; publish_final_results(Path.cwd())"
.venv/bin/python -m nfl_trajectory.final_diagnostics
.venv/bin/python scripts/notebooks.py --publish
```

The first two commands were executed and their repeat verified against unchanged
artifact hashes and modification times. The result publisher validates fit,
inference, seal and score lineage. Diagnostics retain every error row; notebook
publication checks final reports and original research receipts before replacing
canonical outputs. The standalone final exporter is `kaggle/final_export.py`;
its organizer gateway and checkpoint reuse are verified.

The final holdout score is 0.80467 coordinate RMSE on 48 games. Development's
0.68805 remains a separate research result. Review the final notebook's sparse
long-horizon errors and uncertainty before interpreting either score.

## Export only when you choose

The final cell in notebook 02 is off by default. Enabling it checks the current
final bundle and creates `artifacts/kaggle/submission.ipynb` for your download.
The exporter resolves the latest verified inference artifact and checks its source
lineage. The exporter losslessly compresses fitted parameters and embedded source,
and rejects notebooks at or above Kaggle's 1 MB source limit before writing them.
Automated quality exports and sample gateway output live under
`artifacts/quality/` and cannot replace the owner's generated artifact.

The generated notebook uses the organizer inference interface and never submits
to Kaggle. A local sample Parquet is an interface check, not a hidden-test score.
No public model-hosting service is required to review the project.

The [data card](docs/DATA_CARD.md) explains the exact Prediction inventory,
season boundaries, supplied task information, and limits of the evidence.

## Reproduce learned motion supervision

Notebook 02 section **14** reads the verified public
[motion-supervision results](docs/MOTION_SUPERVISION_RESULTS.md). Reviewing the
notebook does not train these models. With the private parent inputs and
checkpoints restored, the locked numerical entry point is:

```bash
uv run --locked scripts/motion_supervision.py --self-test
uv run --locked scripts/motion_supervision.py
```

The second command verifies and reuses completed fits. A changed numerical
signature is rejected instead of overwriting them. The
[completed-fit recovery receipt](docs/results/supervision_recovery.json)
records an actual replay, including unchanged hashes and modification times.
The public [research extension](docs/MOTION_LEARNING_RESEARCH.md) distinguishes
available observations, auxiliary training labels and unresolved feature work.

## Reproduce the matched motion continuation

The [protocol](docs/REPRESENTATION_EXPERIMENT.md),
[result](docs/results/motion_representation.json), and
[recovery receipt](docs/results/representation_recovery.json) identify the exact
sources, inputs, six fitted models and completed-run replay. The following
commands are for reproduction after restoring the private artifacts. Reading
the executed notebook requires no raw tracking data or retraining.

```bash
uv run --locked scripts/motion_representation.py --self-test
uv run --locked scripts/motion_representation.py
uv run --frozen python scripts/audit_roles.py
```

The second command reuses completed fits whose signatures and output hashes
match. It trains missing work only and refuses changed inputs or source lineage.
The feature/checkpoint archive contains content-addressed private objects and a
manifest mapping their hashes to relative paths. Verify each object before
restoring it beside the declared source commit; reject unsafe paths or an
existing destination with different bytes. The preceding verified S3 input
snapshot supplies raw input data and the original frozen attention references.
Maintained source and public reports remain in Git.
