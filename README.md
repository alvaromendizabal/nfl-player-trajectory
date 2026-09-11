# NFL Big Data Bowl 2026 - Prediction

Predict selected players' post-throw x/y trajectories from observed tracking,
the organizer-supplied landing point, player roles, and forecast horizon.
This project implements the [Prediction competition](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction),
not the separate Analytics competition.

**Performance objective: approach 0.46 private coordinate RMSE. The target remains unmet.**
The recorded Kaggle private score is **0.70090**. Internal research results below
are not substitutes for a new Kaggle evaluation or evidence of an official rank.

## Current milestone — 11 September 2026

**Engineering reconstruction is verified; feature and representation research remains open.**
The coordinate/velocity reconstruction, deterministic training loop, bounded
cloud runners, checkpoint recovery checks, and saved canonical notebooks are
being published as an engineering milestone, not a finished predictive model.
Publishing completed work does not close the feature-research gate.

| Evidence | Verified state |
|---|---|
| Tested engineering revision | `d265d9decb4ca5cd829781ceb78bd043480b5c35` |
| Full Quality workflow | [34567518582: passed](https://github.com/alvaromendizabal/nfl-player-trajectory/actions/runs/34567518582) |
| Real-data checkpoint recovery | Both arms resumed from step 3 to step 6 in fresh processes and matched clean checkpoints after private S3 read-back. |
| Last verified Studio synchronization | `nfl-trajectory-dev`: 307 tracked files matched the engineering revision at `2026-09-11T05:55:09Z`. |
| Saved canonical notebooks | 5/5, 9/9, and 27/27 code cells executed; zero stored error outputs. These are saved-execution checks, not a new scientific run. |
| Matched scientific experiment | `nfl-motion-scientific-20260911-055510-d265d9d`: launched on the tested revision; its terminal status and scientific result still require verification. |

The [milestone evidence](https://github.com/alvaromendizabal/nfl-player-trajectory/pull/27#issuecomment-5630177769),
[real-data recovery receipt](docs/results/motion_recovery_verification.json), and
[AWS prerequisite record](docs/results/supervision_aws_readiness.json) distinguish
software checks, engineering optimizer steps, and scientific fits. The earlier
recorded motion model with missing checkpoints has **not** been recovered by
this new implementation.

The last workspace receipt is a historical deployment observation, not a claim
that every later Git commit is already deployed. It also records one preserved,
untracked local receipt, `docs/results/supervision_execution.json`; no tracked
edits were discarded. This publication update changes the README, not the
pinned experiment's features, training settings, or checkpoint format.

## Next bounded decision: finish the existing comparison

Do not launch a duplicate run or a new feature sweep before inspecting the
existing job and independently restoring both final checkpoints. The frozen
comparison uses the same `inner_1` training/evaluation population, initialization,
batch order, reflections, optimizer exposure, and fixed final EMA for both arms.
Only the velocity auxiliary-loss treatment differs.

| Frozen experiment condition | Value |
|---|---|
| Training population | 4,951 plays, 94 games, 193,452 requested rows |
| Evaluation population | All 83,938 requested rows from 41 later games |
| Matched exposure | 16 epochs; 1,248 optimizer updates per arm |
| Velocity auxiliary weight | 0.0 for control; 0.1 for treatment |
| Normalization | Training-only shared-axis velocity RMS |
| Execution limit | One CPU Processing job with a 1,800-second hard runtime cap |
| Continuation rule | At least 1% lower coordinate RMSE **and** paired-game bootstrap upper 95% RMSE-difference bound below zero |

**When both arms finish:** verify matching forecast keys and complete row coverage,
recompute official coordinate RMSE from saved errors, apply the predeclared gate,
and publish the outcome and replay evidence. A passing first fold permits the
next predeclared chronological comparison; it does not establish robustness or
leaderboard improvement. A failed feature gate stops this exact treatment rather
than authorizing validation-driven retuning. An execution failure requires a
diagnosis and verified checkpoint-resume plan before further compute.

Acceleration supervision remains disabled pending the training-label tail audit.
Previously rejected smoothing and fixed soft-coverage treatments remain stopped.
See the [frozen configuration](configs/motion_supervision.json),
[experiment protocol](docs/MOTION_SUPERVISION_EXPERIMENT.md), and
[execution contracts](docs/MOTION_EXECUTION.md).

## Performance evidence — keep evaluation populations separate

| Evidence | Coordinate RMSE (yards) | Interpretation |
|---|---:|---|
| Recorded Kaggle private submission, Version 1 | **0.70090** | Successful after-deadline submission; no official competition rank. |
| Reserved temporal evaluation | **0.80467** | 99,266 rows from 48 later games; not the Kaggle test population. |
| Replayable earlier domain/tree blend | **0.64160** | Three reused chronological research folds. |
| Historical motion-supervision/tree blend | **0.62708** | Recorded result, but six newest checkpoint sets and associated source changes remain missing; not a replayable release. |
| Fixed soft-coverage correction probe | **0.66895** | One reused chronological fold; failed its declared continuation gate. |

The soft-coverage probe improved its matched control from 0.67206 to 0.66895
in 21.8 seconds across three fixed ridge fits. Its 0.463% primary gain missed
the declared 0.5% threshold and its paired-game interval crossed zero. The
negative result rejects this correction interface, not every possible learned
player-relationship representation.

The reserved evaluation was opened only after models, protocol, and predictions
were sealed in private S3. It has now been inspected and must not be described
as an untouched holdout for subsequent research. Detailed results and limitations
are in the [submission record](docs/results/kaggle_submission.json),
[reserved evaluation](docs/results/final_evaluation.json),
[current feature inventory](docs/FEATURE_STATUS.md), and
[recovery status](docs/RECOVERY_STATUS.md).

## Feature engineering with attributable evidence

The tabular bank contains **7,999 candidate columns across 20 families**. Counts
refer to candidates, not distinct raw signals or proven useful features. Every
fitted transform and screen belongs inside its training fold. Historical target
statistics exclude the entire current game date; validation lookups stay frozen.

| Matched research comparison | Reference RMSE | Treatment RMSE | Scope |
|---|---:|---:|---|
| Fixed shallow boosting: landing representation to screened wide inputs without metadata | 0.80120 | **0.68805** | 14.12% feature gain at identical estimator settings on the same development rows. |
| Attention without/with chronological historical target statistics | 0.77414 | **0.72063** | Three matched chronological folds; one seed and reused research data. |
| End-to-end motion control / added smoothed state | 0.71745 | 0.71631 | Failed the declared feature gate despite a small numerical gain. |

Research continues on compact temporal motion representations, learned changing
defender/receiver relationships, role-conditioned arrival behavior, long-horizon
parameterization, and strictly observed forecast-origin augmentation. Historical
NFL data requires explicit rule, license, season, event, role, and evaluation-game
alignment checks before use. These are open hypotheses, not implemented or
validated improvements merely because they appear in the plan.

The [domain evidence and coverage ledger](docs/DOMAIN_RESEARCH.md) records
mechanisms, inference-time availability, completed ablations, rejected treatments,
and remaining gaps. No feature family is declared exhausted based on column
count alone. Experiments change one attributable factor at a time and retain
negative results.

## Review the project

| Notebook | What to inspect |
|---|---|
| [00 · Project readiness](notebooks/00_project_readiness.ipynb) | Runtime and artifact evidence; separation of synthetic checks from private-data work. |
| [01 · Data and football hypotheses](notebooks/01_data_analysis.ipynb) | Observed tracking, information-time boundaries, and domain hypotheses. |
| [02 · Motion benchmarks and research](notebooks/02_motion_benchmarks.ipynb) | Official metrics, controlled ablations, role/horizon diagnostics, Plotly views, and embedded figures. |

The canonical notebooks preserve the historical studies. They do not yet contain
a verified outcome for the newly launched coordinate-versus-velocity experiment.
They should be updated and executed from its saved evidence after result recovery,
without rerunning completed training solely to refresh presentation.

## Validation, preservation, and publication

The primary metric is `sqrt(sum(dx² + dy²) / (2N))`, in yards; lower is better.
ADE, FDE, tail errors, role/horizon slices, and paired whole-game confidence
intervals are diagnostics, not replacement selection metrics. Every requested
row, including long forecasts, remains in the primary evaluation.

Work proceeds as research → hypothesis → implementation → tests → bounded run →
inspection → decision → verified durable checkpoint → publication. Each milestone
records attempted/completed work, actual metrics, failures, elapsed time, artifact
hashes, CI, repository revision, AWS deployment revision, and the next stop/continue
decision. A merge publishes an accepted milestone; it does not certify model
quality, close feature research, deploy code automatically, or authorize unlimited
compute.

The Python 3.11 project uses locked dependencies, UTC progress logs, atomic writes,
source/input hashes, and private content-addressed S3 checkpoints. CI checks lint,
formatting, types, warnings as errors, notebook execution, numerical contracts,
and interruption/recovery behavior. The standalone AWS bootstrap remains compatible
with its Python 3.9 base image before the locked runtime exists.

See [START_HERE](START_HERE.md), the [research plan](docs/RESEARCH_PLAN.md),
[execution discipline](docs/EXECUTION_DISCIPLINE.md),
[model card](docs/MODEL_CARD.md), and [validation record](docs/VALIDATION.md).

## Owner-controlled export and data terms

The final cell of notebook 02 defaults to `GENERATE_EXPORT = False`. Enabling it
exports the verified local final predictor; it does not submit to Kaggle. Export
and unlabelled gateway validation are separate from predictive accuracy and from
this ongoing representation study. No new submission is implied by publication.

Code is MIT licensed. Competition data has separate terms and is not redistributed.
[Sources and attribution](docs/SOURCES.md) document the evidence.
