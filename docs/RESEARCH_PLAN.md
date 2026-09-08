# Research plan

| Phase | Deliverable | Exit gate |
| --- | --- | --- |
| 0 | Tested environment, ingestion, schema audit, metrics, recovery, notebook | All offline checks plus real-data audit and backup |
| 1 — complete | Training EDA, animated play, frozen temporal protocol, five physical baselines and role-conditioned ridge | 0.9896 validation RMSE; tested recovery; published notebooks and report |
| 2 | Physics-informed features and strong tree/residual regressors | Validation improvement with ablations and latency measurements |
| 3 | PyTorch temporal model with player interactions, then attention/graph alternatives | Full optimizer/scheduler/RNG checkpoint restore test, temporal CV, stable training |
| 4 | Hyperparameter research, seed ensembles, uncertainty, robustness | Locked model selection, game-cluster intervals, held-out evaluation |
| 5 | Self-contained Kaggle inference notebook and employer-facing results | Official local gateway pass, resource limits, reproducible model card and report |

This is a research path, not a promise of gold-medal performance. Winning approaches
will be studied and credited. Every new model must beat the reference under the same
data and validation contract. Public leaderboard comparisons require equivalent data
and evaluation settings. Do not compare a local holdout number directly with a public
leaderboard score as proof of rank.

Hugging Face is deferred until a trained, documented model exists and the terms for
sharing weights are reviewed. No Hub repository, GPU job, or public dataset is needed
to start Phase 0. No managed endpoint or always-running service is required.

## Validation and leakage boundaries

- The prediction unit is `(game_id, play_id, nfl_id, output frame_id)`.
- The input ends at the throw; output frame numbering restarts at 1.
- All players, plays, and frames from a game stay in one partition.
- The locked game-date boundaries end training on December 3, 2023, validation on
  December 18, and reserve December 21–January 7 for holdout. See the protocol JSON.
- Do not tune repeatedly on the holdout. Freeze a manifest before model selection.
- Fit scalers, imputation, learned encodings, feature selection, and calibration only
  inside the training portion of each fold. Aggregate historical player features using
  strictly earlier games, with cold-start handling.
- Random frame splits leak neighboring positions and overstate performance.
- The organizer explicitly supplies target receiver, ball landing location, and forecast
  horizon. They are allowed competition inputs. A real-time system that does not know
  the landing location is a different prediction problem and needs separate experiments.
- Supplementary play metadata is audit-only until each candidate feature's timing is
  documented. Post-play outcomes, future positions, and derived labels cannot become inputs.
- Horizontal reflection/rotation must transform positions, vectors, angles, and landing
  coordinates consistently, then restore coordinates before scoring. Test invariance
  before introducing augmentation.
- A player appearing in multiple games is expected. Add a player-disjoint robustness
  analysis if claiming generalization to unseen players; temporal validation alone does
  not establish that claim.

## Metric and figure standards

Report coordinate RMSE (official), ADE/FDE, p95 error, errors by horizon and role,
game-cluster uncertainty, per-play inference latency, total training time, memory,
and AWS billable resources. Classification accuracy, F1, and AUC are not applicable.
Use animated field trajectories, error-versus-time plots, role comparisons, and clear
model ablations. Every figure identifies its split and units. Interpretability must
distinguish model attribution from causal claims.

## Git standards

Use conventional commit titles (`feat:`, `test:`, `docs:`, `refactor:`). Change the
canonical file in place. Do not create alternate `fixed`, `repair`, or numbered patch
files. Keep `main` deployable; use a feature branch per phase. PRs explain why, behavior,
tests, results, limitations, and resume behavior. Merge after CI succeeds, then pull
`main` into Studio. Branch protection is an account-side step after the repository exists.

## Decision after the completed feature experiment

Landing-aware residual ridge achieved 0.9268683892 RMSE on the frozen validation
set, versus 0.9895687826 for role ridge. The interaction model reached 0.9421978915,
but replaced 23 of the landing model's 64 columns. It even removed
`fraction__lateral_speed`, whose standardized lateral coefficient is large in the
landing fit. This observation is a **hypothesis about selection interference**,
not a causal attribution of the metric difference.

Next, retain the landing champion and its entire representation; train a small
additive correction from interaction signals. Compare no correction, a
family-balanced correction, and a role-conditioned correction. Select budgets and
regularization in chronological training-only folds, then compare on the frozen
development validation with paired game-cluster uncertainty. Keep the holdout
unscored. Do not select individual interaction features by their validation error.

The existing univariate screen optimizes individual associations, not conditional
incremental information. Thirty-one selected landing terms derive from `ball_ux`;
family redundancy and temporal feature stability merit evaluation. Defensive
coverage (89.0% of squared error) and forecast seconds two/three (80.2%) are the
priority error regimes. Fourth-second results have only 127 rows.

External evidence: the competition's third-place writeup describes pre-training
and fine-tuning with a small trusted feature set. This is research context, not a
reproduced implementation or score:
https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/3rd-place-solution

The additive, role-conditioned, and subsequent nonlinear challengers have not been
run in this update. Notebook 02 records this decision alongside the measured data.
