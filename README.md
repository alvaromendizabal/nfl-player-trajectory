# NFL Player Trajectory

Predict player movement after the throw using observed motion, the supplied ball
landing point, and player roles. This project combines temporal validation,
interpretable motion models, executed research notebooks, and resumable experiments.

[![Quality](https://github.com/alvaromendizabal/nfl-player-trajectory/actions/workflows/ci.yml/badge.svg)](https://github.com/alvaromendizabal/nfl-player-trajectory/actions/workflows/ci.yml)

**Measured baseline result:** role-conditioned ridge achieved **0.9896 coordinate
RMSE (yards)** on **32 later games**, versus **1.7225** for constant velocity:
**42.6% lower RMSE**. The **48-game holdout remains unscored**. These are local
validation results, not Kaggle leaderboard scores or a state-of-the-art claim.

![Temporal validation benchmark](docs/results/benchmark.png)

## Review the work

Start with [01 · Data analysis](notebooks/01_data_analysis.ipynb), then
[02 · Motion benchmarks](notebooks/02_motion_benchmarks.ipynb). Both include rendered
outputs for GitHub readers; no cloud account or notebook execution is needed.
[00 · Project readiness](notebooks/00_project_readiness.ipynb) is optional orientation.

The research notebooks show coverage, frozen date boundaries, training distributions,
model comparisons, uncertainty, failure slices, and learned weights. They read local
benchmark artifacts when present; otherwise they explicitly label the published
experiment snapshot. They do not silently retrain models.

| Model | Coordinate RMSE / yd | ADE / yd | FDE / yd |
| --- | ---: | ---: | ---: |
| Role-conditioned ridge | **0.9896** | **0.8847** | **1.5057** |
| Constant velocity | 1.7225 | 1.5142 | 2.8696 |
| Constant acceleration | 1.9965 | 1.3073 | 2.6664 |
| Smoothed velocity | 2.0051 | 1.8581 | 3.3798 |
| Ball arrival | 4.0842 | 3.8607 | 5.6192 |
| Last position | 4.4806 | 4.4882 | 6.5526 |

All models score the same **67,857 player-frame positions**. The ridge model's
95% game-cluster bootstrap interval is **0.9218–1.0519 yards**. ADE in the table is
frame-weighted; FDE is trajectory-weighted. See the [model card](docs/MODEL_CARD.md),
[machine-readable results](docs/results/summary.json), and
[validation evidence](docs/VALIDATION.md).

## Model and evaluation

Six equivariant vector features combine terminal velocity, five-frame least-squares
velocity, terminal acceleration, and three polynomial-time terms pointing toward
the supplied landing point. Role-specific ridge weights use training-only RMS
scaling and fixed regularization. An unseen role falls back to the global training
fit. Shared x/y coefficients preserve rotation and translation equivariance.

| Partition | Dates | Games | Purpose |
| --- | --- | ---: | --- |
| Train | 2023-09-07–2023-12-03 | 192 | Feature scaling and coefficient fitting |
| Validation | 2023-12-04–2023-12-18 | 32 | Development comparisons and error analysis |
| Holdout | 2023-12-21–2024-01-07 | 48 | Final evaluation after model selection is locked |

The official metric is coordinate RMSE:

$$\mathrm{RMSE}=\sqrt{\frac{\sum_i[(\hat{x}_i-x_i)^2+(\hat{y}_i-y_i)^2]}{2N}}.$$

Diagnostics include frame- and trajectory-weighted ADE, trajectory-weighted FDE,
p95 displacement, coordinate MAE, and errors by role and forecast time. Confidence
intervals resample whole games 2,000 times; paired comparisons use the same sampled
games. Repeated development selection can make validation scores optimistic, which
is why the later holdout remains reserved.

## Reproduce in the existing project

```bash
git pull --ff-only origin main
python3 scripts/bootstrap.py
.venv/bin/nfl benchmark
.venv/bin/python scripts/notebooks.py --publish
.venv/bin/python kaggle/export.py --model role_ridge
.venv/bin/nfl backup
.venv/bin/nfl status
```

Bootstrap is a quality check, not real-data training. The benchmark computes or
verifies numerical results. `--publish` executes or verifies every notebook and
refreshes the **same canonical filenames** from completed local results. A failed
cell cannot replace the last successful output. Publication is local; reviewed
notebooks and aggregate results reach GitHub through a separate commit and PR.
See [run instructions](START_HERE.md) before updating a working tree with local edits.

Open `artifacts/benchmark/report.html` for the offline interactive field report.
Use **Python (NFL Trajectory)** for interactive notebook work in SageMaker. This
baseline runs on CPU; the pinned environment is recorded in `uv.lock`.

## Engineering evidence

Explicit tests cover numerical scoring, leakage boundaries, geometry, export parity,
notebook execution, publication integrity, and recovery. Ruff, mypy, warnings-as-errors
tests, and notebook execution run in GitHub Actions.

UTC JSONL logs include command/stage/cell elapsed times and a 15-second heartbeat.
Weekly features and sufficient statistics use atomic files and process locks.
Completed stages are reused only when signatures and output hashes match. Notebook
publication validates the complete candidate set before replacing files and records
a completion receipt; each file replacement is atomic, not the whole set at once.
Private content-addressed S3 snapshots retain data and recovery artifacts.

The recorded development run took **33.924 seconds**; a repeat reused **all 33
numerical stages** and took **5.166 seconds**, including report rendering. These are
historical measurements, not runtime guarantees. See
[latency measurements](docs/results/latency.json) and
[recovery instructions](docs/RECOVERY.md).

## Scope, next experiment, and attribution

The ridge model is the established learned baseline. Interaction-feature ablations
and stronger residual models come next, followed by temporal neural models with
optimizer, scheduler, and random-state recovery. See the
[research plan](docs/RESEARCH_PLAN.md). Improvements must be demonstrated on the
same development protocol rather than inferred from architecture names.

Source: Michael Lopez, Tom Bliss, Ally Blake, Yao Yan, Martyna Plomecka, and Addison
Howard. [NFL Big Data Bowl 2026 — Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction). Kaggle, 2025.

The exporter creates a standalone inference notebook with embedded learned weights.
Package/export parity is tested. **Official gateway execution and leaderboard
submission remain pending.** The competition ended; late-submission eligibility
must be checked in the authenticated account. This historical project cannot earn
a new medal. Hugging Face publication is reserved for a later documented release.

Raw NFL data, credentials, cloud identifiers, and private checkpoints stay outside
Git. Published results contain aggregate measurements, figures, and fitted
coefficients. The data page lists CC BY-NC 4.0; competition rules also apply. The
MIT license covers original project code, not NFL data or third-party code.
