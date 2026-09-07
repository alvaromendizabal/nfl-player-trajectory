# NFL Player Trajectory

Player motion prediction after a pass, built around reproducible experiments,
temporal validation, interpretable trajectories, and durable AWS artifacts.

**Status: Phase 0 foundation. No trained model, NFL validation score, or Kaggle medal is claimed.**

This project uses [NFL Big Data Bowl 2026 — Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction).
The competition's final submission deadline was December 3, 2025; results were
published January 6, 2026. Late-submission availability must be checked while signed
in. Historical benchmark work cannot earn a new competition medal.

## Start

See [START_HERE.md](START_HERE.md) for AWS, environment, Kaggle, and GitHub steps.
The first notebook is `notebooks/00_project_readiness.ipynb`. It contains an executed
synthetic example and an explicit real-data readiness check.

```bash
python3 scripts/bootstrap.py
.venv/bin/python scripts/authenticate.py
.venv/bin/nfl download
.venv/bin/nfl audit
.venv/bin/nfl backup
.venv/bin/nfl status
```

Use the `Python (NFL Trajectory)` kernel in SageMaker. Windows users can substitute
`.venv\Scripts\python.exe` and `.venv\Scripts\nfl.exe` for the Linux paths.

## What Phase 0 delivers

- Python 3.11, exact dependency pins, and a complete `uv.lock`.
- Unit and integration tests, Ruff, mypy, executed notebook checks, and GitHub Actions.
- UTC JSONL events, elapsed time, and a 15-second heartbeat during long commands.
- Atomic state, process locks, content hashes, and source/environment fingerprints.
- Official Kaggle browser approval, saved-session reuse, and paginated per-file downloads, with safe ZIP handling.
- Weekly data audits, unique row keys, scored-player and forecast-horizon checks.
- A reproducible game-date split manifest with whole-game isolation.
- Constant-velocity reference implementation and numerical metric tests.
- Offline interactive trajectory report with clearly labeled synthetic data.
- Private content-addressed S3 snapshots and checksum-verified restore.

## Evaluation

The official coordinate RMSE is

\[
\mathrm{RMSE} = \sqrt{\frac{1}{2N}\sum_{i=1}^{N}[(x_i-\hat x_i)^2+(y_i-\hat y_i)^2]}.
\]

We also report frame-weighted and trajectory-weighted average displacement error,
trajectory-weighted final displacement error, 95th-percentile displacement, and
coordinate MAE, all in yards. Classification metrics such as F1 or ROC AUC do not
measure this continuous trajectory prediction task.

Later experiments will report errors by forecast horizon, role, direction, and
season, plus game-cluster bootstrap confidence intervals, inference latency,
runtime, and compute cost. Pool squared errors and row counts across partitions
before taking the square root; do not average fold RMSEs as the global metric.

## Structure

| Path | Purpose |
| --- | --- |
| `src/nfl_trajectory/` | Reusable implementation |
| `tests/` | Numerical, data, restart, and recovery contracts |
| `notebooks/00_project_readiness.ipynb` | Executed introduction and readiness review |
| `kaggle/` | Exporter for a self-contained baseline inference notebook |
| `scripts/` | Environment, quality, notebook, and authentication entry points |
| `docs/` | Methodology, phase plan, and evidence |
| `data/`, `artifacts/`, `logs/`, `.state/` | Generated local work; excluded from Git |

## Resuming

Rerun the same command after resolving a failure. Downloads reuse files only when
metadata and local hashes match. Interrupted downloads retain the official client's
partial file and identity marker; byte-range resume depends on server support.
Audits resume per weekly pair. Changed inputs, source, or `uv.lock` invalidate an
audit checkpoint. A heartbeat indicates that the process is alive; completed-file
and completed-pair events indicate actual progress.

Space storage survives stopping the app, but deleting a space is not a backup
strategy. Run `nfl backup` after each completed phase; keep the returned manifest
identifier. See [docs/RECOVERY.md](docs/RECOVERY.md). Neural epoch/batch checkpointing
will be implemented and tested with the training phase; no trainer exists yet.

## Attribution and scope

Source: Michael Lopez, Tom Bliss, Ally Blake, Yao Yan, Martyna Plomecka, and Addison
Howard. NFL Big Data Bowl 2026 — Prediction. Kaggle, 2025.

The data page lists CC BY-NC 4.0; competition rules also apply. This repository
contains original project code and synthetic examples. Raw NFL data, credentials,
cloud identifiers, and private checkpoints are excluded from version control.
Do not redistribute competition data or publish it to Hugging Face by default.
Code is MIT licensed; that license does not relicense the NFL data or third-party code.

