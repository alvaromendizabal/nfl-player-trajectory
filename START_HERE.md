# Continue the completed benchmark

Preserve the successful baseline, notebooks, and S3 snapshot. The next modeling
command is `.venv/bin/nfl features --checkpoint-s3`, not another baseline training
run. It evaluates 2,843 candidates with training-only screening and three
residual-ridge ablations, retaining at most 64 features per challenger.

Before pulling this update, preserve your locally rendered notebook outputs:

```bash
cd "$HOME/nfl-player-trajectory" &&
git stash push -m "baseline notebook outputs before feature experiment" -- notebooks docs/results &&
git pull --ff-only origin main &&
python3 scripts/bootstrap.py &&
.venv/bin/nfl features --checkpoint-s3 &&
.venv/bin/python scripts/notebooks.py --publish &&
.venv/bin/nfl backup &&
.venv/bin/nfl status
```

Bootstrap checks the updated environment and tests; it does not retrain the baseline.
The stash preserves the previous render; do not pop it over newly generated
notebooks. Your existing S3 snapshot also preserves the successful baseline outputs.
Other source edits are not stashed or discarded; a conflicting pull stops safely.
The pipeline does not delete `data/`, `.state/`, or completed baseline artifacts.

Read `notebooks/01_data_analysis.ipynb`, then `notebooks/02_motion_benchmarks.ipynb`.
Watch for `feature_schema`, `feature_progress`, `features_selected`, and
`feature_experiment_completed`, plus the existing UTC heartbeat. Interrupted weekly
stages resume from verified completion receipts. The checkpoint is per stage, not
mid-matrix operation. Rerun the same modeling command after an interruption.

Publication validates feature model/report hashes, frozen split, source hashes,
baseline provenance, and completed notebooks before updating canonical files.
Generated changes are local until a reviewed commit/PR publishes `notebooks/` and
`docs/results/`. Do not stage `data/`, `artifacts/`, credentials, or logs.

The original role-ridge export already exists. New residual challengers are not yet
wired into the official Kaggle inference gateway; do not treat a feature experiment
as a submitted model or a leaderboard result.

`--checkpoint-s3` uses the existing private bucket in `aws.local.json` (or explicit
`--bucket`). It writes full-workspace snapshots after each prepared week, fitted
model, and feature report. The final backup also includes published notebooks.
Without this flag, checkpoints remain on the current persistent filesystem until
`nfl backup` runs. No new compute instance is created by these commands.
